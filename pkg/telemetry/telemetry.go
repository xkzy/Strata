// pkg/telemetry/telemetry.go - System & Hardware Telemetry Sampler in Go
package telemetry

import (
	"bufio"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

const HistoryCapacity = 60

// drmRoot is where the fallback (non-NVIDIA) GPU path looks. It is a variable so a test can point a collector at
// a fake sysfs tree.
var drmRoot = "/sys/class/drm"

// gpuReading is one GPU sample. present is false when no driver reports a card at all (a card that fell off the
// bus, a wedged driver, a CPU-only host); hasTemp and hasPower are false when the driver does not report that
// value. Nothing here is invented: a value the driver does not give is absent, not zero.
type gpuReading struct {
	used, total       uint64
	util, temp, power float64
	hasTemp, hasPower bool
	present           bool
}

type Reading struct {
	Timestamp float64 `json:"timestamp"`
	CPU       float64 `json:"cpu"`
	RAM       float64 `json:"ram"`
	RAMTotal  float64 `json:"ram_total_gb"`
	RAMUsed   float64 `json:"ram_used_gb"`
	GPU       float64 `json:"gpu"`
	VRAM      float64 `json:"vram"`
	VRAMTotal float64 `json:"vram_total_gb"`
	VRAMUsed  float64 `json:"vram_used_gb"`
	Temp      float64 `json:"temp_c"`
	PowerW    float64 `json:"power_w"`
}

type TelemetryCollector struct {
	mu           sync.RWMutex
	running      bool
	ticker       *time.Ticker
	stopChan     chan struct{}
	prevCPUTotal uint64
	prevCPUIdle  uint64

	haveDisk                    bool
	prevDiskRead, prevDiskWrite uint64
	prevDiskAt                  time.Time
	igpu                        *drmGPU
	pcie                        [3]int // gen, max gen, width of the GPU link (0: unknown)

	// Static hardware facts
	static map[string]interface{}

	// Live current reading (raw units for Monitor tab: bytes, %, °C, W)
	now map[string]interface{}

	// Ring buffer histories for sparklines
	historyGPUUtil  []float64
	historyVRAMUsed []float64
	historyGPUTemp  []float64
	historyGPUPower []float64
	historyTokS     []float64
	historyRAMUtil  []float64
	historyCPUUtil  []float64
	historyIGPUUtil []float64
	historyDiskRead []float64
	historyPcieRx   []float64
	historyPrefill  []float64

	tokSource     func() float64 // decode speed of the running request (nil or 0: idle)
	prefillSource func() float64 // prefill speed of the running request (nil or 0: idle)
}

func NewTelemetryCollector() *TelemetryCollector {
	tc := &TelemetryCollector{
		stopChan:        make(chan struct{}),
		static:          make(map[string]interface{}),
		now:             make(map[string]interface{}),
		historyGPUUtil:  make([]float64, 0, HistoryCapacity),
		historyVRAMUsed: make([]float64, 0, HistoryCapacity),
		historyGPUTemp:  make([]float64, 0, HistoryCapacity),
		historyGPUPower: make([]float64, 0, HistoryCapacity),
		historyTokS:     make([]float64, 0, HistoryCapacity),
		historyPrefill:  make([]float64, 0, HistoryCapacity),
		historyRAMUtil:  make([]float64, 0, HistoryCapacity),
		historyCPUUtil:  make([]float64, 0, HistoryCapacity),
		historyPcieRx:   make([]float64, 0, HistoryCapacity),
	}
	if runtime.GOOS == "linux" {
		tc.igpu = discoverIGPU(drmRoot)
		gen, maxGen, width, _ := discoverDrmPCIe(drmRoot)
		if gen > 0 || maxGen > 0 || width > 0 {
			tc.pcie = [3]int{gen, maxGen, width}
		}
	}
	tc.detectStaticHardware()
	return tc
}

// Static hardware facts do not change while the process runs, and probing them costs one nvidia-smi fork (20 ms
// measured on the development PC) - or, on a host without one, an lspci fork - per collector. The tests build a
// server per test and a multi-model server builds one collector per model, so the probe runs once per process.
var (
	staticOnce  sync.Once
	staticFacts map[string]interface{}
	staticVRAM  uint64
)

func (tc *TelemetryCollector) detectStaticHardware() {
	staticOnce.Do(func() { staticFacts, staticVRAM = probeStaticHardware() })
	for k, v := range staticFacts {
		tc.static[k] = v
	}
	if tc.igpu != nil {
		tc.static["igpu_name"] = tc.igpu.name()
	}
	if staticVRAM > 0 {
		tc.now["gpu_mem_total"] = staticVRAM
	}
}

func probeStaticHardware() (map[string]interface{}, uint64) {
	threads := runtime.NumCPU()
	cpuName := "CPU"
	cores := threads / 2
	if cores <= 0 {
		cores = 1
	}

	if runtime.GOOS == "linux" {
		if data, err := os.ReadFile("/proc/cpuinfo"); err == nil {
			for _, line := range strings.Split(string(data), "\n") {
				if strings.HasPrefix(line, "model name") {
					parts := strings.SplitN(line, ":", 2)
					if len(parts) == 2 {
						cpuName = strings.TrimSpace(parts[1])
						break
					}
				}
			}
		}
	}

	gpuName := ""
	gpuCount := 0
	var vramTotalBytes uint64

	// 1. Try nvidia-smi
	if out, err := exec.Command("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits").Output(); err == nil {
		lines := strings.Split(strings.TrimSpace(string(out)), "\n")
		var names []string
		for _, line := range lines {
			parts := strings.Split(line, ",")
			if len(parts) >= 2 {
				name := strings.TrimSpace(parts[0])
				mib, _ := strconv.ParseUint(strings.TrimSpace(parts[1]), 10, 64)
				names = append(names, name)
				vramTotalBytes += mib * 1024 * 1024
			}
		}
		if len(names) > 0 {
			gpuName = strings.Join(names, " + ")
			gpuCount = len(names)
		}
	}

	// 2. Try AMD sysfs if no NVIDIA GPU
	if gpuName == "" {
		if data, err := os.ReadFile(filepath.Join(drmRoot, "card0/device/mem_info_vram_total")); err == nil {
			bytesVal, _ := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
			if bytesVal > 0 {
				vramTotalBytes = bytesVal
				gpuName = "AMD Radeon GPU (ROCm/HIP)"
				gpuCount = 1
			}
		}
	}

	// 3. Fallback to lspci if needed
	if gpuName == "" {
		if out, err := exec.Command("lspci").Output(); err == nil {
			for _, line := range strings.Split(string(out), "\n") {
				lower := strings.ToLower(line)
				if strings.Contains(lower, "vga compatible") || strings.Contains(lower, "3d controller") {
					if strings.Contains(line, "NVIDIA") {
						gpuName = "NVIDIA GPU"
						gpuCount = 1
						break
					} else if strings.Contains(line, "AMD") || strings.Contains(line, "Advanced Micro Devices") {
						gpuName = "AMD Radeon GPU"
						gpuCount = 1
						break
					}
				}
			}
		}
	}

	computeDevices := []string{}
	if gpuName != "" {
		computeDevices = append(computeDevices, gpuName)
	}

	return map[string]interface{}{
		"gpu_name":        gpuName,
		"gpu_count":       gpuCount,
		"compute_devices": computeDevices,
		"cpu_name":        cpuName,
		"cores":           cores,
		"threads":         threads,
		"psutil":          true,
	}, vramTotalBytes
}

// SetTokSource sets where each sample's tokens-per-second comes from (the server's running request).
func (tc *TelemetryCollector) SetTokSource(f func() float64) {
	tc.mu.Lock()
	tc.tokSource = f
	tc.mu.Unlock()
}

// SetPrefillSource sets where each sample's prompt tokens-per-second comes from (the server's running
// request). The web app draws history.prefill_tok_s_mean beside history.tok_s.
func (tc *TelemetryCollector) SetPrefillSource(f func() float64) {
	tc.mu.Lock()
	tc.prefillSource = f
	tc.mu.Unlock()
}

func (tc *TelemetryCollector) Start() {
	tc.mu.Lock()
	if tc.running {
		tc.mu.Unlock()
		return
	}
	tc.running = true
	tc.ticker = time.NewTicker(1 * time.Second)
	tc.mu.Unlock()

	go tc.sampleLoop()
}

func (tc *TelemetryCollector) Stop() {
	tc.mu.Lock()
	defer tc.mu.Unlock()
	if !tc.running {
		return
	}
	tc.running = false
	if tc.ticker != nil {
		tc.ticker.Stop()
	}
	close(tc.stopChan)
}

func (tc *TelemetryCollector) sampleLoop() {
	for {
		select {
		case <-tc.stopChan:
			return
		case <-tc.ticker.C:
			tc.sampleOnce()
		}
	}
}

func (tc *TelemetryCollector) sampleOnce() {
	cpuPct := tc.readCPU()
	ramUsedBytes, ramTotalBytes, ramPct, ramSystem := tc.readRAM()
	gpu := tc.readGPU()
	diskRead, diskWrite, diskOK := tc.readDisk()

	tc.mu.Lock()
	defer tc.mu.Unlock()

	tc.now["cpu_util"] = cpuPct
	tc.now["cpu"] = cpuPct // the name the web app reads
	tc.now["ram_used"] = ramUsedBytes
	tc.now["ram_total"] = ramTotalBytes
	tc.now["ram_util"] = ramPct
	// ram_system tells consumers whether these bytes are system memory or only this process's Go heap. The
	// fallback path must not be mistaken for available host RAM by an admission controller.
	tc.now["ram_system"] = ramSystem

	tc.publishGPU(gpu)

	if tc.pcie[0] > 0 || tc.pcie[1] > 0 {
		tc.now["gpu_pcie_gen"] = tc.pcie[0]
		tc.now["gpu_pcie_gen_max"] = tc.pcie[1]
		tc.now["gpu_pcie_width"] = tc.pcie[2]
		bw := PcieBandwidthGBps(tc.pcie[1], tc.pcie[2])
		tc.now["gpu_pcie_bw_gb_s"] = mathRound(bw, 1)
	}

	rxMB, txMB, okPcie := tc.readPcieThroughput()
	if okPcie {
		tc.now["gpu_pcie_rx_mb"] = mathRound(rxMB, 1)
		tc.now["gpu_pcie_tx_mb"] = mathRound(txMB, 1)
	}
	var pcieRxMB float64
	if v, ok := tc.now["gpu_pcie_rx_mb"].(float64); ok {
		pcieRxMB = v
	}
	tc.pushHistory(&tc.historyPcieRx, pcieRxMB)

	if diskOK {
		tc.now["disk_read_mb"] = mathRound(diskRead, 1)
		tc.now["disk_write_mb"] = mathRound(diskWrite, 1)
		tc.pushHistory(&tc.historyDiskRead, diskRead)
	}
	if tc.igpu != nil {
		r := tc.igpu.read()
		util, hasUtil := r["util"].(float64)
		if hasUtil {
			tc.now["igpu_util"] = util
		}
		for _, k := range []string{"mem_used", "gtt_used", "gtt_total"} {
			if v, ok := r[k]; ok {
				tc.now["igpu_"+k] = v
			}
		}
		tc.pushHistory(&tc.historyIGPUUtil, util)
	}

	tc.pushHistory(&tc.historyCPUUtil, cpuPct)
	tc.pushHistory(&tc.historyRAMUtil, ramPct)
	tc.pushGPUHistory(gpu)
	tokS := 0.0
	if tc.tokSource != nil {
		tokS = tc.tokSource()
	}
	tc.pushHistory(&tc.historyTokS, tokS)
	prefillS := 0.0
	if tc.prefillSource != nil {
		prefillS = tc.prefillSource()
	}
	tc.pushHistory(&tc.historyPrefill, prefillS)
}

func (tc *TelemetryCollector) pushHistory(slice *[]float64, val float64) {
	if len(*slice) >= HistoryCapacity {
		*slice = (*slice)[1:]
	}
	*slice = append(*slice, val)
}

// publishGPU writes one GPU reading into the current sample. gpu_mem_total is a hardware fact (it stays in the
// map when the live reading goes away); the rest are live readings. A value the driver does not report is
// absent, never invented: the Monitor draws "–" and the controller reads it as unknown. When no card answers at
// all - a fall-off, a wedged driver - the live values are removed rather than replaced by zero, which would read
// as "the GPU is empty". Call with tc.mu held.
func (tc *TelemetryCollector) publishGPU(gpu gpuReading) {
	if !gpu.present {
		delete(tc.now, "gpu_mem_used")
		delete(tc.now, "gpu_util")
		delete(tc.now, "gpu_temp")
		delete(tc.now, "gpu_power")
		return
	}
	tc.now["gpu_mem_total"] = gpu.total
	tc.now["gpu_mem_used"] = gpu.used
	tc.now["gpu_util"] = gpu.util
	if gpu.hasTemp {
		tc.now["gpu_temp"] = gpu.temp
	} else {
		delete(tc.now, "gpu_temp")
	}
	if gpu.hasPower {
		tc.now["gpu_power"] = gpu.power
	} else {
		delete(tc.now, "gpu_power")
	}
}

// pushGPUHistory appends one point per GPU sparkline. A reading the driver did not report keeps the last known
// value instead of drawing a dip to zero that would read as "the GPU went idle". Call with tc.mu held.
func (tc *TelemetryCollector) pushGPUHistory(gpu gpuReading) {
	point := func(h []float64, cur float64, known bool) float64 {
		if known {
			return cur
		}
		if len(h) > 0 {
			return h[len(h)-1]
		}
		return cur
	}
	_, utilKnown := tc.now["gpu_util"].(float64)
	_, vramKnown := tc.now["gpu_mem_used"].(uint64)
	_, tempKnown := tc.now["gpu_temp"].(float64)
	_, powerKnown := tc.now["gpu_power"].(float64)
	tc.pushHistory(&tc.historyGPUUtil, point(tc.historyGPUUtil, gpu.util, utilKnown))
	tc.pushHistory(&tc.historyVRAMUsed, point(tc.historyVRAMUsed, float64(gpu.used), vramKnown))
	tc.pushHistory(&tc.historyGPUTemp, point(tc.historyGPUTemp, gpu.temp, tempKnown))
	tc.pushHistory(&tc.historyGPUPower, point(tc.historyGPUPower, gpu.power, powerKnown))
}

func (tc *TelemetryCollector) readCPU() float64 {
	if runtime.GOOS != "linux" {
		return 0.0
	}

	file, err := os.Open("/proc/stat")
	if err != nil {
		return 0.0
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	if !scanner.Scan() {
		return 0.0
	}
	fields := strings.Fields(scanner.Text())
	if len(fields) < 5 || fields[0] != "cpu" {
		return 0.0
	}

	var total, idle uint64
	for i := 1; i < len(fields); i++ {
		val, _ := strconv.ParseUint(fields[i], 10, 64)
		total += val
		if i == 4 {
			idle = val
		}
	}

	if tc.prevCPUTotal == 0 {
		tc.prevCPUTotal = total
		tc.prevCPUIdle = idle
		return 0.0
	}

	deltaTotal := total - tc.prevCPUTotal
	deltaIdle := idle - tc.prevCPUIdle

	tc.prevCPUTotal = total
	tc.prevCPUIdle = idle

	if deltaTotal == 0 {
		return 0.0
	}

	usage := 100.0 * float64(deltaTotal-deltaIdle) / float64(deltaTotal)
	return mathRound(usage, 1)
}

func (tc *TelemetryCollector) readRAM() (usedBytes, totalBytes uint64, pct float64, system bool) {
	if runtime.GOOS == "linux" {
		if data, err := os.ReadFile("/proc/meminfo"); err == nil {
			var totalKB, freeKB, availKB uint64
			for _, line := range strings.Split(string(data), "\n") {
				parts := strings.Fields(line)
				if len(parts) >= 2 {
					if parts[0] == "MemTotal:" {
						totalKB, _ = strconv.ParseUint(parts[1], 10, 64)
					} else if parts[0] == "MemFree:" {
						freeKB, _ = strconv.ParseUint(parts[1], 10, 64)
					} else if parts[0] == "MemAvailable:" {
						availKB, _ = strconv.ParseUint(parts[1], 10, 64)
					}
				}
			}
			if totalKB > 0 {
				if availKB == 0 {
					availKB = freeKB
				}
				usedKB := totalKB - availKB
				totalBytes = totalKB * 1024
				usedBytes = usedKB * 1024
				pct = mathRound((float64(usedKB)/float64(totalKB))*100.0, 1)
				return usedBytes, totalBytes, pct, true
			}
		}
	}

	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.Alloc, m.Sys, 0.0, false
}

func (tc *TelemetryCollector) readGPU() gpuReading {
	// 1. The shared nvidia-smi query (see nvprobe.go: one probe per window serves every collector).
	if smp, ok := nvidiaGPUQuery(time.Now()); ok {
		tc.pcie = [3]int{smp.pcieGen, smp.pcieGenMax, smp.pcieWidth}
		return gpuReading{present: true, used: smp.usedBytes, total: smp.totalBytes, util: smp.util,
			temp: smp.temp, hasTemp: smp.hasTemp, power: smp.power, hasPower: smp.hasPower}
	}

	// 2. The fallback card: any DRM card that exposes its VRAM in sysfs (AMD and Intel).
	if r := readAMDSysfs(drmRoot); r.present {
		return r
	}

	// 3. Nothing answers - a card that fell off the bus, a wedged driver, or no GPU at all. The sample is
	// unknown, and the caller must publish it as unknown rather than as zero.
	return gpuReading{}
}

func (tc *TelemetryCollector) Snapshot() map[string]interface{} {
	tc.mu.RLock()
	defer tc.mu.RUnlock()

	nowCopy := make(map[string]interface{}, len(tc.now))
	for k, v := range tc.now {
		nowCopy[k] = v
	}

	staticCopy := make(map[string]interface{}, len(tc.static))
	for k, v := range tc.static {
		staticCopy[k] = v
	}

	historyCopy := map[string]interface{}{
		"gpu_util":           copySlice(tc.historyGPUUtil),
		"gpu_mem_used":       copySlice(tc.historyVRAMUsed),
		"gpu_temp":           copySlice(tc.historyGPUTemp),
		"gpu_power":          copySlice(tc.historyGPUPower),
		"tok_s":              copySlice(tc.historyTokS),
		"prefill_tok_s_mean": copySlice(tc.historyPrefill),
		"ram_util":           copySlice(tc.historyRAMUtil),
		"cpu_util":           copySlice(tc.historyCPUUtil),
		"cpu":                copySlice(tc.historyCPUUtil),
		"igpu_util":          copySlice(tc.historyIGPUUtil),
		"disk_read_mb":       copySlice(tc.historyDiskRead),
		"gpu_pcie_rx_mb":     copySlice(tc.historyPcieRx),
	}

	return map[string]interface{}{
		"now":     nowCopy,
		"static":  staticCopy,
		"history": historyCopy,
	}
}

func (tc *TelemetryCollector) GetCurrent() Reading {
	snap := tc.Snapshot()
	now := snap["now"].(map[string]interface{})
	r := Reading{
		Timestamp: float64(time.Now().UnixNano()) / 1e9,
	}
	if v, ok := now["cpu_util"].(float64); ok {
		r.CPU = v
	}
	if v, ok := now["ram_util"].(float64); ok {
		r.RAM = v
	}
	if v, ok := now["ram_total"].(uint64); ok {
		r.RAMTotal = float64(v) / 1024 / 1024 / 1024
	}
	if v, ok := now["ram_used"].(uint64); ok {
		r.RAMUsed = float64(v) / 1024 / 1024 / 1024
	}
	if v, ok := now["gpu_util"].(float64); ok {
		r.GPU = v
	}
	if v, ok := now["gpu_mem_total"].(uint64); ok {
		r.VRAMTotal = float64(v) / 1024 / 1024 / 1024
	}
	if v, ok := now["gpu_mem_used"].(uint64); ok {
		r.VRAMUsed = float64(v) / 1024 / 1024 / 1024
	}
	if v, ok := now["gpu_temp"].(float64); ok {
		r.Temp = v
	}
	if v, ok := now["gpu_power"].(float64); ok {
		r.PowerW = v
	}
	return r
}

func (tc *TelemetryCollector) GetHistory() []Reading {
	return []Reading{tc.GetCurrent()}
}

func copySlice(in []float64) []float64 {
	out := make([]float64, len(in))
	copy(out, in)
	return out
}

func mathRound(val float64, precision int) float64 {
	p := 1.0
	for i := 0; i < precision; i++ {
		p *= 10.0
	}
	return float64(int(val*p+0.5)) / p
}

func (tc *TelemetryCollector) StaticSpecs() map[string]interface{} {
	tc.mu.RLock()
	defer tc.mu.RUnlock()
	res := make(map[string]interface{}, len(tc.static))
	for k, v := range tc.static {
		res[k] = v
	}
	return res
}
