// pkg/telemetry/telemetry.go - System & Hardware Telemetry Sampler in Go
package telemetry

import (
	"bufio"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

const HistoryCapacity = 60

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
		historyRAMUtil:  make([]float64, 0, HistoryCapacity),
		historyCPUUtil:  make([]float64, 0, HistoryCapacity),
	}
	tc.detectStaticHardware()
	return tc
}

func (tc *TelemetryCollector) detectStaticHardware() {
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
		if data, err := os.ReadFile("/sys/class/drm/card0/device/mem_info_vram_total"); err == nil {
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

	tc.static = map[string]interface{}{
		"gpu_name":        gpuName,
		"gpu_count":       gpuCount,
		"compute_devices": computeDevices,
		"cpu_name":        cpuName,
		"cores":           cores,
		"threads":         threads,
		"psutil":          true,
	}

	if vramTotalBytes > 0 {
		tc.now["gpu_mem_total"] = vramTotalBytes
	}
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
	ramUsedBytes, ramTotalBytes, ramPct := tc.readRAM()
	vramUsedBytes, vramTotalBytes, gpuUtil, gpuTemp, gpuPower := tc.readGPU()

	tc.mu.Lock()
	defer tc.mu.Unlock()

	tc.now["cpu_util"] = cpuPct
	tc.now["ram_used"] = ramUsedBytes
	tc.now["ram_total"] = ramTotalBytes
	tc.now["ram_util"] = ramPct

	if vramTotalBytes > 0 {
		tc.now["gpu_mem_total"] = vramTotalBytes
	}
	if vramUsedBytes > 0 || tc.now["gpu_mem_total"] != nil {
		tc.now["gpu_mem_used"] = vramUsedBytes
		tc.now["gpu_util"] = gpuUtil
		tc.now["gpu_temp"] = gpuTemp
		tc.now["gpu_power"] = gpuPower
	}

	tc.pushHistory(&tc.historyCPUUtil, cpuPct)
	tc.pushHistory(&tc.historyRAMUtil, ramPct)
	tc.pushHistory(&tc.historyGPUUtil, gpuUtil)
	tc.pushHistory(&tc.historyVRAMUsed, float64(vramUsedBytes))
	tc.pushHistory(&tc.historyGPUTemp, gpuTemp)
	tc.pushHistory(&tc.historyGPUPower, gpuPower)
	tc.pushHistory(&tc.historyTokS, 0.0)
}

func (tc *TelemetryCollector) pushHistory(slice *[]float64, val float64) {
	if len(*slice) >= HistoryCapacity {
		*slice = (*slice)[1:]
	}
	*slice = append(*slice, val)
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

func (tc *TelemetryCollector) readRAM() (usedBytes, totalBytes uint64, pct float64) {
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
				return usedBytes, totalBytes, pct
			}
		}
	}

	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.Alloc, m.Sys, 0.0
}

func (tc *TelemetryCollector) readGPU() (vramUsedBytes, vramTotalBytes uint64, gpuUtil, gpuTemp, gpuPower float64) {
	// 1. Try nvidia-smi query
	out, err := exec.Command("nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw", "--format=csv,noheader,nounits").Output()
	if err == nil {
		line := strings.TrimSpace(string(out))
		if idx := strings.Index(line, "\n"); idx != -1 {
			line = line[:idx]
		}
		parts := strings.Split(line, ",")
		if len(parts) >= 4 {
			usedMiB, _ := strconv.ParseUint(strings.TrimSpace(parts[0]), 10, 64)
			totalMiB, _ := strconv.ParseUint(strings.TrimSpace(parts[1]), 10, 64)
			util, _ := strconv.ParseFloat(strings.TrimSpace(parts[2]), 64)
			temp, _ := strconv.ParseFloat(strings.TrimSpace(parts[3]), 64)
			power := 0.0
			if len(parts) >= 5 && !strings.Contains(parts[4], "N/A") {
				power, _ = strconv.ParseFloat(strings.TrimSpace(parts[4]), 64)
			}
			return usedMiB * 1024 * 1024, totalMiB * 1024 * 1024, util, temp, power
		}
	}

	// 2. Try AMD sysfs
	if _, err := os.Stat("/sys/class/drm/card0/device/mem_info_vram_used"); err == nil {
		if data, err := os.ReadFile("/sys/class/drm/card0/device/mem_info_vram_used"); err == nil {
			vramUsedBytes, _ = strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
		}
		if data, err := os.ReadFile("/sys/class/drm/card0/device/mem_info_vram_total"); err == nil {
			vramTotalBytes, _ = strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
		}
		if data, err := os.ReadFile("/sys/class/drm/card0/device/gpu_busy_percent"); err == nil {
			gpuUtil, _ = strconv.ParseFloat(strings.TrimSpace(string(data)), 64)
		}
		return vramUsedBytes, vramTotalBytes, gpuUtil, 45.0, 50.0
	}

	return 0, 0, 0, 0, 0
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
		"gpu_util":     copySlice(tc.historyGPUUtil),
		"gpu_mem_used": copySlice(tc.historyVRAMUsed),
		"gpu_temp":     copySlice(tc.historyGPUTemp),
		"gpu_power":    copySlice(tc.historyGPUPower),
		"tok_s":        copySlice(tc.historyTokS),
		"ram_util":     copySlice(tc.historyRAMUtil),
		"cpu_util":     copySlice(tc.historyCPUUtil),
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
