// pkg/autotune/discover.go - hardware discovery that never fails startup.
//
// Every source is optional: a missing /proc entry, an absent nvidia-smi, a container without cgroup v2 or a
// device whose driver does not answer leaves that capability Unknown and the rest of the machine usable. The
// vendor-specific code is confined to the nvidia*/amd* functions; the schema it fills is vendor-neutral.
package autotune

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"strata/pkg/telemetry"
)

// Host is where discovery reads from. The zero value is not usable: call DefaultHost.
type Host struct {
	Proc   string `json:"proc"`   // normally /proc
	Sys    string `json:"sys"`    // normally /sys
	Cgroup string `json:"cgroup"` // normally /sys/fs/cgroup
}

// DefaultHost is the real machine.
func DefaultHost() Host {
	return Host{Proc: "/proc", Sys: "/sys", Cgroup: "/sys/fs/cgroup"}
}

// CommandRunner runs an optional external tool. Discovery never depends on one answering.
type CommandRunner interface {
	Run(ctx context.Context, name string, args ...string) ([]byte, error)
}

// ExecRunner runs the real tool.
type ExecRunner struct{}

func (ExecRunner) Run(ctx context.Context, name string, args ...string) ([]byte, error) {
	return exec.CommandContext(ctx, name, args...).Output()
}

// Discover fills what it can and records what it could not establish. It never returns an error: a machine
// with no readable /proc still yields a conservative CPU-only capability set.
func Discover(ctx context.Context, host Host, runner CommandRunner) Capabilities {
	if runner == nil {
		runner = ExecRunner{}
	}
	if host.Proc == "" || host.Sys == "" {
		host = DefaultHost()
	}
	c := Capabilities{
		Schema:       CapabilitySchema,
		OS:           runtime.GOOS,
		Arch:         runtime.GOARCH,
		Backends:     []string{"cpu"},
		DiscoveredAt: time.Now(),
	}
	discoverCPU(host, &c)
	discoverMemory(host, &c)
	discoverNVIDIA(ctx, host, runner, &c)
	discoverAMD(host, &c)
	discoverIntel(host, &c)
	sort.Strings(c.Unknown)
	return c
}

// ---------------------------------------------------------------------------------------------------------------
// CPU

func discoverCPU(host Host, c *Capabilities) {
	data, err := os.ReadFile(filepath.Join(host.Proc, "cpuinfo"))
	if err != nil {
		c.Unknown = append(c.Unknown, "cpu: /proc/cpuinfo unreadable ("+err.Error()+")")
		c.CPU.Logical = runtime.NumCPU()
		c.CPU.Available = runtime.NumCPU()
		c.CPU.Physical = runtime.NumCPU()
		return
	}
	model, simd, physical := parseCPUInfo(string(data))
	c.CPU.Model = model
	c.CPU.SIMD = simd
	c.CPU.Logical = countCPUSections(string(data))
	if c.CPU.Logical == 0 {
		c.CPU.Logical = runtime.NumCPU()
	}
	if physical > 0 {
		c.CPU.Physical = physical
	} else {
		c.CPU.Physical = c.CPU.Logical
	}
	if avail := parseCPUAffinity(host); avail > 0 {
		c.CPU.Available = avail
	} else {
		c.CPU.Available = runtime.NumCPU()
		if c.CPU.Available > c.CPU.Logical {
			c.CPU.Available = c.CPU.Logical
		}
	}
	if len(c.CPU.SIMD) == 0 {
		c.Unknown = append(c.Unknown, "cpu: SIMD flags not reported")
	}
	c.CPU.NUMA = parseNUMA(host)
	if len(c.CPU.NUMA) == 0 && c.CPU.Physical > 0 {
		c.CPU.NUMA = []NUMANode{{ID: 0, CPUs: c.CPU.Physical}}
		c.Unknown = append(c.Unknown, "cpu: NUMA topology not reported")
	}
}

// countCPUSections counts the blank-line separated processor blocks in /proc/cpuinfo.
func countCPUSections(data string) int {
	n := 0
	for _, block := range strings.Split(data, "\n\n") {
		if strings.Contains(block, "processor") {
			n++
		}
	}
	return n
}

// parseCPUInfo returns the model name, the normalized SIMD feature list and the physical core count (0 when the
// file does not identify cores).
func parseCPUInfo(data string) (model string, simd []string, physical int) {
	// x86 uses "flags", ARM uses "Features"; accept either.
	set := map[string]bool{}
	cores := map[string]bool{}
	var physID, coreID string
	flushCore := func() {
		if coreID != "" {
			cores[physID+":"+coreID] = true
		}
		physID, coreID = "", ""
	}
	for _, line := range strings.Split(data, "\n") {
		key, val, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		key = strings.TrimSpace(key)
		val = strings.TrimSpace(val)
		switch key {
		case "model name", "Model", "cpu model":
			if model == "" {
				model = val
			}
		case "flags", "Features":
			for _, f := range strings.Fields(val) {
				set[strings.ToLower(f)] = true
			}
		case "physical id":
			flushCore()
			physID = val
		case "core id":
			coreID = val
		}
	}
	flushCore()
	// Keep the names the rest of the tree uses (pack.RequiredSIMD, docs) rather than the raw flag names.
	aliases := []struct{ flag, name string }{
		{"sse4_2", "sse4_2"}, {"avx", "avx"}, {"avx2", "avx2"}, {"fma", "fma"},
		{"avx512f", "avx512f"}, {"avx512bw", "avx512bw"}, {"avx512vl", "avx512vl"},
		{"avx512_bf16", "avx512_bf16"}, {"avx512_vnni", "avx512_vnni"},
		{"amx_tile", "amx_tile"}, {"amx_bf16", "amx_bf16"},
		{"neon", "neon"}, {"sve", "sve"}, {"sve2", "sve2"}, {"asimddp", "dotprod"},
	}
	for _, a := range aliases {
		if set[a.flag] {
			simd = append(simd, a.name)
		}
	}
	return model, simd, len(cores)
}

// parseCPUAffinity returns the number of CPUs in this process's affinity mask (Cpus_allowed_list, which honors
// cpusets and taskset), 0 when the file does not say.
func parseCPUAffinity(host Host) int {
	data, err := os.ReadFile(filepath.Join(host.Proc, "self", "status"))
	if err != nil {
		return 0
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "Cpus_allowed_list:") {
			_, list, _ := strings.Cut(line, ":")
			return countCPUList(strings.TrimSpace(list))
		}
	}
	return 0
}

// countCPUList counts "0-3,8,10-11" style lists.
func countCPUList(list string) int {
	n := 0
	for _, part := range strings.Split(list, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		lo, hi, ok := strings.Cut(part, "-")
		if !ok {
			if _, err := strconv.Atoi(part); err == nil {
				n++
			}
			continue
		}
		a, err1 := strconv.Atoi(lo)
		b, err2 := strconv.Atoi(hi)
		if err1 == nil && err2 == nil && b >= a {
			n += b - a + 1
		}
	}
	return n
}

func parseNUMA(host Host) []NUMANode {
	dirs, _ := filepath.Glob(filepath.Join(host.Sys, "devices", "system", "node", "node*"))
	var out []NUMANode
	for _, d := range dirs {
		base := filepath.Base(d)
		id, err := strconv.Atoi(strings.TrimPrefix(base, "node"))
		if err != nil {
			continue
		}
		node := NUMANode{ID: id}
		if b, err := os.ReadFile(filepath.Join(d, "cpulist")); err == nil {
			node.CPUs = countCPUList(strings.TrimSpace(string(b)))
		}
		if b, err := os.ReadFile(filepath.Join(d, "meminfo")); err == nil {
			for _, line := range strings.Split(string(b), "\n") {
				if strings.Contains(line, "MemTotal:") {
					f := strings.Fields(line)
					if len(f) >= 4 {
						if kb, err := strconv.ParseUint(f[3], 10, 64); err == nil {
							node.MemBytes = kb * 1024
						}
					}
				}
			}
		}
		out = append(out, node)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

// ---------------------------------------------------------------------------------------------------------------
// Memory

func discoverMemory(host Host, c *Capabilities) {
	data, err := os.ReadFile(filepath.Join(host.Proc, "meminfo"))
	if err != nil {
		c.Unknown = append(c.Unknown, "memory: /proc/meminfo unreadable")
	} else {
		kib := map[string]uint64{}
		for _, line := range strings.Split(string(data), "\n") {
			f := strings.Fields(line)
			if len(f) < 2 {
				continue
			}
			v, err := strconv.ParseUint(f[1], 10, 64)
			if err == nil {
				kib[strings.TrimSuffix(f[0], ":")] = v
			}
		}
		c.Memory.TotalBytes = kib["MemTotal"] * 1024
		c.Memory.AvailableBytes = kib["MemAvailable"] * 1024
		if c.Memory.AvailableBytes == 0 {
			c.Memory.AvailableBytes = kib["MemFree"] * 1024
		}
		c.Memory.SwapTotalBytes = kib["SwapTotal"] * 1024
		c.Memory.SwapFreeBytes = kib["SwapFree"] * 1024
		// One short second read gives the free-memory spread the solver has to leave as uncertainty. It is a
		// measurement of this machine at this moment, not a universal percentage.
		time.Sleep(50 * time.Millisecond)
		if b, err := os.ReadFile(filepath.Join(host.Proc, "meminfo")); err == nil {
			if avail := memAvailable(string(b)); avail > 0 && c.Memory.AvailableBytes > 0 {
				if avail > c.Memory.AvailableBytes {
					c.Memory.SpreadBytes = avail - c.Memory.AvailableBytes
				} else {
					c.Memory.SpreadBytes = c.Memory.AvailableBytes - avail
				}
			}
		}
	}
	c.Memory.CgroupLimitBytes, c.Memory.CgroupUsedBytes = readCgroupMemory(host)
	c.Memory.PinnedLimitBytes = pinnedMemoryLimit()
	if c.Memory.TotalBytes == 0 {
		c.Unknown = append(c.Unknown, "memory: total memory unknown")
	}
	if c.Memory.AvailableBytes == 0 {
		c.Unknown = append(c.Unknown, "memory: available memory unknown")
	}
	c.Memory.Pressure = memoryPressure(c.Memory)
	if c.Memory.SwapTotalBytes > 0 {
		// Swap is capacity, not RAM: a plan may never rely on it for latency-sensitive inference. Recorded only
		// so /metrics can say the box is swapping.
		used := c.Memory.SwapTotalBytes - c.Memory.SwapFreeBytes
		if used*100/max64(c.Memory.SwapTotalBytes, 1) > 25 {
			c.Memory.Pressure = "critical"
		}
	}
}

// memAvailable re-reads MemAvailable (falling back to MemFree) from a /proc/meminfo body.
func memAvailable(data string) uint64 {
	vals := map[string]uint64{}
	for _, line := range strings.Split(data, "\n") {
		f := strings.Fields(line)
		if len(f) < 2 {
			continue
		}
		v, err := strconv.ParseUint(f[1], 10, 64)
		if err == nil {
			vals[strings.TrimSuffix(f[0], ":")] = v
		}
	}
	if vals["MemAvailable"] > 0 {
		return vals["MemAvailable"] * 1024
	}
	return vals["MemFree"] * 1024
}

// memoryPressure is a policy, documented in docs/optimization/hardware-discovery.md: critical below 10% of total
// (or below 2 GiB, whichever is larger), moderate below 25%.
func memoryPressure(m MemoryInfo) string {
	if m.TotalBytes == 0 || m.AvailableBytes == 0 {
		return "unknown"
	}
	// The floor follows the memory this process can actually use, not the whole box: inside a 8 GiB cgroup a
	// host-derived 10% floor (6.5 GiB here) would call a healthy container critical.
	limit := m.TotalBytes
	if m.CgroupLimitBytes > 0 && m.CgroupLimitBytes < limit {
		limit = m.CgroupLimitBytes
	}
	floor := max64(limit/10, 2<<30)
	avail := m.AvailableBytes
	if m.CgroupLimitBytes > 0 {
		free := uint64(0)
		if m.CgroupUsedBytes < m.CgroupLimitBytes {
			free = m.CgroupLimitBytes - m.CgroupUsedBytes
		}
		if avail == 0 || free < avail {
			avail = free
		}
	}
	if avail < floor {
		return "critical"
	}
	if avail*4 < limit {
		return "moderate"
	}
	return "low"
}

func readCgroupMemory(host Host) (limit, used uint64) {
	// v2: one unified hierarchy; the process's path comes from /proc/self/cgroup ("0::/path").
	path := cgroupPath(host, "0::")
	v2 := filepath.Join(host.Cgroup, strings.TrimPrefix(path, "/"))
	if v, ok := readUint(filepath.Join(v2, "memory.max")); ok {
		limit = v
	}
	if v, ok := readUint(filepath.Join(v2, "memory.current")); ok {
		used = v
	}
	if limit > 0 && used == 0 {
		// memory.max with no current is still a usable ceiling; leave used 0.
	}
	if limit > 1<<60 { // "max" is written as a huge number by some runtimes
		limit = 0
	}
	if limit > 0 {
		return limit, used
	}
	// v1 fallback: /sys/fs/cgroup/memory/memory.limit_in_bytes
	v1path := cgroupPath(host, "memory:")
	dir := filepath.Join(host.Cgroup, "memory", strings.TrimPrefix(v1path, "/"))
	if v, ok := readUint(filepath.Join(dir, "memory.limit_in_bytes")); ok {
		limit = v
	}
	if v, ok := readUint(filepath.Join(dir, "memory.usage_in_bytes")); ok {
		used = v
	}
	if limit > 1<<60 {
		limit = 0
	}
	return limit, used
}

func cgroupPath(host Host, prefix string) string {
	data, err := os.ReadFile(filepath.Join(host.Proc, "self", "cgroup"))
	if err != nil {
		return "/"
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, prefix) {
			_, p, ok := strings.Cut(line, prefix)
			if ok && strings.HasPrefix(p, "/") {
				return p
			}
		}
	}
	return "/"
}

func readUint(path string) (uint64, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	s := strings.TrimSpace(string(b))
	v, err := strconv.ParseUint(s, 10, 64)
	return v, err == nil
}

func max64(a, b uint64) uint64 {
	if a > b {
		return a
	}
	return b
}

// ---------------------------------------------------------------------------------------------------------------
// NVIDIA

// The max link values are the card+slot capability; gen.current is the idle downshift (Gen1 on a healthy card),
// so the bandwidth estimate uses the max values and never mistakes an idle link for a narrow one.
const nvidiaQuery = "index,uuid,name,memory.total,memory.free,compute_cap,pcie.link.gen.current,pcie.link.width.current," +
	"pcie.link.gen.max,pcie.link.width.max,temperature.gpu,utilization.gpu,power.draw,power.limit," +
	"clocks_throttle_reasons.active,driver_version"

func discoverNVIDIA(ctx context.Context, host Host, runner CommandRunner, c *Capabilities) {
	out, err := runner.Run(ctx, "nvidia-smi", "--query-gpu="+nvidiaQuery, "--format=csv,noheader,nounits")
	if err != nil || len(strings.TrimSpace(string(out))) == 0 {
		return // no NVIDIA driver: not an error, simply no capability
	}
	lines := strings.Split(strings.TrimSpace(string(out)), "\n")
	var driver string
	for _, line := range lines {
		f := splitCSV(line)
		if len(f) < 16 {
			c.Unknown = append(c.Unknown, "nvidia: a query line has fewer fields than expected")
			continue
		}
		a := Accelerator{
			ID:        f[1],
			Vendor:    "nvidia",
			Backend:   "cuda",
			Name:      f[2],
			Health:    HealthUnknown,
			DataTypes: []string{"fp32", "fp16", "bf16", "int8", "int4"},
		}
		if n, ok := parseMiB(f[3]); ok {
			a.MemoryTotalBytes = n
		}
		if n, ok := parseMiB(f[4]); ok {
			a.MemoryFreeBytes = n
		}
		a.ComputeClass = f[5]
		a.LinkGen, _ = strconv.Atoi(f[8])
		a.LinkWidth, _ = strconv.Atoi(f[9])
		if a.LinkGen > 0 && a.LinkWidth > 0 {
			a.LinkGBs = telemetry.PcieBandwidthGBps(a.LinkGen, a.LinkWidth)
		}
		a.TemperatureC = parseFloatOr0(f[10])
		a.PowerW = parseFloatOr0(f[12])
		a.PowerLimitW = parseFloatOr0(f[13])
		a.Throttled = throttleFatal(f[14])
		if driver == "" {
			driver = f[15]
		}
		// Health: OK only when every number a plan depends on was reported and nothing says the device is in
		// trouble. A card that is already throttled or hot gets no benchmark work and no new plan.
		switch {
		case a.Throttled:
			a.Health = HealthFailed
			a.HealthReason = "the driver reports a hardware throttle or brake (" + f[14] + ")"
		case a.TemperatureC >= 90:
			a.Health = HealthFailed
			a.HealthReason = fmt.Sprintf("temperature %.0f C at discovery", a.TemperatureC)
		case a.MemoryTotalBytes == 0 || a.MemoryFreeBytes == 0:
			a.Health = HealthUnknown
			a.HealthReason = "the driver did not report memory"
		default:
			a.Health = HealthOK
		}
		c.Accelerators = append(c.Accelerators, a)
	}
	if len(c.Accelerators) == 0 {
		return
	}
	c.Driver = "nvidia " + driver
	if !c.HasBackend("cuda") {
		c.Backends = append(c.Backends, "cuda")
	}
	if list, err := runner.Run(ctx, "nvidia-smi", "-L"); err == nil {
		listed := strings.Count(strings.TrimSpace(string(list)), "GPU ")
		if listed > len(c.Accelerators) {
			c.Unknown = append(c.Unknown, fmt.Sprintf("nvidia: %d card(s) listed but only %d answered the query", listed, len(c.Accelerators)))
		}
	}
	c.Peers = append(c.Peers, nvidiaTopology(ctx, runner, c.Accelerators)...)
}

// throttleFatal interprets clocks_throttle_reasons.active: hardware slowdown/thermal brake (0x8, 0x40, 0x80)
// means the card is protecting itself; software power caps (0x4, 0x20) are normal operation.
func throttleFatal(raw string) bool {
	s := strings.TrimSpace(raw)
	if s == "" || strings.EqualFold(s, "[N/A]") {
		return false
	}
	v, err := strconv.ParseUint(strings.TrimPrefix(strings.ToLower(s), "0x"), 16, 64)
	if err != nil {
		return false
	}
	const hardware = 0x8 | 0x40 | 0x80
	return v&hardware != 0
}

// nvidiaTopology parses `nvidia-smi topo -m`. Only NVLink and same-switch PCIe are treated as usable peer
// paths; PHB/NODE/SYS are host-staged and never claimed as P2P.
func nvidiaTopology(ctx context.Context, runner CommandRunner, accs []Accelerator) []PeerLink {
	out, err := runner.Run(ctx, "nvidia-smi", "topo", "-m")
	if err != nil {
		return nil
	}
	lines := strings.Split(string(out), "\n")
	if len(accs) < 2 {
		return nil
	}
	// Rows look like: "        GPU0    GPU1    NIC0    CPU Affinity    NUMA Affinity"
	// followed by "GPU0     X      NV12    PIX     ...".
	ids := make([]string, len(accs))
	for i, a := range accs {
		ids[i] = a.ID
	}
	var links []PeerLink
	for _, line := range lines {
		f := strings.Fields(line)
		if len(f) < 2 || !strings.HasPrefix(f[0], "GPU") {
			continue
		}
		from, err := strconv.Atoi(strings.TrimPrefix(f[0], "GPU"))
		if err != nil || from < 0 || from >= len(ids) {
			continue
		}
		for j, tok := range f[1:] {
			if j >= len(ids) || j == from {
				continue
			}
			kind, p2p := topoKind(tok)
			if kind == "" {
				continue
			}
			links = append(links, PeerLink{From: ids[from], To: ids[j], Kind: kind, P2P: p2p})
		}
	}
	return links
}

func topoKind(tok string) (kind string, p2p bool) {
	switch {
	case strings.HasPrefix(tok, "NV"):
		return "nvlink", true
	case tok == "PIX" || tok == "PXB":
		return "pcie-switch", true
	case tok == "PHB":
		return "pcie-host", false
	case tok == "NODE":
		return "numa", false
	case tok == "SYS":
		return "host-staged", false
	}
	return "", false
}

func parseMiB(s string) (uint64, bool) {
	s = strings.TrimSpace(s)
	if s == "" || strings.EqualFold(s, "[N/A]") || strings.EqualFold(s, "N/A") {
		return 0, false
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil || v <= 0 {
		return 0, false
	}
	return uint64(v * 1024 * 1024), true
}

func parseFloatOr0(s string) float64 {
	s = strings.TrimSpace(s)
	if s == "" || strings.EqualFold(s, "[N/A]") || strings.EqualFold(s, "N/A") {
		return 0
	}
	v, _ := strconv.ParseFloat(s, 64)
	return v
}

func splitCSV(line string) []string {
	parts := strings.Split(line, ",")
	out := make([]string, len(parts))
	for i, p := range parts {
		out[i] = strings.TrimSpace(p)
	}
	return out
}

// ---------------------------------------------------------------------------------------------------------------
// AMD / Intel (sysfs; no vendor tool required)

var pciAddress = regexp.MustCompile(`^[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]$`)

func discoverAMD(host Host, c *Capabilities) {
	discoverAMDSysfs(host, c)
}

func discoverIntel(host Host, c *Capabilities) {
	entries, err := os.ReadDir(filepath.Join(host.Sys, "class", "drm"))
	if err != nil {
		return
	}
	for _, e := range entries {
		if !strings.HasPrefix(e.Name(), "card") || strings.Contains(e.Name(), "-") {
			continue
		}
		dev := filepath.Join(host.Sys, "class", "drm", e.Name(), "device")
		if sysfsString(dev, "vendor") != "0x8086" {
			continue
		}
		c.Unknown = append(c.Unknown, "intel: a display device was found but no Intel compute backend adapter exists")
		return
	}
}

func sysfsString(dir, name string) string {
	b, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

func sysfsUint(dir, name string) (uint64, bool) {
	return readUint(filepath.Join(dir, name))
}

// discoverAMDSysfs finds AMD GPUs from /sys/class/drm without rocm-smi.
func discoverAMDSysfs(host Host, c *Capabilities) {
	entries, err := os.ReadDir(filepath.Join(host.Sys, "class", "drm"))
	if err != nil {
		return
	}
	haveKFD := false
	if _, err := os.Stat("/dev/kfd"); err == nil {
		haveKFD = true
	}
	for _, e := range entries {
		name := e.Name()
		if !strings.HasPrefix(name, "card") || strings.Contains(name, "-") {
			continue
		}
		dev := filepath.Join(host.Sys, "class", "drm", name, "device")
		if sysfsString(dev, "vendor") != "0x1002" {
			continue
		}
		real, err := filepath.EvalSymlinks(dev)
		if err != nil {
			real = dev
		}
		id := filepath.Base(real)
		if !pciAddress.MatchString(id) {
			id = name
		}
		a := Accelerator{ID: id, Vendor: "amd", Backend: "rocm", Name: sysfsString(dev, "product_name"), Health: HealthUnknown}
		if a.Name == "" {
			a.Name = "AMD GPU"
		}
		if v, ok := sysfsUint(dev, "mem_info_vram_total"); ok {
			a.MemoryTotalBytes = v
		}
		if v, ok := sysfsUint(dev, "mem_info_vram_used"); ok && a.MemoryTotalBytes >= v {
			a.MemoryFreeBytes = a.MemoryTotalBytes - v
		}
		if b, err := os.ReadFile(filepath.Join(dev, "current_link_speed")); err == nil {
			a.LinkGen = parseGtToGen(string(b))
		}
		if v, ok := sysfsUint(dev, "current_link_width"); ok {
			a.LinkWidth = int(v)
		}
		if a.LinkGen > 0 && a.LinkWidth > 0 {
			a.LinkGBs = telemetry.PcieBandwidthGBps(a.LinkGen, a.LinkWidth)
		}
		if temps, _ := filepath.Glob(filepath.Join(dev, "hwmon", "hwmon*", "temp1_input")); len(temps) > 0 {
			if b, err := os.ReadFile(temps[0]); err == nil {
				if milli, err := strconv.Atoi(strings.TrimSpace(string(b))); err == nil {
					a.TemperatureC = float64(milli) / 1000
				}
			}
		}
		switch {
		case a.MemoryTotalBytes == 0:
			a.Health = HealthUnknown
			a.HealthReason = "sysfs did not report VRAM"
		case a.TemperatureC >= 90:
			a.Health = HealthFailed
			a.HealthReason = fmt.Sprintf("temperature %.0f C at discovery", a.TemperatureC)
		default:
			a.Health = HealthOK
		}
		c.Accelerators = append(c.Accelerators, a)
		if haveKFD && !c.HasBackend("rocm") {
			c.Backends = append(c.Backends, "rocm")
		}
	}
}

// parseGtToGen is duplicated from telemetry (unexported there) for the AMD link speed string.
func parseGtToGen(s string) int {
	s = strings.ToLower(strings.TrimSpace(s))
	for _, p := range []struct {
		needle string
		gen    int
	}{{"64.0", 6}, {"32.0", 5}, {"16.0", 4}, {"8.0", 3}, {"5.0", 2}, {"2.5", 1}} {
		if strings.Contains(s, p.needle) {
			return p.gen
		}
	}
	return 0
}
