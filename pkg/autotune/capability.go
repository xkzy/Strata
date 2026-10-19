// pkg/autotune/capability.go - the vendor-neutral description of the machine Strata runs on.
//
// Everything the solver needs to reject an infeasible configuration is here. A field whose value could not be
// established stays at its zero value and the reason is listed in Capabilities.Unknown; nothing is assumed.
package autotune

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sort"
	"strings"
	"time"
)

// Health is a device's state as far as discovery could establish it. Unknown is not OK: a plan may use a device
// only when its state is exactly OK.
type Health string

const (
	HealthOK      Health = "ok"
	HealthUnknown Health = "unknown"
	HealthFailed  Health = "failed"
)

// NUMANode is one Linux NUMA node.
type NUMANode struct {
	ID       int    `json:"id"`
	CPUs     int    `json:"cpus"`
	MemBytes uint64 `json:"mem_bytes,omitempty"`
}

// CPUInfo is the processor as discovery found it. Counts are what the process may use, not what the box has.
type CPUInfo struct {
	Model              string     `json:"model,omitempty"`
	Logical            int        `json:"logical"`
	Physical           int        `json:"physical"`
	Available          int        `json:"available"` // logical CPUs in this process's affinity mask
	NUMA               []NUMANode `json:"numa,omitempty"`
	SIMD               []string   `json:"simd,omitempty"` // avx2, avx512f, avx512_bf16, amx_tile, fma, sse4_2, neon ...
	MemBandwidthGBs    float64    `json:"mem_bandwidth_gbs,omitempty"`
	MemBandwidthSource string     `json:"mem_bandwidth_source,omitempty"` // measured | estimated
}

// MemoryInfo is host memory plus the limits a container, cgroup or rlimit imposes.
type MemoryInfo struct {
	TotalBytes       uint64 `json:"total_bytes"`
	AvailableBytes   uint64 `json:"available_bytes"`
	SwapTotalBytes   uint64 `json:"swap_total_bytes,omitempty"`
	SwapFreeBytes    uint64 `json:"swap_free_bytes,omitempty"`
	CgroupLimitBytes uint64 `json:"cgroup_limit_bytes,omitempty"`
	CgroupUsedBytes  uint64 `json:"cgroup_used_bytes,omitempty"`
	PinnedLimitBytes uint64 `json:"pinned_limit_bytes,omitempty"`
	SpreadBytes      uint64 `json:"spread_bytes,omitempty"` // observed change in available memory over 50 ms
	Pressure         string `json:"pressure,omitempty"`     // low | moderate | critical
}

// UsableBytes is the memory a plan may budget against: the smallest of the host, cgroup and available figures.
func (m MemoryInfo) UsableBytes() uint64 {
	usable := m.AvailableBytes
	if usable == 0 {
		if m.TotalBytes > 0 {
			usable = m.TotalBytes / 2 // no availability figure: half is the only defensible conservative guess
		}
	}
	if m.CgroupLimitBytes > 0 {
		free := m.CgroupLimitBytes
		if m.CgroupUsedBytes < free {
			free -= m.CgroupUsedBytes
		} else {
			free = 0
		}
		if free < usable || usable == 0 {
			usable = free
		}
	}
	return usable
}

// PeerLink is one accelerator-to-accelerator connection as discovery found it.
type PeerLink struct {
	From         string  `json:"from"`
	To           string  `json:"to"`
	Kind         string  `json:"kind"` // nvlink | pcie-switch | pcie-host | host-staged | unknown
	P2P          bool    `json:"p2p"`
	BandwidthGBs float64 `json:"bandwidth_gbs,omitempty"` // unidirectional estimate; 0: unknown
}

// Accelerator is one compute device. Vendor-specific discovery fills it; the solver only reads these fields.
type Accelerator struct {
	ID               string   `json:"id"`      // stable identity: PCI address or UUID
	Vendor           string   `json:"vendor"`  // nvidia | amd | intel | unknown
	Backend          string   `json:"backend"` // cuda | rocm | oneapi | none
	Name             string   `json:"name,omitempty"`
	MemoryTotalBytes uint64   `json:"memory_total_bytes,omitempty"`
	MemoryFreeBytes  uint64   `json:"memory_free_bytes,omitempty"`
	ComputeClass     string   `json:"compute_class,omitempty"` // e.g. NVIDIA compute capability 12.0; empty: unknown
	DataTypes        []string `json:"data_types,omitempty"`
	LinkGen          int      `json:"link_gen,omitempty"`
	LinkWidth        int      `json:"link_width,omitempty"`
	LinkGBs          float64  `json:"link_gbs,omitempty"`   // estimated usable unidirectional bandwidth
	MemoryGBs        float64  `json:"memory_gbs,omitempty"` // estimated usable bandwidth
	Streams          int      `json:"streams,omitempty"`
	Health           Health   `json:"health"`
	HealthReason     string   `json:"health_reason,omitempty"`
	TemperatureC     float64  `json:"temperature_c,omitempty"`
	PowerW           float64  `json:"power_w,omitempty"`
	PowerLimitW      float64  `json:"power_limit_w,omitempty"`
	Throttled        bool     `json:"throttled,omitempty"`
}

// Capabilities is the whole discovered machine. It is immutable; discovery returns a fresh value.
type Capabilities struct {
	Schema       int           `json:"schema"`
	OS           string        `json:"os"`
	Arch         string        `json:"arch"`
	CPU          CPUInfo       `json:"cpu"`
	Memory       MemoryInfo    `json:"memory"`
	Accelerators []Accelerator `json:"accelerators,omitempty"`
	Peers        []PeerLink    `json:"peers,omitempty"`
	Backends     []string      `json:"backends,omitempty"` // cpu plus any accelerator backend; empty: cpu only
	Driver       string        `json:"driver,omitempty"`
	Unknown      []string      `json:"unknown,omitempty"` // capabilities discovery could not establish
	DiscoveredAt time.Time     `json:"discovered_at"`
}

// CapabilitySchema is bumped whenever a field's meaning changes, which invalidates persisted tuning records.
const CapabilitySchema = 1

// HasBackend reports whether a backend was established (cpu is always present after discovery).
func (c Capabilities) HasBackend(name string) bool {
	for _, b := range c.Backends {
		if b == name {
			return true
		}
	}
	return false
}

// GPU returns the accelerator with this ID, if discovery found it.
func (c Capabilities) GPU(id string) (Accelerator, bool) {
	for _, a := range c.Accelerators {
		if a.ID == id {
			return a, true
		}
	}
	return Accelerator{}, false
}

// HealthyGPUs returns the devices a plan may use: health exactly OK and memory figures reported.
func (c Capabilities) HealthyGPUs() []Accelerator {
	var out []Accelerator
	for _, a := range c.Accelerators {
		if a.Health == HealthOK && a.MemoryTotalBytes > 0 {
			out = append(out, a)
		}
	}
	return out
}

// Peer returns the link in this direction.
func (c Capabilities) Peer(from, to string) (PeerLink, bool) {
	for _, p := range c.Peers {
		if p.From == from && p.To == to {
			return p, true
		}
	}
	return PeerLink{Kind: "unknown", P2P: false}, false
}

// Fingerprint identifies the hardware+driver+backend combination a tuning record belongs to. It deliberately
// excludes free memory, temperatures and utilisation: those change between runs and would invalidate every
// record. Accelerators are sorted so that discovery order cannot change the hash.
func (c Capabilities) Fingerprint() string {
	accs := append([]Accelerator(nil), c.Accelerators...)
	sort.Slice(accs, func(i, j int) bool { return accs[i].ID < accs[j].ID })
	var b strings.Builder
	fmt.Fprintf(&b, "schema=%d\nos=%s/%s\ndriver=%s\n", CapabilitySchema, c.OS, c.Arch, c.Driver)
	fmt.Fprintf(&b, "cpu=%s logical=%d physical=%d avail=%d simd=%s\n",
		c.CPU.Model, c.CPU.Logical, c.CPU.Physical, c.CPU.Available, strings.Join(sortedCopy(c.CPU.SIMD), ","))
	for _, n := range c.CPU.NUMA {
		fmt.Fprintf(&b, "numa=%d cpus=%d mem=%d\n", n.ID, n.CPUs, n.MemBytes)
	}
	fmt.Fprintf(&b, "mem_total=%d cgroup=%d pinned=%d\n", c.Memory.TotalBytes, c.Memory.CgroupLimitBytes, c.Memory.PinnedLimitBytes)
	for _, a := range accs {
		fmt.Fprintf(&b, "acc=%s vendor=%s backend=%s name=%s mem=%d compute=%s dt=%s link=%d/%d\n",
			a.ID, a.Vendor, a.Backend, a.Name, a.MemoryTotalBytes, a.ComputeClass,
			strings.Join(sortedCopy(a.DataTypes), ","), a.LinkGen, a.LinkWidth)
	}
	for _, p := range sortedPeers(c.Peers) {
		fmt.Fprintf(&b, "peer=%s->%s kind=%s p2p=%t\n", p.From, p.To, p.Kind, p.P2P)
	}
	sum := sha256.Sum256([]byte(b.String()))
	return hex.EncodeToString(sum[:])[:24]
}

func sortedCopy(in []string) []string {
	out := append([]string(nil), in...)
	sort.Strings(out)
	return out
}

func sortedPeers(in []PeerLink) []PeerLink {
	out := append([]PeerLink(nil), in...)
	sort.Slice(out, func(i, j int) bool {
		if out[i].From != out[j].From {
			return out[i].From < out[j].From
		}
		return out[i].To < out[j].To
	})
	return out
}

// reserveBytes is the memory a plan must leave free on a device or host. It is not a universal percentage:
//
//	OS floor    fixed headroom for the driver, allocator metadata and the host process
//	uncertainty measured free-memory spread over the sampling window (the caller passes it in)
//	workspace   the backend's per-stream scratch, which grows with concurrency
//
// The result is rounded up to 64 MiB so that small measurements cannot produce a razor-thin reserve.
func reserveBytes(floor uint64, uncertainty uint64, workspace uint64) uint64 {
	const mib = uint64(1) << 20
	total := floor + uncertainty + workspace
	return ((total + 63*mib) / (64 * mib)) * (64 * mib)
}
