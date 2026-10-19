package autotune

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

type fakeRunner struct {
	answers map[string][]byte
	errs    map[string]error
}

func (f fakeRunner) Run(_ context.Context, name string, args ...string) ([]byte, error) {
	key := name + " " + strings.Join(args, " ")
	if err, ok := f.errs[key]; ok {
		return nil, err
	}
	if out, ok := f.answers[key]; ok {
		return out, nil
	}
	// A prefix match keeps the test independent of the exact query column list.
	for k, out := range f.answers {
		if strings.HasPrefix(key, k) {
			return out, nil
		}
	}
	return nil, errors.New("not found")
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func fixtureHost(t *testing.T) Host {
	t.Helper()
	root := t.TempDir()
	host := Host{Proc: filepath.Join(root, "proc"), Sys: filepath.Join(root, "sys"), Cgroup: filepath.Join(root, "cgroup")}
	writeFile(t, filepath.Join(host.Proc, "cpuinfo"), `processor	: 0
model name	: Test CPU 9000
physical id	: 0
core id		: 0
flags		: fpu sse4_2 avx avx2 fma avx512f avx512bw avx512vl

processor	: 1
model name	: Test CPU 9000
physical id	: 0
core id		: 1
flags		: fpu sse4_2 avx avx2 fma avx512f avx512bw avx512vl

`)
	writeFile(t, filepath.Join(host.Proc, "self", "status"), "Name:\tstrata\nCpus_allowed_list:\t0-3\n")
	writeFile(t, filepath.Join(host.Proc, "self", "cgroup"), "0::/\n")
	writeFile(t, filepath.Join(host.Proc, "meminfo"), "MemTotal:       65536000 kB\nMemAvailable:   48000000 kB\nSwapTotal:       1000000 kB\nSwapFree:         900000 kB\n")
	writeFile(t, filepath.Join(host.Cgroup, "memory.max"), "8589934592\n")
	writeFile(t, filepath.Join(host.Cgroup, "memory.current"), "2147483648\n")
	writeFile(t, filepath.Join(host.Sys, "devices", "system", "node", "node0", "cpulist"), "0-3\n")
	writeFile(t, filepath.Join(host.Sys, "devices", "system", "node", "node0", "meminfo"), "Node 0 MemTotal:       65536000 kB\n")
	return host
}

const (
	nvQuery = "nvidia-smi --query-gpu="
	nvTopo  = "nvidia-smi topo -m"
	nvList  = "nvidia-smi -L"
)

func TestDiscoverCPUAndMemory(t *testing.T) {
	host := fixtureHost(t)
	caps := Discover(context.Background(), host, fakeRunner{})
	if caps.CPU.Model != "Test CPU 9000" {
		t.Errorf("model: %q", caps.CPU.Model)
	}
	if caps.CPU.Logical != 2 || caps.CPU.Physical != 2 {
		t.Errorf("logical=%d physical=%d, want 2/2", caps.CPU.Logical, caps.CPU.Physical)
	}
	if caps.CPU.Available != 4 { // the affinity list says 0-3 while cpuinfo has 2 blocks
		t.Errorf("available=%d, want 4", caps.CPU.Available)
	}
	for _, s := range []string{"avx2", "avx512f", "fma"} {
		if !hasSIMD(caps.CPU.SIMD, s) {
			t.Errorf("missing SIMD %s in %v", s, caps.CPU.SIMD)
		}
	}
	if caps.Memory.TotalBytes != 65536000*1024 || caps.Memory.AvailableBytes == 0 {
		t.Errorf("memory: %+v", caps.Memory)
	}
	if caps.Memory.CgroupLimitBytes != 8<<30 {
		t.Errorf("cgroup limit: %d", caps.Memory.CgroupLimitBytes)
	}
	if len(caps.CPU.NUMA) != 1 || caps.CPU.NUMA[0].CPUs != 4 {
		t.Errorf("numa: %+v", caps.CPU.NUMA)
	}
	if caps.Memory.Pressure != "low" {
		t.Errorf("pressure: %s", caps.Memory.Pressure)
	}
}

func TestDiscoverNVIDIAAndTopology(t *testing.T) {
	host := fixtureHost(t)
	runner := fakeRunner{answers: map[string][]byte{
		nvQuery: []byte("0, GPU-aaa, NVIDIA GeForce RTX 5060 Ti, 16384, 15000, 12.0, 1, 8, 4, 16, 39, 2, 4.0, 180.0, 0x0000000000000001, 595.91.07\n" +
			"1, GPU-bbb, NVIDIA GeForce RTX 4060, 8188, 7000, 8.9, 1, 4, 3, 8, 42, 0, [N/A], 115.0, 0x0000000000000000, 595.91.07\n"),
		nvTopo: []byte("\tGPU0\tGPU1\tCPU Affinity\tNUMA Affinity\nGPU0\t X \tPHB\t0-15\t0\nGPU1\tPHB\t X \t0-15\t0\n"),
		nvList: []byte("GPU 0: NVIDIA GeForce RTX 5060 Ti (UUID: GPU-aaa)\nGPU 1: NVIDIA GeForce RTX 4060 (UUID: GPU-bbb)\n"),
	}}
	caps := Discover(context.Background(), host, runner)
	if len(caps.Accelerators) != 2 {
		t.Fatalf("accelerators: %d", len(caps.Accelerators))
	}
	a := caps.Accelerators[0]
	if a.ID != "GPU-aaa" || a.Vendor != "nvidia" || a.Backend != "cuda" || a.Health != HealthOK {
		t.Errorf("gpu0: %+v", a)
	}
	if a.MemoryTotalBytes != 16384<<20 || a.MemoryFreeBytes != 15000<<20 {
		t.Errorf("gpu0 memory: %d/%d", a.MemoryFreeBytes, a.MemoryTotalBytes)
	}
	if a.LinkGBs <= 0 || a.LinkGen != 4 || a.LinkWidth != 16 { // max capability, not the idle Gen1 downshift
		t.Errorf("gpu0 link: %d/%d %.1f", a.LinkGen, a.LinkWidth, a.LinkGBs)
	}
	if caps.Accelerators[1].PowerW != 0 || caps.Accelerators[1].Health != HealthOK {
		t.Errorf("gpu1 with [N/A] power must stay healthy: %+v", caps.Accelerators[1])
	}
	if !caps.HasBackend("cuda") || caps.Driver != "nvidia 595.91.07" {
		t.Errorf("backend/driver: %v %q", caps.Backends, caps.Driver)
	}
	if len(caps.Peers) != 2 {
		t.Fatalf("peers: %+v", caps.Peers)
	}
	for _, p := range caps.Peers {
		if p.Kind != "pcie-host" || p.P2P {
			t.Errorf("PHB must be host-staged, not P2P: %+v", p)
		}
	}
	// Free memory is dynamic and must not change the fingerprint.
	before := caps.Fingerprint()
	caps.Accelerators[0].MemoryFreeBytes = 1
	if after := caps.Fingerprint(); after != before {
		t.Errorf("free memory changed the fingerprint: %s -> %s", before, after)
	}
}

func TestDiscoverNVLinkIsPeer(t *testing.T) {
	host := fixtureHost(t)
	runner := fakeRunner{answers: map[string][]byte{
		nvQuery: []byte("0, GPU-a, A, 24576, 20000, 9.0, 1, 16, 4, 16, 35, 0, 300, 320, 0x0, 1.0\n1, GPU-b, B, 24576, 20000, 9.0, 1, 16, 4, 16, 35, 0, 300, 320, 0x0, 1.0\n"),
		nvTopo:  []byte("GPU0\tGPU1\nGPU0\t X \tNV12\nGPU1\tNV12\t X \n"),
	}}
	caps := Discover(context.Background(), host, runner)
	if len(caps.Peers) == 0 || !caps.Peers[0].P2P || caps.Peers[0].Kind != "nvlink" {
		t.Fatalf("NVLink must be a peer path: %+v", caps.Peers)
	}
}

func TestDiscoverPartialFailureIsConservative(t *testing.T) {
	host := Host{Proc: filepath.Join(t.TempDir(), "missing"), Sys: filepath.Join(t.TempDir(), "missing"), Cgroup: filepath.Join(t.TempDir(), "missing")}
	caps := Discover(context.Background(), host, fakeRunner{})
	if caps.CPU.Logical <= 0 || caps.CPU.Available <= 0 {
		t.Errorf("CPU counts must fall back to the runtime: %+v", caps.CPU)
	}
	if len(caps.Unknown) == 0 {
		t.Error("what could not be established must be recorded")
	}
	if len(caps.Accelerators) != 0 {
		t.Errorf("no tool answered but devices were invented: %+v", caps.Accelerators)
	}
	if !caps.HasBackend("cpu") {
		t.Error("cpu backend must always be present")
	}
	if caps.Fingerprint() == "" {
		t.Error("a fingerprint must still exist")
	}
}

func TestDiscoverThrottledGPUIsFailed(t *testing.T) {
	host := fixtureHost(t)
	runner := fakeRunner{answers: map[string][]byte{
		nvQuery: []byte("0, GPU-a, A, 16384, 15000, 12.0, 1, 8, 4, 16, 39, 2, 4.0, 180.0, 0x0000000000000040, 1.0\n"),
	}}
	caps := Discover(context.Background(), host, runner)
	if len(caps.Accelerators) != 1 || caps.Accelerators[0].Health != HealthFailed {
		t.Fatalf("a hardware thermal brake must fail the card for planning: %+v", caps.Accelerators)
	}
}

func TestDiscoverAMDSysfs(t *testing.T) {
	host := fixtureHost(t)
	real := filepath.Join(host.Sys, "devices", "pci0000:00", "0000:03:00.0")
	writeFile(t, filepath.Join(real, "vendor"), "0x1002\n")
	writeFile(t, filepath.Join(real, "product_name"), "Radeon Test\n")
	writeFile(t, filepath.Join(real, "mem_info_vram_total"), "17163091968\n")
	writeFile(t, filepath.Join(real, "mem_info_vram_used"), "1073741824\n")
	writeFile(t, filepath.Join(real, "current_link_width"), "16\n")
	writeFile(t, filepath.Join(real, "current_link_speed"), "16.0 GT/s PCIe\n")
	writeFile(t, filepath.Join(real, "hwmon", "hwmon0", "temp1_input"), "45000\n")
	if err := os.MkdirAll(filepath.Join(host.Sys, "class", "drm", "card1"), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(real, filepath.Join(host.Sys, "class", "drm", "card1", "device")); err != nil {
		t.Fatal(err)
	}
	caps := Discover(context.Background(), host, fakeRunner{})
	var amd *Accelerator
	for i := range caps.Accelerators {
		if caps.Accelerators[i].Vendor == "amd" {
			amd = &caps.Accelerators[i]
		}
	}
	if amd == nil {
		t.Fatalf("AMD card not discovered: %+v", caps.Accelerators)
	}
	if amd.ID != "0000:03:00.0" || amd.MemoryTotalBytes != 17163091968 || amd.MemoryFreeBytes != 17163091968-1073741824 {
		t.Errorf("amd: %+v", amd)
	}
	if amd.TemperatureC != 45 || amd.Health != HealthOK {
		t.Errorf("amd health: %+v", amd)
	}
}
