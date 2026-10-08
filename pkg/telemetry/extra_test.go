package telemetry

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

// The Monitor tab's iGPU, Disk read and PCIe cards read hardware.igpu_util, hardware.disk_read_mb and
// hardware.gpu_pcie_gen / gpu_pcie_width; without them the cards stay "–".

const diskstatsSample = `   7       0 loop0 100 0 800 5 0 0 0 0 0 0 0
 259       0 nvme0n1 5000 10 2000000 900 3000 5 1000000 800 0 700 1700
 259       1 nvme0n1p1 4000 0 1500000 800 2000 0 900000 700 0 600 1500
 259       2 nvme0n1p2 900 0 400000 100 900 0 100000 100 0 100 200
   8       0 sda 200 0 4000 50 100 0 2000 20 0 60 70
   8       1 sda1 190 0 3900 49 99 0 1900 19 0 59 68
 253       0 dm-0 4000 0 1500000 800 2000 0 900000 700 0 600 1500
   9       0 md0 10 0 80 1 10 0 80 1 0 2 2
 252       0 zram0 50 0 400 1 50 0 400 1 0 1 1
`

func TestParseDiskstatsCountsWholePhysicalDisksOnly(t *testing.T) {
	read, write := parseDiskstats(diskstatsSample)
	// nvme0n1 + sda: partitions, loop, dm, md and zram would count the same bytes twice (or are not disks)
	if read != 2000000+4000 || write != 1000000+2000 {
		t.Errorf("sectors read/written = %d/%d, want %d/%d", read, write, 2004000, 1002000)
	}
	if r, w := parseDiskstats("garbage\n\n1 2\n"); r != 0 || w != 0 {
		t.Errorf("unreadable input must count nothing, got %d/%d", r, w)
	}
}

func TestDiskRates(t *testing.T) {
	tc := &TelemetryCollector{}
	t0 := time.Unix(1000, 0)
	if _, _, ok := tc.diskRates(1000, 500, t0); ok {
		t.Error("the first sample has nothing to compare with")
	}
	// 8 MiB read and 4 MiB written in 2 seconds: 16384 and 8192 sectors of 512 bytes
	r, w, ok := tc.diskRates(1000+16384, 500+8192, t0.Add(2*time.Second))
	if !ok || r != 4 || w != 2 {
		t.Errorf("rates = %v/%v ok=%v, want 4 and 2 MB/s", r, w, ok)
	}
	// a counter that went backwards (a disk removed) must not give a negative or huge rate
	if r, w, ok := tc.diskRates(10, 10, t0.Add(4*time.Second)); ok && (r < 0 || w < 0) {
		t.Errorf("negative rate %v/%v", r, w)
	}
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

func fakeDRM(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	// card0: a discrete AMD card (8 GiB of VRAM), card1: an APU (512 MiB carve-out, a large GTT), card2: NVIDIA
	for name, v := range map[string]map[string]string{
		"card0": {"vendor": "0x1002", "mem_info_vram_total": "8589934592", "mem_info_vram_used": "1000", "mem_info_gtt_total": "17179869184", "gpu_busy_percent": "90", "product_name": "Radeon RX 7800"},
		"card1": {"vendor": "0x1002", "mem_info_vram_total": "536870912", "mem_info_vram_used": "268435456", "mem_info_gtt_total": "17179869184", "mem_info_gtt_used": "1073741824", "gpu_busy_percent": "37", "product_name": "Radeon Graphics"},
		"card2": {"vendor": "0x10de"},
	} {
		for file, content := range v {
			writeFile(t, filepath.Join(root, name, "device", file), content+"\n")
		}
	}
	writeFile(t, filepath.Join(root, "card1-DP-1", "status"), "disconnected\n") // a connector, not a card
	return root
}

func TestDiscoverIGPUFindsTheApuNotTheDiscreteCard(t *testing.T) {
	g := discoverIGPU(fakeDRM(t))
	if g == nil {
		t.Fatal("the APU (card1) must be found")
	}
	if g.name() != "Radeon Graphics" {
		t.Errorf("name = %q", g.name())
	}
	r := g.read()
	if r["util"] != 37.0 || r["mem_used"] != uint64(268435456) || r["gtt_used"] != uint64(1073741824) || r["gtt_total"] != uint64(17179869184) {
		t.Errorf("read = %v", r)
	}
	if discoverIGPU(t.TempDir()) != nil {
		t.Error("no cards: no iGPU")
	}
}

func TestParseNvidiaSmiLine(t *testing.T) {
	s, ok := parseNvidiaSmiLine("8032, 8188, 12, 41, 23.50, 4, 4, 16")
	if !ok || s.usedBytes != 8032<<20 || s.totalBytes != 8188<<20 || s.util != 12 || s.temp != 41 || s.power != 23.5 ||
		s.pcieGen != 4 || s.pcieGenMax != 4 || s.pcieWidth != 16 {
		t.Errorf("parsed %+v ok=%v", s, ok)
	}
	// power and PCIe fields not supported by the card or the driver
	s, ok = parseNvidiaSmiLine("100, 200, 3, 40, [N/A], [N/A], [N/A], [N/A]")
	if !ok || s.power != 0 || s.pcieGen != 0 || s.pcieWidth != 0 || s.totalBytes != 200<<20 {
		t.Errorf("N/A fields: %+v ok=%v", s, ok)
	}
	// the five-field line an older query gave
	if s, ok = parseNvidiaSmiLine("100, 200, 3, 40, 10.0"); !ok || s.power != 10 || s.pcieGen != 0 {
		t.Errorf("five fields: %+v ok=%v", s, ok)
	}
	if _, ok = parseNvidiaSmiLine("not csv"); ok {
		t.Error("a line that is not the query's output must be refused")
	}
}

func TestCollectorReportsDiskAndKeepsItsHistory(t *testing.T) {
	if runtime.GOOS != "linux" {
		t.Skip("disk counters come from /proc/diskstats")
	}
	tc := NewTelemetryCollector()
	tc.sampleOnce()
	time.Sleep(60 * time.Millisecond)
	tc.sampleOnce()
	snap := tc.Snapshot()
	now := snap["now"].(map[string]interface{})
	if _, ok := now["disk_read_mb"].(float64); !ok {
		t.Errorf("hardware.disk_read_mb missing after two samples: %v", now["disk_read_mb"])
	}
	if _, ok := now["disk_write_mb"].(float64); !ok {
		t.Error("hardware.disk_write_mb missing")
	}
	hist := snap["history"].(map[string]interface{})
	if h, _ := hist["disk_read_mb"].([]float64); len(h) == 0 {
		t.Errorf("history.disk_read_mb missing: %v", hist["disk_read_mb"])
	}
}
