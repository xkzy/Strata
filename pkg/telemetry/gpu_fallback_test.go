package telemetry

import (
	"os"
	"path/filepath"
	"testing"
)

func writeSysfs(t *testing.T, root string, files map[string]string) {
	t.Helper()
	for rel, content := range files {
		p := filepath.Join(root, rel)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(content), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

// The fallback card must report the values its driver gives, including temperature and power from hwmon.
func TestAMDFallbackReadsRealTempAndPower(t *testing.T) {
	root := t.TempDir()
	writeSysfs(t, root, map[string]string{
		"card0/device/mem_info_vram_total":         "17163091968",
		"card0/device/mem_info_vram_used":          "123456789",
		"card0/device/gpu_busy_percent":            "63",
		"card0/device/hwmon/hwmon7/temp1_input":    "45000",
		"card0/device/hwmon/hwmon7/power1_average": "50000000",
	})
	r := readAMDSysfs(root)
	if !r.present || r.total != 17163091968 || r.used != 123456789 || r.util != 63 {
		t.Fatalf("fallback reading = %+v", r)
	}
	if !r.hasTemp || r.temp != 45 {
		t.Errorf("temperature from hwmon: %v (present %v)", r.temp, r.hasTemp)
	}
	if !r.hasPower || r.power != 50 {
		t.Errorf("power from hwmon: %v (present %v)", r.power, r.hasPower)
	}
}

// A driver that exposes no hwmon must be reported as not knowing the temperature and power. The old path
// returned a fixed 45 C and 50 W, which the controller would have trusted.
func TestAMDFallbackDoesNotInventTempOrPower(t *testing.T) {
	root := t.TempDir()
	writeSysfs(t, root, map[string]string{
		"card0/device/mem_info_vram_total": "8589934592",
		"card0/device/mem_info_vram_used":  "1000",
		"card0/device/gpu_busy_percent":    "5",
	})
	r := readAMDSysfs(root)
	if !r.present {
		t.Fatal("the card is there, only its hwmon is missing")
	}
	if r.hasTemp || r.hasPower {
		t.Errorf("no temperature or power was reported, but the reading claims one: %+v", r)
	}
	tc := NewTelemetryCollector()
	tc.mu.Lock()
	tc.now["gpu_mem_total"], tc.now["gpu_mem_used"] = r.total, r.used
	tc.now["gpu_util"] = r.util
	tc.now["gpu_temp"] = 40.0
	tc.now["gpu_power"] = 50.0
	tc.publishGPU(r)
	_, tempSet := tc.now["gpu_temp"]
	_, powerSet := tc.now["gpu_power"]
	tc.mu.Unlock()
	if tempSet || powerSet {
		t.Errorf("a value the driver does not report must be absent, not kept: temp=%v power=%v", tempSet, powerSet)
	}
}

// The card with the most VRAM is the one counted.
func TestAMDFallbackPicksTheBiggerCard(t *testing.T) {
	root := t.TempDir()
	writeSysfs(t, root, map[string]string{
		"card0/device/mem_info_vram_total": "2147483648",
		"card0/device/mem_info_vram_used":  "111",
		"card1/device/mem_info_vram_total": "17163091968",
		"card1/device/mem_info_vram_used":  "222",
	})
	r := readAMDSysfs(root)
	if !r.present || r.total != 17163091968 || r.used != 222 {
		t.Errorf("fallback card = %+v, want the one with the most VRAM", r)
	}
}

// A card that stops answering (a fall-off, a wedged driver, no GPU at all) must not be published as empty: the
// live values are unknown. The total stays, because it is a hardware fact from the static probe.
func TestLostCardIsNotPublishedAsAnEmptyGPU(t *testing.T) {
	tc := NewTelemetryCollector()
	tc.mu.Lock()
	tc.now["gpu_mem_total"] = uint64(16 << 30)
	tc.publishGPU(gpuReading{present: true, total: 16 << 30, used: 4 << 30, util: 37, temp: 44, hasTemp: true, power: 60, hasPower: true})
	tc.pushGPUHistory(gpuReading{present: true, util: 37, used: 4 << 30, temp: 44, power: 60})
	tc.publishGPU(gpuReading{}) // what readGPU returns when no driver answers
	_, used := tc.now["gpu_mem_used"]
	_, util := tc.now["gpu_util"]
	_, temp := tc.now["gpu_temp"]
	_, power := tc.now["gpu_power"]
	total := tc.now["gpu_mem_total"]
	tc.pushGPUHistory(gpuReading{})
	histUtil := tc.historyGPUUtil[len(tc.historyGPUUtil)-1]
	tc.mu.Unlock()
	if used || util || temp || power {
		t.Errorf("a lost card must publish no live values: used=%v util=%v temp=%v power=%v", used, util, temp, power)
	}
	if total != uint64(16<<30) {
		t.Errorf("the static total is a hardware fact and stays: %v", total)
	}
	if histUtil != 37 {
		t.Errorf("the sparkline must keep the last known value, not draw a dip to zero: %v", histUtil)
	}
}
