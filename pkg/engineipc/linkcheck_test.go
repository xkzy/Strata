package engineipc

import (
	"os"
	"path/filepath"
	"testing"
)

func fakeDev(t *testing.T, root, name, vendor, class, cur, max string) {
	d := filepath.Join(root, name)
	if err := os.MkdirAll(d, 0o755); err != nil {
		t.Fatal(err)
	}
	for k, v := range map[string]string{"vendor": vendor, "class": class, "current_link_width": cur, "max_link_width": max,
		"current_link_speed": "16.0 GT/s PCIe", "max_link_speed": "32.0 GT/s PCIe"} {
		if err := os.WriteFile(filepath.Join(d, k), []byte(v+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
}

func TestLinkWarnings(t *testing.T) {
	root := t.TempDir()
	fakeDev(t, root, "0000:01:00.0", "0x10de", "0x030000", "8", "16") // x8 of x16: an 8-lane card is normal, say so
	fakeDev(t, root, "0000:02:00.0", "0x10de", "0x030000", "4", "16") // x4 of x16: narrow for any RTX card
	fakeDev(t, root, "0000:05:00.0", "0x10de", "0x030000", "8", "8")  // full width: quiet
	fakeDev(t, root, "0000:00:01.1", "0x1022", "0x060400", "4", "16") // not an NVIDIA display device: ignored
	fakeDev(t, root, "0000:01:00.1", "0x10de", "0x040300", "4", "16") // its audio function: ignored
	w := LinkWarnings(root)
	if len(w) != 2 {
		t.Fatalf("warnings: %v", w)
	}
	// sysfs gives the slot's max, not the card's: the x8 message must not claim the card allows x16 (this PC's
	// RTX 5060 Ti is a PCIe 5.0 x8 card in a Gen4 x16 slot and has linked at x8 since it was installed).
	if !contains(w[0], "0000:01:00.0") || !contains(w[0], "x8") || !contains(w[0], "x16") ||
		!contains(w[0], "8-lane card") || contains(w[0], "the slot and card allow") {
		t.Fatalf("x8 of x16: %v", w[0])
	}
	if !contains(w[1], "0000:02:00.0") || !contains(w[1], "narrower than any") {
		t.Fatalf("x4 of x16: %v", w[1])
	}
	if len(LinkWarnings(t.TempDir())) != 0 {
		t.Fatal("an empty tree must be quiet")
	}
}

func contains(s, sub string) bool {
	for i := 0; i+len(sub) <= len(s); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func TestLinkWarningsTopologyAndDeadCard(t *testing.T) {
	root := t.TempDir()
	mk := func(rel, class, cur, max string) string {
		d := filepath.Join(root, "devices", "pci0000:00", filepath.FromSlash(rel))
		fakeDev(t, filepath.Dir(d), filepath.Base(d), "0x10de", class, cur, max)
		if err := os.MkdirAll(filepath.Join(root, "bus"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(d, filepath.Join(root, "bus", filepath.Base(d))); err != nil {
			t.Fatal(err)
		}
		return d
	}
	mk("0000:00:01.1/0000:01:00.0", "0x030000", "16", "16")
	mk("0000:00:02.1/0000:03:00.2/0000:05:00.0", "0x030000", "4", "8")
	w := LinkWarnings(filepath.Join(root, "bus"))
	found := false
	for _, m := range w {
		if contains(m, "different root ports") {
			found = true
		}
	}
	if !found {
		t.Fatalf("two cards on different root ports must be reported: %v", w)
	}
	// both under one bridge: quiet
	root2 := t.TempDir()
	for _, n := range []string{"0000:02:00.0", "0000:03:00.0"} {
		d := filepath.Join(root2, "devices", "pci0000:00", "0000:00:01.1", "0000:01:00.0", n)
		fakeDev(t, filepath.Dir(d), n, "0x10de", "0x030000", "16", "16")
		os.MkdirAll(filepath.Join(root2, "bus"), 0o755)
		os.Symlink(d, filepath.Join(root2, "bus", n))
	}
	for _, m := range LinkWarnings(filepath.Join(root2, "bus")) {
		if contains(m, "different root ports") {
			t.Fatalf("cards under one bridge must not be reported: %v", m)
		}
	}
	// a card that fell off the bus
	root3 := t.TempDir()
	fakeDev(t, root3, "0000:01:00.0", "0x10de", "0x030000", "63", "63")
	if w := LinkWarnings(root3); len(w) != 1 || !contains(w[0], "does not answer") {
		t.Fatalf("dead card: %v", w)
	}
}
