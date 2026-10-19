// pkg/engineipc/linkcheck.go - warns at start when an NVIDIA card's PCIe link trained narrower than the card can do.
// A card at half its lanes (a loose card, dirty contacts, a slot shared or split in the BIOS) is slower, and a link that is
// marginal can drop the card off the bus under load (Xid 79).  The check only reads Linux sysfs.
//
// sysfs gives the slot's max_link_width, not the card's: a card that uses only 8 lanes (RTX 4060/4060 Ti, 5060/5060 Ti)
// links at x8 in a x16 slot by design, and reading the card's own Link Capabilities needs root (the config file stops
// at the 64-byte header for an unprivileged reader). So an x8 link is reported with that caveat rather than as a fault.
package engineipc

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

func sysfsValue(dir, name string) string {
	b, err := os.ReadFile(filepath.Join(dir, name))
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(b))
}

func linkWidth(s string) int {
	n, err := strconv.Atoi(strings.TrimPrefix(strings.TrimSpace(s), "x"))
	if err != nil {
		return 0
	}
	return n
}

// LinkWarnings looks at every NVIDIA display device under root (normally /sys/bus/pci/devices).
func LinkWarnings(root string) []string {
	dirs, _ := filepath.Glob(filepath.Join(root, "*"))
	var out []string
	var gpus []string
	for _, d := range dirs {
		if sysfsValue(d, "vendor") != "0x10de" || !strings.HasPrefix(sysfsValue(d, "class"), "0x03") {
			continue
		}
		cur, max := linkWidth(sysfsValue(d, "current_link_width")), linkWidth(sysfsValue(d, "max_link_width"))
		if cur > 16 || max > 16 { // a card that fell off the bus reads all ones (x63)
			out = append(out, fmt.Sprintf("GPU %s: the card does not answer on the PCIe bus (link reads x%d). It needs a "+
				"full power-off before it can be used again; that does not fix the cause.", filepath.Base(d), cur))
			continue
		}
		gpus = append(gpus, d)
		if real, err := filepath.EvalSymlinks(d); err == nil { // the slot's own port may offer fewer lanes than the card
			if pm := linkWidth(sysfsValue(filepath.Dir(real), "max_link_width")); pm > 0 && pm < max {
				max = pm
			}
		}
		if cur > 0 && max > 0 && cur < max {
			if cur >= 8 {
				out = append(out, fmt.Sprintf("GPU %s: the PCIe link is x%d of the slot's x%d. An 8-lane card "+
					"(RTX 4060/4060 Ti, 5060/5060 Ti) links at x8 by design; a 16-lane card at x8 has lost half its lanes: "+
					"reseat it, check its power cable and the BIOS (no x8/x8 split on this slot). Slow, and a marginal link "+
					"can drop the card off the bus under load (Xid 79).", filepath.Base(d), cur, max))
			} else {
				out = append(out, fmt.Sprintf("GPU %s: the PCIe link trained at x%d of the slot's x%d, narrower than any "+
					"desktop RTX card uses. Reseat the card and check its power cable and the BIOS (no x8/x8 split on this "+
					"slot). Slow, and a marginal link can drop the card off the bus under load (Xid 79).",
					filepath.Base(d), cur, max))
			}
		}
	}
	return append(out, topologyNotes(gpus)...)
}

// topologyNotes says, for each pair of cards, where they meet on the PCIe tree.  Only a bridge below the root complex that both
// hang under allows a direct card-to-card path; at the host bridge alone ("PHB") the traffic goes through the CPU and host memory.
func topologyNotes(gpus []string) []string {
	var out []string
	for i := 0; i < len(gpus); i++ {
		for j := i + 1; j < len(gpus); j++ {
			a, errA := filepath.EvalSymlinks(gpus[i])
			b, errB := filepath.EvalSymlinks(gpus[j])
			if errA != nil || errB != nil {
				continue
			}
			shared := commonDir(a, b)
			if strings.HasPrefix(filepath.Base(shared), "pci") || filepath.Base(shared) == "devices" || shared == "/" {
				out = append(out, fmt.Sprintf("GPUs %s and %s sit on different root ports (they meet only at the CPU's host bridge): no direct "+
					"card-to-card path, so the second GPU's traffic goes through host memory. A faster prompt path needs both cards under one "+
					"PCIe switch/bridge, which most desktop boards do not have.", filepath.Base(gpus[i]), filepath.Base(gpus[j])))
			}
		}
	}
	return out
}

func commonDir(a, b string) string {
	pa, pb := strings.Split(a, string(filepath.Separator)), strings.Split(b, string(filepath.Separator))
	n := 0
	for n < len(pa) && n < len(pb) && pa[n] == pb[n] {
		n++
	}
	if n == 0 {
		return "/"
	}
	return strings.Join(pa[:n], string(filepath.Separator))
}
