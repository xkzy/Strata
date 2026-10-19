// pkg/autotune/apply.go - apply only the parts of a plan that have evidence.
//
// The tuner never rewrites a flag it did not measure or prove:
//
//	--pool-workers  needs a measured CPU scaling curve
//	--pcie-frac     needs a measured engine rate for that exact value (the engine's own probe is better than a guess)
//	--spec-min-p    needs a measured engine rate for that exact value
//
// Device placement is reported in the plan and in /metrics but is not rewritten here: there is no engine argument
// for it (the engine chooses the card; the server exposes the visible devices through its environment), and
// changing CUDA_VISIBLE_DEVICES from the tuner would be a blind change with a failure mode the tuner cannot
// validate. That limitation is documented in docs/optimization/startup-benchmark.md.
package autotune

import "strings"

// ApplyEvidence is what the caller is allowed to apply.
type ApplyEvidence struct {
	PoolWorkersMeasured bool `json:"pool_workers_measured"`
	PCIeMeasured        bool `json:"pcie_measured"`
	SpecMeasured        bool `json:"spec_measured"`
}

// ApplyArgs returns a copy of args with the measured controls set. Existing occurrences are replaced in place so
// the engine's last-wins parsing cannot be confused by an appended duplicate.
func ApplyArgs(args []string, selected Candidate, ev ApplyEvidence) []string {
	out := append([]string(nil), args...)
	if ev.PoolWorkersMeasured && selected.PoolWorkers > 0 {
		out = setFlag(out, "--pool-workers", itoa(selected.PoolWorkers))
	}
	if ev.PCIeMeasured && selected.PCIeFrac >= 0 {
		out = setFlag(out, "--pcie-frac", format2(selected.PCIeFrac))
	}
	if ev.SpecMeasured && selected.SpecMinP > 0 {
		out = setFlag(out, "--spec-min-p", format2(selected.SpecMinP))
	}
	return out
}

// EvidenceFromMeasurements derives the apply permission from what was actually measured for this candidate.
func EvidenceFromMeasurements(c Candidate, m Measurements) ApplyEvidence {
	key := c.Key()
	_, measured := m.DecodeTokS[key]
	_, pool := m.BestPoolWorkers()
	return ApplyEvidence{
		PoolWorkersMeasured: pool && c.PoolWorkers > 0,
		PCIeMeasured:        measured && c.PCIeFrac >= 0,
		SpecMeasured:        measured && c.SpecMinP > 0,
	}
}

func setFlag(args []string, flag, value string) []string {
	for i := 0; i < len(args); i++ {
		if args[i] == flag {
			if i+1 < len(args) {
				args[i+1] = value
				return args
			}
			return append(args, value)
		}
		if strings.HasPrefix(args[i], flag+"=") {
			args[i] = flag + "=" + value
			return args
		}
	}
	return append(args, flag, value)
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	neg := v < 0
	if neg {
		v = -v
	}
	var b [20]byte
	i := len(b)
	for v > 0 {
		i--
		b[i] = byte('0' + v%10)
		v /= 10
	}
	if neg {
		i--
		b[i] = '-'
	}
	return string(b[i:])
}

func format2(v float64) string {
	// The engine parses with atof; two decimals is the precision its own probe prints.
	n := int(v*100 + 0.5)
	if v < 0 {
		n = int(v*100 - 0.5)
	}
	sign := ""
	if n < 0 {
		sign, n = "-", -n
	}
	return sign + itoa(n/100) + "." + pad2(n%100)
}

func pad2(v int) string {
	if v < 10 {
		return "0" + itoa(v)
	}
	return itoa(v)
}
