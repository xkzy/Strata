// Package mathruntime provides the scientific formula knowledge registry for Strata Unified CAS.
package mathruntime

import (
	"fmt"
	"math"
	"strings"
	"sync"
)

type CASDomainType string

const (
	DomainMathematics      CASDomainType = "MATHEMATICS"
	DomainMechanics        CASDomainType = "MECHANICS"
	DomainElectromagnetism CASDomainType = "ELECTROMAGNETISM"
	DomainThermodynamics   CASDomainType = "THERMODYNAMICS"
	DomainRelativity       CASDomainType = "RELATIVITY"
	DomainQuantumMechanics CASDomainType = "QUANTUM_MECHANICS"
	DomainCircuits         CASDomainType = "CIRCUITS"
	DomainSignalsControls  CASDomainType = "SIGNALS_CONTROLS"
	DomainOptics           CASDomainType = "OPTICS"
)

type ScientificFormula struct {
	ID          string          `json:"id"`
	Name        string          `json:"name"`
	Domain      CASDomainType   `json:"domain"`
	Equation    string          `json:"equation"`
	Symbols     []string        `json:"symbols"`
	TargetVar   string          `json:"target_var"`
	Dimensions  DimensionVector `json:"dimensions"`
	Unit        string          `json:"unit"`
	Assumptions []string        `json:"assumptions"`
	Literature  string          `json:"literature"`
	Version     string          `json:"version"`
	SolveFunc   func(vars map[string]float64) (float64, error) `json:"-"`
}

type FormulaRegistry struct {
	formulas map[string]ScientificFormula
	lookup   map[string]string
	mu       sync.RWMutex
}

var (
	defaultFormulaRegistry *FormulaRegistry
	formulaOnce            sync.Once
)

func GetFormulaRegistry() *FormulaRegistry {
	formulaOnce.Do(func() {
		defaultFormulaRegistry = NewFormulaRegistry()
	})
	return defaultFormulaRegistry
}

func NewFormulaRegistry() *FormulaRegistry {
	r := &FormulaRegistry{
		formulas: make(map[string]ScientificFormula),
		lookup:   make(map[string]string),
	}
	r.registerAllFormulas()
	return r
}

func (r *FormulaRegistry) registerAllFormulas() {
	r.formulas = make(map[string]ScientificFormula)
	r.lookup = make(map[string]string)

	defs := []ScientificFormula{
		// =====================================================================
		// 1. Classical Mechanics & Kinematics
		// =====================================================================
		{
			ID:          "newton_second_law",
			Name:        "Newton's Second Law of Motion",
			Domain:      DomainMechanics,
			Equation:    "F = m * a",
			Symbols:     []string{"F", "m", "a"},
			TargetVar:   "F",
			Dimensions:  ForceDimension,
			Unit:        "N",
			Assumptions: []string{"constant mass", "inertial frame", "non-relativistic (v << c)"},
			Literature:  "Isaac Newton, Philosophiæ Naturalis Principia Mathematica (1687)",
			Version:     "1.0.0",
			SolveFunc: func(vars map[string]float64) (float64, error) {
				m, okM := vars["m"]
				a, okA := vars["a"]
				if okM && okA {
					return m * a, nil
				}
				F, okF := vars["F"]
				if okF && okM && m != 0 {
					return F / m, nil
				}
				if okF && okA && a != 0 {
					return F / a, nil
				}
				return 0, fmt.Errorf("insufficient variables for Newton's 2nd Law (need 2 of {F, m, a})")
			},
		},
		{
			ID:          "linear_momentum",
			Name:        "Linear Momentum",
			Domain:      DomainMechanics,
			Equation:    "p = m * v",
			Symbols:     []string{"p", "m", "v"},
			TargetVar:   "p",
			Dimensions:  DimensionVector{1, 1, -1, 0, 0, 0, 0},
			Unit:        "kg*m/s",
			Assumptions: []string{"v << c"},
			Literature:  "Newtonian Mechanics",
			Version:     "1.0.0",
			SolveFunc: func(vars map[string]float64) (float64, error) {
				m, okM := vars["m"]
				v, okV := vars["v"]
				if okM && okV {
					return m * v, nil
				}
				p, okP := vars["p"]
				if okP && okM && m != 0 {
					return p / m, nil
				}
				if okP && okV && v != 0 {
					return p / v, nil
				}
				return 0, fmt.Errorf("need 2 of {p, m, v}")
			},
		},
		{
			ID:          "kinetic_energy",
			Name:        "Kinetic Energy",
			Domain:      DomainMechanics,
			Equation:    "E_k = 0.5 * m * v^2",
			Symbols:     []string{"E_k", "m", "v"},
			TargetVar:   "E_k",
			Dimensions:  EnergyDimension,
			Unit:        "J",
			Assumptions: []string{"translational motion", "v << c"},
			Literature:  "Gottfried Leibniz (vis viva, 1686); Thomas Young (1807)",
			Version:     "1.0.0",
			SolveFunc: func(vars map[string]float64) (float64, error) {
				m, okM := vars["m"]
				v, okV := vars["v"]
				if okM && okV {
					return 0.5 * m * v * v, nil
				}
				return 0, fmt.Errorf("need m and v")
			},
		},
		{
			ID:          "gravitational_potential_energy",
			Name:        "Gravitational Potential Energy (Uniform Field)",
			Domain:      DomainMechanics,
			Equation:    "U = m * g * h",
			Symbols:     []string{"U", "m", "g", "h"},
			TargetVar:   "U",
			Dimensions:  EnergyDimension,
			Unit:        "J",
			Assumptions: []string{"uniform gravitational field g ~ 9.80665 m/s^2"},
			Literature:  "Classical Mechanics",
			Version:     "1.0.0",
			SolveFunc: func(vars map[string]float64) (float64, error) {
				m, okM := vars["m"]
				h, okH := vars["h"]
				g := 9.80665
				if vG, ok := vars["g"]; ok {
					g = vG
				}
				if okM && okH {
					return m * g * h, nil
				}
				return 0, fmt.Errorf("need m and h")
			},
		},
		{
			ID:          "mechanical_work",
			Name:        "Mechanical Work",
			Domain:      DomainMechanics,
			Equation:    "W = F * d",
			Symbols:     []string{"W", "F", "d"},
			TargetVar:   "W",
			Dimensions:  EnergyDimension,
			Unit:        "J",
			Assumptions: []string{"constant collinear force"},
			Literature:  "Gaspard-Gustave de Coriolis (1826)",
			Version:     "1.0.0",
			SolveFunc: func(vars map[string]float64) (float64, error) {
				f, okF := vars["F"]
				d, okD := vars["d"]
				if okF && okD {
					return f * d, nil
				}
				return 0, fmt.Errorf("need F and d")
			},
		},
		{
			ID:          "mechanical_power",
			Name:        "Mechanical Power",
			Domain:      DomainMechanics,
			Equation:    "P = W / t",
			Symbols:     []string{"P", "W", "t"},
			TargetVar:   "P",
			Dimensions:  PowerDimension,
			Unit:        "W",
			Assumptions: []string{"average power over interval t > 0"},
			Literature:  "James Watt (1782)",
			Version:     "1.0.0",
			SolveFunc: func(vars map[string]float64) (float64, error) {
				w, okW := vars["W"]
				t, okT := vars["t"]
				if okW && okT && t != 0 {
					return w / t, nil
				}
				return 0, fmt.Errorf("need W and t > 0")
			},
		},
		{
			ID:          "kinematics_position",
			Name:        "Kinematics Position Equation",
			Domain:      DomainMechanics,
			Equation:    "x = x0 + v0 * t + 0.5 * a * t^2",
			Symbols:     []string{"x", "x0", "v0", "t", "a"},
			TargetVar:   "x",
			Dimensions:  LengthDimension,
			Unit:        "m",
			Assumptions: []string{"constant acceleration a"},
			Literature:  "Galileo Galilei (1638)",
			Version:     "1.0.0",
			SolveFunc: func(vars map[string]float64) (float64, error) {
				t, okT := vars["t"]
				a, okA := vars["a"]
				if !okT || !okA {
					return 0, fmt.Errorf("need t and a")
				}
				x0 := vars["x0"]
				v0 := vars["v0"]
				return x0 + v0*t + 0.5*a*t*t, nil
			},
		},
		{
			ID:          "kinematics_velocity",
			Name:        "Kinematics Velocity Equation",
			Domain:      DomainMechanics,
			Equation:    "v = v0 + a * t",
			Symbols:     []string{"v", "v0", "a", "t"},
			TargetVar:   "v",
			Dimensions:  VelocityDimension,
			Unit:        "m/s",
			Assumptions: []string{"constant acceleration a"},
			Literature:  "Galileo Galilei (1638)",
			Version:     "1.0.0",
			SolveFunc: func(vars map[string]float64) (float64, error) {
				a, okA := vars["a"]
				t, okT := vars["t"]
				if okA && okT {
					v0 := vars["v0"]
					return v0 + a*t, nil
				}
				return 0, fmt.Errorf("need a and t")
			},
		},

		// =====================================================================
		// 2. Electromagnetism & Circuits
		// =====================================================================
		{
			ID:          "ohms_law",
			Name:        "Ohm's Law",
			Domain:      DomainCircuits,
			Equation:    "V = I * R",
			Symbols:     []string{"V", "I", "R"},
			TargetVar:   "V",
			Dimensions:  VoltageDimension,
			Unit:        "V",
			Assumptions: []string{"linear ohmic conductor", "constant temperature"},
			Literature:  "Georg Simon Ohm (1827)",
			Version:     "1.0.0",
			SolveFunc: func(vars map[string]float64) (float64, error) {
				i, okI := vars["I"]
				r, okR := vars["R"]
				if okI && okR {
					return i * r, nil
				}
				v, okV := vars["V"]
				if okV && okR && r != 0 {
					return v / r, nil
				}
				if okV && okI && i != 0 {
					return v / i, nil
				}
				return 0, fmt.Errorf("need 2 of {V, I, R}")
			},
		},
		{
			ID:          "electric_power",
			Name:        "Electric Power (Joule Heating)",
			Domain:      DomainCircuits,
			Equation:    "P = V * I",
			Symbols:     []string{"P", "V", "I"},
			TargetVar:   "P",
			Dimensions:  PowerDimension,
			Unit:        "W",
			Assumptions: []string{"steady-state DC or instantaneous AC"},
			Literature:  "James Prescott Joule (1841)",
			Version:     "1.0.0",
			SolveFunc: func(vars map[string]float64) (float64, error) {
				v, okV := vars["V"]
				i, okI := vars["I"]
				if okV && okI {
					return v * i, nil
				}
				return 0, fmt.Errorf("need V and I")
			},
		},
		{
			ID:          "capacitance_charge",
			Name:        "Capacitor Charge Relation",
			Domain:      DomainCircuits,
			Equation:    "Q = C * V",
			Symbols:     []string{"Q", "C", "V"},
			TargetVar:   "Q",
			Dimensions:  ChargeDimension,
			Unit:        "C",
			Assumptions: []string{"linear dielectric"},
			Literature:  "Michael Faraday (1837)",
			Version:     "1.0.0",
			SolveFunc: func(vars map[string]float64) (float64, error) {
				c, okC := vars["C"]
				v, okV := vars["V"]
				if okC && okV {
					return c * v, nil
				}
				return 0, fmt.Errorf("need C and V")
			},
		},
		{
			ID:          "rc_time_constant",
			Name:        "RC Circuit Time Constant",
			Domain:      DomainCircuits,
			Equation:    "tau = R * C",
			Symbols:     []string{"tau", "R", "C"},
			TargetVar:   "tau",
			Dimensions:  TimeDimension,
			Unit:        "s",
			Assumptions: []string{"first-order RC low/high pass filter"},
			Literature:  "Linear Circuit Theory",
			Version:     "1.0.0",
			SolveFunc: func(vars map[string]float64) (float64, error) {
				r, okR := vars["R"]
				c, okC := vars["C"]
				if okR && okC {
					return r * c, nil
				}
				return 0, fmt.Errorf("need R and C")
			},
		},
		{
			ID:          "rl_time_constant",
			Name:        "RL Circuit Time Constant",
			Domain:      DomainCircuits,
			Equation:    "tau = L / R",
			Symbols:     []string{"tau", "L", "R"},
			TargetVar:   "tau",
			Dimensions:  TimeDimension,
			Unit:        "s",
			Assumptions: []string{"first-order RL circuit", "R > 0"},
			Literature:  "Linear Circuit Theory",
			Version:     "1.0.0",
			SolveFunc: func(vars map[string]float64) (float64, error) {
				l, okL := vars["L"]
				r, okR := vars["R"]
				if okL && okR && r != 0 {
					return l / r, nil
				}
				return 0, fmt.Errorf("need L and R > 0")
			},
		},
		{
			ID:          "lc_resonance_frequency",
			Name:        "LC Resonant Frequency",
			Domain:      DomainCircuits,
			Equation:    "f_0 = 1 / (2 * pi * sqrt(L * C))",
			Symbols:     []string{"f_0", "L", "C"},
			TargetVar:   "f_0",
			Dimensions:  FrequencyDimension,
			Unit:        "Hz",
			Assumptions: []string{"ideal LC tank circuit without resistance"},
			Literature:  "Oliver Lodge (1889); Heinrich Hertz (1887)",
			Version:     "1.0.0",
			SolveFunc: func(vars map[string]float64) (float64, error) {
				l, okL := vars["L"]
				c, okC := vars["C"]
				if okL && okC && l > 0 && c > 0 {
					importMath := 3.141592653589793
					return 1.0 / (2.0 * importMath * math.Sqrt(l*c)), nil
				}
				return 0, fmt.Errorf("need L > 0 and C > 0")
			},
		},

		// =====================================================================
		// 3. Thermodynamics
		// =====================================================================
		{
			ID:          "ideal_gas_law",
			Name:        "Ideal Gas Law",
			Domain:      DomainThermodynamics,
			Equation:    "P * V = n * R * T",
			Symbols:     []string{"P", "V", "n", "R", "T"},
			TargetVar:   "P",
			Dimensions:  PressureDimension,
			Unit:        "Pa",
			Assumptions: []string{"ideal gas (point particles, elastic collisions)", "T in Kelvin"},
			Literature:  "Benoît Paul Émile Clapeyron (1834)",
			Version:     "1.0.0",
			SolveFunc: func(vars map[string]float64) (float64, error) {
				n, okN := vars["n"]
				t, okT := vars["T"]
				v, okV := vars["V"]
				R := 8.314462618
				if rVal, ok := vars["R"]; ok {
					R = rVal
				}
				if okN && okT && okV && v != 0 {
					return (n * R * t) / v, nil
				}
				return 0, fmt.Errorf("need n, T, and V > 0")
			},
		},
		{
			ID:          "first_law_thermodynamics",
			Name:        "First Law of Thermodynamics (Energy Conservation)",
			Domain:      DomainThermodynamics,
			Equation:    "delta_U = Q - W",
			Symbols:     []string{"delta_U", "Q", "W"},
			TargetVar:   "delta_U",
			Dimensions:  EnergyDimension,
			Unit:        "J",
			Assumptions: []string{"closed system", "W = work done by system on surroundings"},
			Literature:  "Rudolf Clausius (1850); William Rankine (1859)",
			Version:     "1.0.0",
			SolveFunc: func(vars map[string]float64) (float64, error) {
				q, okQ := vars["Q"]
				w, okW := vars["W"]
				if okQ && okW {
					return q - w, nil
				}
				return 0, fmt.Errorf("need Q and W")
			},
		},
		{
			ID:          "carnot_efficiency",
			Name:        "Carnot Heat Engine Maximum Efficiency",
			Domain:      DomainThermodynamics,
			Equation:    "eta = 1 - T_C / T_H",
			Symbols:     []string{"eta", "T_C", "T_H"},
			TargetVar:   "eta",
			Dimensions:  DimlessDimension,
			Unit:        "1",
			Assumptions: []string{"reversible Carnot cycle", "T_H > T_C > 0 in Kelvin"},
			Literature:  "Nicolas Léonard Sadi Carnot (1824)",
			Version:     "1.0.0",
			SolveFunc: func(vars map[string]float64) (float64, error) {
				tc, okC := vars["T_C"]
				th, okH := vars["T_H"]
				if okC && okH && th > 0 {
					return 1.0 - (tc / th), nil
				}
				return 0, fmt.Errorf("need T_C and T_H > 0 (in Kelvin)")
			},
		},

		// =====================================================================
		// 4. Relativity & Modern Physics
		// =====================================================================
		{
			ID:          "mass_energy_equivalence",
			Name:        "Mass-Energy Equivalence",
			Domain:      DomainRelativity,
			Equation:    "E = m * c^2",
			Symbols:     []string{"E", "m", "c"},
			TargetVar:   "E",
			Dimensions:  EnergyDimension,
			Unit:        "J",
			Assumptions: []string{"rest frame of mass m"},
			Literature:  "Albert Einstein, Annalen der Physik (1905)",
			Version:     "1.0.0",
			SolveFunc: func(vars map[string]float64) (float64, error) {
				m, okM := vars["m"]
				c := 299792458.0
				if cVal, ok := vars["c"]; ok {
					c = cVal
				}
				if okM {
					return m * c * c, nil
				}
				return 0, fmt.Errorf("need mass m")
			},
		},
		{
			ID:          "lorentz_factor",
			Name:        "Lorentz Gamma Factor",
			Domain:      DomainRelativity,
			Equation:    "gamma = 1 / sqrt(1 - (v / c)^2)",
			Symbols:     []string{"gamma", "v", "c"},
			TargetVar:   "gamma",
			Dimensions:  DimlessDimension,
			Unit:        "1",
			Assumptions: []string{"speed v < c"},
			Literature:  "Hendrik Lorentz (1892); Albert Einstein (1905)",
			Version:     "1.0.0",
			SolveFunc: func(vars map[string]float64) (float64, error) {
				v, okV := vars["v"]
				c := 299792458.0
				if cVal, ok := vars["c"]; ok {
					c = cVal
				}
				if okV && math.Abs(v) < c {
					beta := v / c
					return 1.0 / math.Sqrt(1.0-beta*beta), nil
				}
				return 0, fmt.Errorf("need v < c")
			},
		},

		// =====================================================================
		// 5. Quantum Mechanics
		// =====================================================================
		{
			ID:          "planck_einstein_relation",
			Name:        "Planck-Einstein Photon Energy",
			Domain:      DomainQuantumMechanics,
			Equation:    "E = h * f",
			Symbols:     []string{"E", "h", "f"},
			TargetVar:   "E",
			Dimensions:  EnergyDimension,
			Unit:        "J",
			Assumptions: []string{"single photon of frequency f"},
			Literature:  "Max Planck (1900); Albert Einstein (1905)",
			Version:     "1.0.0",
			SolveFunc: func(vars map[string]float64) (float64, error) {
				f, okF := vars["f"]
				h := 6.62607015e-34
				if hVal, ok := vars["h"]; ok {
					h = hVal
				}
				if okF {
					return h * f, nil
				}
				return 0, fmt.Errorf("need frequency f")
			},
		},
		{
			ID:          "de_broglie_wavelength",
			Name:        "de Broglie Matter Wavelength",
			Domain:      DomainQuantumMechanics,
			Equation:    "lambda = h / p",
			Symbols:     []string{"lambda", "h", "p"},
			TargetVar:   "lambda",
			Dimensions:  LengthDimension,
			Unit:        "m",
			Assumptions: []string{"momentum p > 0"},
			Literature:  "Louis de Broglie (1924)",
			Version:     "1.0.0",
			SolveFunc: func(vars map[string]float64) (float64, error) {
				p, okP := vars["p"]
				h := 6.62607015e-34
				if hVal, ok := vars["h"]; ok {
					h = hVal
				}
				if okP && p > 0 {
					return h / p, nil
				}
				return 0, fmt.Errorf("need momentum p > 0")
			},
		},
	}

	for _, f := range defs {
		r.formulas[f.ID] = f
		r.lookup[strings.ToLower(f.ID)] = f.ID
		r.lookup[strings.ToLower(f.Name)] = f.ID
	}

	aliases := map[string]string{
		"mech_f_ma":           "newton_second_law",
		"f_ma":                "newton_second_law",
		"f=ma":                "newton_second_law",
		"mech_kinetic_energy": "kinetic_energy",
		"ek":                  "kinetic_energy",
		"circ_ohms_law":       "ohms_law",
		"v_ir":                "ohms_law",
		"v=ir":                "ohms_law",
		"circ_electric_power": "electric_power",
		"p_vi":                "electric_power",
		"circ_rc_tau":         "rc_time_constant",
		"rc_tau":              "rc_time_constant",
		"circ_lc_resonance":   "lc_resonance_frequency",
		"lc_resonance":        "lc_resonance_frequency",
		"thermo_ideal_gas_p":  "ideal_gas_law",
		"ideal_gas_p":         "ideal_gas_law",
		"ideal_gas_law_p":     "ideal_gas_law",
		"rel_energy_mass":     "mass_energy_equivalence",
		"e=mc^2":              "mass_energy_equivalence",
		"rel_lorentz_gamma":   "lorentz_factor",
		"lorentz_gamma":       "lorentz_factor",
		"gamma":               "lorentz_factor",
		"quant_photon_energy": "planck_einstein_relation",
		"photon_energy":       "planck_einstein_relation",
		"e=hf":                "planck_einstein_relation",
		"quant_de_broglie":    "de_broglie_wavelength",
	}
	for alias, target := range aliases {
		r.lookup[strings.ToLower(alias)] = target
	}
}

func (r *FormulaRegistry) Get(idOrName string) (ScientificFormula, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	key := strings.ToLower(strings.TrimSpace(idOrName))
	if id, ok := r.lookup[key]; ok {
		return r.formulas[id], true
	}
	return ScientificFormula{}, false
}

func (r *FormulaRegistry) ListByDomain(domain CASDomainType) []ScientificFormula {
	r.mu.RLock()
	defer r.mu.RUnlock()
	var res []ScientificFormula
	for _, f := range r.formulas {
		if f.Domain == domain {
			res = append(res, f)
		}
	}
	return res
}

func (r *FormulaRegistry) ListAll() []ScientificFormula {
	r.mu.RLock()
	defer r.mu.RUnlock()
	res := make([]ScientificFormula, 0, len(r.formulas))
	for _, f := range r.formulas {
		res = append(res, f)
	}
	return res
}
