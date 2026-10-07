package mathruntime

import (
	"context"
	"math"
	"strings"
	"testing"
)

func TestDimensionalAlgebra(t *testing.T) {
	// 10 m
	q1, err := NewQuantity(10, "m")
	if err != nil {
		t.Fatalf("failed to create q1: %v", err)
	}

	// 5 s
	q2, err := NewQuantity(5, "s")
	if err != nil {
		t.Fatalf("failed to create q2: %v", err)
	}

	// Incompatible addition: 10 m + 5 s -> ERROR
	_, err = q1.Add(q2)
	if err == nil {
		t.Errorf("expected error adding incompatible dimensions (m and s), got nil")
	}

	// Division: 10 m / 5 s -> 2 m/s (Velocity)
	v, err := q1.Divide(q2)
	if err != nil {
		t.Fatalf("failed to divide q1 by q2: %v", err)
	}
	if v.Dimension != VelocityDimension {
		t.Errorf("expected velocity dimension, got %v", v.Dimension)
	}
	if math.Abs(v.NumericValue-2.0) > 1e-9 {
		t.Errorf("expected numeric value 2.0, got %f", v.NumericValue)
	}
	if v.UnitSymbol != "m/s" {
		t.Errorf("expected unit 'm/s', got '%s'", v.UnitSymbol)
	}

	// Force = Mass * Acceleration
	// 5 kg * 2 m/s^2 -> 10 N
	m, _ := NewQuantity(5, "kg")
	a, _ := NewQuantity(2, "m/s^2")
	f := m.Multiply(a)
	if f.Dimension != ForceDimension {
		t.Errorf("expected force dimension, got %v", f.Dimension)
	}
	if math.Abs(f.NumericValue-10.0) > 1e-9 {
		t.Errorf("expected 10 N, got %f", f.NumericValue)
	}
	if f.UnitSymbol != "N" {
		t.Errorf("expected unit 'N', got '%s'", f.UnitSymbol)
	}

	// Energy = Force * Distance
	// 10 N * 3 m -> 30 J
	d, _ := NewQuantity(3, "m")
	e := f.Multiply(d)
	if e.Dimension != EnergyDimension {
		t.Errorf("expected energy dimension, got %v", e.Dimension)
	}
	if math.Abs(e.NumericValue-30.0) > 1e-9 {
		t.Errorf("expected 30 J, got %f", e.NumericValue)
	}
	if e.UnitSymbol != "J" {
		t.Errorf("expected unit 'J', got '%s'", e.UnitSymbol)
	}
}

func TestConstantRegistry(t *testing.T) {
	reg := GetConstantRegistry()
	c, ok := reg.Get("c")
	if !ok {
		t.Fatalf("constant c not found")
	}
	if math.Abs(c.Value-299792458) > 1e-6 {
		t.Errorf("expected c = 299792458, got %f", c.Value)
	}
	if c.Dimension != VelocityDimension {
		t.Errorf("expected velocity dimension for c, got %v", c.Dimension)
	}

	// Planck constant
	h, ok := reg.Get("h")
	if !ok {
		t.Fatalf("constant h not found")
	}
	if math.Abs(h.Value-6.62607015e-34) > 1e-40 {
		t.Errorf("expected h = 6.62607015e-34, got %e", h.Value)
	}
}

func TestFormulaRegistrySolvers(t *testing.T) {
	reg := GetFormulaRegistry()

	// 1. Mechanics: F = m*a
	fFormula, ok := reg.Get("mech_f_ma")
	if !ok {
		t.Fatalf("mech_f_ma formula not found")
	}
	fVal, err := fFormula.SolveFunc(map[string]float64{"m": 12.5, "a": 4.0})
	if err != nil {
		t.Fatalf("failed to solve F=ma: %v", err)
	}
	if math.Abs(fVal-50.0) > 1e-6 {
		t.Errorf("expected F = 50.0 N, got %f", fVal)
	}

	// 2. Mechanics: Kinetic Energy Ek = 0.5 * m * v^2
	ekFormula, ok := reg.Get("mech_kinetic_energy")
	if !ok {
		t.Fatalf("mech_kinetic_energy formula not found")
	}
	ekVal, err := ekFormula.SolveFunc(map[string]float64{"m": 4.0, "v": 10.0})
	if err != nil {
		t.Fatalf("failed to solve Ek: %v", err)
	}
	if math.Abs(ekVal-200.0) > 1e-6 {
		t.Errorf("expected Ek = 200.0 J, got %f", ekVal)
	}

	// 3. Circuits: Ohm's Law V = I * R
	ohmFormula, ok := reg.Get("circ_ohms_law")
	if !ok {
		t.Fatalf("circ_ohms_law formula not found")
	}
	vVal, err := ohmFormula.SolveFunc(map[string]float64{"I": 2.5, "R": 100.0})
	if err != nil {
		t.Fatalf("failed to solve V=IR: %v", err)
	}
	if math.Abs(vVal-250.0) > 1e-6 {
		t.Errorf("expected V = 250.0 V, got %f", vVal)
	}

	// 4. Circuits: RC time constant tau = R * C
	rcFormula, ok := reg.Get("circ_rc_tau")
	if !ok {
		t.Fatalf("circ_rc_tau formula not found")
	}
	tauVal, err := rcFormula.SolveFunc(map[string]float64{"R": 1000.0, "C": 1e-6})
	if err != nil {
		t.Fatalf("failed to solve tau=RC: %v", err)
	}
	if math.Abs(tauVal-0.001) > 1e-9 {
		t.Errorf("expected tau = 0.001 s (1 ms), got %f", tauVal)
	}

	// 5. Circuits: LC Resonance f0 = 1 / (2*pi*sqrt(L*C))
	lcFormula, ok := reg.Get("circ_lc_resonance")
	if !ok {
		t.Fatalf("circ_lc_resonance formula not found")
	}
	// L = 100 uH (1e-4 H), C = 100 pF (1e-10 F) -> LC = 1e-14 -> sqrt(LC) = 1e-7
	// f0 = 1 / (2*pi*1e-7) ~= 1.591549e6 Hz (~1.59 MHz)
	f0Val, err := lcFormula.SolveFunc(map[string]float64{"L": 1e-4, "C": 1e-10})
	if err != nil {
		t.Fatalf("failed to solve LC resonance: %v", err)
	}
	expectedF0 := 1.0 / (2.0 * math.Pi * 1e-7)
	if math.Abs(f0Val-expectedF0)/expectedF0 > 1e-5 {
		t.Errorf("expected f0 = %f Hz, got %f", expectedF0, f0Val)
	}

	// 6. Thermodynamics: Ideal Gas PV = nRT -> P = nRT/V
	idealPFormula, ok := reg.Get("thermo_ideal_gas_p")
	if !ok {
		t.Fatalf("thermo_ideal_gas_p formula not found")
	}
	// n = 1 mol, T = 300 K, V = 0.02494 m^3 -> P ~= 100000 Pa (1 bar)
	pVal, err := idealPFormula.SolveFunc(map[string]float64{"n": 1.0, "T": 300.0, "V": 0.02494})
	if err != nil {
		t.Fatalf("failed to solve Ideal Gas P: %v", err)
	}
	if pVal < 99000 || pVal > 101000 {
		t.Errorf("expected P ~100000 Pa, got %f", pVal)
	}

	// 7. Relativity: E = m * c^2
	relEFormula, ok := reg.Get("rel_energy_mass")
	if !ok {
		t.Fatalf("rel_energy_mass formula not found")
	}
	// m = 1 kg -> E = c^2 ~= 8.98755e16 J
	eVal, err := relEFormula.SolveFunc(map[string]float64{"m": 1.0})
	if err != nil {
		t.Fatalf("failed to solve E=mc^2: %v", err)
	}
	cVal := 299792458.0
	if math.Abs(eVal-(cVal*cVal)) > 1.0 {
		t.Errorf("expected E = %e J, got %e", cVal*cVal, eVal)
	}

	// 8. Quantum: Photon Energy E = h * f
	qPhotonFormula, ok := reg.Get("quant_photon_energy")
	if !ok {
		t.Fatalf("quant_photon_energy formula not found")
	}
	// f = 5e14 Hz (green light) -> E = h * 5e14 ~= 3.313e-19 J
	qEVal, err := qPhotonFormula.SolveFunc(map[string]float64{"f": 5e14})
	if err != nil {
		t.Fatalf("failed to solve E=hf: %v", err)
	}
	hVal := 6.62607015e-34
	if math.Abs(qEVal-(hVal*5e14)) > 1e-30 {
		t.Errorf("expected E = %e J, got %e", hVal*5e14, qEVal)
	}
}

func TestUnifiedCASProblemSolver(t *testing.T) {
	cas := GetUnifiedCASEngine()
	ctx := context.Background()

	// Problem 1: Mechanics Force
	prob1 := CASProblem{
		Domain:    DomainMechanics,
		FormulaID: "mech_f_ma",
		Variables: map[string]string{"m": "5.0 kg", "a": "9.8 m/s^2"},
		TargetVar: "F",
	}
	res1, err := cas.Solve(ctx, prob1)
	if err != nil {
		t.Fatalf("cas.Solve failed for prob1: %v", err)
	}
	if res1.Status != StatusSuccess || !strings.Contains(res1.ExactResult, "49 N") {
		t.Errorf("expected 49 N, got %+v", res1)
	}
	if res1.Dimension != ForceDimension.String() {
		t.Errorf("expected ForceDimension, got %v", res1.Dimension)
	}

	// Problem 2: Relativistic Gamma factor
	prob2 := CASProblem{
		Domain:    DomainRelativity,
		FormulaID: "rel_lorentz_gamma",
		Variables: map[string]string{"v": "239833966.4"}, // 0.8 c -> gamma = 1 / sqrt(1 - 0.64) = 1/0.6 = 1.666667
		TargetVar: "gamma",
	}
	res2, err := cas.Solve(ctx, prob2)
	if err != nil {
		t.Fatalf("cas.Solve failed for prob2: %v", err)
	}
	if !strings.Contains(res2.NumericResult, "1.66666") && !strings.Contains(res2.ExactResult, "1.66666") {
		t.Errorf("expected gamma ~ 1.666667, got %+v", res2)
	}
}

func TestPhysicalClaimVerification(t *testing.T) {
	cas := GetUnifiedCASEngine()
	ctx := context.Background()

	// True claim: 10 kg * 5 m/s^2 = 50 N
	v1, err := cas.VerifyPhysicalClaim(ctx, "10 kg * 5 m/s^2 = 50 N")
	if err != nil || !v1.Matches {
		t.Errorf("expected matching claim, got result: %+v, err: %v", v1, err)
	}

	// Contradictory claim: 10 kg * 5 m/s^2 = 100 N
	v2, err := cas.VerifyPhysicalClaim(ctx, "10 kg * 5 m/s^2 = 100 N")
	if err != nil {
		t.Fatalf("unexpected error verifying claim2: %v", err)
	}
	if v2.Matches {
		t.Errorf("expected non-matching claim, got matches=true")
	}

	// Dimensional error claim: 5 m + 3 s
	v3, err := cas.VerifyPhysicalClaim(ctx, "5 m + 3 s = 8 m")
	if err != nil {
		t.Fatalf("unexpected error verifying claim3: %v", err)
	}
	if v3.Matches {
		t.Errorf("expected dimensional error to fail matching")
	}
	if !strings.Contains(v3.DiscrepancyDetails, "DIMENSION_ERROR") {
		t.Errorf("expected DIMENSION_ERROR in details, got: %s", v3.DiscrepancyDetails)
	}
}

func TestScientificIntentInterception(t *testing.T) {
	cas := GetUnifiedCASEngine()

	// Query containing mass and acceleration
	text := "What is the net force on 25 kg at 4 m/s^2?"
	res := cas.InterceptScientificIntent(text)
	if res == nil {
		t.Fatalf("expected scientific intent to be intercepted")
	}
	if !strings.Contains(res.ExactResult, "100 N") {
		t.Errorf("expected verified fact containing '100 N', got %+v", res)
	}
}
