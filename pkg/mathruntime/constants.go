// Package mathruntime provides the versioned physical constants registry for Strata Unified CAS.
package mathruntime

import (
	"fmt"
	"strings"
	"sync"
)

type PhysicalConstant struct {
	ID          string          `json:"id"`
	Name        string          `json:"name"`
	Symbol      string          `json:"symbol"`
	Value       float64         `json:"value"`
	Unit        string          `json:"unit"`
	Dimension   DimensionVector `json:"dimension"`
	Uncertainty float64         `json:"uncertainty"`
	Source      string          `json:"source"`
	Version     string          `json:"version"`
}

type ConstantRegistry struct {
	constants map[string]PhysicalConstant
	lookup    map[string]string
	mu        sync.RWMutex
}

var (
	defaultConstantRegistry *ConstantRegistry
	constOnce               sync.Once
)

func GetConstantRegistry() *ConstantRegistry {
	constOnce.Do(func() {
		defaultConstantRegistry = NewConstantRegistry()
	})
	return defaultConstantRegistry
}

func NewConstantRegistry() *ConstantRegistry {
	r := &ConstantRegistry{
		constants: make(map[string]PhysicalConstant),
		lookup:    make(map[string]string),
	}
	r.registerAllConstants()
	return r
}

func (r *ConstantRegistry) registerAllConstants() {
	r.constants = make(map[string]PhysicalConstant)
	r.lookup = make(map[string]string)

	consts := []PhysicalConstant{
		{
			ID:          "speed_of_light",
			Name:        "Speed of Light in Vacuum",
			Symbol:      "c",
			Value:       299792458.0,
			Unit:        "m/s",
			Dimension:   VelocityDimension,
			Uncertainty: 0.0,
			Source:      "CODATA 2022 (exact SI definition)",
			Version:     "2022.1",
		},
		{
			ID:          "gravitational_constant",
			Name:        "Newtonian Constant of Gravitation",
			Symbol:      "G",
			Value:       6.67430e-11,
			Unit:        "m^3/(kg*s^2)",
			Dimension:   DimensionVector{3, -1, -2, 0, 0, 0, 0},
			Uncertainty: 1.5e-15,
			Source:      "CODATA 2022",
			Version:     "2022.1",
		},
		{
			ID:          "planck_constant",
			Name:        "Planck Constant",
			Symbol:      "h",
			Value:       6.62607015e-34,
			Unit:        "J*s",
			Dimension:   DimensionVector{2, 1, -1, 0, 0, 0, 0},
			Uncertainty: 0.0,
			Source:      "CODATA 2022 (exact SI definition)",
			Version:     "2022.1",
		},
		{
			ID:          "reduced_planck_constant",
			Name:        "Reduced Planck Constant (Dirac Constant)",
			Symbol:      "hbar",
			Value:       1.054571817e-34,
			Unit:        "J*s",
			Dimension:   DimensionVector{2, 1, -1, 0, 0, 0, 0},
			Uncertainty: 0.0,
			Source:      "CODATA 2022 (h / 2*pi)",
			Version:     "2022.1",
		},
		{
			ID:          "elementary_charge",
			Name:        "Elementary Charge",
			Symbol:      "e",
			Value:       1.602176634e-19,
			Unit:        "C",
			Dimension:   ChargeDimension,
			Uncertainty: 0.0,
			Source:      "CODATA 2022 (exact SI definition)",
			Version:     "2022.1",
		},
		{
			ID:          "boltzmann_constant",
			Name:        "Boltzmann Constant",
			Symbol:      "k_B",
			Value:       1.380649e-23,
			Unit:        "J/K",
			Dimension:   DimensionVector{2, 1, -2, 0, -1, 0, 0},
			Uncertainty: 0.0,
			Source:      "CODATA 2022 (exact SI definition)",
			Version:     "2022.1",
		},
		{
			ID:          "avogadro_constant",
			Name:        "Avogadro Constant",
			Symbol:      "N_A",
			Value:       6.02214076e23,
			Unit:        "1/mol",
			Dimension:   DimensionVector{0, 0, 0, 0, 0, -1, 0},
			Uncertainty: 0.0,
			Source:      "CODATA 2022 (exact SI definition)",
			Version:     "2022.1",
		},
		{
			ID:          "molar_gas_constant",
			Name:        "Molar Gas Constant",
			Symbol:      "R",
			Value:       8.314462618,
			Unit:        "J/(mol*K)",
			Dimension:   DimensionVector{2, 1, -2, 0, -1, -1, 0},
			Uncertainty: 0.0,
			Source:      "CODATA 2022 (N_A * k_B)",
			Version:     "2022.1",
		},
		{
			ID:          "vacuum_permittivity",
			Name:        "Vacuum Electric Permittivity",
			Symbol:      "epsilon_0",
			Value:       8.8541878128e-12,
			Unit:        "F/m",
			Dimension:   DimensionVector{-3, -1, 4, 2, 0, 0, 0},
			Uncertainty: 1.3e-21,
			Source:      "CODATA 2022 (1 / (mu_0 * c^2))",
			Version:     "2022.1",
		},
		{
			ID:          "vacuum_permeability",
			Name:        "Vacuum Magnetic Permeability",
			Symbol:      "mu_0",
			Value:       1.25663706212e-6,
			Unit:        "H/m",
			Dimension:   DimensionVector{1, 1, -2, -2, 0, 0, 0},
			Uncertainty: 1.9e-15,
			Source:      "CODATA 2022",
			Version:     "2022.1",
		},
		{
			ID:          "standard_gravity",
			Name:        "Standard Acceleration of Gravity",
			Symbol:      "g",
			Value:       9.80665,
			Unit:        "m/s^2",
			Dimension:   AccelerationDimension,
			Uncertainty: 0.0,
			Source:      "Standard Standard ISO 80000-3",
			Version:     "2022.1",
		},
		{
			ID:          "electron_mass",
			Name:        "Electron Rest Mass",
			Symbol:      "m_e",
			Value:       9.1093837015e-31,
			Unit:        "kg",
			Dimension:   MassDimension,
			Uncertainty: 2.8e-40,
			Source:      "CODATA 2022",
			Version:     "2022.1",
		},
		{
			ID:          "proton_mass",
			Name:        "Proton Rest Mass",
			Symbol:      "m_p",
			Value:       1.67262192369e-27,
			Unit:        "kg",
			Dimension:   MassDimension,
			Uncertainty: 5.1e-37,
			Source:      "CODATA 2022",
			Version:     "2022.1",
		},
		{
			ID:          "stefan_boltzmann_constant",
			Name:        "Stefan-Boltzmann Constant",
			Symbol:      "sigma_SB",
			Value:       5.670374419e-8,
			Unit:        "W/(m^2*K^4)",
			Dimension:   DimensionVector{0, 1, -3, 0, -4, 0, 0},
			Uncertainty: 0.0,
			Source:      "CODATA 2022",
			Version:     "2022.1",
		},
		{
			ID:          "rydberg_constant",
			Name:        "Rydberg Constant",
			Symbol:      "R_inf",
			Value:       10973731.568160,
			Unit:        "1/m",
			Dimension:   DimensionVector{-1, 0, 0, 0, 0, 0, 0},
			Uncertainty: 2.1e-5,
			Source:      "CODATA 2022",
			Version:     "2022.1",
		},
	}

	for _, c := range consts {
		r.constants[c.ID] = c
		r.lookup[strings.ToLower(c.ID)] = c.ID
		r.lookup[strings.ToLower(c.Symbol)] = c.ID
		r.lookup[strings.ToLower(c.Name)] = c.ID
	}
}

func (r *ConstantRegistry) Get(idOrSymbol string) (PhysicalConstant, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	key := strings.ToLower(strings.TrimSpace(idOrSymbol))
	if id, ok := r.lookup[key]; ok {
		return r.constants[id], true
	}
	return PhysicalConstant{}, false
}

func (r *ConstantRegistry) GetQuantity(idOrSymbol string) (PhysicalQuantity, error) {
	c, ok := r.Get(idOrSymbol)
	if !ok {
		return PhysicalQuantity{}, fmt.Errorf("constant %q not found", idOrSymbol)
	}
	return PhysicalQuantity{
		NumericValue: c.Value,
		Dimension:    c.Dimension,
		UnitSymbol:   c.Unit,
	}, nil
}

func (r *ConstantRegistry) ListAll() []PhysicalConstant {
	r.mu.RLock()
	defer r.mu.RUnlock()
	res := make([]PhysicalConstant, 0, len(r.constants))
	for _, c := range r.constants {
		res = append(res, c)
	}
	return res
}
