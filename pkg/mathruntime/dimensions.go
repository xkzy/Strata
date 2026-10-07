// Package mathruntime provides the deterministic mathematical and physical CAS engine for Strata.
package mathruntime

import (
	"fmt"
	"math/big"
	"strings"
)

// DimensionVector represents powers of SI base dimensions:
// [Length, Mass, Time, Current, Temperature, Amount, LuminousIntensity]
type DimensionVector [7]int8

const (
	DimLength = 0
	DimMass   = 1
	DimTime   = 2
	DimCurrent = 3
	DimTemp   = 4
	DimAmount = 5
	DimLum    = 6
)

var (
	DimlessDimension    = DimensionVector{0, 0, 0, 0, 0, 0, 0}
	LengthDimension     = DimensionVector{1, 0, 0, 0, 0, 0, 0}
	MassDimension       = DimensionVector{0, 1, 0, 0, 0, 0, 0}
	TimeDimension       = DimensionVector{0, 0, 1, 0, 0, 0, 0}
	CurrentDimension    = DimensionVector{0, 0, 0, 1, 0, 0, 0}
	TempDimension       = DimensionVector{0, 0, 0, 0, 1, 0, 0}
	AmountDimension     = DimensionVector{0, 0, 0, 0, 0, 1, 0}
	LuminousDimension   = DimensionVector{0, 0, 0, 0, 0, 0, 1}

	// Derived dimensions
	VelocityDimension     = DimensionVector{1, 0, -1, 0, 0, 0, 0}     // L T^-1
	AccelerationDimension = DimensionVector{1, 0, -2, 0, 0, 0, 0}     // L T^-2
	ForceDimension        = DimensionVector{1, 1, -2, 0, 0, 0, 0}     // M L T^-2
	EnergyDimension       = DimensionVector{2, 1, -2, 0, 0, 0, 0}     // M L^2 T^-2
	PowerDimension        = DimensionVector{2, 1, -3, 0, 0, 0, 0}     // M L^2 T^-3
	PressureDimension     = DimensionVector{-1, 1, -2, 0, 0, 0, 0}    // M L^-1 T^-2
	ChargeDimension       = DimensionVector{0, 0, 1, 1, 0, 0, 0}      // I T
	VoltageDimension      = DimensionVector{2, 1, -3, -1, 0, 0, 0}    // M L^2 T^-3 I^-1
	ResistanceDimension   = DimensionVector{2, 1, -3, -2, 0, 0, 0}    // M L^2 T^-3 I^-2
	CapacitanceDimension  = DimensionVector{-2, -1, 4, 2, 0, 0, 0}    // M^-1 L^-2 T^4 I^2
	InductanceDimension   = DimensionVector{2, 1, -2, -2, 0, 0, 0}    // M L^2 T^-2 I^-2
	FrequencyDimension    = DimensionVector{0, 0, -1, 0, 0, 0, 0}     // T^-1
	MagneticFieldDimension= DimensionVector{0, 1, -2, -1, 0, 0, 0}    // M T^-2 I^-1 (Tesla)
)

func (d DimensionVector) IsDimensionless() bool {
	return d == DimlessDimension
}

func (d DimensionVector) Multiply(other DimensionVector) DimensionVector {
	var res DimensionVector
	for i := 0; i < 7; i++ {
		res[i] = d[i] + other[i]
	}
	return res
}

func (d DimensionVector) Divide(other DimensionVector) DimensionVector {
	var res DimensionVector
	for i := 0; i < 7; i++ {
		res[i] = d[i] - other[i]
	}
	return res
}

func (d DimensionVector) Pow(n int8) DimensionVector {
	var res DimensionVector
	for i := 0; i < 7; i++ {
		res[i] = d[i] * n
	}
	return res
}

func (d DimensionVector) String() string {
	if d.IsDimensionless() {
		return "1"
	}
	var parts []string
	names := [7]string{"L", "M", "T", "I", "Theta", "N", "J"}
	for i, p := range d {
		if p == 1 {
			parts = append(parts, names[i])
		} else if p != 0 {
			parts = append(parts, fmt.Sprintf("%s^%d", names[i], p))
		}
	}
	return strings.Join(parts, "*")
}

// UnitDefinition defines a measurement unit with scale factor to SI base
type UnitDefinition struct {
	Symbol     string
	Name       string
	Dimension  DimensionVector
	ScaleToSI  float64
	OffsetToSI float64
}

var standardUnits = map[string]UnitDefinition{
	// Length
	"m":   {Symbol: "m", Name: "meter", Dimension: LengthDimension, ScaleToSI: 1.0},
	"km":  {Symbol: "km", Name: "kilometer", Dimension: LengthDimension, ScaleToSI: 1000.0},
	"cm":  {Symbol: "cm", Name: "centimeter", Dimension: LengthDimension, ScaleToSI: 0.01},
	"mm":  {Symbol: "mm", Name: "millimeter", Dimension: LengthDimension, ScaleToSI: 0.001},
	"um":  {Symbol: "um", Name: "micrometer", Dimension: LengthDimension, ScaleToSI: 1e-6},
	"nm":  {Symbol: "nm", Name: "nanometer", Dimension: LengthDimension, ScaleToSI: 1e-9},
	"pm":  {Symbol: "pm", Name: "picometer", Dimension: LengthDimension, ScaleToSI: 1e-12},
	"ly":  {Symbol: "ly", Name: "light-year", Dimension: LengthDimension, ScaleToSI: 9.4607304725808e15},
	"au":  {Symbol: "au", Name: "astronomical unit", Dimension: LengthDimension, ScaleToSI: 1.495978707e11},
	"inch":{Symbol: "in", Name: "inch", Dimension: LengthDimension, ScaleToSI: 0.0254},
	"ft":  {Symbol: "ft", Name: "foot", Dimension: LengthDimension, ScaleToSI: 0.3048},
	"mi":  {Symbol: "mi", Name: "mile", Dimension: LengthDimension, ScaleToSI: 1609.344},

	// Mass
	"kg":  {Symbol: "kg", Name: "kilogram", Dimension: MassDimension, ScaleToSI: 1.0},
	"g":   {Symbol: "g", Name: "gram", Dimension: MassDimension, ScaleToSI: 0.001},
	"mg":  {Symbol: "mg", Name: "milligram", Dimension: MassDimension, ScaleToSI: 1e-6},
	"ug":  {Symbol: "ug", Name: "microgram", Dimension: MassDimension, ScaleToSI: 1e-9},
	"tonne": {Symbol: "t", Name: "tonne", Dimension: MassDimension, ScaleToSI: 1000.0},
	"lb":  {Symbol: "lb", Name: "pound", Dimension: MassDimension, ScaleToSI: 0.45359237},
	"oz":  {Symbol: "oz", Name: "ounce", Dimension: MassDimension, ScaleToSI: 0.028349523125},

	// Time
	"s":   {Symbol: "s", Name: "second", Dimension: TimeDimension, ScaleToSI: 1.0},
	"ms":  {Symbol: "ms", Name: "millisecond", Dimension: TimeDimension, ScaleToSI: 0.001},
	"us":  {Symbol: "us", Name: "microsecond", Dimension: TimeDimension, ScaleToSI: 1e-6},
	"ns":  {Symbol: "ns", Name: "nanosecond", Dimension: TimeDimension, ScaleToSI: 1e-9},
	"ps":  {Symbol: "ps", Name: "picosecond", Dimension: TimeDimension, ScaleToSI: 1e-12},
	"min": {Symbol: "min", Name: "minute", Dimension: TimeDimension, ScaleToSI: 60.0},
	"h":   {Symbol: "h", Name: "hour", Dimension: TimeDimension, ScaleToSI: 3600.0},
	"hr":  {Symbol: "hr", Name: "hour", Dimension: TimeDimension, ScaleToSI: 3600.0},
	"day": {Symbol: "day", Name: "day", Dimension: TimeDimension, ScaleToSI: 86400.0},
	"year":{Symbol: "yr", Name: "year", Dimension: TimeDimension, ScaleToSI: 31557600.0},

	// Current
	"A":   {Symbol: "A", Name: "ampere", Dimension: CurrentDimension, ScaleToSI: 1.0},
	"mA":  {Symbol: "mA", Name: "milliampere", Dimension: CurrentDimension, ScaleToSI: 0.001},
	"uA":  {Symbol: "uA", Name: "microampere", Dimension: CurrentDimension, ScaleToSI: 1e-6},
	"kA":  {Symbol: "kA", Name: "kiloampere", Dimension: CurrentDimension, ScaleToSI: 1000.0},

	// Temperature
	"K":   {Symbol: "K", Name: "kelvin", Dimension: TempDimension, ScaleToSI: 1.0},
	"C":   {Symbol: "degC", Name: "celsius", Dimension: TempDimension, ScaleToSI: 1.0, OffsetToSI: 273.15},

	// Amount
	"mol": {Symbol: "mol", Name: "mole", Dimension: AmountDimension, ScaleToSI: 1.0},
	"kmol":{Symbol: "kmol", Name: "kilomole", Dimension: AmountDimension, ScaleToSI: 1000.0},

	// Derived: Velocity
	"m/s":  {Symbol: "m/s", Name: "meter per second", Dimension: VelocityDimension, ScaleToSI: 1.0},
	"km/h": {Symbol: "km/h", Name: "kilometer per hour", Dimension: VelocityDimension, ScaleToSI: 1.0 / 3.6},
	"mph":  {Symbol: "mph", Name: "mile per hour", Dimension: VelocityDimension, ScaleToSI: 0.44704},
	"ft/s": {Symbol: "ft/s", Name: "foot per second", Dimension: VelocityDimension, ScaleToSI: 0.3048},

	// Derived: Acceleration
	"m/s^2": {Symbol: "m/s^2", Name: "meter per second squared", Dimension: AccelerationDimension, ScaleToSI: 1.0},
	"m/s2":  {Symbol: "m/s^2", Name: "meter per second squared", Dimension: AccelerationDimension, ScaleToSI: 1.0},
	"ft/s^2":{Symbol: "ft/s^2", Name: "foot per second squared", Dimension: AccelerationDimension, ScaleToSI: 0.3048},
	"g_acc": {Symbol: "g", Name: "standard gravity", Dimension: AccelerationDimension, ScaleToSI: 9.80665},

	// Derived: Force
	"N":   {Symbol: "N", Name: "newton", Dimension: ForceDimension, ScaleToSI: 1.0},
	"kN":  {Symbol: "kN", Name: "kilonewton", Dimension: ForceDimension, ScaleToSI: 1000.0},
	"MN":  {Symbol: "MN", Name: "meganewton", Dimension: ForceDimension, ScaleToSI: 1e6},
	"dyn": {Symbol: "dyn", Name: "dyne", Dimension: ForceDimension, ScaleToSI: 1e-5},
	"lbf": {Symbol: "lbf", Name: "pound-force", Dimension: ForceDimension, ScaleToSI: 4.4482216152605},

	// Derived: Energy / Work
	"J":   {Symbol: "J", Name: "joule", Dimension: EnergyDimension, ScaleToSI: 1.0},
	"kJ":  {Symbol: "kJ", Name: "kilojoule", Dimension: EnergyDimension, ScaleToSI: 1000.0},
	"MJ":  {Symbol: "MJ", Name: "megajoule", Dimension: EnergyDimension, ScaleToSI: 1e6},
	"GJ":  {Symbol: "GJ", Name: "gigajoule", Dimension: EnergyDimension, ScaleToSI: 1e9},
	"Wh":  {Symbol: "Wh", Name: "watt-hour", Dimension: EnergyDimension, ScaleToSI: 3600.0},
	"kWh": {Symbol: "kWh", Name: "kilowatt-hour", Dimension: EnergyDimension, ScaleToSI: 3.6e6},
	"MWh": {Symbol: "MWh", Name: "megawatt-hour", Dimension: EnergyDimension, ScaleToSI: 3.6e9},
	"eV":  {Symbol: "eV", Name: "electronvolt", Dimension: EnergyDimension, ScaleToSI: 1.602176634e-19},
	"keV": {Symbol: "keV", Name: "kiloelectronvolt", Dimension: EnergyDimension, ScaleToSI: 1.602176634e-16},
	"MeV": {Symbol: "MeV", Name: "megaelectronvolt", Dimension: EnergyDimension, ScaleToSI: 1.602176634e-13},
	"GeV": {Symbol: "GeV", Name: "gigaelectronvolt", Dimension: EnergyDimension, ScaleToSI: 1.602176634e-10},
	"cal": {Symbol: "cal", Name: "calorie", Dimension: EnergyDimension, ScaleToSI: 4.184},
	"kcal":{Symbol: "kcal", Name: "kilocalorie", Dimension: EnergyDimension, ScaleToSI: 4184.0},
	"BTU": {Symbol: "BTU", Name: "british thermal unit", Dimension: EnergyDimension, ScaleToSI: 1055.06},

	// Derived: Power
	"W":   {Symbol: "W", Name: "watt", Dimension: PowerDimension, ScaleToSI: 1.0},
	"mW":  {Symbol: "mW", Name: "milliwatt", Dimension: PowerDimension, ScaleToSI: 0.001},
	"kW":  {Symbol: "kW", Name: "kilowatt", Dimension: PowerDimension, ScaleToSI: 1000.0},
	"MW":  {Symbol: "MW", Name: "megawatt", Dimension: PowerDimension, ScaleToSI: 1e6},
	"GW":  {Symbol: "GW", Name: "gigawatt", Dimension: PowerDimension, ScaleToSI: 1e9},
	"hp":  {Symbol: "hp", Name: "horsepower", Dimension: PowerDimension, ScaleToSI: 745.69987158227022},

	// Derived: Pressure
	"Pa":  {Symbol: "Pa", Name: "pascal", Dimension: PressureDimension, ScaleToSI: 1.0},
	"kPa": {Symbol: "kPa", Name: "kilopascal", Dimension: PressureDimension, ScaleToSI: 1000.0},
	"MPa": {Symbol: "MPa", Name: "megapascal", Dimension: PressureDimension, ScaleToSI: 1e6},
	"GPa": {Symbol: "GPa", Name: "gigapascal", Dimension: PressureDimension, ScaleToSI: 1e9},
	"bar": {Symbol: "bar", Name: "bar", Dimension: PressureDimension, ScaleToSI: 1e5},
	"atm": {Symbol: "atm", Name: "standard atmosphere", Dimension: PressureDimension, ScaleToSI: 101325.0},
	"torr":{Symbol: "torr", Name: "torr", Dimension: PressureDimension, ScaleToSI: 133.32236842105},
	"psi": {Symbol: "psi", Name: "pound per square inch", Dimension: PressureDimension, ScaleToSI: 6894.757293168},

	// Derived: Electromagnetism & Circuits
	"C_charge": {Symbol: "C", Name: "coulomb", Dimension: ChargeDimension, ScaleToSI: 1.0},
	"V":   {Symbol: "V", Name: "volt", Dimension: VoltageDimension, ScaleToSI: 1.0},
	"mV":  {Symbol: "mV", Name: "millivolt", Dimension: VoltageDimension, ScaleToSI: 0.001},
	"kV":  {Symbol: "kV", Name: "kilovolt", Dimension: VoltageDimension, ScaleToSI: 1000.0},
	"ohm": {Symbol: "ohm", Name: "ohm", Dimension: ResistanceDimension, ScaleToSI: 1.0},
	"kohm":{Symbol: "kohm", Name: "kiloohm", Dimension: ResistanceDimension, ScaleToSI: 1000.0},
	"Mohm":{Symbol: "Mohm", Name: "megaohm", Dimension: ResistanceDimension, ScaleToSI: 1e6},
	"F":   {Symbol: "F", Name: "farad", Dimension: CapacitanceDimension, ScaleToSI: 1.0},
	"uF":  {Symbol: "uF", Name: "microfarad", Dimension: CapacitanceDimension, ScaleToSI: 1e-6},
	"nF":  {Symbol: "nF", Name: "nanofarad", Dimension: CapacitanceDimension, ScaleToSI: 1e-9},
	"pF":  {Symbol: "pF", Name: "picofarad", Dimension: CapacitanceDimension, ScaleToSI: 1e-12},
	"H":   {Symbol: "H", Name: "henry", Dimension: InductanceDimension, ScaleToSI: 1.0},
	"mH":  {Symbol: "mH", Name: "millihenry", Dimension: InductanceDimension, ScaleToSI: 0.001},
	"uH":  {Symbol: "uH", Name: "microhenry", Dimension: InductanceDimension, ScaleToSI: 1e-6},
	"Hz":  {Symbol: "Hz", Name: "hertz", Dimension: FrequencyDimension, ScaleToSI: 1.0},
	"kHz": {Symbol: "kHz", Name: "kilohertz", Dimension: FrequencyDimension, ScaleToSI: 1000.0},
	"MHz": {Symbol: "MHz", Name: "megahertz", Dimension: FrequencyDimension, ScaleToSI: 1e6},
	"GHz": {Symbol: "GHz", Name: "gigahertz", Dimension: FrequencyDimension, ScaleToSI: 1e9},
	"T_mag":{Symbol: "T", Name: "tesla", Dimension: MagneticFieldDimension, ScaleToSI: 1.0},
	"G_mag":{Symbol: "G", Name: "gauss", Dimension: MagneticFieldDimension, ScaleToSI: 1e-4},
}

// PhysicalQuantity is a first-class quantity combining a numerical/exact value and physical dimension
type PhysicalQuantity struct {
	NumericValue float64
	ExactValue   *big.Rat
	Dimension    DimensionVector
	UnitSymbol   string
}

func NewQuantity(val float64, unitStr string) (PhysicalQuantity, error) {
	unitStr = strings.TrimSpace(unitStr)
	if unitStr == "" || unitStr == "1" {
		return PhysicalQuantity{
			NumericValue: val,
			ExactValue:   new(big.Rat).SetFloat64(val),
			Dimension:    DimlessDimension,
			UnitSymbol:   "",
		}, nil
	}

	u, ok := standardUnits[unitStr]
	if !ok {
		// Try case-insensitive lookup
		for k, def := range standardUnits {
			if strings.EqualFold(k, unitStr) || strings.EqualFold(def.Name, unitStr) {
				u = def
				ok = true
				break
			}
		}
	}
	if !ok {
		return PhysicalQuantity{}, fmt.Errorf("unknown unit %q", unitStr)
	}

	siVal := val*u.ScaleToSI + u.OffsetToSI
	return PhysicalQuantity{
		NumericValue: siVal,
		ExactValue:   new(big.Rat).SetFloat64(siVal),
		Dimension:    u.Dimension,
		UnitSymbol:   u.Symbol,
	}, nil
}

func (q PhysicalQuantity) Add(other PhysicalQuantity) (PhysicalQuantity, error) {
	if q.Dimension != other.Dimension {
		return PhysicalQuantity{}, fmt.Errorf("dimensional mismatch: cannot add %s (%s) and %s (%s)",
			q.UnitSymbol, q.Dimension.String(), other.UnitSymbol, other.Dimension.String())
	}
	sum := q.NumericValue + other.NumericValue
	var exact *big.Rat
	if q.ExactValue != nil && other.ExactValue != nil {
		exact = new(big.Rat).Add(q.ExactValue, other.ExactValue)
	}
	return PhysicalQuantity{
		NumericValue: sum,
		ExactValue:   exact,
		Dimension:    q.Dimension,
		UnitSymbol:   q.UnitSymbol,
	}, nil
}

func (q PhysicalQuantity) Multiply(other PhysicalQuantity) PhysicalQuantity {
	prod := q.NumericValue * other.NumericValue
	var exact *big.Rat
	if q.ExactValue != nil && other.ExactValue != nil {
		exact = new(big.Rat).Mul(q.ExactValue, other.ExactValue)
	}
	return PhysicalQuantity{
		NumericValue: prod,
		ExactValue:   exact,
		Dimension:    q.Dimension.Multiply(other.Dimension),
		UnitSymbol:   determineCanonicalUnit(q.Dimension.Multiply(other.Dimension)),
	}
}

func (q PhysicalQuantity) Divide(other PhysicalQuantity) (PhysicalQuantity, error) {
	if other.NumericValue == 0 {
		return PhysicalQuantity{}, fmt.Errorf("division by zero in physical quantity")
	}
	quot := q.NumericValue / other.NumericValue
	var exact *big.Rat
	if q.ExactValue != nil && other.ExactValue != nil && other.ExactValue.Sign() != 0 {
		exact = new(big.Rat).Quo(q.ExactValue, other.ExactValue)
	}
	return PhysicalQuantity{
		NumericValue: quot,
		ExactValue:   exact,
		Dimension:    q.Dimension.Divide(other.Dimension),
		UnitSymbol:   determineCanonicalUnit(q.Dimension.Divide(other.Dimension)),
	}, nil
}

func determineCanonicalUnit(d DimensionVector) string {
	switch d {
	case LengthDimension:
		return "m"
	case MassDimension:
		return "kg"
	case TimeDimension:
		return "s"
	case CurrentDimension:
		return "A"
	case TempDimension:
		return "K"
	case VelocityDimension:
		return "m/s"
	case AccelerationDimension:
		return "m/s^2"
	case ForceDimension:
		return "N"
	case EnergyDimension:
		return "J"
	case PowerDimension:
		return "W"
	case PressureDimension:
		return "Pa"
	case ChargeDimension:
		return "C"
	case VoltageDimension:
		return "V"
	case ResistanceDimension:
		return "ohm"
	case CapacitanceDimension:
		return "F"
	case InductanceDimension:
		return "H"
	case FrequencyDimension:
		return "Hz"
	case MagneticFieldDimension:
		return "T"
	default:
		return d.String()
	}
}

func (q PhysicalQuantity) String() string {
	valStr := fmt.Sprintf("%g", q.NumericValue)
	if q.ExactValue != nil && q.ExactValue.IsInt() {
		valStr = q.ExactValue.Num().String()
	}
	if q.UnitSymbol == "" {
		return valStr
	}
	return fmt.Sprintf("%s %s", valStr, q.UnitSymbol)
}
