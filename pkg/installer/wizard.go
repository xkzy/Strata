package installer

import (
	"fmt"
	"math"
	"path/filepath"
	"strings"
)

// SetupFlags encapsulates all CLI arguments provided to strata setup
type SetupFlags struct {
	Check             bool
	Family            string
	Model             string
	Context           int
	LowRAM            string
	ResidentBudgetGiB int
	Yes               bool
	GGUFDir           string
	NoStart           bool
}

// ExecuteSetup executes hardware probing, recommendation, and configuration writing
func ExecuteSetup(flags SetupFlags) error {
	specs := DetectHardware()

	fmt.Println("=== Step 1: Checking your PC ===")
	fmt.Printf("  [ok] CPU: %d cores\n", specs.CPUCores)
	fmt.Printf("  [ok] System RAM: %.1f GB\n", specs.RAMTotalGB)
	if specs.GPUVendor != GPUNone {
		fmt.Printf("  [ok] GPU: %s (%.1f GB VRAM, %s)\n", specs.GPUName, specs.VRAMGB, specs.GPUVendor)
	} else {
		fmt.Printf("  [!]  GPU: None detected (CPU-only execution)\n")
	}

	if flags.Check {
		return runCheckReport(specs)
	}

	fmt.Println("\n=== Step 2: Model Configuration ===")
	// Pick model option
	modelChoice := flags.Model
	familyChoice := flags.Family
	if familyChoice == "" {
		familyChoice = "openai"
	}
	if modelChoice == "" {
		modelChoice = "GPT_OSS_20B"
	}

	fmt.Printf("  [ok] Model Family: %s\n", familyChoice)
	fmt.Printf("  [ok] Model Size:   %s\n", modelChoice)

	// Context calculation
	ctx := flags.Context
	if ctx <= 0 {
		if specs.VRAMGB >= 16 {
			ctx = 65536
		} else if specs.VRAMGB >= 10 {
			ctx = 32768
		} else {
			ctx = 16384
		}
	}
	fmt.Printf("  [ok] Context length: %d tokens\n", ctx)

	// Memory & budget safety calculation (at least 10 GiB headroom for system OS)
	headroomGB := 10.0
	availForModel := specs.RAMTotalGB - headroomGB
	budgetGiB := flags.ResidentBudgetGiB
	if budgetGiB <= 0 {
		if availForModel >= 8.0 {
			budgetGiB = int(math.Min(availForModel, 12.0))
		} else {
			budgetGiB = 4
		}
	}
	fmt.Printf("  [ok] RAM budget: %d GiB prefetched into memory (safe headroom: %.1f GB)\n", budgetGiB, headroomGB)

	// Directories
	modelsDir := GetModelsDir()
	if flags.GGUFDir != "" {
		modelsDir = flags.GGUFDir
	}
	packDir := filepath.Join(GetPacksDir(), strings.ToLower(modelChoice))
	nativeGGUF := filepath.Join(modelsDir, fmt.Sprintf("%s.gguf", strings.ToLower(modelChoice)))
	configFile := fmt.Sprintf("strata-%s.json", strings.ToLower(modelChoice))
	launcherScript := "run.sh"

	opts := SetupConfigOptions{
		ModelTag:           strings.ToLower(modelChoice),
		PackDir:            packDir,
		NativeGGUF:         nativeGGUF,
		MaxContext:         ctx,
		ResidentBudgetGiB:  budgetGiB,
		UseMmapExperts:     true,
		UseResidentExperts: false,
		SpecDraft:          4,
		PowerPolicy:        "BALANCED",
		ResourceAdapt:      true,
		PoolWorkers:        10,
		PCIeFrac:           0.35,
	}

	cfg, err := GenerateRunConfig(opts)
	if err != nil {
		return fmt.Errorf("generate run config: %w", err)
	}

	if err := SaveRunConfig(cfg, configFile); err != nil {
		return fmt.Errorf("save run config: %w", err)
	}
	fmt.Printf("  [ok] Configuration written: %s\n", configFile)

	if err := WriteLauncherScript(configFile, launcherScript); err != nil {
		return fmt.Errorf("write launcher script: %w", err)
	}
	fmt.Printf("  [ok] Launcher script created: %s\n", launcherScript)

	fmt.Println("\nAll set. Run ./run.sh (or bin/strata serve) to start Strata.")
	return nil
}

func runCheckReport(specs HardwareSpecs) error {
	fmt.Println("\nModel Fit Analysis for this PC:")
	curated := GetCuratedHuggingFaceModels()
	for _, m := range curated {
		fit := "does not fit"
		if specs.RAMTotalGB >= m.MinRAMGB {
			fit = fmt.Sprintf("fits (needs >= %.0f GB RAM)", m.MinRAMGB)
		} else if specs.RAMTotalGB >= m.MinRAMGB-8 {
			fit = fmt.Sprintf("tight (fits with low-RAM mmap mode)")
		}
		fmt.Printf("  %-30s %s\n", m.Name, fit)
	}
	fmt.Println("\nThis PC can run Strata. Run strata setup to configure.")
	return nil
}
