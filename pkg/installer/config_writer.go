package installer

import (
	"encoding/json"
	"fmt"
	"os"
	"runtime"
)

// SetupConfigOptions contains parameters for generating Strata engine run configs
type SetupConfigOptions struct {
	ModelTag           string
	PackDir            string
	NativeGGUF         string
	MaxContext         int
	ResidentBudgetGiB  int
	UseMmapExperts     bool
	UseResidentExperts bool
	SpecDraft          int
	MTPPath            string
	PowerPolicy        string
	ResourceAdapt      bool
	PoolWorkers        int
	PCIeFrac           float64
}

// StrataRunConfig defines the JSON structure read by strata serve and the engine
type StrataRunConfig struct {
	Args          []string `json:"args"`
	PowerPolicy   string   `json:"power_policy,omitempty"`
	ResourceAdapt bool     `json:"resource_adapt"` // written even when false: an explicit switch, not an absent key
	Parallel      int      `json:"parallel,omitempty"`
}

// GenerateRunConfig creates the engine arguments and runtime options
func GenerateRunConfig(opts SetupConfigOptions) (StrataRunConfig, error) {
	args := []string{
		"--pack", opts.PackDir,
		"--native", opts.NativeGGUF,
		"--expert-cache", "auto",
		"--prefill", "auto",
	}

	if opts.MaxContext > 0 {
		args = append(args, "--max-context", fmt.Sprintf("%d", opts.MaxContext))
	}
	if opts.SpecDraft > 0 {
		args = append(args, "--spec", fmt.Sprintf("%d", opts.SpecDraft), "--spec-min-p", "0.5")
	}
	if opts.MTPPath != "" {
		args = append(args, "--mtp", opts.MTPPath)
	}

	if opts.UseMmapExperts {
		args = append(args, "--mmap-experts")
	} else if opts.UseResidentExperts {
		args = append(args, "--resident-experts")
	}

	if opts.ResidentBudgetGiB > 0 {
		args = append(args, "--resident-budget-gib", fmt.Sprintf("%d", opts.ResidentBudgetGiB))
	}

	if opts.PoolWorkers > 0 {
		args = append(args, "--pool-workers", fmt.Sprintf("%d", opts.PoolWorkers))
	} else {
		args = append(args, "--pool-workers", "10")
	}

	if opts.PCIeFrac > 0 {
		args = append(args, "--pcie-frac", fmt.Sprintf("%.2f", opts.PCIeFrac))
	} else {
		args = append(args, "--pcie-frac", "0.35")
	}

	policy := opts.PowerPolicy
	if policy == "" {
		policy = "BALANCED"
	}

	return StrataRunConfig{
		Args:          args,
		PowerPolicy:   policy,
		ResourceAdapt: opts.ResourceAdapt,
		Parallel:      1,
	}, nil
}

// SaveRunConfig writes the config to disk formatted with indentation
func SaveRunConfig(cfg StrataRunConfig, path string) error {
	data, err := json.MarshalIndent(cfg, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// WriteLauncherScript generates a native platform launch script (run.sh or run.bat)
func WriteLauncherScript(configPath, scriptPath string) error {
	var content string
	if runtime.GOOS == "windows" {
		content = fmt.Sprintf("@echo off\nbin\\strata.exe serve --config \"%s\"\npause\n", configPath)
	} else {
		content = fmt.Sprintf("#!/bin/sh\nexec ./bin/strata serve --config \"%s\" \"$@\"\n", configPath)
	}
	return os.WriteFile(scriptPath, []byte(content), 0755)
}
