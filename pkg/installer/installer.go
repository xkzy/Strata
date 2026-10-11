// pkg/installer/installer.go - Native Hardware Detection, Model Recommendation & Downloader in Go
package installer

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"
)

type GPUVendor string

const (
	GPUNvidia GPUVendor = "NVIDIA"
	GPUAMD    GPUVendor = "AMD"
	GPUIntel  GPUVendor = "Intel"
	GPUNone   GPUVendor = "CPU"
)

type ModelSource string

const (
	SourceHuggingFace ModelSource = "huggingface"
	SourceOllama      ModelSource = "ollama"
	SourceLocal       ModelSource = "local"
	SourceCustom      ModelSource = "custom"
)

type HardwareSpecs struct {
	CPUCores   int       `json:"cpu_cores"`
	RAMTotalGB float64   `json:"ram_total_gb"`
	GPUVendor  GPUVendor `json:"gpu_vendor"`
	GPUName    string    `json:"gpu_name"`
	VRAMGB     float64   `json:"vram_gb"`
}

type ModelOption struct {
	Name          string      `json:"name"`
	RepoID        string      `json:"repo_id"`
	Filename      string      `json:"filename"`
	DownloadURL   string      `json:"download_url"`
	SizeGB        float64     `json:"size_gb"`
	MinRAMGB      float64     `json:"min_ram_gb"`
	Recommended   bool        `json:"recommended"`
	Description   string      `json:"description"`
	Source        ModelSource `json:"source"`
	Category      string      `json:"category"` // "coder", "reasoning", "general", "lightweight", "local"
	Tags          []string    `json:"tags,omitempty"`
	Quantization  string      `json:"quantization,omitempty"`
	ParamSize     string      `json:"param_size,omitempty"`
	ContextLength int         `json:"context_length,omitempty"`
}

// DetectHardware inspects CPU, RAM, and GPU hardware on the host
func DetectHardware() HardwareSpecs {
	specs := HardwareSpecs{
		CPUCores:  runtime.NumCPU(),
		GPUVendor: GPUNone,
		GPUName:   "CPU Only",
	}

	// 1. RAM Detection on Linux
	if runtime.GOOS == "linux" {
		if data, err := os.ReadFile("/proc/meminfo"); err == nil {
			for _, line := range strings.Split(string(data), "\n") {
				if strings.HasPrefix(line, "MemTotal:") {
					fields := strings.Fields(line)
					if len(fields) >= 2 {
						kb, _ := strconv.ParseUint(fields[1], 10, 64)
						specs.RAMTotalGB = float64(kb) / 1024 / 1024
					}
					break
				}
			}
		}
	} else {
		specs.RAMTotalGB = 16.0 // fallback
	}

	// 2. GPU Detection
	// 2.1 NVIDIA query via nvidia-smi
	if out, err := exec.Command("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits").Output(); err == nil {
		lines := strings.Split(strings.TrimSpace(string(out)), "\n")
		var names []string
		var totalMiB float64
		for _, line := range lines {
			parts := strings.Split(line, ",")
			if len(parts) >= 2 {
				name := strings.TrimSpace(parts[0])
				mib, _ := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
				names = append(names, name)
				totalMiB += mib
			}
		}
		if len(names) > 0 {
			specs.GPUVendor = GPUNvidia
			specs.GPUName = strings.Join(names, " + ")
			specs.VRAMGB = float64(int((totalMiB/1024.0)*10+0.5)) / 10.0
		}
	} else if _, err := os.Stat("/dev/nvidia0"); err == nil {
		specs.GPUVendor = GPUNvidia
		specs.GPUName = "NVIDIA CUDA GPU"
		specs.VRAMGB = 8.0
	}

	// 2.2 AMD sysfs
	if specs.GPUVendor == GPUNone {
		if data, err := os.ReadFile("/sys/class/drm/card0/device/mem_info_vram_total"); err == nil {
			bytes, _ := strconv.ParseUint(strings.TrimSpace(string(data)), 10, 64)
			if bytes > 0 {
				specs.GPUVendor = GPUAMD
				specs.GPUName = "AMD Radeon GPU (ROCm/HIP)"
				specs.VRAMGB = float64(int((float64(bytes)/1024/1024/1024)*10+0.5)) / 10.0
			}
		}
	}

	return specs
}

// ollamaManifest matches the OCI/Docker manifest structure used by Ollama
type ollamaManifest struct {
	SchemaVersion int `json:"schemaVersion"`
	Layers        []struct {
		MediaType string `json:"mediaType"`
		Digest    string `json:"digest"`
		Size      int64  `json:"size"`
	} `json:"layers"`
}

// ScanLocalOllamaModels discovers models installed in the local Ollama repository
func ScanLocalOllamaModels() []ModelOption {
	var results []ModelOption

	var possibleDirs []string
	if env := os.Getenv("OLLAMA_MODELS"); env != "" {
		possibleDirs = append(possibleDirs, env)
	}

	homeDir, _ := os.UserHomeDir()
	if homeDir != "" {
		possibleDirs = append(possibleDirs,
			filepath.Join(homeDir, ".ollama", "models"),
			filepath.Join(homeDir, "AppData", "Local", "Programs", "Ollama", "models"),
		)
	}
	possibleDirs = append(possibleDirs, "/usr/share/ollama/.ollama/models")

	seenBlobs := make(map[string]bool)

	for _, baseDir := range possibleDirs {
		manifestsRoot := filepath.Join(baseDir, "manifests")
		blobsRoot := filepath.Join(baseDir, "blobs")
		if _, err := os.Stat(manifestsRoot); err != nil {
			continue
		}

		_ = filepath.Walk(manifestsRoot, func(path string, info os.FileInfo, err error) error {
			if err != nil || info == nil || info.IsDir() {
				return nil
			}

			rel, err := filepath.Rel(manifestsRoot, path)
			if err != nil {
				return nil
			}

			// Clean manifest rel path: e.g. "registry.ollama.ai/library/deepseek-coder/latest"
			parts := strings.Split(filepath.ToSlash(rel), "/")
			if len(parts) < 2 {
				return nil
			}

			modelName := parts[len(parts)-2]
			tag := parts[len(parts)-1]
			if len(parts) >= 3 && parts[len(parts)-3] != "library" && parts[len(parts)-3] != "registry.ollama.ai" {
				modelName = parts[len(parts)-3] + "/" + modelName
			}

			data, err := os.ReadFile(path)
			if err != nil {
				return nil
			}

			var mf ollamaManifest
			if err := json.Unmarshal(data, &mf); err != nil {
				return nil
			}

			for _, layer := range mf.Layers {
				if layer.MediaType == "application/vnd.ollama.image.model" && strings.HasPrefix(layer.Digest, "sha256:") {
					hash := strings.TrimPrefix(layer.Digest, "sha256:")
					blobPath := filepath.Join(blobsRoot, "sha256-"+hash)
					if fi, err := os.Stat(blobPath); err == nil && !seenBlobs[blobPath] {
						seenBlobs[blobPath] = true
						sizeGB := float64(fi.Size()) / (1024 * 1024 * 1024)
						minRAM := sizeGB * 1.3
						if minRAM < 8.0 {
							minRAM = 8.0
						}

						cat := "general"
						lower := strings.ToLower(modelName)
						if strings.Contains(lower, "coder") || strings.Contains(lower, "code") {
							cat = "coder"
						} else if strings.Contains(lower, "r1") || strings.Contains(lower, "deepseek") || strings.Contains(lower, "reason") {
							cat = "reasoning"
						} else if sizeGB < 4.0 {
							cat = "lightweight"
						}

						displayName := fmt.Sprintf("%s:%s (Ollama)", modelName, tag)
						results = append(results, ModelOption{
							Name:          displayName,
							RepoID:        fmt.Sprintf("ollama/%s:%s", modelName, tag),
							Filename:      blobPath,
							DownloadURL:   "",
							SizeGB:        float64(int(sizeGB*10+0.5)) / 10.0,
							MinRAMGB:      float64(int(minRAM*10+0.5)) / 10.0,
							Description:   fmt.Sprintf("Local Ollama model %s:%s (Instant Activation, No Download)", modelName, tag),
							Source:        SourceOllama,
							Category:      cat,
							Tags:          []string{"Ollama", "Local", "Instant"},
							Quantization:  tag,
							ParamSize:     extractParamSize(modelName + " " + tag),
							ContextLength: 32768,
						})
					}
				}
			}
			return nil
		})
	}

	return results
}

// ScanLocalCustomModels discovers existing local Strata packs and GGUFs
func ScanLocalCustomModels() []ModelOption {
	var results []ModelOption

	candidates := []struct {
		name        string
		path        string
		sizeGB      float64
		minRAMGB    float64
		desc        string
		category    string
		tags        []string
		recommended bool
	}{
		{
			name:        "Qwen3.8-Flash-Next-Coder-IQ1_M (Local Pack)",
			path:        "/home/khing/Downloads/Strata-data/packs/coder-iq1_m",
			sizeGB:      26.0,
			minRAMGB:    16.0,
			desc:        "Existing local compiled pack in Strata-data (Instant Activation, Optimized KV)",
			category:    "coder",
			tags:        []string{"Strata Pack", "MoE", "Local", "Fast"},
			recommended: true,
		},
		{
			name:        "Qwen3.8-Flash-Next-Coder-IQ1_M (Local GGUF)",
			path:        "/home/khing/Downloads/Strata-data/models/coder-IQ1_M",
			sizeGB:      58.4,
			minRAMGB:    24.0,
			desc:        "Existing local GGUF shards in Strata-data directory",
			category:    "coder",
			tags:        []string{"GGUF Shards", "Local"},
			recommended: false,
		},
	}

	for _, c := range candidates {
		if _, err := os.Stat(c.path); err == nil {
			results = append(results, ModelOption{
				Name:          c.name,
				RepoID:        "local/" + filepath.Base(c.path),
				Filename:      c.path,
				DownloadURL:   "",
				SizeGB:        c.sizeGB,
				MinRAMGB:      c.minRAMGB,
				Recommended:   c.recommended,
				Description:   c.desc,
				Source:        SourceLocal,
				Category:      c.category,
				Tags:          c.tags,
				Quantization:  "IQ1_M",
				ParamSize:     "MoE",
				ContextLength: 32768,
			})
		}
	}

	// Scan local models/ directory for downloaded .gguf files
	if entries, err := os.ReadDir("models"); err == nil {
		for _, e := range entries {
			if !e.IsDir() && strings.HasSuffix(strings.ToLower(e.Name()), ".gguf") {
				filePath := filepath.Join("models", e.Name())
				fi, _ := e.Info()
				sizeGB := 0.0
				if fi != nil {
					sizeGB = float64(fi.Size()) / (1024 * 1024 * 1024)
				}
				name := strings.TrimSuffix(e.Name(), filepath.Ext(e.Name()))
				results = append(results, ModelOption{
					Name:          name + " (Local)",
					RepoID:        "local/models/" + e.Name(),
					Filename:      filePath,
					DownloadURL:   "",
					SizeGB:        float64(int(sizeGB*10+0.5)) / 10.0,
					MinRAMGB:      float64(int((sizeGB*1.2+2)*10+0.5)) / 10.0,
					Description:   fmt.Sprintf("Downloaded GGUF file in models/ directory (%s)", e.Name()),
					Source:        SourceLocal,
					Category:      "local",
					Tags:          []string{"Local", "GGUF", "Ready"},
					Quantization:  extractQuant(e.Name()),
					ParamSize:     extractParamSize(e.Name()),
					ContextLength: 32768,
				})
			}
		}
	}

	return results
}

// GetCuratedHuggingFaceModels returns a rich curated catalog of popular GGUF models
func GetCuratedHuggingFaceModels() []ModelOption {
	return []ModelOption{
		// --- Coding ---
		{
			Name:          "Qwen2.5-Coder-7B-Instruct-Q4_K_M",
			RepoID:        "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
			Filename:      "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
			DownloadURL:   "https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
			SizeGB:        4.7,
			MinRAMGB:      12.0,
			Description:   "Industry-standard 7B coding model, balanced precision and speed (16GB RAM)",
			Source:        SourceHuggingFace,
			Category:      "coder",
			Tags:          []string{"Coding", "7B", "Q4_K_M", "Hugging Face"},
			Quantization:  "Q4_K_M",
			ParamSize:     "7B",
			ContextLength: 32768,
		},
		{
			Name:          "Qwen2.5-Coder-7B-Instruct-Q8_0",
			RepoID:        "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
			Filename:      "Qwen2.5-Coder-7B-Instruct-Q8_0.gguf",
			DownloadURL:   "https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q8_0.gguf",
			SizeGB:        8.1,
			MinRAMGB:      24.0,
			Description:   "High-precision 8-bit quantization for workstations with >= 32GB RAM",
			Source:        SourceHuggingFace,
			Category:      "coder",
			Tags:          []string{"Coding", "7B", "Q8_0", "High Precision"},
			Quantization:  "Q8_0",
			ParamSize:     "7B",
			ContextLength: 32768,
		},
		{
			Name:          "Qwen2.5-Coder-14B-Instruct-Q4_K_M",
			RepoID:        "bartowski/Qwen2.5-Coder-14B-Instruct-GGUF",
			Filename:      "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
			DownloadURL:   "https://huggingface.co/bartowski/Qwen2.5-Coder-14B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
			SizeGB:        9.0,
			MinRAMGB:      20.0,
			Description:   "Advanced 14B coding engine with deep syntax and refactoring capability",
			Source:        SourceHuggingFace,
			Category:      "coder",
			Tags:          []string{"Coding", "14B", "Q4_K_M", "Deep Reasoning"},
			Quantization:  "Q4_K_M",
			ParamSize:     "14B",
			ContextLength: 32768,
		},
		{
			Name:          "Qwen2.5-Coder-1.5B-Instruct-Q4_K_M",
			RepoID:        "bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF",
			Filename:      "Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf",
			DownloadURL:   "https://huggingface.co/bartowski/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-1.5B-Instruct-Q4_K_M.gguf",
			SizeGB:        1.1,
			MinRAMGB:      6.0,
			Description:   "Ultra-lightweight 1.5B coding model for ultra-low latency & compact laptops",
			Source:        SourceHuggingFace,
			Category:      "coder",
			Tags:          []string{"Coding", "1.5B", "Fast", "Ultra-Light"},
			Quantization:  "Q4_K_M",
			ParamSize:     "1.5B",
			ContextLength: 32768,
		},
		{
			Name:          "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M",
			RepoID:        "bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF",
			Filename:      "DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
			DownloadURL:   "https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/resolve/main/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf",
			SizeGB:        9.1,
			MinRAMGB:      20.0,
			Description:   "16B MoE (2.4B active) coding model supporting 300+ programming languages",
			Source:        SourceHuggingFace,
			Category:      "coder",
			Tags:          []string{"Coding", "MoE", "16B", "DeepSeek"},
			Quantization:  "Q4_K_M",
			ParamSize:     "16B MoE",
			ContextLength: 65536,
		},

		// --- Reasoning / R1 ---
		{
			Name:          "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M",
			RepoID:        "bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF",
			Filename:      "DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
			DownloadURL:   "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf",
			SizeGB:        4.7,
			MinRAMGB:      12.0,
			Description:   "State-of-the-art chain-of-thought reasoning distilled into Qwen 7B",
			Source:        SourceHuggingFace,
			Category:      "reasoning",
			Tags:          []string{"Reasoning", "R1", "7B", "Chain-of-Thought"},
			Quantization:  "Q4_K_M",
			ParamSize:     "7B",
			ContextLength: 32768,
		},
		{
			Name:          "DeepSeek-R1-Distill-Qwen-8B-Q4_K_M",
			RepoID:        "bartowski/DeepSeek-R1-Distill-Qwen-8B-GGUF",
			Filename:      "DeepSeek-R1-Distill-Qwen-8B-Q4_K_M.gguf",
			DownloadURL:   "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-8B-Q4_K_M.gguf",
			SizeGB:        5.2,
			MinRAMGB:      14.0,
			Description:   "DeepSeek R1 reasoning distilled on Qwen 8B base architecture",
			Source:        SourceHuggingFace,
			Category:      "reasoning",
			Tags:          []string{"Reasoning", "R1", "8B", "Math & Logic"},
			Quantization:  "Q4_K_M",
			ParamSize:     "8B",
			ContextLength: 32768,
		},
		{
			Name:          "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M",
			RepoID:        "bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF",
			Filename:      "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
			DownloadURL:   "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF/resolve/main/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
			SizeGB:        9.0,
			MinRAMGB:      20.0,
			Description:   "Exceptional mathematical reasoning and deep logical problem solving (14B)",
			Source:        SourceHuggingFace,
			Category:      "reasoning",
			Tags:          []string{"Reasoning", "R1", "14B", "Advanced"},
			Quantization:  "Q4_K_M",
			ParamSize:     "14B",
			ContextLength: 32768,
		},
		{
			Name:          "DeepSeek-R1-Distill-Llama-8B-Q4_K_M",
			RepoID:        "bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF",
			Filename:      "DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
			DownloadURL:   "https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF/resolve/main/DeepSeek-R1-Distill-Llama-8B-Q4_K_M.gguf",
			SizeGB:        4.9,
			MinRAMGB:      14.0,
			Description:   "DeepSeek R1 reasoning distilled onto Meta Llama-3.1 8B foundation",
			Source:        SourceHuggingFace,
			Category:      "reasoning",
			Tags:          []string{"Reasoning", "R1", "8B", "Llama"},
			Quantization:  "Q4_K_M",
			ParamSize:     "8B",
			ContextLength: 32768,
		},
		{
			Name:          "MiMo-V2.6-Flash-RL-Q2_K",
			RepoID:        "ggml-org/MiMo-V2.6-Flash-RL-GGUF",
			Filename:      "MiMo-V2.6-Flash-RL-Q2_K-00001-of-00002.gguf",
			DownloadURL:   "https://huggingface.co/ggml-org/MiMo-V2.6-Flash-RL-GGUF/resolve/main/MiMo-V2.6-Flash-RL-Q2_K-00001-of-00002.gguf",
			SizeGB:        126.2,
			MinRAMGB:      64.0,
			Description:   "Xiaomi MiMo-V2.6-Flash-RL 2-bit K-quant MoE with RL reasoning & tool call support",
			Source:        SourceHuggingFace,
			Category:      "reasoning",
			Tags:          []string{"Reasoning", "MoE", "RL", "MiMo", "Q2_K"},
			Quantization:  "Q2_K",
			ParamSize:     "256-Expert MoE",
			ContextLength: 1048576,
		},
		{
			Name:          "MiMo-V2.6-Flash-RL-MXFP4",
			RepoID:        "ggml-org/MiMo-V2.6-Flash-RL-GGUF",
			Filename:      "MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf",
			DownloadURL:   "https://huggingface.co/ggml-org/MiMo-V2.6-Flash-RL-GGUF/resolve/main/MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf",
			SizeGB:        167.4,
			MinRAMGB:      64.0,
			Description:   "Xiaomi MiMo-V2.6-Flash-RL 4-bit microscaling format MoE with high fidelity RL reasoning",
			Source:        SourceHuggingFace,
			Category:      "reasoning",
			Tags:          []string{"Reasoning", "MoE", "RL", "MiMo", "MXFP4"},
			Quantization:  "MXFP4",
			ParamSize:     "256-Expert MoE",
			ContextLength: 1048576,
		},

		// --- General & Lightweight ---
		{
			Name:          "Llama-3.2-3B-Instruct-Q4_K_M",
			RepoID:        "bartowski/Llama-3.2-3B-Instruct-GGUF",
			Filename:      "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
			DownloadURL:   "https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.gguf",
			SizeGB:        2.0,
			MinRAMGB:      8.0,
			Description:   "Meta's latest high-efficiency 3B general assistant, fast & low memory",
			Source:        SourceHuggingFace,
			Category:      "lightweight",
			Tags:          []string{"General", "3B", "Fast", "Meta"},
			Quantization:  "Q4_K_M",
			ParamSize:     "3B",
			ContextLength: 32768,
		},
		{
			Name:          "Llama-3.2-1B-Instruct-Q4_K_M",
			RepoID:        "bartowski/Llama-3.2-1B-Instruct-GGUF",
			Filename:      "Llama-3.2-1B-Instruct-Q4_K_M.gguf",
			DownloadURL:   "https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf",
			SizeGB:        0.8,
			MinRAMGB:      4.0,
			Description:   "Ultra-compact 1B model, fits in minimal memory with instant responses",
			Source:        SourceHuggingFace,
			Category:      "lightweight",
			Tags:          []string{"General", "1B", "Ultra-Fast", "Minimal"},
			Quantization:  "Q4_K_M",
			ParamSize:     "1B",
			ContextLength: 32768,
		},
		{
			Name:          "Llama-3.1-8B-Instruct-Q4_K_M",
			RepoID:        "bartowski/Meta-Llama-3.1-8B-Instruct-GGUF",
			Filename:      "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
			DownloadURL:   "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
			SizeGB:        4.9,
			MinRAMGB:      14.0,
			Description:   "Meta's flagship 8B general purpose model with 128K native context support",
			Source:        SourceHuggingFace,
			Category:      "general",
			Tags:          []string{"General", "8B", "128K Context", "Meta"},
			Quantization:  "Q4_K_M",
			ParamSize:     "8B",
			ContextLength: 131072,
		},
		{
			Name:          "Mistral-7B-Instruct-v0.3-Q4_K_M",
			RepoID:        "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
			Filename:      "Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
			DownloadURL:   "https://huggingface.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF/resolve/main/Mistral-7B-Instruct-v0.3-Q4_K_M.gguf",
			SizeGB:        4.4,
			MinRAMGB:      12.0,
			Description:   "Mistral 7B v0.3 with function calling and extended 32K context",
			Source:        SourceHuggingFace,
			Category:      "general",
			Tags:          []string{"General", "7B", "Mistral", "Tools"},
			Quantization:  "Q4_K_M",
			ParamSize:     "7B",
			ContextLength: 32768,
		},
		{
			Name:          "gemma-2-9b-it-Q4_K_M",
			RepoID:        "bartowski/gemma-2-9b-it-GGUF",
			Filename:      "gemma-2-9b-it-Q4_K_M.gguf",
			DownloadURL:   "https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf",
			SizeGB:        5.8,
			MinRAMGB:      16.0,
			Description:   "Google's high-performance Gemma 2 9B model with sliding window attention",
			Source:        SourceHuggingFace,
			Category:      "general",
			Tags:          []string{"General", "9B", "Google", "Gemma 2"},
			Quantization:  "Q4_K_M",
			ParamSize:     "9B",
			ContextLength: 8192,
		},
		{
			Name:          "Qwen2.5-Coder-7B-Instruct-Q2_K",
			RepoID:        "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
			Filename:      "Qwen2.5-Coder-7B-Instruct-Q2_K.gguf",
			DownloadURL:   "https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q2_K.gguf",
			SizeGB:        2.9,
			MinRAMGB:      8.0,
			Description:   "Lightweight 2-bit quantization for compact systems with 8GB-12GB RAM",
			Source:        SourceHuggingFace,
			Category:      "coder",
			Tags:          []string{"Coding", "7B", "Q2_K", "Low Memory"},
			Quantization:  "Q2_K",
			ParamSize:     "7B",
			ContextLength: 32768,
		},
	}
}

// GetAvailableModels returns combined catalog of local packs, Ollama models, and curated Hugging Face models
func GetAvailableModels() []ModelOption {
	var models []ModelOption

	// 1. Local packs and downloaded files
	models = append(models, ScanLocalCustomModels()...)

	// 2. Local Ollama models
	models = append(models, ScanLocalOllamaModels()...)

	// 3. Online downloadable Hugging Face catalog
	models = append(models, GetCuratedHuggingFaceModels()...)

	return models
}

// ResolveHuggingFaceInput parses a user-entered Hugging Face URL or Repo ID into a valid ModelOption
func ResolveHuggingFaceInput(rawInput string) (*ModelOption, error) {
	rawInput = strings.TrimSpace(rawInput)
	if rawInput == "" {
		return nil, fmt.Errorf("empty input")
	}

	var downloadURL string
	var filename string
	var repoID string

	if strings.HasPrefix(rawInput, "http://") || strings.HasPrefix(rawInput, "https://") {
		u, err := url.Parse(rawInput)
		if err != nil {
			return nil, fmt.Errorf("invalid URL: %w", err)
		}
		host := strings.ToLower(u.Host)
		if host != "huggingface.co" && host != "hf.co" && host != "www.huggingface.co" {
			return nil, fmt.Errorf("only huggingface.co URLs are supported for custom models (got %s)", host)
		}

		path := strings.Trim(u.Path, "/")
		parts := strings.Split(path, "/")
		if len(parts) < 2 {
			return nil, fmt.Errorf("invalid Hugging Face model repository path: %s", path)
		}

		repoID = parts[0] + "/" + parts[1]

		if len(parts) >= 4 && (parts[2] == "resolve" || parts[2] == "blob") {
			filename = filepath.Base(path)
			branch := parts[3]
			downloadURL = fmt.Sprintf("https://huggingface.co/%s/resolve/%s/%s", repoID, branch, filename)
		} else if strings.HasSuffix(strings.ToLower(path), ".gguf") {
			filename = filepath.Base(path)
			downloadURL = fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repoID, filename)
		} else {
			// Auto-infer default Q4_K_M GGUF filename from repo name
			cleanName := parts[1]
			cleanName = strings.TrimSuffix(cleanName, "-GGUF")
			cleanName = strings.TrimSuffix(cleanName, "-gguf")
			filename = cleanName + "-Q4_K_M.gguf"
			downloadURL = fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repoID, filename)
		}
	} else if strings.Contains(rawInput, "/") {
		// Repo ID or repo:filename format
		parts := strings.Split(rawInput, ":")
		repoID = strings.TrimSpace(parts[0])
		if len(parts) >= 2 {
			filename = strings.TrimSpace(parts[1])
		} else {
			rParts := strings.Split(repoID, "/")
			modelPart := rParts[len(rParts)-1]
			cleanName := strings.TrimSuffix(modelPart, "-GGUF")
			cleanName = strings.TrimSuffix(cleanName, "-gguf")
			filename = cleanName + "-Q4_K_M.gguf"
		}
		downloadURL = fmt.Sprintf("https://huggingface.co/%s/resolve/main/%s", repoID, filename)
	} else {
		return nil, fmt.Errorf("unrecognized format: please enter a Hugging Face repo (e.g. 'bartowski/Qwen2.5-Coder-7B-Instruct-GGUF') or direct GGUF URL")
	}

	// Security: Sanitize filename to prevent directory traversal
	filename = filepath.Base(filepath.Clean(filename))
	if filename == "." || filename == "/" || strings.Contains(filename, "..") {
		return nil, fmt.Errorf("invalid filename extracted from input")
	}
	if !strings.HasSuffix(strings.ToLower(filename), ".gguf") {
		filename += ".gguf"
	}

	name := strings.TrimSuffix(filename, ".gguf")
	quant := extractQuant(filename)
	param := extractParamSize(filename)

	cat := "general"
	lower := strings.ToLower(name)
	if strings.Contains(lower, "coder") || strings.Contains(lower, "code") {
		cat = "coder"
	} else if strings.Contains(lower, "r1") || strings.Contains(lower, "deepseek") || strings.Contains(lower, "reason") {
		cat = "reasoning"
	}

	return &ModelOption{
		Name:          name + " (Custom HF)",
		RepoID:        repoID,
		Filename:      filename,
		DownloadURL:   downloadURL,
		SizeGB:        4.5,
		MinRAMGB:      12.0,
		Description:   fmt.Sprintf("Custom Hugging Face model from %s", repoID),
		Source:        SourceCustom,
		Category:      cat,
		Tags:          []string{"Custom", "Hugging Face", quant, param},
		Quantization:  quant,
		ParamSize:     param,
		ContextLength: 32768,
	}, nil
}

func extractQuant(s string) string {
	upper := strings.ToUpper(s)
	for _, q := range []string{"Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S", "Q4_0", "Q3_K_M", "Q3_K_S", "Q2_K", "IQ4_XS", "IQ3_M", "IQ2_M", "IQ1_M"} {
		if strings.Contains(upper, q) {
			return q
		}
	}
	return "Q4_K_M"
}

func extractParamSize(s string) string {
	upper := strings.ToUpper(s)
	for _, p := range []string{"1.5B", "3B", "7B", "8B", "9B", "14B", "16B", "27B", "30B", "32B", "70B"} {
		if strings.Contains(upper, p) {
			return p
		}
	}
	return "7B"
}

// RecommendModel picks the optimal model based on detected RAM
func RecommendModel(specs HardwareSpecs) ModelOption {
	models := GetAvailableModels()
	if len(models) == 0 {
		return ModelOption{}
	}

	if specs.RAMTotalGB >= 28.0 {
		for i := range models {
			if strings.Contains(models[i].Name, "Q8_0") {
				models[i].Recommended = true
				return models[i]
			}
		}
	} else if specs.RAMTotalGB >= 12.0 {
		for i := range models {
			if strings.Contains(models[i].Name, "Q4_K_M") && strings.Contains(models[i].Name, "7B") {
				models[i].Recommended = true
				return models[i]
			}
		}
	} else {
		for i := range models {
			if strings.Contains(models[i].Name, "Q2_K") {
				models[i].Recommended = true
				return models[i]
			}
		}
	}
	models[0].Recommended = true
	return models[0]
}

type DownloadStatus string

const (
	StatusIdle        DownloadStatus = "idle"
	StatusDownloading DownloadStatus = "downloading"
	StatusCompleted   DownloadStatus = "completed"
	StatusFailed      DownloadStatus = "failed"
)

type DownloadProgress struct {
	Status     DownloadStatus `json:"status"`
	ModelName  string         `json:"model_name"`
	Downloaded int64          `json:"downloaded_bytes"`
	Total      int64          `json:"total_bytes"`
	Percent    float64        `json:"percent"`
	SpeedMBs   float64        `json:"speed_mb_s"`
	DestPath   string         `json:"dest_path"`
	Error      string         `json:"error,omitempty"`
}

type SetupManager struct {
	mu         sync.RWMutex
	progress   DownloadProgress
	cancelCtx  context.Context
	cancelFunc context.CancelFunc
}

func NewSetupManager() *SetupManager {
	return &SetupManager{
		progress: DownloadProgress{
			Status: StatusIdle,
		},
	}
}

func (sm *SetupManager) GetStatus() DownloadProgress {
	sm.mu.RLock()
	defer sm.mu.RUnlock()
	return sm.progress
}

func (sm *SetupManager) CancelDownload() {
	sm.mu.Lock()
	defer sm.mu.Unlock()
	if sm.cancelFunc != nil {
		sm.cancelFunc()
		sm.cancelFunc = nil
	}
	if sm.progress.Status == StatusDownloading {
		sm.progress.Status = StatusIdle
		sm.progress.Error = "Download cancelled by user"
	}
}

func (sm *SetupManager) StartDownload(modelName, url, destPath string) error {
	sm.mu.Lock()
	if sm.progress.Status == StatusDownloading {
		sm.mu.Unlock()
		return fmt.Errorf("a download is already in progress")
	}

	ctx, cancel := context.WithCancel(context.Background())
	sm.cancelCtx = ctx
	sm.cancelFunc = cancel
	sm.progress = DownloadProgress{
		Status:    StatusDownloading,
		ModelName: modelName,
		DestPath:  destPath,
	}
	sm.mu.Unlock()

	go func() {
		lastBytes := int64(0)
		lastTime := time.Now()

		err := DownloadWithResume(url, destPath, func(downloaded, total int64) {
			sm.mu.Lock()
			defer sm.mu.Unlock()

			now := time.Now()
			duration := now.Sub(lastTime).Seconds()
			if duration >= 0.5 {
				bytesDelta := downloaded - lastBytes
				sm.progress.SpeedMBs = float64(bytesDelta) / 1024 / 1024 / duration
				lastBytes = downloaded
				lastTime = now
			}

			sm.progress.Downloaded = downloaded
			sm.progress.Total = total
			if total > 0 {
				sm.progress.Percent = (float64(downloaded) / float64(total)) * 100.0
			}
		})

		sm.mu.Lock()
		defer sm.mu.Unlock()
		if err != nil {
			sm.progress.Status = StatusFailed
			sm.progress.Error = err.Error()
		} else {
			sm.progress.Status = StatusCompleted
			sm.progress.Percent = 100.0
			sm.progress.SpeedMBs = 0
		}
	}()

	return nil
}

// DownloadWithResume downloads a model weight file with HTTP Range resume support
func DownloadWithResume(url, destPath string, progress func(downloaded, total int64)) error {
	var startOffset int64 = 0
	if fi, err := os.Stat(destPath); err == nil {
		startOffset = fi.Size()
	}

	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}

	// Attach HF_TOKEN if available
	hfToken := os.Getenv("HF_TOKEN")
	if hfToken == "" {
		hfToken = os.Getenv("HUGGING_FACE_HUB_TOKEN")
	}
	if hfToken != "" {
		req.Header.Set("Authorization", "Bearer "+hfToken)
	}
	req.Header.Set("User-Agent", "Strata-Setup/1.0")

	if startOffset > 0 {
		req.Header.Set("Range", fmt.Sprintf("bytes=%d-", startOffset))
	}

	client := &http.Client{
		Timeout: 0, // No global timeout for large model downloads
	}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK && resp.StatusCode != http.StatusPartialContent {
		if resp.StatusCode == http.StatusRequestedRangeNotSatisfiable {
			// Already fully downloaded
			return nil
		}
		if resp.StatusCode == http.StatusUnauthorized {
			return fmt.Errorf("HTTP error 401 Unauthorized: Hugging Face model repository requires authentication. Please set the HF_TOKEN environment variable or use a direct public model URL")
		}
		if resp.StatusCode == http.StatusForbidden {
			return fmt.Errorf("HTTP error 403 Forbidden: access denied to model file")
		}
		if resp.StatusCode == http.StatusNotFound {
			return fmt.Errorf("HTTP error 404 Not Found: model weight file does not exist at %s", url)
		}
		return fmt.Errorf("HTTP error %s", resp.Status)
	}

	totalSize := resp.ContentLength
	if resp.StatusCode == http.StatusPartialContent {
		totalSize += startOffset
	}

	os.MkdirAll(filepath.Dir(destPath), 0755)
	flags := os.O_CREATE | os.O_WRONLY
	if startOffset > 0 && resp.StatusCode == http.StatusPartialContent {
		flags |= os.O_APPEND
	} else {
		flags |= os.O_TRUNC
		startOffset = 0
	}

	file, err := os.OpenFile(destPath, flags, 0644)
	if err != nil {
		return err
	}
	defer file.Close()

	buf := make([]byte, 64*1024)
	current := startOffset

	for {
		n, readErr := resp.Body.Read(buf)
		if n > 0 {
			if _, writeErr := file.Write(buf[:n]); writeErr != nil {
				return writeErr
			}
			current += int64(n)
			if progress != nil {
				progress(current, totalSize)
			}
		}
		if readErr != nil {
			if readErr == io.EOF {
				break
			}
			return readErr
		}
	}

	return nil
}

// VerifyFileSHA256 checks cryptographic SHA256 checksum of a downloaded file
func VerifyFileSHA256(filePath, expectedSHA256 string) (bool, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return false, err
	}
	defer file.Close()

	h := sha256.New()
	if _, err := io.Copy(h, file); err != nil {
		return false, err
	}

	computed := hex.EncodeToString(h.Sum(nil))
	return strings.EqualFold(computed, expectedSHA256), nil
}
