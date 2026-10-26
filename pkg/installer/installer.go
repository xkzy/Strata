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
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
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
	Shards        []string    `json:"shards,omitempty"`
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
	Installed     bool        `json:"installed,omitempty"`
	HasFiles      bool        `json:"has_files,omitempty"`
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
							Installed:     true,
							HasFiles:      true,
						})
					}
				}
			}
			return nil
		})
	}

	return results
}

// GetDataDir returns the root Strata-data directory where models, packs, and mtp layers are stored.
func GetDataDir() string {
	if d := os.Getenv("STRATA_DATA_DIR"); d != "" {
		return d
	}
	cwd, _ := os.Getwd()
	// Priority 1: parent directory / Strata-data (standard layout: ../Strata-data next to Strata/)
	parentData := filepath.Join(filepath.Dir(cwd), "Strata-data")
	if fi, err := os.Stat(parentData); err == nil && fi.IsDir() {
		return parentData
	}
	// Priority 2: user home ~/Downloads/Strata-data
	if home, err := os.UserHomeDir(); err == nil && home != "" {
		homeData := filepath.Join(home, "Downloads", "Strata-data")
		if fi, err := os.Stat(homeData); err == nil && fi.IsDir() {
			return homeData
		}
	}
	// Priority 3: local ./Strata-data
	localData := filepath.Join(cwd, "Strata-data")
	if fi, err := os.Stat(localData); err == nil && fi.IsDir() {
		return localData
	}
	return parentData
}

// GetModelsDir returns the directory where GGUF model files are stored ($STRATA_DATA_DIR/models).
func GetModelsDir() string {
	dir := filepath.Join(GetDataDir(), "models")
	_ = os.MkdirAll(dir, 0755)
	return dir
}

// GetPacksDir returns the directory where compiled Strata packs are stored ($STRATA_DATA_DIR/packs).
func GetPacksDir() string {
	dir := filepath.Join(GetDataDir(), "packs")
	_ = os.MkdirAll(dir, 0755)
	return dir
}

func calcDirSizeGB(dir string) float64 {
	var total int64
	_ = filepath.Walk(dir, func(_ string, info os.FileInfo, err error) error {
		if err == nil && !info.IsDir() {
			total += info.Size()
		}
		return nil
	})
	return float64(int((float64(total)/(1024*1024*1024))*10+0.5)) / 10.0
}

func formatPackName(n string) string {
	n = strings.ReplaceAll(n, "-", " ")
	n = strings.ReplaceAll(n, "_", " ")
	words := strings.Fields(n)
	for i, w := range words {
		if len(w) > 0 {
			words[i] = strings.ToUpper(w[:1]) + w[1:]
		}
	}
	return strings.Join(words, "-")
}

// ScanLocalCustomModels discovers existing local Strata packs and GGUFs in Strata-data
func ScanLocalCustomModels() []ModelOption {
	var results []ModelOption
	seenPaths := make(map[string]bool)

	// 1. Scan packs in GetPacksDir()
	packsDir := GetPacksDir()
	if entries, err := os.ReadDir(packsDir); err == nil {
		for _, e := range entries {
			if e.IsDir() {
				packPath := filepath.Join(packsDir, e.Name())
				if seenPaths[packPath] {
					continue
				}
				hasTok := false
				if _, err := os.Stat(filepath.Join(packPath, "tokenizer", "vocab.json")); err == nil {
					hasTok = true
				}
				hasIndex := false
				if _, err := os.Stat(filepath.Join(packPath, "index.txt")); err == nil {
					hasIndex = true
				}
				hasNative := false
				if _, err := os.Stat(filepath.Join(packPath, "native_experts.txt")); err == nil {
					hasNative = true
				}
				if hasTok || hasIndex || hasNative {
					seenPaths[packPath] = true
					sizeGB := calcDirSizeGB(packPath)
					name := e.Name()
					results = append(results, ModelOption{
						Name:          formatPackName(name) + " (Local Pack)",
						RepoID:        "local/packs/" + name,
						Filename:      packPath,
						DownloadURL:   "",
						SizeGB:        sizeGB,
						MinRAMGB:      max(16.0, sizeGB*0.8),
						Recommended:   strings.Contains(strings.ToLower(name), "coder"),
						Description:   fmt.Sprintf("Existing local compiled pack in Strata-data/packs/%s (Instant Activation, Optimized KV)", name),
						Source:        SourceLocal,
						Category:      "coder",
						Tags:          []string{"Strata Pack", "MoE", "Local", "Fast"},
						Quantization:  extractQuant(name),
						ParamSize:     "MoE",
						ContextLength: 32768,
						Installed:     true,
						HasFiles:      true,
					})
				}
			}
		}
	}

	// 2. Scan GGUFs in GetModelsDir() and local "models"
	modelsDirs := []string{GetModelsDir(), "models"}
	reShard := regexp.MustCompile(`-(\d{5})-of-(\d{5})\.gguf$`)

	for _, mDir := range modelsDirs {
		entries, err := os.ReadDir(mDir)
		if err != nil {
			continue
		}
		for _, e := range entries {
			fullPath := filepath.Join(mDir, e.Name())
			if seenPaths[fullPath] {
				continue
			}

			if e.IsDir() {
				subEntries, err := os.ReadDir(fullPath)
				if err != nil {
					continue
				}
				var firstShard string
				var totalBytes int64
				for _, se := range subEntries {
					if strings.HasSuffix(strings.ToLower(se.Name()), ".gguf") {
						fi, _ := se.Info()
						if fi != nil {
							totalBytes += fi.Size()
						}
						if firstShard == "" || strings.Contains(se.Name(), "-00001-of-") {
							firstShard = filepath.Join(fullPath, se.Name())
						}
					}
				}
				if firstShard != "" {
					seenPaths[fullPath] = true
					sizeGB := float64(totalBytes) / (1024 * 1024 * 1024)
					results = append(results, ModelOption{
						Name:          formatPackName(e.Name()) + " (Local GGUF)",
						RepoID:        "local/models/" + e.Name(),
						Filename:      fullPath,
						DownloadURL:   "",
						SizeGB:        float64(int(sizeGB*10+0.5)) / 10.0,
						MinRAMGB:      float64(int((sizeGB*0.5+8)*10+0.5)) / 10.0,
						Description:   fmt.Sprintf("Existing local GGUF shards in Strata-data/models/%s", e.Name()),
						Source:        SourceLocal,
						Category:      "coder",
						Tags:          []string{"GGUF Shards", "Local"},
						Quantization:  extractQuant(e.Name()),
						ParamSize:     "MoE",
						ContextLength: 32768,
						Installed:     true,
						HasFiles:      true,
					})
				}
			} else if strings.HasSuffix(strings.ToLower(e.Name()), ".gguf") {
				sm := reShard.FindStringSubmatch(e.Name())
				if len(sm) == 3 {
					idx, _ := strconv.Atoi(sm[1])
					if idx > 1 {
						continue
					}
				}
				seenPaths[fullPath] = true
				shards := FindAllShardPaths(fullPath)
				var totalBytes int64
				for _, s := range shards {
					if fi, err := os.Stat(s); err == nil {
						totalBytes += fi.Size()
					}
				}
				sizeGB := float64(totalBytes) / (1024 * 1024 * 1024)
				name := strings.TrimSuffix(e.Name(), filepath.Ext(e.Name()))
				if len(sm) == 3 {
					name = strings.TrimSuffix(name, "-"+sm[1]+"-of-"+sm[2])
				}
				results = append(results, ModelOption{
					Name:          name + " (Local)",
					RepoID:        "local/models/" + e.Name(),
					Filename:      fullPath,
					DownloadURL:   "",
					SizeGB:        float64(int(sizeGB*10+0.5)) / 10.0,
					MinRAMGB:      float64(int((sizeGB*1.2+2)*10+0.5)) / 10.0,
					Description:   fmt.Sprintf("Downloaded GGUF file in Strata-data/models (%s)", e.Name()),
					Source:        SourceLocal,
					Category:      "local",
					Tags:          []string{"Local", "GGUF", "Ready"},
					Quantization:  extractQuant(e.Name()),
					ParamSize:     extractParamSize(e.Name()),
					ContextLength: 32768,
					Installed:     AllShardsExist(fullPath),
					HasFiles:      true,
				})
			}
		}
	}

	return results
}

// GetCuratedHuggingFaceModels returns a rich curated catalog of popular GGUF models
func GetCuratedHuggingFaceModels() []ModelOption {
	return []ModelOption{
		// --- Xiaomi MiMo (Hybrid MoE + Native RL) ---
		{
			Name:        "MiMo-V2.6-Flash-RL-Q2_K",
			RepoID:      "ggml-org/MiMo-V2.6-Flash-RL-GGUF",
			Filename:    "MiMo-V2.6-Flash-RL-Q2_K-00001-of-00002.gguf",
			DownloadURL: "https://huggingface.co/ggml-org/MiMo-V2.6-Flash-RL-GGUF/resolve/main/MiMo-V2.6-Flash-RL-Q2_K-00001-of-00002.gguf",
			Shards: []string{
				"https://huggingface.co/ggml-org/MiMo-V2.6-Flash-RL-GGUF/resolve/main/MiMo-V2.6-Flash-RL-Q2_K-00001-of-00002.gguf",
				"https://huggingface.co/ggml-org/MiMo-V2.6-Flash-RL-GGUF/resolve/main/MiMo-V2.6-Flash-RL-Q2_K-00002-of-00002.gguf",
			},
			SizeGB:        126.2,
			MinRAMGB:      28.0,
			Description:   "Xiaomi MiMo-V2.6-Flash-RL 2-bit K-quant MoE with native RL reasoning (2 shards, 126.2 GB, fits current disk)",
			Source:        SourceHuggingFace,
			Category:      "mimo",
			Tags:          []string{"MiMo", "Xiaomi", "Reasoning", "MoE", "RL", "Q2_K"},
			Quantization:  "Q2_K",
			ParamSize:     "256-Expert MoE",
			ContextLength: 1048576,
		},
		{
			Name:        "MiMo-V2.6-Flash-RL-MXFP4",
			RepoID:      "ggml-org/MiMo-V2.6-Flash-RL-GGUF",
			Filename:    "MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf",
			DownloadURL: "https://huggingface.co/ggml-org/MiMo-V2.6-Flash-RL-GGUF/resolve/main/MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf",
			Shards: []string{
				"https://huggingface.co/ggml-org/MiMo-V2.6-Flash-RL-GGUF/resolve/main/MiMo-V2.6-Flash-RL-MXFP4-00001-of-00002.gguf",
				"https://huggingface.co/ggml-org/MiMo-V2.6-Flash-RL-GGUF/resolve/main/MiMo-V2.6-Flash-RL-MXFP4-00002-of-00002.gguf",
			},
			SizeGB:        167.4,
			MinRAMGB:      28.0,
			Description:   "Xiaomi MiMo-V2.6-Flash-RL 4-bit microscaling format MoE with high fidelity RL reasoning (2 shards, 167.4 GB)",
			Source:        SourceHuggingFace,
			Category:      "mimo",
			Tags:          []string{"MiMo", "Xiaomi", "Reasoning", "MoE", "RL", "MXFP4"},
			Quantization:  "MXFP4",
			ParamSize:     "256-Expert MoE",
			ContextLength: 1048576,
		},

		// --- DeepSeek-V4.1-Flash (552B-backbone multimodal MoE, 40 text layers, 384 experts) ---
		{
			Name:          "DeepSeek-V4.1-Flash-GSQ-RCO-3.0bit",
			RepoID:        "pfeifferj/DeepSeek-V4.1-Flash-GSQ-RCO-GGUF",
			Filename:      "DeepSeek-V4.1-Flash-GSQ-RCO-3.0bit.gguf",
			DownloadURL:   "https://huggingface.co/pfeifferj/DeepSeek-V4.1-Flash-GSQ-RCO-GGUF/resolve/main/DeepSeek-V4.1-Flash-GSQ-RCO-3.0bit.gguf",
			SizeGB:        415.8,
			MinRAMGB:      448.0,
			Description:   "DeepSeek-V4.1-Flash 552B-backbone multimodal MoE in GSQ-RCO 3.0-bit (single 415.8 GB file, 1M-token context; separate DSpark draft 9.3 GB and mmproj vision 1.0 GB in the same repo)",
			Source:        SourceHuggingFace,
			Category:      "reasoning",
			Tags:          []string{"DeepSeek", "MoE", "552B", "1M Context", "GSQ-RCO", "Vision"},
			Quantization:  "GSQ-RCO-3.0bit",
			ParamSize:     "552B-Backbone MoE",
			ContextLength: 1048576,
		},

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
			Name:          "GPT-OSS-20B-MXFP4",
			RepoID:        "openai/gpt-oss-20b",
			Filename:      "gpt-oss-20b-mxfp4.gguf",
			DownloadURL:   "https://huggingface.co/openai/gpt-oss-20b/resolve/main/gpt-oss-20b-mxfp4.gguf",
			SizeGB:        13.5,
			MinRAMGB:      16.0,
			Description:   "OpenAI GPT-OSS 21B MoE (3.6B active) in native MXFP4 quantization with full CoT reasoning",
			Source:        SourceHuggingFace,
			Category:      "reasoning",
			Tags:          []string{"OpenAI", "GPT-OSS", "Reasoning", "MoE", "MXFP4", "20B"},
			Quantization:  "MXFP4",
			ParamSize:     "21B MoE (3.6B Active)",
			ContextLength: 131072,
		},
		{
			Name:          "GPT-OSS-20B-UD-Q4_K_XL",
			RepoID:        "unsloth/gpt-oss-20b-GGUF",
			Filename:      "gpt-oss-20b-UD-Q4_K_XL.gguf",
			DownloadURL:   "https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-UD-Q4_K_XL.gguf",
			SizeGB:        11.9,
			MinRAMGB:      16.0,
			Description:   "OpenAI GPT-OSS-20B Unsloth Dynamic Q4_K_XL quantization (maximum 4-bit fidelity, 131k context)",
			Source:        SourceHuggingFace,
			Category:      "reasoning",
			Tags:          []string{"OpenAI", "Unsloth", "UD-Q4_K_XL", "21B MoE", "Dynamic", "Reasoning"},
			Quantization:  "UD-Q4_K_XL",
			ParamSize:     "21B MoE (3.6B Active)",
			ContextLength: 131072,
		},
		{
			Name:          "GPT-OSS-20B-Q4_K_M",
			RepoID:        "unsloth/gpt-oss-20b-GGUF",
			Filename:      "gpt-oss-20b-Q4_K_M.gguf",
			DownloadURL:   "https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q4_K_M.gguf",
			SizeGB:        11.6,
			MinRAMGB:      16.0,
			Description:   "OpenAI GPT-OSS-20B balanced Q4_K_M quantization from Unsloth (131k context, CoT reasoning)",
			Source:        SourceHuggingFace,
			Category:      "reasoning",
			Tags:          []string{"OpenAI", "Unsloth", "Q4_K_M", "21B MoE", "Reasoning"},
			Quantization:  "Q4_K_M",
			ParamSize:     "21B MoE (3.6B Active)",
			ContextLength: 131072,
		},
		{
			Name:          "GPT-OSS-20B-Q2_K",
			RepoID:        "unsloth/gpt-oss-20b-GGUF",
			Filename:      "gpt-oss-20b-Q2_K.gguf",
			DownloadURL:   "https://huggingface.co/unsloth/gpt-oss-20b-GGUF/resolve/main/gpt-oss-20b-Q2_K.gguf",
			SizeGB:        11.5,
			MinRAMGB:      16.0,
			Description:   "OpenAI GPT-OSS-20B ultra-compact 2-bit quantization from Unsloth",
			Source:        SourceHuggingFace,
			Category:      "reasoning",
			Tags:          []string{"OpenAI", "Unsloth", "Q2_K", "21B MoE", "Compact"},
			Quantization:  "Q2_K",
			ParamSize:     "21B MoE (3.6B Active)",
			ContextLength: 131072,
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

	curated := GetCuratedHuggingFaceModels()
	curatedFiles := make(map[string]bool)
	for i := range curated {
		curatedFiles[curated[i].Filename] = true
		curatedFiles[filepath.Base(curated[i].Filename)] = true
		dest := filepath.Join("models", curated[i].Filename)
		totalGB := TotalShardsSizeGB(dest)
		if totalGB > 0 || HasAnyShardOrPart(dest) {
			curated[i].HasFiles = true
		}
		if AllShardsExist(dest) && (curated[i].SizeGB == 0 || totalGB >= curated[i].SizeGB*0.95) {
			curated[i].Installed = true
		}
	}

	// 1. Local packs and downloaded files (avoid duplicates already represented in curated catalog)
	for _, local := range ScanLocalCustomModels() {
		base := filepath.Base(local.Filename)
		if curatedFiles[base] {
			continue
		}
		models = append(models, local)
	}

	// 2. Local Ollama models
	models = append(models, ScanLocalOllamaModels()...)

	// 3. Online downloadable Hugging Face catalog
	models = append(models, curated...)

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
	for _, p := range []string{"1.5B", "3B", "7B", "8B", "9B", "14B", "16B", "27B", "30B", "31B", "32B", "70B"} {
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

// FindAllShardPaths expands a shard-1 destination path to all shard paths if multi-shard pattern is detected
func FindAllShardPaths(primaryPath string) []string {
	re := regexp.MustCompile(`-(\d{5})-of-(\d{5})\.gguf$`)
	m := re.FindStringSubmatch(primaryPath)
	if len(m) == 3 {
		total, err := strconv.Atoi(m[2])
		if err == nil && total > 1 {
			prefix := primaryPath[:len(primaryPath)-len(m[0])]
			var paths []string
			for i := 1; i <= total; i++ {
				paths = append(paths, fmt.Sprintf("%s-%05d-of-%05d.gguf", prefix, i, total))
			}
			return paths
		}
	}
	return []string{primaryPath}
}

// TotalShardsSizeGB calculates total size in GB across all shards of a model path
func TotalShardsSizeGB(primaryPath string) float64 {
	shards := FindAllShardPaths(primaryPath)
	var total int64
	for _, s := range shards {
		if fi, err := os.Stat(s); err == nil {
			total += fi.Size()
		}
	}
	return float64(total) / (1024 * 1024 * 1024)
}

// ResolveShardURLs returns all shard URLs for a multi-shard model download
func ResolveShardURLs(primaryURL string, explicitShards []string) []string {
	if len(explicitShards) > 0 {
		return explicitShards
	}
	re := regexp.MustCompile(`-(\d{5})-of-(\d{5})\.gguf`)
	m := re.FindStringSubmatch(primaryURL)
	if len(m) == 3 {
		total, err := strconv.Atoi(m[2])
		if err == nil && total > 1 {
			var urls []string
			for i := 1; i <= total; i++ {
				u := re.ReplaceAllString(primaryURL, fmt.Sprintf("-%05d-of-%05d.gguf", i, total))
				urls = append(urls, u)
			}
			return urls
		}
	}
	return []string{primaryURL}
}

// AllShardsExist verifies whether all shards of a multi-shard GGUF exist and are non-empty
func AllShardsExist(primaryPath string) bool {
	shards := FindAllShardPaths(primaryPath)
	if len(shards) == 0 {
		return false
	}
	for i, s := range shards {
		fi, err := os.Stat(s)
		if err != nil || fi.Size() == 0 {
			return false
		}
		// If it is a multi-shard model and not shard 1, check that it contains substantial weight data (> 100 MB)
		if len(shards) > 1 && i > 0 && fi.Size() < 100*1024*1024 {
			return false
		}
	}
	return true
}

// GetFreeDiskSpaceGB returns available storage space in GB on the filesystem containing dir
func GetFreeDiskSpaceGB(dir string) float64 {
	p := dir
	if p == "" {
		p = "."
	}
	if err := os.MkdirAll(p, 0755); err != nil {
		p = "."
	}
	var stat syscall.Statfs_t
	if err := syscall.Statfs(p, &stat); err == nil {
		freeBytes := stat.Bavail * uint64(stat.Bsize)
		return float64(freeBytes) / (1024 * 1024 * 1024)
	}
	return 999.0
}

type DownloadOptions struct {
	ModelName string
	URL       string
	DestPath  string
	SizeGB    float64
	Shards    []string
}

func (sm *SetupManager) StartDownload(modelName, url, destPath string) error {
	return sm.StartDownloadWithOptions(DownloadOptions{
		ModelName: modelName,
		URL:       url,
		DestPath:  destPath,
	})
}

func (sm *SetupManager) StartDownloadWithOptions(opts DownloadOptions) error {
	sm.mu.Lock()
	if sm.progress.Status == StatusDownloading {
		sm.mu.Unlock()
		return fmt.Errorf("a download is already in progress")
	}

	destDir := filepath.Dir(opts.DestPath)
	freeGB := GetFreeDiskSpaceGB(destDir)
	if opts.SizeGB > 0 && freeGB < opts.SizeGB {
		sm.mu.Unlock()
		return fmt.Errorf("insufficient disk space: %s requires ~%.1f GB, but only %.1f GB is available in %s. Please free up disk space or choose a smaller model (such as MiMo-V2.6-Flash-RL-Q2_K which requires 126.2 GB)", opts.ModelName, opts.SizeGB, freeGB, destDir)
	}

	shardURLs := ResolveShardURLs(opts.URL, opts.Shards)
	shardPaths := FindAllShardPaths(opts.DestPath)
	if len(shardPaths) != len(shardURLs) {
		shardPaths = make([]string, len(shardURLs))
		for i, u := range shardURLs {
			shardPaths[i] = filepath.Join(destDir, filepath.Base(u))
		}
	}

	ctx, cancel := context.WithCancel(context.Background())
	sm.cancelCtx = ctx
	sm.cancelFunc = cancel
	sm.progress = DownloadProgress{
		Status:    StatusDownloading,
		ModelName: opts.ModelName,
		DestPath:  opts.DestPath,
	}
	sm.mu.Unlock()

	go func() {
		totalShards := len(shardURLs)
		lastBytes := int64(0)
		lastTime := time.Now()

		overallTotal := int64(opts.SizeGB * 1024 * 1024 * 1024)
		priorBytes := int64(0)

		for i := 0; i < totalShards; i++ {
			select {
			case <-ctx.Done():
				sm.mu.Lock()
				sm.progress.Status = StatusIdle
				sm.progress.Error = "Download cancelled by user"
				sm.mu.Unlock()
				return
			default:
			}

			currentURL := shardURLs[i]
			currentDest := shardPaths[i]

			sm.mu.Lock()
			if totalShards > 1 {
				sm.progress.ModelName = fmt.Sprintf("%s (Shard %d/%d)", opts.ModelName, i+1, totalShards)
			}
			sm.mu.Unlock()

			err := DownloadWithResume(currentURL, currentDest, func(downloaded, shardTotal int64) {
				sm.mu.Lock()
				defer sm.mu.Unlock()

				now := time.Now()
				duration := now.Sub(lastTime).Seconds()
				totalDownloaded := priorBytes + downloaded

				if duration >= 0.5 {
					bytesDelta := totalDownloaded - lastBytes
					sm.progress.SpeedMBs = float64(bytesDelta) / 1024 / 1024 / duration
					lastBytes = totalDownloaded
					lastTime = now
				}

				sm.progress.Downloaded = totalDownloaded
				if overallTotal > 0 {
					sm.progress.Total = overallTotal
					sm.progress.Percent = (float64(totalDownloaded) / float64(overallTotal)) * 100.0
				} else if shardTotal > 0 {
					sm.progress.Total = priorBytes + shardTotal
					sm.progress.Percent = (float64(totalDownloaded) / float64(priorBytes+shardTotal)) * 100.0
				}
			})

			if err != nil {
				sm.mu.Lock()
				sm.progress.Status = StatusFailed
				sm.progress.Error = fmt.Sprintf("Failed downloading shard %d/%d (%s): %v", i+1, totalShards, filepath.Base(currentDest), err)
				sm.mu.Unlock()
				return
			}

			if fi, err := os.Stat(currentDest); err == nil {
				priorBytes += fi.Size()
			}
		}

		sm.mu.Lock()
		defer sm.mu.Unlock()
		sm.progress.Status = StatusCompleted
		sm.progress.Percent = 100.0
		sm.progress.SpeedMBs = 0
		sm.progress.ModelName = opts.ModelName
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

func mustAbs(p string) string {
	a, err := filepath.Abs(p)
	if err != nil {
		return p
	}
	return a
}

// HasAnyShardOrPart checks whether any shard or partial download file exists for the primaryPath
func HasAnyShardOrPart(primaryPath string) bool {
	if _, err := os.Stat(primaryPath); err == nil {
		return true
	}
	if _, err := os.Stat(primaryPath + ".part"); err == nil {
		return true
	}
	shards := FindAllShardPaths(primaryPath)
	for _, s := range shards {
		if _, err := os.Stat(s); err == nil {
			return true
		}
		if _, err := os.Stat(s + ".part"); err == nil {
			return true
		}
	}
	return false
}

// IsSafeModelPath validates that target path is confined within allowed model or pack directories
func IsSafeModelPath(target string) bool {
	if strings.TrimSpace(target) == "" {
		return false
	}
	abs, err := filepath.Abs(target)
	if err != nil {
		return false
	}
	abs = filepath.Clean(abs)

	allowedDirs := []string{
		mustAbs("models"),
		mustAbs("packs"),
		GetModelsDir(),
		GetPacksDir(),
	}

	if home, err := os.UserHomeDir(); err == nil && home != "" {
		allowedDirs = append(allowedDirs,
			filepath.Join(home, "Downloads", "Strata-data", "models"),
			filepath.Join(home, "Downloads", "Strata-data", "packs"),
			filepath.Join(home, ".strata"),
		)
	}

	for _, allowed := range allowedDirs {
		cleanAllowed := filepath.Clean(allowed)
		if abs == cleanAllowed {
			// Never permit deleting the root container directory itself
			return false
		}
		if strings.HasPrefix(abs, cleanAllowed+string(filepath.Separator)) {
			return true
		}
	}
	return false
}

// DeleteModelFiles securely deletes all model weights, shards, and partial download files
// for a given model path or filename. Returns total bytes freed, deleted file paths, and error.
func DeleteModelFiles(target string) (int64, []string, error) {
	if strings.TrimSpace(target) == "" {
		return 0, nil, fmt.Errorf("empty model target path")
	}

	// Resolve path candidate
	candidates := []string{target}
	if !filepath.IsAbs(target) {
		candidates = append([]string{
			filepath.Join(GetModelsDir(), target),
			filepath.Join("models", target),
		}, candidates...)
	}

	var resolvedPath string
	for _, cand := range candidates {
		if _, err := os.Stat(cand); err == nil {
			resolvedPath = cand
			break
		}
		// Check if part file exists
		if _, err := os.Stat(cand + ".part"); err == nil {
			resolvedPath = cand
			break
		}
		// Check if any shard exists
		for _, s := range FindAllShardPaths(cand) {
			if _, err := os.Stat(s); err == nil {
				resolvedPath = cand
				break
			}
			if _, err := os.Stat(s + ".part"); err == nil {
				resolvedPath = cand
				break
			}
		}
		if resolvedPath != "" {
			break
		}
	}

	if resolvedPath == "" {
		resolvedPath = candidates[0]
	}

	if !IsSafeModelPath(resolvedPath) {
		return 0, nil, fmt.Errorf("security violation: path %q is outside allowed model directories", target)
	}

	var deletedPaths []string
	var freedBytes int64

	// If it's a directory (e.g. a local pack directory inside packs/ or Strata-data/packs/)
	fi, statErr := os.Stat(resolvedPath)
	if statErr == nil && fi.IsDir() {
		_ = filepath.Walk(resolvedPath, func(path string, info os.FileInfo, err error) error {
			if err == nil && !info.IsDir() {
				freedBytes += info.Size()
			}
			return nil
		})
		if err := os.RemoveAll(resolvedPath); err != nil {
			return freedBytes, nil, fmt.Errorf("failed to remove directory %s: %w", resolvedPath, err)
		}
		deletedPaths = append(deletedPaths, resolvedPath)
		return freedBytes, deletedPaths, nil
	}

	// Otherwise, find all potential shard paths and .part files
	shards := FindAllShardPaths(resolvedPath)
	seen := make(map[string]bool)

	var filesToDelete []string
	for _, s := range shards {
		if !seen[s] {
			seen[s] = true
			filesToDelete = append(filesToDelete, s)
		}
		partPath := s + ".part"
		if !seen[partPath] {
			seen[partPath] = true
			filesToDelete = append(filesToDelete, partPath)
		}
		offsetPath := s + ".part.offset"
		if !seen[offsetPath] {
			seen[offsetPath] = true
			filesToDelete = append(filesToDelete, offsetPath)
		}
	}

	// Also check base path .part and .part.offset
	basePart := resolvedPath + ".part"
	if !seen[basePart] {
		seen[basePart] = true
		filesToDelete = append(filesToDelete, basePart)
	}

	for _, f := range filesToDelete {
		if info, err := os.Stat(f); err == nil {
			size := info.Size()
			if err := os.Remove(f); err == nil {
				freedBytes += size
				deletedPaths = append(deletedPaths, f)
			} else {
				return freedBytes, deletedPaths, fmt.Errorf("failed to delete file %s: %w", f, err)
			}
		}
	}

	if len(deletedPaths) == 0 {
		return 0, nil, fmt.Errorf("no model files found to delete for %s", target)
	}

	return freedBytes, deletedPaths, nil
}

