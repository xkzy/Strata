// pkg/installer/installer.go - Native Hardware Detection, Model Recommendation & Downloader in Go
package installer

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
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

type HardwareSpecs struct {
	CPUCores   int       `json:"cpu_cores"`
	RAMTotalGB float64   `json:"ram_total_gb"`
	GPUVendor  GPUVendor `json:"gpu_vendor"`
	GPUName    string    `json:"gpu_name"`
	VRAMGB     float64   `json:"vram_gb"`
}

type ModelOption struct {
	Name        string  `json:"name"`
	RepoID      string  `json:"repo_id"`
	Filename    string  `json:"filename"`
	DownloadURL string  `json:"download_url"`
	SizeGB      float64 `json:"size_gb"`
	MinRAMGB    float64 `json:"min_ram_gb"`
	Recommended bool    `json:"recommended"`
	Description string  `json:"description"`
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

// GetAvailableModels returns supported model variants and scans for existing local packs
func GetAvailableModels() []ModelOption {
	models := []ModelOption{}

	// 1. Scan for existing local downloaded models/packs
	localPackPath := "/home/khing/Downloads/Strata-data/packs/coder-iq1_m"
	if _, err := os.Stat(localPackPath); err == nil {
		models = append(models, ModelOption{
			Name:        "Qwen3.8-Flash-Next-Coder-IQ1_M (Local Pack)",
			RepoID:      "local/Strata-data/packs/coder-iq1_m",
			Filename:    localPackPath,
			DownloadURL: "",
			SizeGB:      26.0,
			MinRAMGB:    16.0,
			Recommended: true,
			Description: "Existing local compiled pack in Strata-data (Instant Activation, No Download Needed)",
		})
	}

	localModelDir := "/home/khing/Downloads/Strata-data/models/coder-IQ1_M"
	if _, err := os.Stat(localModelDir); err == nil {
		models = append(models, ModelOption{
			Name:        "Qwen3.8-Flash-Next-Coder-IQ1_M (Local GGUF)",
			RepoID:      "local/Strata-data/models/coder-IQ1_M",
			Filename:    localModelDir,
			DownloadURL: "",
			SizeGB:      58.4,
			MinRAMGB:    24.0,
			Description: "Existing local downloaded GGUF shards in Strata-data",
		})
	}

	// 2. Online downloadable options
	models = append(models, []ModelOption{
		{
			Name:        "Qwen2.5-Coder-7B-Instruct-Q8_0",
			RepoID:      "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
			Filename:    "Qwen2.5-Coder-7B-Instruct-Q8_0.gguf",
			DownloadURL: "https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q8_0.gguf",
			SizeGB:      8.1,
			MinRAMGB:    24.0,
			Description: "High-precision 8-bit quantization for systems with >= 32GB RAM",
		},
		{
			Name:        "Qwen2.5-Coder-7B-Instruct-Q4_K_M",
			RepoID:      "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
			Filename:    "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
			DownloadURL: "https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
			SizeGB:      4.7,
			MinRAMGB:    12.0,
			Description: "Standard 4-bit quantization, balanced for 16GB-32GB RAM",
		},
		{
			Name:        "Qwen2.5-Coder-7B-Instruct-Q2_K",
			RepoID:      "bartowski/Qwen2.5-Coder-7B-Instruct-GGUF",
			Filename:    "Qwen2.5-Coder-7B-Instruct-Q2_K.gguf",
			DownloadURL: "https://huggingface.co/bartowski/Qwen2.5-Coder-7B-Instruct-GGUF/resolve/main/Qwen2.5-Coder-7B-Instruct-Q2_K.gguf",
			SizeGB:      2.9,
			MinRAMGB:    8.0,
			Description: "Lightweight 2-bit quantization for compact systems with 8GB-12GB RAM",
		},
	}...)

	return models
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
			if strings.Contains(models[i].Name, "Q4_K_M") {
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
	mu          sync.RWMutex
	progress    DownloadProgress
	cancelCtx   context.Context
	cancelFunc  context.CancelFunc
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
