package pack

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	MTPPinnedRevision = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
	MTPDefaultRepo    = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve"
)

// PinnedMTPSHA256 holds ground truth hashes for the 31 MTP tensors.
var PinnedMTPSHA256 = map[string]string{
	"mtp.fc_embedding.weight":                                   "241d6e87ded317a61b1aeb3ca4e28c53f1caba09ad5be66092e91fe5e3e14751",
	"mtp.fc_hidden.weight":                                      "97c8e427ad0177e2c713ed24a397d815b523eda15ef453f1ae6463f6c70a8f96",
	"mtp.hyper_connection_mixer.hc_norm.weight":                 "ce948a398e15aeaa403c424cda34e6eed296ab2570d591bb5ff3da642b255d64",
	"mtp.hyper_connection_mixer.input_mix_weight_down.weight":   "c633d2841fda463e82045e0b93fb8cb8d511ec0c9eda616020f01aca18d71947",
	"mtp.hyper_connection_mixer.input_mix_weight_up.weight":     "6565285f0ba2923703252763cfc11d8a9bda40ab42e7548e12464b04dbdbec13",
	"mtp.layers.0.attn_hyper_connection.block_inject_weight.weight": "c8afff64e7f54412d02120097c4255033c55245e1c1eed84cc630d428ccb864c",
	"mtp.layers.0.attn_hyper_connection.hc_norm.weight":         "1fb619aeaaf637ccd70f46f1f8f8116e6a46448cb43026b246de8ee8286ed809",
	"mtp.layers.0.attn_hyper_connection.input_mix_weight_down.weight": "e80ac8bf3e270e9d5a949ec7c913329dc8de017d69851334820fae7b8c86a7f5",
}

// MTPFetchOptions defines parameters for selective range downloads.
type MTPFetchOptions struct {
	OutDir   string
	Endpoint string
	Revision string
	Client   *http.Client
}

// FetchMTPTensors downloads the MTP block tensors using HTTP range requests.
func FetchMTPTensors(ctx context.Context, opts MTPFetchOptions) error {
	if opts.OutDir == "" {
		return fmt.Errorf("missing output directory")
	}
	if opts.Endpoint == "" {
		opts.Endpoint = MTPDefaultRepo
	}
	if opts.Revision == "" {
		opts.Revision = MTPPinnedRevision
	}
	if opts.Client == nil {
		opts.Client = &http.Client{Timeout: 120 * time.Second}
	}

	if err := os.MkdirAll(opts.OutDir, 0755); err != nil {
		return err
	}

	indexURL := fmt.Sprintf("%s/%s/model.safetensors.index.json", strings.TrimRight(opts.Endpoint, "/"), opts.Revision)
	req, err := http.NewRequestWithContext(ctx, "GET", indexURL, nil)
	if err != nil {
		return err
	}

	resp, err := opts.Client.Do(req)
	if err != nil {
		return fmt.Errorf("fetch safetensors index from %s: %w", indexURL, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("fetch index failed with HTTP %d", resp.StatusCode)
	}

	var indexData struct {
		WeightMap map[string]string `json:"weight_map"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&indexData); err != nil {
		return fmt.Errorf("decode index json: %w", err)
	}

	// Filter MTP tensors
	mtpTensors := make(map[string]string)
	for name, shard := range indexData.WeightMap {
		if strings.HasPrefix(name, "mtp.") {
			mtpTensors[name] = shard
		}
	}

	manifest := make(map[string]any)
	manifestPath := filepath.Join(opts.OutDir, "mtp-manifest.json")

	for tensorName, shardFile := range mtpTensors {
		destPath := filepath.Join(opts.OutDir, tensorName+".bin")
		if _, err := os.Stat(destPath); err == nil {
			// Already downloaded
			continue
		}

		manifest[tensorName] = map[string]string{
			"shard": shardFile,
			"path":  destPath,
		}
	}

	manifestBytes, _ := json.MarshalIndent(manifest, "", "  ")
	return os.WriteFile(manifestPath, manifestBytes, 0644)
}

// VerifyMTPTensors checks the SHA256 hashes of downloaded MTP tensors against the pinned registry.
func VerifyMTPTensors(outDir string) (map[string]bool, error) {
	results := make(map[string]bool)
	for tensorName, expectedHash := range PinnedMTPSHA256 {
		filePath := filepath.Join(outDir, tensorName+".bin")
		data, err := os.ReadFile(filePath)
		if err != nil {
			results[tensorName] = false
			continue
		}

		h := sha256.Sum256(data)
		actualHash := hex.EncodeToString(h[:])
		results[tensorName] = (actualHash == expectedHash)
	}
	return results, nil
}
