//go:build !cuda

package densecpu

import (
	"errors"

	"strata/pkg/gguf"
)

// GPUStats says where the weights ended up (all zero without the cuda build tag).
type GPUStats struct {
	GPUTensors, CPUTensors int
	GPUBytes, CPUBytes     uint64
	PerDevice              []uint64
}

// EnableGPU needs the cuda build tag: go test -tags cuda ./pkg/densecpu (after cuda/build.sh).
func (*weights) EnableGPU() (GPUStats, error) {
	return GPUStats{}, errors.New("densecpu: built without the cuda tag")
}

// gpuMatmul is the CUDA backend's handle; without the cuda build tag it is empty and every matmul stays on the CPU.
type gpuMatmul struct{}

func (*gpuMatmul) free() {}

func (*gpuMatmul) matmul(string, *gguf.TensorInfo, [][]float32) ([][]float32, bool) {
	return nil, false
}

func (*gpuMatmul) attn(string, [][]float32, *kvCache, int, int, int, int, int, int, float32) ([][]float32, bool) {
	return nil, false
}

func (*gpuMatmul) gdn(string, *gdnState, [][]float32, [][]float32, [][]float32, int, int, int, float32) ([][]float32, bool) {
	return nil, false
}

// devKV and devS are the CUDA backend's resident cache and state; without the tag there are none.
type devKV struct{}
type devS struct{}

func (*gdnState) fetch() {}

func (*gpuMatmul) ffn(string, [][]float32, bool) ([][]float32, bool) { return nil, false }

func (*gpuMatmul) multi([]string, [][]float32) ([][][]float32, bool) { return nil, false }

func (*devKV) release() {}
func (*devS) release()  {}
