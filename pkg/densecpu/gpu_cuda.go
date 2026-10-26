//go:build cuda

package densecpu

/*
#cgo CFLAGS: -I${SRCDIR}/cuda
#cgo LDFLAGS: -L${SRCDIR}/cuda -ldensegpu -Wl,-rpath,${SRCDIR}/cuda
#include "densegpu.h"
*/
import "C"

import (
	"fmt"
	"sync/atomic"
	"time"
	"unsafe"

	"strata/pkg/gguf"
)

// GPUStats says where the weights ended up.
type GPUStats struct {
	GPUTensors, CPUTensors int
	GPUBytes, CPUBytes     uint64
	PerDevice              []uint64 // bytes on each CUDA device
}

type gpuTensor struct {
	dev      int
	ptr      unsafe.Pointer
	rowBytes int
}

// gpuMatmul holds the uploaded weight matrices; matmul runs a tensor on its device or reports it is not resident.
type gpuMatmul struct{ res map[string]gpuTensor }

// GPUCallNanos and GPUCalls count the time and number of matmul calls made on the GPU (copy, kernel, copy back).
var GPUCallNanos, GPUCalls atomic.Int64

const gpuHeadroom = 2 << 30 // per card: CUDA context, activation scratch, the display

// EnableGPU copies every supported 2-D weight matrix to the card with the most free memory (largest first is not
// needed: the model is placed in file order). What does not fit stays on the CPU path.
func (m *weights) EnableGPU() (GPUStats, error) {
	var st GPUStats
	n := int(C.dg_device_count())
	if n == 0 {
		return st, fmt.Errorf("densecpu: no CUDA device")
	}
	free := make([]uint64, n)
	st.PerDevice = make([]uint64, n)
	for d := 0; d < n; d++ {
		var f, t C.size_t
		if C.dg_mem(C.int(d), &f, &t) != 0 {
			return st, fmt.Errorf("densecpu: cannot query CUDA device %d", d)
		}
		if uint64(f) > gpuHeadroom {
			free[d] = uint64(f) - gpuHeadroom
		}
	}
	g := &gpuMatmul{res: map[string]gpuTensor{}}
	for _, t := range m.g.Tensors {
		elems, bytes, err := blockInfo(t.Type)
		if err != nil || len(t.Dims) != 2 || fusedDot(t.Type) == nil || int(t.Dims[0])%elems != 0 {
			continue
		}
		rowBytes := int(t.Dims[0]) / elems * bytes
		size := uint64(rowBytes) * t.Dims[1]
		best := 0
		for d := range free {
			if free[d] > free[best] {
				best = d
			}
		}
		if free[best] < size {
			st.CPUTensors++
			st.CPUBytes += size
			continue
		}
		off := m.g.DataStart + int64(t.Offset)
		p := C.dg_upload(C.int(best), unsafe.Pointer(&m.mm[off]), C.size_t(size))
		if p == nil {
			return st, fmt.Errorf("densecpu: upload of %s (%d bytes) to device %d failed", t.Name, size, best)
		}
		g.res[t.Name] = gpuTensor{dev: best, ptr: p, rowBytes: rowBytes}
		free[best] -= size
		st.GPUTensors++
		st.GPUBytes += size
		st.PerDevice[best] += size
	}
	m.gpu = g
	return st, nil
}

// free releases every uploaded matrix.
func (g *gpuMatmul) free() {
	for _, r := range g.res {
		C.dg_free(C.int(r.dev), r.ptr)
	}
	g.res = nil
}

// rawMatvec uploads a raw quantized matrix (rows x cols, type typ) to device dev and multiplies it by X; the
// kernel parity test uses it, the model path does not.
func rawMatvec(dev int, typ uint32, w []byte, rows, cols int, X [][]float32) ([][]float32, error) {
	elems, bytes, err := blockInfo(typ)
	if err != nil {
		return nil, err
	}
	rowBytes := cols / elems * bytes
	if len(w) < rows*rowBytes {
		return nil, fmt.Errorf("densecpu: %d bytes is too few for %d rows", len(w), rows)
	}
	p := C.dg_upload(C.int(dev), unsafe.Pointer(&w[0]), C.size_t(rows*rowBytes))
	if p == nil {
		return nil, fmt.Errorf("densecpu: upload failed")
	}
	t := &gguf.TensorInfo{Name: "raw", Dims: []uint64{uint64(cols), uint64(rows)}, Type: typ}
	g := &gpuMatmul{res: map[string]gpuTensor{"raw": {dev: dev, ptr: p, rowBytes: rowBytes}}}
	Y, _ := g.matmul("raw", t, X)
	return Y, nil
}

func (g *gpuMatmul) matmul(name string, t *gguf.TensorInfo, X [][]float32) ([][]float32, bool) {
	r, ok := g.res[name]
	if !ok {
		return nil, false
	}
	cols, rows, n := int(t.Dims[0]), int(t.Dims[1]), len(X)
	x := make([]float32, 0, n*cols)
	for _, v := range X {
		x = append(x, v...)
	}
	y := make([]float32, n*rows)
	defer func(t0 time.Time) { GPUCallNanos.Add(int64(time.Since(t0))); GPUCalls.Add(1) }(time.Now())
	if rc := C.dg_matvec(C.int(r.dev), C.int(t.Type), r.ptr, C.size_t(r.rowBytes), C.int(rows), C.int(cols),
		(*C.float)(unsafe.Pointer(&x[0])), C.int(n), (*C.float)(unsafe.Pointer(&y[0]))); rc != 0 {
		panic(fmt.Sprintf("densecpu: CUDA matvec of %s failed with code %d", name, rc))
	}
	Y := make([][]float32, n)
	for i := range Y {
		Y[i] = y[i*rows : (i+1)*rows]
	}
	return Y, true
}
