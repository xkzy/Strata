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
	"strings"
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
	typ      uint32
	rows     int
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
	// A layer's tensors stay together on one card (cards are filled in turn, not tensor by tensor), so a block's
	// activations never have to cross the bus between its own matrices.
	layerKey := func(name string) string {
		if strings.HasPrefix(name, "blk.") {
			if i := strings.Index(name[4:], "."); i >= 0 {
				return name[:4+i]
			}
		}
		return name
	}
	groupSize := map[string]uint64{}
	for _, t := range m.g.Tensors {
		if elems, bytes, err := blockInfo(t.Type); err == nil && len(t.Dims) == 2 && fusedDot(t.Type) != nil && int(t.Dims[0])%elems == 0 {
			groupSize[layerKey(t.Name)] += uint64(int(t.Dims[0])/elems*bytes) * t.Dims[1]
		}
	}
	mostFree := func() int {
		best := 0
		for d := range free {
			if free[d] > free[best] {
				best = d
			}
		}
		return best
	}
	cur, lastKey := mostFree(), ""
	for _, t := range m.g.Tensors {
		elems, bytes, err := blockInfo(t.Type)
		if err != nil || len(t.Dims) != 2 || fusedDot(t.Type) == nil || int(t.Dims[0])%elems != 0 {
			continue
		}
		rowBytes := int(t.Dims[0]) / elems * bytes
		size := uint64(rowBytes) * t.Dims[1]
		if k := layerKey(t.Name); k != lastKey {
			lastKey = k
			if free[cur] < groupSize[k] {
				cur = mostFree()
			}
		}
		best := cur
		if free[best] < size {
			best = mostFree()
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
		g.res[t.Name] = gpuTensor{dev: best, ptr: p, rowBytes: rowBytes, typ: t.Type, rows: int(t.Dims[1])}
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

// attn runs causal attention for the n rows of Q (cache rows first.. of c; window > 0 limits each token to the last
// window rows) on device 0; false means the caller must use the CPU path. K and V are copied to the card on every call.
func (g *gpuMatmul) attn(Q [][]float32, c *kvCache, first, window, nHead, nKV, hd int, scale float32) ([][]float32, bool) {
	n := len(Q)
	q := make([]float32, 0, n*nHead*hd)
	for _, r := range Q {
		q = append(q, r...)
	}
	k, v := make([]uint16, 0, len(c.K)*nKV*hd), make([]uint16, 0, len(c.V)*nKV*hd)
	for j := range c.K {
		k, v = append(k, c.K[j]...), append(v, c.V[j]...)
	}
	out := make([]float32, n*nHead*hd)
	defer func(t0 time.Time) { GPUCallNanos.Add(int64(time.Since(t0))); GPUCalls.Add(1) }(time.Now())
	if C.dg_attn(0, C.int(n), C.int(first), C.int(window), C.int(nHead), C.int(nKV), C.int(hd), C.float(scale), (*C.float)(&q[0]),
		(*C.ushort)(&k[0]), (*C.ushort)(&v[0]), (*C.float)(&out[0])) != 0 {
		return nil, false
	}
	R := make([][]float32, n)
	for i := range R {
		R[i] = out[i*nHead*hd : (i+1)*nHead*hd]
	}
	return R, true
}

// gdn runs the delta-net recurrence for all n tokens on the GPU. conv holds the normalised q/k/v rows, decay and
// beta are [n][numV]; the state S (one slice per value head) is advanced in place and the output is [n][numV*hv].
// False means the caller must use the CPU loop (S is untouched).
func (g *gpuMatmul) gdn(conv, decay, beta [][]float32, S [][]float32, numK, hk, hv int, qs float32) ([][]float32, bool) {
	n, numV := len(conv), len(S)
	flat := func(rows [][]float32) []float32 {
		f := make([]float32, 0, len(rows)*len(rows[0]))
		for _, r := range rows {
			f = append(f, r...)
		}
		return f
	}
	s, out := flat(S), make([]float32, n*numV*hv)
	c, d, b := flat(conv), flat(decay), flat(beta)
	defer func(t0 time.Time) { GPUCallNanos.Add(int64(time.Since(t0))); GPUCalls.Add(1) }(time.Now())
	if C.dg_gdn(0, C.int(n), C.int(numK), C.int(numV), C.int(hk), C.int(hv), C.float(qs), (*C.float)(&c[0]),
		(*C.float)(&d[0]), (*C.float)(&b[0]), (*C.float)(&s[0]), (*C.float)(&out[0])) != 0 {
		return nil, false
	}
	for h := range S {
		copy(S[h], s[h*hk*hv:(h+1)*hk*hv])
	}
	R := make([][]float32, n)
	for t := range R {
		R[t] = out[t*numV*hv : (t+1)*numV*hv]
	}
	return R, true
}

// MatvecStats reports where dg_matvec spent its time since the last reset: seconds copying activations in, running the
// kernels and copying results out, plus bytes moved each way and the number of calls.
func MatvecStats(reset bool) (in, kern, out, bytesIn, bytesOut float64, calls int64) {
	var t [9]C.double
	var b [6]C.double
	var n [3]C.long
	C.dg_stats(&t[0], &b[0], &n[0])
	if reset {
		C.dg_stats_reset()
	}
	return float64(t[0]), float64(t[1]), float64(t[2]), float64(b[0]), float64(b[1]), int64(n[0])
}

// ffn runs down(act(gate x) * up x) for the layer prefix p on one card; gelu picks the Gemma activation. False means
// one of the three weights is not resident on the same card, and the caller does the three matmuls itself.
func (g *gpuMatmul) ffn(p string, X [][]float32, gelu bool) ([][]float32, bool) {
	gt, ut, dt := g.res[p+"ffn_gate.weight"], g.res[p+"ffn_up.weight"], g.res[p+"ffn_down.weight"]
	if gt.ptr == nil || ut.ptr == nil || dt.ptr == nil || gt.dev != ut.dev || gt.dev != dt.dev {
		return nil, false
	}
	n, nEmbd := len(X), len(X[0])
	nFF := gt.rows
	x := make([]float32, 0, n*nEmbd)
	for _, v := range X {
		x = append(x, v...)
	}
	y := make([]float32, n*nEmbd)
	act := C.int(0)
	if gelu {
		act = 1
	}
	defer func(t0 time.Time) { GPUCallNanos.Add(int64(time.Since(t0))); GPUCalls.Add(1) }(time.Now())
	if C.dg_ffn(C.int(gt.dev), act, C.int(gt.typ), gt.ptr, C.size_t(gt.rowBytes), C.int(ut.typ), ut.ptr, C.size_t(ut.rowBytes),
		C.int(dt.typ), dt.ptr, C.size_t(dt.rowBytes), C.int(nFF), C.int(nEmbd), (*C.float)(&x[0]), C.int(n), (*C.float)(&y[0])) != 0 {
		return nil, false
	}
	Y := make([][]float32, n)
	for i := range Y {
		Y[i] = y[i*nEmbd : (i+1)*nEmbd]
	}
	return Y, true
}

// multi applies the named matrices to X with one upload of X. False means they are not all resident on one card (or
// not all supported), and the caller multiplies them one by one.
func (g *gpuMatmul) multi(names []string, X [][]float32) ([][][]float32, bool) {
	cnt := len(names)
	if cnt == 0 || cnt > 8 || len(X) == 0 {
		return nil, false
	}
	ts := make([]gpuTensor, cnt)
	types, rbs, rows := make([]C.int, cnt), make([]C.size_t, cnt), make([]C.int, cnt)
	ws := make([]unsafe.Pointer, cnt)
	total := 0
	for k, nm := range names {
		ts[k] = g.res[nm]
		if ts[k].ptr == nil || ts[k].dev != ts[0].dev {
			return nil, false
		}
		types[k], rbs[k], rows[k], ws[k] = C.int(ts[k].typ), C.size_t(ts[k].rowBytes), C.int(ts[k].rows), ts[k].ptr
		total += ts[k].rows
	}
	n, cols := len(X), len(X[0])
	x := make([]float32, 0, n*cols)
	for _, v := range X {
		x = append(x, v...)
	}
	y := make([]float32, n*total)
	defer func(t0 time.Time) { GPUCallNanos.Add(int64(time.Since(t0))); GPUCalls.Add(1) }(time.Now())
	if C.dg_multi(C.int(ts[0].dev), C.int(cnt), &types[0], (*unsafe.Pointer)(unsafe.Pointer(&ws[0])), &rbs[0], &rows[0],
		C.int(cols), (*C.float)(&x[0]), C.int(n), (*C.float)(&y[0])) != 0 {
		return nil, false
	}
	out := make([][][]float32, cnt)
	off := 0
	for k := range out {
		r := ts[k].rows
		out[k] = make([][]float32, n)
		for i := range out[k] {
			out[k][i] = y[off+i*r : off+(i+1)*r]
		}
		off += n * r
	}
	return out, true
}
