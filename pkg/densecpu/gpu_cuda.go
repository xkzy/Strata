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
	"runtime"
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

// devKV is the device copy of one attention layer's K and V (half bits, one row per position, rows laid out as on the
// host). It only ever grows by appended rows, so each call uploads just the new ones.
type devKV struct {
	dev     int
	k, v    unsafe.Pointer
	capRows int
	rows    int // rows already on the card
}

// devS is the device copy of one delta-net block's recurrent state; while it exists, the host copy is stale.
type devS struct {
	dev int
	ptr unsafe.Pointer
}

func flatHalf(rows [][]uint16) []uint16 {
	f := make([]uint16, 0, len(rows)*len(rows[0]))
	for _, r := range rows {
		f = append(f, r...)
	}
	return f
}

// resident brings the device copy of cache c up to date and returns it, or nil when the card has no room (the caller
// then uploads the cache for this call only, or uses the CPU).
func (g *gpuMatmul) resident(c *kvCache, dev, rowElems int) *devKV {
	rowBytes, need := rowElems*2, len(c.K)
	d := c.dev
	if d == nil {
		d = &devKV{dev: dev}
		runtime.SetFinalizer(d, func(d *devKV) { C.dg_free(C.int(d.dev), d.k); C.dg_free(C.int(d.dev), d.v) })
		c.dev = d
	}
	if need > d.capRows {
		nc := max(2*need, 1024)
		k, v := C.dg_alloc(C.int(dev), C.size_t(nc*rowBytes)), C.dg_alloc(C.int(dev), C.size_t(nc*rowBytes))
		if k == nil || v == nil {
			C.dg_free(C.int(dev), k)
			C.dg_free(C.int(dev), v)
			return nil
		}
		if d.rows > 0 {
			C.dg_d2d(C.int(dev), k, d.k, C.size_t(d.rows*rowBytes))
			C.dg_d2d(C.int(dev), v, d.v, C.size_t(d.rows*rowBytes))
		}
		C.dg_free(C.int(dev), d.k)
		C.dg_free(C.int(dev), d.v)
		d.k, d.v, d.capRows = k, v, nc
	}
	if need > d.rows {
		hk, hv := flatHalf(c.K[d.rows:need]), flatHalf(c.V[d.rows:need])
		off, n := C.size_t(d.rows*rowBytes), C.size_t((need-d.rows)*rowBytes)
		if C.dg_h2d(C.int(dev), d.k, off, unsafe.Pointer(&hk[0]), n) != 0 || C.dg_h2d(C.int(dev), d.v, off, unsafe.Pointer(&hv[0]), n) != 0 {
			return nil
		}
		d.rows = need
	}
	return d
}

// attn runs causal attention for the n rows of Q (cache rows first..; window > 0 limits each token to the last window
// rows) on the card holding layer p's weights. Caches without a window stay resident on the card; a windowed cache is
// uploaded for the call (so only for batches of 4 or more). False means the caller must use the CPU path.
func (g *gpuMatmul) attn(p string, Q [][]float32, c *kvCache, first, window, nHead, nKV, hd int, scale float32) ([][]float32, bool) {
	n, dev, rowElems := len(Q), g.res[p+"attn_q.weight"].dev, nKV*hd
	var dk, dv unsafe.Pointer
	if d := (*devKV)(nil); window == 0 {
		if d = g.resident(c, dev, rowElems); d != nil {
			dk, dv = d.k, d.v
		}
	}
	if dk == nil {
		if n < 4 {
			return nil, false
		}
		hk, hv := flatHalf(c.K), flatHalf(c.V)
		dk, dv = C.dg_alloc(C.int(dev), C.size_t(len(hk)*2)), C.dg_alloc(C.int(dev), C.size_t(len(hv)*2))
		defer C.dg_free(C.int(dev), dk)
		defer C.dg_free(C.int(dev), dv)
		if dk == nil || dv == nil || C.dg_h2d(C.int(dev), dk, 0, unsafe.Pointer(&hk[0]), C.size_t(len(hk)*2)) != 0 ||
			C.dg_h2d(C.int(dev), dv, 0, unsafe.Pointer(&hv[0]), C.size_t(len(hv)*2)) != 0 {
			return nil, false
		}
	}
	q := make([]float32, 0, n*nHead*hd)
	for _, r := range Q {
		q = append(q, r...)
	}
	out := make([]float32, n*nHead*hd)
	defer func(t0 time.Time) { GPUCallNanos.Add(int64(time.Since(t0))); GPUCalls.Add(1) }(time.Now())
	if C.dg_attn_dev(C.int(dev), C.int(n), C.int(first), C.int(window), C.int(nHead), C.int(nKV), C.int(hd), C.float(scale),
		(*C.float)(&q[0]), dk, dv, (*C.float)(&out[0])) != 0 {
		return nil, false
	}
	R := make([][]float32, n)
	for i := range R {
		R[i] = out[i*nHead*hd : (i+1)*nHead*hd]
	}
	return R, true
}

// gdn runs the delta-net recurrence for all n tokens on the card holding layer p's weights. conv holds the normalised
// q/k/v rows, decay and beta are [n][numV]; the state stays on the card between calls (st.dev) and the output is
// [n][numV*hv]. False means the caller must use the CPU loop (after st.fetch).
func (g *gpuMatmul) gdn(p string, st *gdnState, conv, decay, beta [][]float32, numK, hk, hv int, qs float32) ([][]float32, bool) {
	n, numV, dev := len(conv), len(st.S), g.res[p+"attn_qkv.weight"].dev
	if st.dev == nil {
		bytes := numV * hk * hv * 4
		ptr := C.dg_alloc(C.int(dev), C.size_t(bytes))
		if ptr == nil {
			return nil, false
		}
		flat := make([]float32, 0, numV*hk*hv)
		for _, s := range st.S {
			flat = append(flat, s...)
		}
		if C.dg_h2d(C.int(dev), ptr, 0, unsafe.Pointer(&flat[0]), C.size_t(bytes)) != 0 {
			C.dg_free(C.int(dev), ptr)
			return nil, false
		}
		st.dev = &devS{dev: dev, ptr: ptr}
		runtime.SetFinalizer(st.dev, func(d *devS) { C.dg_free(C.int(d.dev), d.ptr) })
	}
	flatten := func(rows [][]float32) []float32 {
		f := make([]float32, 0, len(rows)*len(rows[0]))
		for _, r := range rows {
			f = append(f, r...)
		}
		return f
	}
	out := make([]float32, n*numV*hv)
	c, d, b := flatten(conv), flatten(decay), flatten(beta)
	defer func(t0 time.Time) { GPUCallNanos.Add(int64(time.Since(t0))); GPUCalls.Add(1) }(time.Now())
	if C.dg_gdn_dev(C.int(dev), C.int(n), C.int(numK), C.int(numV), C.int(hk), C.int(hv), C.float(qs), (*C.float)(&c[0]),
		(*C.float)(&d[0]), (*C.float)(&b[0]), (*C.float)(st.dev.ptr), (*C.float)(&out[0])) != 0 {
		return nil, false
	}
	R := make([][]float32, n)
	for t := range R {
		R[t] = out[t*numV*hv : (t+1)*numV*hv]
	}
	return R, true
}

// fetch makes the host copy of the state current again and drops the device copy, so the CPU loop can continue it.
func (s *gdnState) fetch() {
	d := s.dev
	if d == nil {
		return
	}
	hk := len(s.S[0])
	flat := make([]float32, len(s.S)*hk)
	if C.dg_d2h(C.int(d.dev), unsafe.Pointer(&flat[0]), d.ptr, C.size_t(len(flat)*4)) != 0 {
		panic("densecpu: cannot read the delta-net state back from the card")
	}
	for h := range s.S {
		copy(s.S[h], flat[h*hk:(h+1)*hk])
	}
	C.dg_free(C.int(d.dev), d.ptr)
	d.ptr = nil
	s.dev = nil
}
