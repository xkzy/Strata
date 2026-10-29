//go:build cuda

package densecpu

/*
#cgo CFLAGS: -I${SRCDIR}/cuda
#cgo LDFLAGS: -L${SRCDIR}/cuda -ldensegpu -Wl,-rpath,${SRCDIR}/cuda
#include <stdlib.h>
#include "densegpu.h"
*/
import "C"

import (
	"fmt"
	"runtime/debug"
	"strconv"
	"unsafe"
)

// QwenGPU is Qwen3.8 dense with the whole forward pass on the GPUs: layers are split over the cards in proportion to their
// memory bandwidth and only the hidden state crosses the bus. One session at a time (Reset starts a new one).
type QwenGPU struct {
	m      *Qwen
	cfg    C.dg_cfg
	segs   []qseg
	pos    int
	devMem map[unsafe.Pointer]int // device pointer -> device, for freeing
	state  []stateBuf
	chunk  int
}

type qseg struct {
	dev    int
	layers *C.dg_layer // C-allocated array
	n      int
}

type stateBuf struct {
	dev   int
	p     unsafe.Pointer
	bytes int
}

func (g *QwenGPU) up(dev int, data []float32) *C.float {
	p := C.dg_upload(C.int(dev), unsafe.Pointer(&data[0]), C.size_t(len(data)*4))
	if p == nil {
		panic("densecpu: upload failed")
	}
	g.devMem[p] = dev
	return (*C.float)(p)
}

func (g *QwenGPU) mat(dev int, name string) C.dg_mat {
	t, b := g.m.raw(name)
	if t == nil || len(t.Dims) != 2 {
		panic("densecpu: missing tensor " + name)
	}
	elems, bytes, err := blockInfo(t.Type)
	if err != nil {
		panic(err)
	}
	cols, rows := int(t.Dims[0]), int(t.Dims[1])
	rb := cols / elems * bytes
	p := C.dg_upload(C.int(dev), unsafe.Pointer(&b[0]), C.size_t(rb*rows))
	if p == nil {
		panic("densecpu: upload of " + name + " failed")
	}
	g.devMem[p] = dev
	return C.dg_mat{_type: C.int(t.Type), w: p, rb: C.size_t(rb), rows: C.int(rows), cols: C.int(cols)}
}

func (g *QwenGPU) zero(dev, bytes int) unsafe.Pointer {
	p := C.dg_alloc_zero(C.int(dev), C.size_t(bytes))
	if p == nil {
		panic("densecpu: device allocation failed")
	}
	g.devMem[p] = dev
	g.state = append(g.state, stateBuf{dev, p, bytes})
	return p
}

// EnableResident uploads the model and allocates the state for contexts up to maxCtx tokens.
func (m *Qwen) EnableResident(maxCtx int) (g *QwenGPU, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("%v\n%s", r, debug.Stack())
		}
	}()
	nd := int(C.dg_device_count())
	if nd == 0 {
		return nil, fmt.Errorf("densecpu: no CUDA device")
	}
	g = &QwenGPU{m: m, devMem: map[unsafe.Pointer]int{}, chunk: 512}
	qt, _ := m.raw("blk.3.attn_q.weight")
	kt, _ := m.raw("blk.3.attn_k.weight")
	nq, _ := m.raw("blk.3.attn_q_norm.weight")
	hd := int(nq.Dims[0])
	ffn, _ := m.raw("blk.0.ffn_gate.weight")
	g.cfg = C.dg_cfg{n_embd: C.int(len(m.vec("output_norm.weight"))), n_head: C.int(int(qt.Dims[1]) / (2 * hd)), n_kv: C.int(int(kt.Dims[1]) / hd),
		hd: C.int(hd), n_rot: C.int(m.nRot), hk: C.int(m.headK), nk: C.int(m.numK), nv: C.int(m.numV), conv_k: C.int(m.convK),
		d_inner: C.int(m.dInner), n_ff: C.int(ffn.Dims[1]), cap: C.int(maxCtx), eps: C.float(m.eps), base: C.float(m.base)}

	// bytes per layer, and which device takes which layers: consecutive, in proportion to measured bandwidth, the output matrix
	// counted on the last device
	size := func(name string) uint64 {
		t, _ := m.raw(name)
		if t == nil {
			return 0
		}
		elems, bytes, _ := blockInfo(t.Type)
		rows := uint64(1)
		for _, d := range t.Dims[1:] {
			rows *= d
		}
		return uint64(int(t.Dims[0])/elems*bytes) * rows
	}
	lb := make([]uint64, m.nLayer)
	var total uint64
	for l := range lb {
		for _, t := range m.g.Tensors {
			if len(t.Name) > 4 && t.Name[:4] == "blk." {
				var idx int
				fmt.Sscanf(t.Name, "blk.%d.", &idx)
				if idx == l {
					lb[l] += size(t.Name)
				}
			}
		}
		total += lb[l]
	}
	outB := size("output.weight")
	total += outB
	bw := make([]float64, nd)
	var bwSum float64
	for d := range bw {
		bw[d] = float64(C.dg_bandwidth(C.int(d)))
		bwSum += bw[d]
	}
	devOf := make([]int, m.nLayer)
	d, acc, cum := 0, uint64(0), bw[0]
	for l := range lb {
		for d < nd-1 && float64(acc+lb[l]/2) > float64(total)*cum/bwSum {
			d++
			cum += bw[d]
		}
		devOf[l] = d
		acc += lb[l]
	}

	for l := 0; l < m.nLayer; {
		e := l
		for e < m.nLayer && devOf[e] == devOf[l] {
			e++
		}
		arr := (*C.dg_layer)(C.calloc(C.size_t(e-l), C.size_t(unsafe.Sizeof(C.dg_layer{}))))
		ls := unsafe.Slice(arr, e-l)
		for i := l; i < e; i++ {
			g.layer(&ls[i-l], devOf[l], i, maxCtx)
		}
		g.segs = append(g.segs, qseg{dev: devOf[l], layers: arr, n: e - l})
		l = e
	}
	last := g.segs[len(g.segs)-1].dev
	t, b := m.raw("output.weight")
	elems, bytes, _ := blockInfo(t.Type)
	rb := int(t.Dims[0]) / elems * bytes
	p := C.dg_upload(C.int(last), unsafe.Pointer(&b[0]), C.size_t(rb)*C.size_t(t.Dims[1]))
	if p == nil {
		return nil, fmt.Errorf("densecpu: upload of output.weight failed")
	}
	g.devMem[p] = last
	m.gpu = &gpuMatmul{res: map[string]gpuTensor{"output.weight": {dev: last, ptr: p, rowBytes: rb}}}
	g.Reset()
	return g, nil
}

func (g *QwenGPU) layer(L *C.dg_layer, dev, i, maxCtx int) {
	m := g.m
	p := "blk." + strconv.Itoa(i) + "."
	L.attn_norm, L.post_norm = g.up(dev, m.vec(p+"attn_norm.weight")), g.up(dev, m.vec(p+"post_attention_norm.weight"))
	L.ffn_gate, L.ffn_up, L.ffn_down = g.mat(dev, p+"ffn_gate.weight"), g.mat(dev, p+"ffn_up.weight"), g.mat(dev, p+"ffn_down.weight")
	if t, _ := m.raw(p + "ssm_out.weight"); t != nil {
		for k, n := range []string{"attn_qkv", "attn_gate", "ssm_beta", "ssm_alpha", "ssm_out"} {
			L.m[k] = g.mat(dev, p+n+".weight")
		}
		L.f[0], L.f[1] = g.up(dev, m.vec(p+"ssm_dt.bias")), g.up(dev, m.vec(p+"ssm_a"))
		L.f[2], L.f[3] = g.up(dev, m.vec(p+"ssm_conv1d.weight")), g.up(dev, m.vec(p+"ssm_norm.weight"))
		L.S = (*C.float)(g.zero(dev, m.numV*m.headK*(m.dInner/m.numV)*4))
		L.tail = (*C.float)(g.zero(dev, (m.convK-1)*(2*m.numK*m.headK+m.dInner)*4))
		return
	}
	L.attn = 1
	for k, n := range []string{"attn_q", "attn_k", "attn_v", "attn_output"} {
		L.m[k] = g.mat(dev, p+n+".weight")
	}
	L.f[0], L.f[1] = g.up(dev, m.vec(p+"attn_q_norm.weight")), g.up(dev, m.vec(p+"attn_k_norm.weight"))
	kvBytes := maxCtx * int(g.cfg.n_kv) * int(g.cfg.hd) * 2
	L.Kc, L.Vc = g.zero(dev, kvBytes), g.zero(dev, kvBytes)
}

// Reset starts a new session.
func (g *QwenGPU) Reset() {
	for _, s := range g.state {
		C.dg_zero(C.int(s.dev), s.p, C.size_t(s.bytes))
	}
	g.pos = 0
}

// Close frees the device memory.
func (g *QwenGPU) Close() {
	for p, d := range g.devMem {
		C.dg_free(C.int(d), p)
	}
	for _, s := range g.segs {
		C.free(unsafe.Pointer(s.layers))
	}
	g.m.gpu.res = nil
	g.m.gpu = nil
}

// Step feeds the tokens to the session and returns the logits of the last one.
func (g *QwenGPU) Step(tokens []int) []float32 {
	m := g.m
	emb, eb := m.raw("token_embd.weight")
	ec := int(emb.Dims[0])
	elems, bytes, _ := blockInfo(emb.Type)
	rowBytes := ec / elems * bytes
	var last []float32
	for lo := 0; lo < len(tokens); lo += g.chunk {
		hi := min(lo+g.chunk, len(tokens))
		n := hi - lo
		h := make([]float32, n*ec)
		for i, tk := range tokens[lo:hi] {
			if err := DequantRow(emb.Type, eb[tk*rowBytes:], ec, h[i*ec:(i+1)*ec]); err != nil {
				panic(err)
			}
		}
		for _, s := range g.segs {
			if rc := C.dg_qwen_run(C.int(s.dev), &g.cfg, s.layers, C.int(s.n), (*C.float)(unsafe.Pointer(&h[0])), C.int(n), C.int(g.pos)); rc != 0 {
				panic(fmt.Sprintf("densecpu: dg_qwen_run on device %d failed with code %d", s.dev, rc))
			}
		}
		g.pos += n
		last = h[(n-1)*ec:]
	}
	x := rms(last, m.vec("output_norm.weight"), m.eps)
	return m.matmul("output.weight", [][]float32{x})[0]
}

// ProfDump prints (stderr) and clears the per-section GPU times collected when DG_PROF is set.
func ProfDump() { C.dg_prof_dump() }
