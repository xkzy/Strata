//go:build cuda

package densecpu

/*
#cgo CFLAGS: -I${SRCDIR}/cuda
#include <stdlib.h>
#include "densegpu.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

// mtpGPU is the model's multi-token-prediction block on the last card: position i reads the final hidden state of position i and the
// embedding of the token at i+1 and predicts the token at i+2. It has its own attention cache.
type mtpGPU struct {
	seg qseg
	pos int // positions already in its cache
}

const mtpPrefix = "blk.%d."

// EnableMTP uploads the MTP block (needs EnableResident first). False when the file has none.
func (g *QwenGPU) EnableMTP() (ok bool, err error) {
	defer func() {
		if r := recover(); r != nil {
			err = fmt.Errorf("%v", r)
		}
	}()
	m := g.m
	p := fmt.Sprintf(mtpPrefix, m.nLayer)
	if t, _ := m.raw(p + "nextn.eh_proj.weight"); t == nil {
		return false, nil
	}
	dev := g.segs[len(g.segs)-1].dev
	arr := (*C.dg_layer)(C.calloc(1, C.size_t(unsafe.Sizeof(C.dg_layer{}))))
	g.layer(arr, dev, m.nLayer, int(g.cfg.cap))
	g.mtp = &mtpGPU{seg: qseg{dev: dev, layers: arr, n: 1}}
	name := p + "nextn.eh_proj.weight"
	mt := g.mat(dev, name)
	m.gpu.res[name] = gpuTensor{dev: dev, ptr: mt.w, rowBytes: int(mt.rb)}
	return true, nil
}

// mtpRun appends positions pos.. to the MTP cache. rows are the post-final-norm hidden states of those positions of the main model and
// toks the tokens that follow each of them; it returns the logits (for the token after toks[last]) of the last row.
func (g *QwenGPU) mtpRun(rows [][]float32, toks []int) []float32 {
	m := g.m
	p := fmt.Sprintf(mtpPrefix, m.nLayer)
	enorm, hnorm := m.vec(p+"nextn.enorm.weight"), m.vec(p+"nextn.hnorm.weight")
	var last []float32
	for lo := 0; lo < len(rows); lo += g.chunk {
		hi := min(lo+g.chunk, len(rows))
		in := make([][]float32, hi-lo)
		for i := range in {
			in[i] = append(rms(g.embed(toks[lo+i]), enorm, m.eps), rms(rows[lo+i], hnorm, m.eps)...)
		}
		x := m.matmul(p+"nextn.eh_proj.weight", in)
		n := hi - lo
		flat := make([]float32, 0, n*len(x[0]))
		for _, r := range x {
			flat = append(flat, r...)
		}
		s := g.mtp.seg
		if rc := C.dg_qwen_run(C.int(s.dev), &g.cfg, s.layers, 1, (*C.float)(unsafe.Pointer(&flat[0])), C.int(n), C.int(g.mtp.pos), 0); rc != 0 {
			panic(fmt.Sprintf("densecpu: MTP block failed with code %d", rc))
		}
		g.mtp.pos += n
		last = flat[(n-1)*len(x[0]):]
	}
	return m.matmul("output.weight", [][]float32{rms(last, m.vec(p+"nextn.shared_head_norm.weight"), m.eps)})[0]
}

// SpecStats counts a Generate run.
type SpecStats struct {
	Cycles, Accepted int // verify calls, drafts that were right
}

// Generate decodes greedily after prompt, up to maxNew tokens or until eos (< 0: none). With spec the MTP block drafts the token after the
// next one and a 2-token call verifies it: a right draft gives 2 tokens for one pass over the weights.
func (g *QwenGPU) Generate(prompt []int, maxNew, eos int, spec bool) ([]int, SpecStats) {
	var st SpecStats
	g.Reset()
	if g.mtp != nil {
		g.mtp.pos = 0
	}
	rows := g.normed(g.forward(prompt, false))
	next := argmax(g.m.matmul("output.weight", rows[len(rows)-1:])[0])
	out := []int{next}
	if !spec || g.mtp == nil {
		for len(out) < maxNew && next != eos {
			next = argmax(g.Step([]int{next}))
			out = append(out, next)
		}
		return out, st
	}
	draft := argmax(g.mtpRun(rows, append(append([]int(nil), prompt[1:]...), next)))
	for len(out) < maxNew && out[len(out)-1] != eos {
		p := g.pos // position of the pending token
		hs := g.normed(g.forward([]int{next, draft}, true))
		lg := g.m.matmul("output.weight", hs)
		st.Cycles++
		y0 := argmax(lg[0])
		out = append(out, y0)
		if y0 == draft {
			st.Accepted++
			y1 := argmax(lg[1])
			out = append(out, y1)
			draft = argmax(g.mtpRun(hs, []int{y0, y1}))
			next = y1
			continue
		}
		g.rollback(p + 1)
		g.mtp.pos = p
		draft = argmax(g.mtpRun(hs[:1], []int{y0}))
		next = y0
	}
	return out[:min(len(out), maxNew)], st
}
