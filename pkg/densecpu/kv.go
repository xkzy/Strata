package densecpu

import (
	"math"
	"runtime"
	"sync"
)

// kvCache holds the post-rope keys and the values of one attention layer, one row per position, as IEEE half floats
// (half the memory of float32; the rounding error is about 0.05% per value).
type kvCache struct {
	K, V [][]uint16
	dev  *devKV // the same rows on the card (cuda build); only unwindowed caches
}

var halfTab = func() (t [65536]float32) {
	for i := range t {
		t[i] = f16([]byte{byte(i), byte(i >> 8)})
	}
	return
}()

// toHalf rounds to nearest even; values beyond the half range saturate to infinity.
func toHalf(x float32) uint16 {
	b := math.Float32bits(x)
	sign, e, man := uint16(b>>16)&0x8000, int32(b>>23&0xff)-127+15, b&0x7fffff
	switch {
	case b>>23&0xff == 0xff: // inf / nan
		if man != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	case e >= 31:
		return sign | 0x7c00
	case e <= 0:
		if e < -10 {
			return sign
		}
		man |= 0x800000
		sh := uint(14 - e)
		h := man >> sh
		if r := man & (1<<sh - 1); r > 1<<(sh-1) || r == 1<<(sh-1) && h&1 == 1 {
			h++
		}
		return sign | uint16(h)
	}
	h := uint32(e)<<10 | man>>13
	if r := man & 0x1fff; r > 0x1000 || r == 0x1000 && h&1 == 1 {
		h++ // a carry into the exponent is the correct result
	}
	return sign | uint16(h)
}

func packHalf(rows [][]float32) [][]uint16 {
	out := make([][]uint16, len(rows))
	for i, r := range rows {
		out[i] = make([]uint16, len(r))
		for j, v := range r {
			out[i][j] = toHalf(v)
		}
	}
	return out
}

// parFor runs fn(0..n-1) on up to NumCPU goroutines (attention heads and delta-net heads are independent).
func parFor(n int, fn func(i int)) {
	w := runtime.NumCPU()
	if w > n {
		w = n
	}
	var wg sync.WaitGroup
	next := make(chan int, n)
	for i := 0; i < n; i++ {
		next <- i
	}
	close(next)
	for ; w > 0; w-- {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				fn(i)
			}
		}()
	}
	wg.Wait()
}

// gdnState is the memory of one gated-delta-net block: the last convK-1 pre-conv rows and the per-head state.
type gdnState struct {
	tail [][]float32
	S    [][]float32
	dev  *devS // the state on the card (cuda build); while set, S is stale until fetch
}

// QwenState is a running Qwen session: pass it to Step with only the new tokens.
type QwenState struct {
	pos int
	gdn []*gdnState
	kv  []*kvCache
}

// Free releases the session's card memory (the cache and delta-net state kept in VRAM); the garbage collector cannot
// see VRAM. The session must not be used afterwards.
func (s *QwenState) Free() {
	for _, c := range s.kv {
		if c != nil {
			c.dev.release()
			c.dev = nil
		}
	}
	for _, g := range s.gdn {
		if g != nil {
			g.dev.release()
			g.dev = nil
		}
	}
}

// NewState starts an empty session.
func (m *Qwen) NewState() *QwenState {
	return &QwenState{gdn: make([]*gdnState, m.nLayer), kv: make([]*kvCache, m.nLayer)}
}

// Pos is the number of tokens the session has consumed.
func (s *QwenState) Pos() int { return s.pos }

// GemmaState is a running Gemma session; sliding layers keep only the last window of rows.
type GemmaState struct {
	pos  int
	kv   []*kvCache
	base []int // position of kv[l].K[0]
}

// Free releases the session's card memory, as QwenState.Free does.
func (s *GemmaState) Free() {
	for _, c := range s.kv {
		c.dev.release()
		c.dev = nil
	}
}

// NewState starts an empty session.
func (m *Gemma) NewState() *GemmaState {
	s := &GemmaState{kv: make([]*kvCache, m.nLayer), base: make([]int, m.nLayer)}
	for l := range s.kv {
		s.kv[l] = &kvCache{}
	}
	return s
}

// Pos is the number of tokens the session has consumed.
func (s *GemmaState) Pos() int { return s.pos }

// h2f32 converts a half float without the lookup table (no inf/nan: cached keys and values are finite).
func h2f32(h uint16) float32 {
	v := math.Float32frombits(uint32(h&0x7fff)<<13) * 0x1p112
	if h&0x8000 != 0 {
		return -v
	}
	return v
}

// dotHalf is q . k with four accumulators (len(q) is a multiple of 4).
func dotHalf(q []float32, k []uint16) float32 {
	var a, b, c, d float32
	k = k[:len(q)]
	for i := 0; i+3 < len(q); i += 4 {
		a += q[i] * h2f32(k[i])
		b += q[i+1] * h2f32(k[i+1])
		c += q[i+2] * h2f32(k[i+2])
		d += q[i+3] * h2f32(k[i+3])
	}
	return (a + b) + (c + d)
}

// axpyHalf adds w*v to o.
func axpyHalf(o []float32, w float32, v []uint16) {
	v = v[:len(o)]
	for i := range o {
		o[i] += w * h2f32(v[i])
	}
}
