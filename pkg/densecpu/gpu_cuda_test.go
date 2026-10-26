//go:build cuda

package densecpu

import (
	"math"
	"math/rand"
	"testing"
)

// Every CUDA kernel must agree with the CPU kernel on random matrices, on every card, for 1 and several inputs.
func TestCUDAKernelsMatchCPU(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	for dev := 0; dev < 2; dev++ {
		for _, c := range []struct {
			typ   uint32
			f16at []int
		}{{tQ4_K, []int{0, 2}}, {tQ5_K, []int{0, 2}}, {tQ6_K, []int{208}}, {tQ8_0, []int{0}}, {tBF16, nil}} {
			elems, bytes, _ := blockInfo(c.typ)
			rows, cols := 37, 1024
			w := make([]byte, rows*cols/elems*bytes)
			rng.Read(w)
			for b := 0; b < rows*cols/elems; b++ {
				for _, o := range c.f16at {
					w[b*bytes+o], w[b*bytes+o+1] = byte(rng.Intn(256)), 0x2c+byte(rng.Intn(8))
				}
				if c.typ == tBF16 {
					w[b*2+1] &= 0x3f
				}
			}
			for _, n := range []int{1, 2, 3, 5, 8, 19} {
				X := make([][]float32, n)
				for i := range X {
					X[i] = make([]float32, cols)
					for j := range X[i] {
						X[i][j] = rng.Float32()*2 - 1
					}
				}
				Y, err := rawMatvec(dev, c.typ, w, rows, cols, X)
				if err != nil {
					t.Skipf("no CUDA device %d: %v", dev, err)
				}
				rowBytes := cols / elems * bytes
				for i := range X {
					x := X[i]
					if c.typ != tBF16 {
						x = quantRef(x) // every batch size runs the int8 kernels on quantized inputs (cols is a multiple of 256)
					}
					xs := sums32(x)
					for r := 0; r < rows; r++ {
						want := float64(fusedDot(c.typ)(w[r*rowBytes:(r+1)*rowBytes], x, xs))
						if math.Abs(float64(Y[i][r])-want) > 1e-3*(1+math.Abs(want)) {
							t.Fatalf("dev %d type %d n=%d row %d: gpu %v cpu %v", dev, c.typ, n, r, Y[i][r], want)
						}
					}
				}
			}
		}
	}
}

// quantRef applies the kernel's input quantization: per 32 values, scale max|x|/127, round to nearest even.
func quantRef(x []float32) []float32 {
	out := make([]float32, len(x))
	for b := 0; b < len(x); b += 32 {
		var amax float32
		for _, v := range x[b : b+32] {
			amax = float32(math.Max(float64(amax), math.Abs(float64(v))))
		}
		d := amax / 127
		if d == 0 {
			continue
		}
		for i := b; i < b+32; i++ {
			out[i] = float32(math.RoundToEven(float64(x[i]/d))) * d
		}
	}
	return out
}

// TestCUDAAttnMatchesCPU checks the attention kernel against the CPU loops in Qwen.attn on random half caches.
func TestCUDAAttnMatchesCPU(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	for _, c := range []struct{ n, base, window, nHead, nKV, hd int }{{5, 0, 0, 8, 2, 64}, {7, 300, 0, 24, 4, 256}, {4, 1000, 0, 32, 4, 512}, {6, 900, 128, 32, 16, 256}, {5, 40, 128, 32, 4, 512}} {
		ctx := c.base + c.n
		rnd := func(rows, w int) [][]float32 {
			r := make([][]float32, rows)
			for i := range r {
				r[i] = make([]float32, w)
				for j := range r[i] {
					r[i][j] = rng.Float32()*2 - 1
				}
			}
			return r
		}
		Q, K, V := rnd(c.n, c.nHead*c.hd), rnd(ctx, c.nKV*c.hd), rnd(ctx, c.nKV*c.hd)
		kc := &kvCache{K: packHalf(K), V: packHalf(V)}
		scale := float32(1 / math.Sqrt(float64(c.hd)))
		got, ok := testGPU().attn("", Q, kc, 0, c.base, c.window, c.nHead, c.nKV, c.hd, scale)
		if !ok {
			t.Skip("no CUDA device")
		}
		for i := 0; i < c.n; i++ {
			for h := 0; h < c.nHead; h++ {
				kv := h / (c.nHead / c.nKV)
				lo := 0
				if c.window > 0 && c.base+i-c.window+1 > 0 {
					lo = c.base + i - c.window + 1
				}
				sc := make([]float64, c.base+i+1-lo)
				mx := math.Inf(-1)
				for j := range sc {
					sc[j] = float64(dotHalf(Q[i][h*c.hd:(h+1)*c.hd], kc.K[lo+j][kv*c.hd:(kv+1)*c.hd]) * scale)
					mx = math.Max(mx, sc[j])
				}
				var sum float64
				for j := range sc {
					sc[j] = math.Exp(sc[j] - mx)
					sum += sc[j]
				}
				want := make([]float32, c.hd)
				for j := range sc {
					axpyHalf(want, float32(sc[j]/sum), kc.V[lo+j][kv*c.hd:(kv+1)*c.hd])
				}
				for d, w := range want {
					if g := got[i][h*c.hd+d]; math.Abs(float64(g-w)) > 2e-4 {
						t.Fatalf("case %+v token %d head %d dim %d: gpu %v cpu %v", c, i, h, d, g, w)
					}
				}
			}
		}
	}
}

// testGPU is a gpuMatmul that only knows layer "" lives on card 0.
func testGPU() *gpuMatmul {
	return &gpuMatmul{res: map[string]gpuTensor{"attn_q.weight": {}}}
}

// TestCUDAAttnResidentGrows feeds one cache to the kernel in steps (so the resident copy is extended, and regrown past
// its first capacity) and checks each answer is identical to one computed from a fresh cache of the same rows.
func TestCUDAAttnResidentGrows(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	const nHead, nKV, hd, total = 8, 2, 64, 3000
	row := func() []float32 {
		r := make([]float32, nKV*hd)
		for i := range r {
			r[i] = rng.Float32()*2 - 1
		}
		return r
	}
	K, V := make([][]float32, total), make([][]float32, total)
	for i := range K {
		K[i], V[i] = row(), row()
	}
	kc, pos := &kvCache{}, 0
	for _, n := range []int{700, 1, 1, 3, 600, 1, 1000, 1, 693} {
		kc.K, kc.V = append(kc.K, packHalf(K[pos:pos+n])...), append(kc.V, packHalf(V[pos:pos+n])...)
		Q := make([][]float32, n)
		for i := range Q {
			Q[i] = make([]float32, nHead*hd)
			for j := range Q[i] {
				Q[i][j] = rng.Float32()*2 - 1
			}
		}
		got, ok := testGPU().attn("", Q, kc, 0, pos, 0, nHead, nKV, hd, 0.125)
		if !ok {
			t.Skip("no CUDA device")
		}
		fresh := &kvCache{K: kc.K, V: kc.V}
		want, ok2 := testGPU().attn("", Q, fresh, 0, pos, 0, nHead, nKV, hd, 0.125)
		if !ok2 {
			t.Fatalf("fresh cache of %d rows (n=%d, first=%d): attn declined", len(fresh.K), n, pos)
		}
		for i := range got {
			for j := range got[i] {
				if got[i][j] != want[i][j] {
					t.Fatalf("after %d rows, token %d dim %d: resident %v, fresh %v", pos+n, i, j, got[i][j], want[i][j])
				}
			}
		}
		pos += n
	}
}

// TestCUDAAttnRing runs a sliding-window cache the way Gemma does (host rows trimmed to the window after every call)
// through enough single tokens and batches that the ring wraps several times, against a CPU reference over all rows.
func TestCUDAAttnRing(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	const nHead, nKV, hd, window, total = 8, 2, 64, 100, 2500
	rnd := func(w int) []float32 {
		r := make([]float32, w)
		for i := range r {
			r[i] = rng.Float32()*2 - 1
		}
		return r
	}
	allK, allV := make([][]uint16, total), make([][]uint16, total)
	for i := range allK {
		allK[i], allV[i] = packHalf([][]float32{rnd(nKV * hd)})[0], packHalf([][]float32{rnd(nKV * hd)})[0]
	}
	kc, b0, pos := &kvCache{}, 0, 0
	for _, n := range []int{300, 1, 1, 1, 5, 1, 400, 1, 1, 7, 512, 1, 600, 1} {
		kc.K, kc.V = append(kc.K, allK[pos:pos+n]...), append(kc.V, allV[pos:pos+n]...)
		Q := make([][]float32, n)
		for i := range Q {
			Q[i] = rnd(nHead * hd)
		}
		got, ok := testGPU().attn("", Q, kc, b0, pos-b0, window, nHead, nKV, hd, 0.125)
		if !ok {
			t.Skip("no CUDA device")
		}
		for i := 0; i < n; i++ {
			last := pos + i
			lo := max(0, last-window+1)
			for h := 0; h < nHead; h++ {
				kv := h / (nHead / nKV)
				sc := make([]float64, last-lo+1)
				mx := math.Inf(-1)
				for j := range sc {
					sc[j] = float64(dotHalf(Q[i][h*hd:(h+1)*hd], allK[lo+j][kv*hd:(kv+1)*hd]) * 0.125)
					mx = math.Max(mx, sc[j])
				}
				var sum float64
				for j := range sc {
					sc[j] = math.Exp(sc[j] - mx)
					sum += sc[j]
				}
				want := make([]float32, hd)
				for j := range sc {
					axpyHalf(want, float32(sc[j]/sum), allV[lo+j][kv*hd:(kv+1)*hd])
				}
				for d, w := range want {
					if g := got[i][h*hd+d]; math.Abs(float64(g-w)) > 2e-4 {
						t.Fatalf("batch ending at %d, token %d head %d dim %d: gpu %v cpu %v", pos+n, i, h, d, g, w)
					}
				}
			}
		}
		pos += n
		if drop := len(kc.K) - window; drop > 0 { // Gemma's trim
			kc.K, kc.V = append([][]uint16(nil), kc.K[drop:]...), append([][]uint16(nil), kc.V[drop:]...)
			b0 += drop
		}
	}
}
