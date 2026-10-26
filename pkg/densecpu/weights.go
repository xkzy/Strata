package densecpu

import (
	"fmt"
	"math"
	"os"
	"runtime"
	"sync"
	"syscall"

	"strata/pkg/gguf"
)

// weights is an open GGUF with its payload memory-mapped (never copied).
type weights struct {
	g   *gguf.GGUFFile
	mm  []byte
	gpu *gpuMatmul // nil: everything runs on the CPU
}

// openWeights opens path and checks general.architecture is arch.
func openWeights(path, arch string) (*weights, error) {
	g, err := gguf.Open(path)
	if err != nil {
		return nil, err
	}
	if a, _ := g.Metadata["general.architecture"].(string); a != arch {
		g.Close()
		return nil, fmt.Errorf("densecpu: %s is %q, not %s", path, a, arch)
	}
	f, err := os.Open(path)
	if err != nil {
		g.Close()
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		g.Close()
		return nil, err
	}
	mm, err := syscall.Mmap(int(f.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		g.Close()
		return nil, err
	}
	return &weights{g: g, mm: mm}, nil
}

func metaF(g *gguf.GGUFFile, key string) float32 {
	switch v := g.Metadata[key].(type) {
	case float32:
		return v
	case float64:
		return float32(v)
	}
	return 0
}

func metaI(g *gguf.GGUFFile, key string) int {
	switch v := g.Metadata[key].(type) {
	case uint32:
		return int(v)
	case int32:
		return int(v)
	case uint64:
		return int(v)
	case int64:
		return int(v)
	}
	return 0
}

// Close releases the mapping.
func (m *weights) Close() {
	if m.gpu != nil {
		m.gpu.free()
	}
	syscall.Munmap(m.mm)
	m.g.Close()
}

func (m *weights) raw(name string) (*gguf.TensorInfo, []byte) {
	t, ok := m.g.GetTensor(name)
	if !ok {
		return nil, nil
	}
	return t, m.mm[m.g.DataStart+int64(t.Offset):]
}

// vec dequantizes a whole (small) tensor.
func (m *weights) vec(name string) []float32 {
	t, b := m.raw(name)
	if t == nil {
		return nil
	}
	n := 1
	for _, d := range t.Dims {
		n *= int(d)
	}
	out := make([]float32, n)
	if err := DequantRow(t.Type, b, n, out); err != nil {
		panic(err)
	}
	return out
}

// matmul returns Y[i][r] = sum_c W[r][c] * X[i][c] for the 2-D tensor name (dims [cols, rows]).
// Each weight row is dequantized once and used for every input row.
func (m *weights) matmul(name string, X [][]float32) [][]float32 {
	t, b := m.raw(name)
	if t == nil || len(t.Dims) != 2 {
		panic("densecpu: missing or non-2D tensor " + name)
	}
	cols, rows := int(t.Dims[0]), int(t.Dims[1])
	elems, bytes, err := blockInfo(t.Type)
	if err != nil {
		panic(err)
	}
	rowBytes := cols / elems * bytes
	if m.gpu != nil {
		if Y, ok := m.gpu.matmul(name, t, X); ok {
			return Y
		}
	}
	fused := fusedDot(t.Type)
	var xs [][]float32
	if fused != nil {
		for _, x := range X {
			xs = append(xs, sums32(x))
		}
	}
	Y := make([][]float32, len(X))
	for i := range Y {
		Y[i] = make([]float32, rows)
	}
	workers := runtime.NumCPU()
	chunk := (rows + workers*4 - 1) / (workers * 4)
	var wg sync.WaitGroup
	next := make(chan int, workers*4)
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			buf := make([]float32, cols)
			for lo := range next {
				for r := lo; r < lo+chunk && r < rows; r++ {
					if fused != nil {
						for i, x := range X {
							Y[i][r] = fused(b[r*rowBytes:(r+1)*rowBytes], x, xs[i])
						}
						continue
					}
					if err := DequantRow(t.Type, b[r*rowBytes:], cols, buf); err != nil {
						panic(err)
					}
					for i, x := range X {
						var s float32
						for c, w := range buf {
							s += w * x[c]
						}
						Y[i][r] = s
					}
				}
			}
		}()
	}
	for lo := 0; lo < rows; lo += chunk {
		next <- lo
	}
	close(next)
	wg.Wait()
	return Y
}

func rms(x, w []float32, eps float32) []float32 {
	var ss float64
	for _, v := range x {
		ss += float64(v) * float64(v)
	}
	k := float32(1 / math.Sqrt(ss/float64(len(x))+float64(eps)))
	out := make([]float32, len(x))
	for i, v := range x {
		out[i] = v * k
		if w != nil {
			out[i] *= w[i]
		}
	}
	return out
}

func geluTanh(x float32) float32 {
	f := float64(x)
	return float32(0.5 * f * (1 + math.Tanh(0.7978845608028654*(f+0.044715*f*f*f))))
}

// ropeKey identifies one set of rotation angles; the heads of a token all share it, so one entry is enough.
type ropeKey struct {
	pos, nRot int
	base      float32
	ff        *float32
}

var ropeMemo struct {
	sync.Mutex
	key    ropeKey
	cs, sn []float64
}

// ropeAngles returns cos and sin of theta_i = pos*base^(-2i/nRot)/ff[i] for i < nRot/2.
func ropeAngles(pos int, nRot int, base float32, ff []float32) (cs, sn []float64) {
	k := ropeKey{pos: pos, nRot: nRot, base: base}
	if ff != nil {
		k.ff = &ff[0]
	}
	ropeMemo.Lock()
	defer ropeMemo.Unlock()
	if ropeMemo.key == k && ropeMemo.cs != nil {
		return ropeMemo.cs, ropeMemo.sn
	}
	half := nRot / 2
	cs, sn = make([]float64, half), make([]float64, half)
	for i := range cs {
		theta := float64(pos) * math.Pow(float64(base), -2*float64(i)/float64(nRot))
		if ff != nil {
			theta /= float64(ff[i])
		}
		sn[i], cs[i] = math.Sincos(theta)
	}
	ropeMemo.key, ropeMemo.cs, ropeMemo.sn = k, cs, sn
	return cs, sn
}

// ropeNeox rotates v (one head, len hd) in place: pairs (i, i+nRot/2), theta = pos*base^(-2i/nRot)/ff[i].
func ropeNeox(v []float32, pos int, nRot int, base float32, ff []float32) {
	half := nRot / 2
	cs, sn := ropeAngles(pos, nRot, base, ff)
	for i := 0; i < half; i++ {
		a, b := float64(v[i]), float64(v[i+half])
		v[i], v[i+half] = float32(a*cs[i]-b*sn[i]), float32(a*sn[i]+b*cs[i])
	}
}

// ffn is the feed-forward block down(act(gate x) * up x), act being silu or (gelu) the tanh gelu. With the weights on
// one card the whole block runs there and only x and the result cross the bus.
func (m *weights) ffn(p string, X [][]float32, gelu bool) [][]float32 {
	if m.gpu != nil {
		if Y, ok := m.gpu.ffn(p, X, gelu); ok {
			return Y
		}
	}
	g, u := m.matmul(p+"ffn_gate.weight", X), m.matmul(p+"ffn_up.weight", X)
	for i := range g {
		for j := range g[i] {
			if gelu {
				g[i][j] = geluTanh(g[i][j]) * u[i][j]
			} else {
				g[i][j] = silu(g[i][j]) * u[i][j]
			}
		}
	}
	return m.matmul(p+"ffn_down.weight", g)
}

// matmulMulti applies several matrices that take the same input; on the GPU the input is uploaded once.
func (m *weights) matmulMulti(names []string, X [][]float32) [][][]float32 {
	if m.gpu != nil {
		if Y, ok := m.gpu.multi(names, X); ok {
			return Y
		}
	}
	Y := make([][][]float32, len(names))
	for k, nm := range names {
		Y[k] = m.matmul(nm, X)
	}
	return Y
}
