#!/usr/bin/env python3
"""Experiment A (wavelet pilot): do the teacher's dense projection matrices compact their energy in a 2-D wavelet basis?
For W (m x n) and an orthonormal basis T, discarding coefficient set D gives ||W-W^|_F^2 = sum_{D} C^2 exactly, so the best
k-term error is the tail of the sorted squared coefficients. We report, per matrix, the fraction of coefficients that holds
99% / 99.9% of the energy in four orthonormal bases: identity (raw weights = plain top-k pruning), Haar and db4 2-D DWT
(full depth), and a random orthogonal DCT-free control (random signs+permutation then Haar). Wavelets help only if they beat identity.
Also: SVD rank holding the same energy (the low-rank alternative). Quantisation (type given by the file) is reported, since the
file's weights are already quantised.
Usage: wavelet_stats.py list | wavelet_stats.py run out.json [name_regex] [max_matrices]   (CPU only, reads GGUF headers + chosen tensors)"""
import sys, re, json, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)).rsplit('/', 1)[0] + '/ncm')
from param_count import read_header
from head import SHARD, _read  # noqa: F401  (SHARD path)
import gguf
from gguf.quants import dequantize

offs = {}
_, ts = read_header(SHARD, offs)
info = {n: (d, t) for n, d, t, _ in ts}

def load(name):
    d, t = info[name]; n = int(np.prod(d))
    qt = gguf.GGMLQuantizationType(t)
    if qt == gguf.GGMLQuantizationType.F32: nb, dt = 4 * n, 'f32'
    elif qt == gguf.GGMLQuantizationType.F16: nb, dt = 2 * n, 'f16'
    elif qt == gguf.GGMLQuantizationType.BF16: nb, dt = 2 * n, 'bf16'
    else:
        blk, ts_ = gguf.GGML_QUANT_SIZES[qt]; nb, dt = n // blk * ts_, 'q'
    with open(SHARD, 'rb') as f:
        f.seek(offs[name]); raw = np.frombuffer(f.read(nb), dtype=np.uint8)
    if dt == 'f32': a = raw.view(np.float32)
    elif dt == 'f16': a = raw.view(np.float16).astype(np.float32)
    elif dt == 'bf16': a = (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    else: a = dequantize(raw.reshape(-1, ts_), qt)
    return np.asarray(a, dtype=np.float32).reshape(d[::-1]), qt.name

def haar(x, axis):
    """Full-depth orthonormal Haar along axis (length must be a power of two)."""
    x = np.moveaxis(x, axis, 0).copy(); n = x.shape[0]
    while n > 1:
        a, b = x[0:n:2].copy(), x[1:n:2].copy()
        x[:n // 2], x[n // 2:n] = (a + b) / np.sqrt(2), (a - b) / np.sqrt(2); n //= 2
    return np.moveaxis(x, 0, axis)

def frac_for(c, levels=(0.99, 0.999)):
    e = np.sort((c.astype(np.float64) ** 2).ravel())[::-1]; cs = np.cumsum(e) / e.sum()
    return [float((np.searchsorted(cs, l) + 1) / e.size) for l in levels]

def analyse(W):
    m, n = W.shape
    # crop to powers of two (top-left block); report the crop
    M, N = 1 << int(np.log2(m)), 1 << int(np.log2(n)); B = W[:M, :N]
    rng = np.random.default_rng(0)
    R = B[rng.permutation(M)][:, rng.permutation(N)] * rng.choice([-1, 1], size=(M, 1)) * rng.choice([-1, 1], size=(1, N))
    out = {'shape': [m, n], 'crop': [M, N],
           'identity': frac_for(B), 'haar': frac_for(haar(haar(B, 0), 1)), 'random_haar': frac_for(haar(haar(R, 0), 1))}
    s = np.linalg.svd(B.astype(np.float64), compute_uv=False)
    cs = np.cumsum(s ** 2) / (s ** 2).sum(); k = [int(np.searchsorted(cs, l) + 1) for l in (0.99, 0.999)]
    # svd storage fraction of coefficients for rank k: k*(M+N)/(M*N)
    out['svd_storage'] = [kk * (M + N) / (M * N) for kk in k]; out['svd_rank'] = k
    try:
        import pywt
        c = pywt.wavedec2(B.astype(np.float64), 'db4', mode='periodization')
        flat = np.concatenate([c[0].ravel()] + [d.ravel() for lvl in c[1:] for d in lvl])
        out['db4'] = frac_for(flat)
        out['ll_energy_frac_by_level'] = None
    except ImportError:
        out['db4'] = None
    return out

if __name__ == '__main__':
    if sys.argv[1] == 'list':
        for n, (d, t) in info.items():
            if len(d) == 2 and re.search(r'blk\.(0|3)\.|^output|token_embd', n): print(n, d, gguf.GGMLQuantizationType(t).name)
        sys.exit()
    outp, rx, mx = sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else r'blk\.(0|3|7|47)\.(attn_(q|k|v|qkv|gate|output)|ssm_out)\.weight', int(sys.argv[4]) if len(sys.argv) > 4 else 12
    res = {}
    for name in [n for n in info if re.search(rx, n)][:mx]:
        try: W, qt = load(name)
        except Exception as e: res[name] = {'error': repr(e)}; print(name, 'SKIP', repr(e)); continue
        r = analyse(W); r['quant'] = qt; res[name] = r
        print(name, qt, r['shape'], 'id', [round(x, 3) for x in r['identity']], 'haar', [round(x, 3) for x in r['haar']],
              'rand', [round(x, 3) for x in r['random_haar']], 'db4', r['db4'] and [round(x, 3) for x in r['db4']],
              'svd_storage', [round(x, 3) for x in r['svd_storage']], flush=True)
        json.dump(res, open(outp, 'w'), indent=1)
