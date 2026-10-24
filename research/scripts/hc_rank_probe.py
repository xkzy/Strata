#!/usr/bin/env python3
"""Weight-only probe of the hyper-connection bottleneck (hc_*_down [320 x 10240], hc_*_up [10240 x 320]), BF16 at source.
Reads the exact tensors from shard 1. Unit j's contribution to the output is U[:,j] * act(D[j,:]·x): its weight-only
importance is s_j = ||D_j|| * ||U_:j||. This is a PROXY: function error also depends on the activation distribution (not measured)."""
import json, sys
import numpy as np

H = json.load(open('research/data/teacher_gguf_headers.json'))[0]
path, base = H['file'], H['data_start']
T = {t['name']: t for t in H['tensors']}

def bf16(name):
    t = T[name]; n = int(np.prod(t['shape']))
    raw = np.fromfile(path, dtype=np.uint16, count=n, offset=base + t['off'])
    return (raw.astype(np.uint32) << 16).view(np.float32).reshape(t['shape'][::-1])  # numpy shape = reversed ne

# sanity: BF16 weights of a normalised tensor must be finite and O(1); a wrong data_start gives garbage/NaN
x = bf16('blk.0.hc_attn_down.weight')
assert np.isfinite(x).all() and 1e-4 < np.abs(x).mean() < 10, ('bad offset', np.abs(x).mean())
print('sanity ok: blk.0.hc_attn_down', x.shape, 'mean|w|=%.4f' % np.abs(x).mean())

out = {}
for role in ('attn', 'ffn'):
    rows = []
    for l in range(48):
        D = bf16(f'blk.{l}.hc_{role}_down.weight')   # [320, 10240]
        U = bf16(f'blk.{l}.hc_{role}_up.weight')     # [10240, 320]
        assert D.shape == (320, 10240) and U.shape == (10240, 320), (D.shape, U.shape)
        s = np.linalg.norm(D, axis=1) * np.linalg.norm(U, axis=0)
        e = np.sort(s ** 2)[::-1]; c = np.cumsum(e) / e.sum()
        r = {q: int(np.searchsorted(c, q) + 1) for q in (0.90, 0.95, 0.99)}
        sv = np.linalg.svd(D, compute_uv=False); sc = np.cumsum(sv ** 2) / (sv ** 2).sum()
        rs = {q: int(np.searchsorted(sc, q) + 1) for q in (0.90, 0.99)}
        rows.append((l, r[0.90], r[0.95], r[0.99], rs[0.90], rs[0.99], float((sv ** 2).sum() ** 2 / (sv ** 4).sum())))
    a = np.array(rows, dtype=float)
    print(f'\nhc_{role}: bottleneck units (of 320) holding X of sum s_j^2, mean over 48 layers [min..max]; and SVD(D) rank for energy')
    for i, nm in ((1, 'units 90%'), (2, 'units 95%'), (3, 'units 99%'), (4, 'svd(D) 90%'), (5, 'svd(D) 99%'), (6, 'stable rank')):
        print(f'  {nm:12s} {a[:, i].mean():6.1f}  [{a[:, i].min():.0f}..{a[:, i].max():.0f}]')
    out[role] = a.tolist()
json.dump(out, open('research/data/hc_rank_probe.json', 'w'))
