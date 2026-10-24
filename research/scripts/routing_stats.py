#!/usr/bin/env python3
"""Real expert-routing statistics from the teacher's captured routes (cap1/routing_*.bin).
Record = (layer:i32, k:i32, ids:i32[k], weights:f32[k]); weights are 0 in this dump so only ids are used.
Order inside a layer is assumed to be processing order of positions (ASSUMPTION, see doc)."""
import struct, glob, json, sys
import numpy as np

E, K, L = 256, 10, 48
rng = np.random.default_rng(0)

def load(path):
    b = open(path, 'rb').read(); off = 0; per = [[] for _ in range(L)]
    while off + 8 <= len(b):
        layer, k = struct.unpack_from('<ii', b, off)
        per[layer].append(struct.unpack_from(f'<{k}i', b, off + 8)); off += 8 + 8 * k
    n = {len(p) for p in per}
    assert len(n) == 1, f'unequal positions per layer {n}'
    return np.array(per, dtype=np.int16)  # [L, T, K]

files = sorted(glob.glob('/home/khing/Downloads/Strata-data/ncm/cap1/routing_*.bin'))
runs = [load(f) for f in files]
T_tot = sum(r.shape[1] for r in runs)
print(f'files={len(runs)} positions per file={[r.shape[1] for r in runs]} total={T_tot}')
assert all(((r >= 0) & (r < E)).all() for r in runs)

# 1. adjacent-position overlap and n-window distinct experts, per layer, pooled over files
def window_distinct(run, n):
    L_, T, _ = run.shape
    out = []
    for s in range(0, T - n + 1):
        w = run[:, s:s + n, :].reshape(L_, -1)
        out.append([len(set(row.tolist())) for row in w])
    return np.array(out)  # [windows, L]

res = {'n': {}}
uni = lambda n: E * (1 - (1 - K / E) ** n)
for n in (1, 2, 3, 4, 6, 8):
    per_file = [window_distinct(r, n).mean() for r in runs if r.shape[1] >= n]
    m = float(np.mean(per_file)); ci = float(1.96 * np.std(per_file, ddof=1) / np.sqrt(len(per_file)))
    res['n'][n] = dict(distinct_mean=m, ci95=ci, uniform_model=uni(n), ratio_to_n1=m / res['n'][1]['distinct_mean'] if n > 1 else 1.0)
    print(f'n={n}: distinct experts/layer = {m:6.2f} ±{ci:.2f} (uniform model {uni(n):6.2f}); rho_expert-bound = {res["n"][n]["ratio_to_n1"]:.2f} (uniform {uni(n)/K:.2f})')

# 2. expert usage skew: share of activations covered by the top-N experts of each layer (static profile, in-sample AND held-out files)
def topn_share(train, test, N):
    shares = []
    for l in range(L):
        cnt = np.bincount(np.concatenate([r[l].ravel() for r in train]), minlength=E)
        top = set(np.argsort(-cnt)[:N].tolist())
        t = np.concatenate([r[l].ravel() for r in test])
        shares.append(np.isin(t, list(top)).mean())
    return float(np.mean(shares))
idx = np.arange(len(runs)); rng.shuffle(idx); tr, te = [runs[i] for i in idx[:8]], [runs[i] for i in idx[8:]]
res['topn_heldout'] = {}
for N in (16, 32, 64, 128):
    res['topn_heldout'][N] = topn_share(tr, te, N)
    print(f'top-{N:3d}/layer profile from 8 files covers {res["topn_heldout"][N]:.1%} of activations in 4 held-out files (uniform would be {N/E:.1%})')

# 3. effect on speculation: speedup = E[tok]/rho  at alpha=0.77, gamma=n-1, fully expert-bound pass
a = 0.77
for n in (2, 3, 4, 5, 7):
    if n in res['n']:
        e_tok = (1 - a ** n) / (1 - a); rho = res['n'][n]['ratio_to_n1']
        print(f'gamma={n-1}: E[tok]={e_tok:.2f} rho={rho:.2f} -> expert-bound speedup {e_tok/rho:.2f}x')
json.dump(res, open('research/data/routing_stats.json', 'w'), indent=1, default=float)
