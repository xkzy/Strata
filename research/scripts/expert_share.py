#!/usr/bin/env python
"""Expert-by-expert pilot. For a layer's routed-expert tensor X [E=256, 640, 2560] (gate or up), measure how much structure the experts share:
  mean_frac   energy of the mean expert:  E*||mean||^2 / sum_e ||x_e||^2   (a single shared S plus residuals Delta_e = x_e - S)
  top_k_frac  energy in the top-k cross-expert components (Gram eigenvalues; k/E is what unstructured experts give)
  cos         pairwise cosine between flattened experts (mean |off-diag|, max, mean nearest-neighbour)
  cluster_res residual energy after subtracting the cluster mean (k-means, k=8/32 on the top-64 Gram coordinates) vs the same k with random labels
  haar        fraction of one expert's coefficients holding 99% / 99.9% energy in 2-D Haar vs raw (as in wavelet_stats.py)
Weight-space similarity only: experts that compute similar functions under a hidden-unit permutation look dissimilar here (limit).
Usage: expert_share.py OUT.json [layers=3,20,47] [kinds=gate,up]    (CPU)"""
import sys, json, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wavelet_stats as ws

def kmeans(P, k, it=30, seed=0):
    r = np.random.default_rng(seed); c = P[r.choice(len(P), k, replace=False)]
    for _ in range(it):
        lab = ((P[:, None] - c[None]) ** 2).sum(-1).argmin(1)
        for j in range(k):
            if (lab == j).any(): c[j] = P[lab == j].mean(0)
    return lab

def analyse(X):
    E = X.shape[0]; F = X.reshape(E, -1); en = (F.astype(np.float64) ** 2).sum(1); tot = en.sum()
    mean = F.mean(0, dtype=np.float64)
    out = {'mean_frac': float(E * (mean ** 2).sum() / tot)}
    G = (F @ F.T).astype(np.float64); w, V = np.linalg.eigh(G); w = w[::-1]; V = V[:, ::-1]
    out['top_k_frac'] = {str(k): float(w[:k].sum() / w.sum()) for k in (1, 8, 32, 128)}
    d = np.sqrt(np.diag(G)); C = G / d[:, None] / d[None]; np.fill_diagonal(C, np.nan)
    out['cos'] = {'mean_abs': float(np.nanmean(np.abs(C))), 'max': float(np.nanmax(C)), 'mean_nearest': float(np.nanmax(C, axis=1).mean())}
    P = V[:, :64] * np.sqrt(np.maximum(w[:64], 0)); rng = np.random.default_rng(1); out['cluster_res'] = {}
    for k in (8, 32):
        lab = kmeans(P, k); rl = rng.integers(0, k, E)
        def res(l):
            r = 0.0
            for j in range(k):
                s = l == j
                if s.any(): r += ((F[s].astype(np.float64) - F[s].mean(0, dtype=np.float64)) ** 2).sum()
            return float(r / tot)
        out['cluster_res'][str(k)] = {'kmeans': res(lab), 'random_labels': res(rl)}
    B = X[0][:512, :2048]
    out['haar_expert0'] = {'raw': ws.frac_for(B), 'haar': ws.frac_for(ws.haar(ws.haar(B, 0), 1))}
    return out

if __name__ == '__main__':
    outp = sys.argv[1]
    layers = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else '3,20,47').split(',')]
    kinds = (sys.argv[3] if len(sys.argv) > 3 else 'gate,up').split(',')
    res = {}
    for l in layers:
        for kd in kinds:
            name = f'blk.{l}.ffn_{kd}_exps.weight'
            try: X, qt = ws.load(name)
            except Exception as e: print(name, 'SKIP', repr(e)); continue
            r = analyse(X); r['quant'] = qt; res[name] = r; del X
            print(name, qt, 'mean_frac', round(r['mean_frac'], 4), 'top_k', {k: round(v, 3) for k, v in r['top_k_frac'].items()},
                  'cos', {k: round(v, 3) for k, v in r['cos'].items()}, 'cl', r['cluster_res'], flush=True)
            json.dump(res, open(outp, 'w'), indent=1)
