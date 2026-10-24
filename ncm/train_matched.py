"""Parameter-matched control for train_ms2.py (same split, PCA, steps, seeds).

The v2 comparison gave the filter bank two codebook sets, so ~2x the parameters of plain RVQ at the same code bits. Here:
  A  rvq K            K books                          (K*codes*R params,  K*8 bits/pos)
  Bs fb+rvq shared K  coarse and detail share K books  (same codebook params as A K, same bits/pos as A K) + filter weights
  B  fb+rvq K+K       two sets (2K books)              (2x params, K*8 bits/pos)           <- the v2 comparison
  A2 rvq 2K           2K books                         (same codebook params as B K+K, 2x the bits/pos)
usage: python ncm/train_matched.py CAP_DIR OUT.json
"""
import json, sys
import numpy as np, torch
args, sys.argv = sys.argv[1:], sys.argv[:1]  # train_ms2 reads R/CODES/STEPS from argv at import
import train_ms2 as t
from multiscale import LiftingFilterBank, ResidualVQ


def run_shared(K, seed, ztr, zte, P, xtr, xte):
    torch.manual_seed(seed)
    fb, rq = LiftingFilterBank(ztr.shape[-1]), ResidualVQ(ztr.shape[-1], K, t.CODES)
    with torch.no_grad():
        s, d = fb.analyze(ztr)
        t.init_books(rq, torch.cat([s, d], 1))

    def f(z):
        s, d = fb.analyze(z)
        sh, _, qs = rq(s)
        dh, _, qd = rq(d)
        return fb.synthesize(sh, dh, z.shape[1]), qs + qd
    params = list(fb.parameters()) + list(rq.parameters())
    t.fit(f, params, ztr)
    with torch.no_grad():
        return t.nmse(f(ztr)[0] @ P.T, xtr), t.nmse(f(zte)[0] @ P.T, xte), sum(p.numel() for p in params)


def nparams(method, K1, K2):
    D = t.R
    if method == "A":
        return K1 * t.CODES * D
    return (K1 + K2) * t.CODES * D + sum(p.numel() for p in LiftingFilterBank(D).parameters())


if __name__ == "__main__":
    cap, out = args
    xtr, xte, P = t.prep(cap)
    ztr, zte = xtr @ P, xte @ P
    rows = []
    for K in (1, 2, 4):
        for name, fn, bits in [("A", lambda s: t.run("A", K, 0, s, ztr, zte, P, xtr, xte)[:2] + (nparams("A", K, 0),), K * t.BITS),
                               ("Bs shared", lambda s: run_shared(K, s, ztr, zte, P, xtr, xte), K * t.BITS),
                               ("B two sets", lambda s: t.run("B", K, K, s, ztr, zte, P, xtr, xte)[:2] + (nparams("B", K, K),), K * t.BITS),
                               ("A2 rvq 2K", lambda s: t.run("A", 2 * K, 0, s, ztr, zte, P, xtr, xte)[:2] + (nparams("A", 2 * K, 0),), 2 * K * t.BITS)]:
            res = np.array([fn(s) for s in (0, 1, 2)])
            rows.append(dict(K=K, method=name, bits_per_pos=bits, params=int(res[0, 2]), train_nmse=[float(res[:, 0].mean()), float(res[:, 0].std())],
                             test_nmse=[float(res[:, 1].mean()), float(res[:, 1].std())]))
            print(f"K={K} {name:11s} bits/pos {bits:3d} params {int(res[0,2]):8d} train {res[:,0].mean():.3f}±{res[:,0].std():.3f} test {res[:,1].mean():.3f}±{res[:,1].std():.3f}", flush=True)
    json.dump(dict(R=t.R, codes=t.CODES, steps=t.STEPS, seeds=[0, 1, 2], results=rows), open(out, "w"), indent=1)
