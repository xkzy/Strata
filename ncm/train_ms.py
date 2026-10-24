"""Stage 2 ablation on REAL teacher residuals: residual-VQ alone (A) vs lifting filter bank + residual-VQ (B), at equal bits.

Data: final-layer residuals captured from the teacher (ncm/capture_batch.sh), mean over the 4 hyper-connection streams -> [T, 2560],
standardised per dimension with TRAIN statistics. Metric: NMSE = ||x - x_hat||^2 / ||x||^2 on held-out sequences (fraction of the
standardised energy not recovered; 1.0 = predicting the train mean). Bits/position = levels * log2(codes), amortised over positions.

usage: python ncm/train_ms.py CAP_DIR OUT.json
"""
import json, sys, hashlib, glob, os
import numpy as np, torch
from dataset import load_sequences
from multiscale import LiftingFilterBank, ResidualVQ

T_LEN, CODES, STEPS, N_TRAIN = 96, 64, 400, 9  # first 9 sequences train, the rest held out
BITS = int(np.log2(CODES))


def prep(cap):
    seqs = load_sequences(cap)
    keys = sorted(seqs)
    x = torch.tensor(np.stack([seqs[k]["rows"][:T_LEN].mean(1) for k in keys]))  # [S, T, D]
    mu, sd = x[:N_TRAIN].mean((0, 1)), x[:N_TRAIN].std((0, 1)).clamp_min(1e-3)
    return (x - mu) / sd, keys


@torch.no_grad()
def init_books(rvq, z):  # data-dependent init: each level starts from random residual vectors (avoids dead codes)
    res = z.reshape(-1, z.shape[-1])
    for book in rvq.books:
        book.weight.copy_(res[torch.randperm(len(res))[: CODES]])
        res = res - book(torch.cdist(res, book.weight).argmin(-1))


def nmse(a, b):
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


def run_A(xtr, xte, K, seed):
    torch.manual_seed(seed)
    rvq = ResidualVQ(xtr.shape[-1], K, CODES)
    init_books(rvq, xtr)
    opt = torch.optim.Adam(rvq.parameters(), 1e-3)
    for _ in range(STEPS):
        zh, _, ql = rvq(xtr)
        loss = ((zh - xtr) ** 2).mean() + ql
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return nmse(rvq(xtr)[0], xtr), nmse(rvq(xte)[0], xte), K * BITS


def run_B(xtr, xte, K1, K2, seed):
    torch.manual_seed(seed)
    D = xtr.shape[-1]
    fb, rs, rd = LiftingFilterBank(D), ResidualVQ(D, K1, CODES), ResidualVQ(D, K2, CODES)
    with torch.no_grad():
        s, d = fb.analyze(xtr)
        init_books(rs, s); init_books(rd, d)

    def fwd(x):
        s, d = fb.analyze(x)
        sh, _, qs = rs(s)
        dh, _, qd = rd(d)
        return fb.synthesize(sh, dh, x.shape[1]), qs + qd

    opt = torch.optim.Adam(list(fb.parameters()) + list(rs.parameters()) + list(rd.parameters()), 1e-3)
    for _ in range(STEPS):
        xh, ql = fwd(xtr)
        loss = ((xh - xtr) ** 2).mean() + ql
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return nmse(fwd(xtr)[0], xtr), nmse(fwd(xte)[0], xte), (K1 + K2) * BITS / 2


if __name__ == "__main__":
    cap, out = sys.argv[1], sys.argv[2]
    x, keys = prep(cap)
    xtr, xte = x[:N_TRAIN], x[N_TRAIN:]
    rows = []
    for K in (1, 2, 3):
        cfgs = [("A rvq", lambda s, K=K: run_A(xtr, xte, K, s), f"K={K}"),
                ("B fb+rvq even", lambda s, K=K: run_B(xtr, xte, K, K, s), f"K1=K2={K}"),
                ("B fb+rvq coarse-heavy", lambda s, K=K: run_B(xtr, xte, K + 1, K - 1, s) if K > 1 else None, f"K1={K+1},K2={K-1}")]
        for name, fn, tag in cfgs:
            res = [fn(s) for s in (0, 1, 2)]
            if res[0] is None:
                continue
            a = np.array(res)
            rows.append(dict(method=name, cfg=tag, bits_per_pos=float(a[0, 2]), train_nmse=[float(a[:, 0].mean()), float(a[:, 0].std())],
                             test_nmse=[float(a[:, 1].mean()), float(a[:, 1].std())]))
            print(f"{name:24s} {tag:10s} bits/pos {a[0,2]:4.1f}  train {a[:,0].mean():.3f}±{a[:,0].std():.3f}  test {a[:,1].mean():.3f}±{a[:,1].std():.3f}", flush=True)
    meta = dict(codes=CODES, steps=STEPS, seq_len=T_LEN, n_train_seq=N_TRAIN, n_test_seq=len(keys) - N_TRAIN, seeds=[0, 1, 2], torch=torch.__version__,
                files={os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest()[:16] for f in sorted(glob.glob(f"{cap}/final_r_*.bin"))})
    json.dump(dict(meta=meta, results=rows), open(out, "w"), indent=1)
