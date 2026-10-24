"""Stage 2 v2: RVQ vs filter bank + RVQ on a PCA-reduced teacher residual, with a held-out split from the capture manifest.

Target per position: final-layer residual, mean over the 4 hyper-connection streams -> 2560-d, standardised with TRAIN stats, then
projected on the top-R train PCA directions. NMSE is always measured back in the standardised 2560-d space, so it includes the PCA
truncation (`pca_only` is the floor: no quantisation at all). 1.0 = the train mean. Sequences are cut into non-overlapping windows of T.

usage: python ncm/train_ms2.py CAP_DIR OUT.json [R=128] [CODES=256] [STEPS=600]
"""
import glob, hashlib, json, os, sys
import numpy as np, torch
from dataset import load_sequences
from multiscale import LiftingFilterBank, ResidualVQ

T = 128
DEV = torch.device(os.environ.get("NCM_DEVICE", "cpu"))  # the committed results were produced on cpu; cuda rounds differently
R = int(sys.argv[3]) if len(sys.argv) > 3 else 128
CODES = int(sys.argv[4]) if len(sys.argv) > 4 else 256
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 600
BITS = int(np.log2(CODES))


def prep(cap):
    man = json.load(open(f"{cap}/manifest.json"))
    seqs = load_sequences(cap)
    if os.environ.get("NCM_SPLIT_SEED"):  # a different random held-out set: 20% of each source's prompts, instead of every 5th
        import random
        rng, by_src = random.Random(int(os.environ["NCM_SPLIT_SEED"])), {}
        for i, m in enumerate(man):
            by_src.setdefault(m["source"], []).append(i)
        for ids in by_src.values():
            rng.shuffle(ids)
            for j, i in enumerate(ids):
                man[i]["split"] = "test" if j < round(len(ids) / 5) else "train"
    tr, te, src = [], [], []
    for k, s in sorted(seqs.items()):
        m = man[int(k)]
        x = s["rows"].mean(1)  # [P, 2560]
        for w in range(len(x) // T):
            (te if m["split"] == "test" else tr).append(x[w * T:(w + 1) * T])
            src.append((m["source"], m["split"]))
    tr, te = torch.tensor(np.stack(tr), device=DEV), torch.tensor(np.stack(te), device=DEV)
    mu, sd = tr.mean((0, 1)), tr.std((0, 1)).clamp_min(1e-3)
    tr, te = (tr - mu) / sd, (te - mu) / sd
    flat = tr.reshape(-1, tr.shape[-1])
    _, evals, evecs = torch.pca_lowrank(flat, q=R, center=False, niter=4)
    return tr, te, evecs[:, :R]  # evecs [2560, R] orthonormal


def nmse(xh, x):
    return float(((xh - x) ** 2).sum() / (x ** 2).sum())


@torch.no_grad()
def init_books(rvq, z):
    res = z.reshape(-1, z.shape[-1])
    for book in rvq.books:
        book.weight.copy_(res[torch.randperm(len(res))[:CODES]])
        res = res - book(torch.cdist(res, book.weight).argmin(-1))


def fit(model_fwd, params, ztr, steps=STEPS):
    opt = torch.optim.Adam(params, 1e-3)
    for _ in range(steps):
        zh, ql = model_fwd(ztr)
        loss = ((zh - ztr) ** 2).mean() + ql
        opt.zero_grad(); loss.backward(); opt.step()


def run(method, K1, K2, seed, ztr, zte, P, xtr, xte, return_zte=False):
    torch.manual_seed(seed)
    D = ztr.shape[-1]
    if method == "A":
        rvq = ResidualVQ(D, K1, CODES).to(DEV)
        init_books(rvq, ztr)
        f = lambda z: (lambda o: (o[0], o[2]))(rvq(z))
        params, bits = list(rvq.parameters()), K1 * BITS
    else:
        fb, rs, rd = LiftingFilterBank(D).to(DEV), ResidualVQ(D, K1, CODES).to(DEV), ResidualVQ(D, K2, CODES).to(DEV)
        with torch.no_grad():
            s, d = fb.analyze(ztr)
            init_books(rs, s); init_books(rd, d)

        def f(z):
            s, d = fb.analyze(z)
            sh, _, qs = rs(s)
            dh, _, qd = rd(d)
            return fb.synthesize(sh, dh, z.shape[1]), qs + qd
        params, bits = list(fb.parameters()) + list(rs.parameters()) + list(rd.parameters()), (K1 + K2) * BITS / 2
    fit(f, params, ztr)
    with torch.no_grad():
        if return_zte:  # head_agreement.py: the reconstructed held-out codes, in PCA space
            return f(zte)[0]
        return nmse(f(ztr)[0] @ P.T, xtr), nmse(f(zte)[0] @ P.T, xte), bits


if __name__ == "__main__":
    cap, out = sys.argv[1], sys.argv[2]
    xtr, xte, P = prep(cap)
    ztr, zte = xtr @ P, xte @ P
    print(f"train windows {len(xtr)}, test windows {len(xte)}, T={T}, R={R}, codes={CODES}", flush=True)
    rows = [dict(method="train_mean", cfg="-", bits_per_pos=0, train_nmse=[1.0, 0], test_nmse=[nmse(torch.zeros_like(xte), xte), 0]),
            dict(method="pca_only", cfg=f"R={R}", bits_per_pos=R * 32, train_nmse=[nmse(ztr @ P.T, xtr), 0], test_nmse=[nmse(zte @ P.T, xte), 0])]
    for K in (1, 2, 3, 4):
        cfgs = [("A rvq", "A", K, 0), ("B fb+rvq even", "B", K, K)] + ([("B fb+rvq coarse-heavy", "B", K + 1, K - 1)] if K > 1 else [])
        for name, m, k1, k2 in cfgs:
            res = np.array([run(m, k1, k2, s, ztr, zte, P, xtr, xte) for s in (0, 1, 2)])
            rows.append(dict(method=name, cfg=f"K1={k1}" + (f",K2={k2}" if k2 else ""), bits_per_pos=float(res[0, 2]),
                             train_nmse=[float(res[:, 0].mean()), float(res[:, 0].std())], test_nmse=[float(res[:, 1].mean()), float(res[:, 1].std())]))
            print(f"{name:24s} {rows[-1]['cfg']:10s} bits/pos {res[0,2]:5.1f} train {res[:,0].mean():.3f}±{res[:,0].std():.3f} test {res[:,1].mean():.3f}±{res[:,1].std():.3f}", flush=True)
    for r in rows[:2]:
        print(r)
    meta = dict(T=T, R=R, codes=CODES, steps=STEPS, seeds=[0, 1, 2], train_windows=len(xtr), test_windows=len(xte), torch=torch.__version__,
                files={os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest()[:16] for f in sorted(glob.glob(f"{cap}/final_r_*.bin"))})
    json.dump(dict(meta=meta, results=rows), open(out, "w"), indent=1)
