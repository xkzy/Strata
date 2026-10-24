"""Next-token agreement: pass reconstructed final residuals through the teacher's real LM head (engine `--head-eval`).

usage: python ncm/head_agreement.py prepare CAP_DIR WORK_DIR     # trains the quantizers (seed 0), writes WORK_DIR/rows.f32 + meta.npz
       <run the engine: strata ... --head-eval WORK_DIR/rows.f32 WORK_DIR/argmax.i32>
       python ncm/head_agreement.py score WORK_DIR OUT.json

Held-out rows are the test windows of train_ms2.prep (same split, same train statistics, T=128, PCA R=128, 256 codes). The quantizers
see only the stream-mean residual, so each variant's 4-stream row is  R[s] - mean_s(R) + x_hat  (the true per-stream deviations are kept)
unless the variant name says "bcast". Agreement = share of positions where the head's argmax equals the teacher's own recorded next token.
"""
import json, sys
import numpy as np, torch
args, sys.argv = sys.argv[1:], sys.argv[:1]  # train_ms2 reads R/CODES/STEPS from argv at import; keep its defaults
import train_ms2 as t
from dataset import load_sequences

VARIANTS = [("A", 1, 0), ("B", 1, 1), ("A", 4, 0), ("B", 4, 4)]  # method, K1, K2 (8 and 32 bits/pos)


def test_windows(cap):
    man = json.load(open(f"{cap}/manifest.json"))
    tr, te, nx = [], [], []
    for k, s in sorted(load_sequences(cap).items()):
        split = man[int(k)]["split"]
        for w in range(len(s["rows"]) // t.T):
            sl = slice(w * t.T, (w + 1) * t.T)
            (te if split == "test" else tr).append(s["rows"][sl])
            if split == "test":
                nx.append(s["nxt"][sl])
    return np.stack(tr), np.stack(te), np.stack(nx)  # [N, T, hc, n_embd], [N, T]


def prepare(cap, work):
    torch.manual_seed(0)
    xtr, xte, P = t.prep(cap)
    rtr, rte, nxt = test_windows(cap)
    mu, sd = torch.tensor(rtr.mean(2)).mean((0, 1)), torch.tensor(rtr.mean(2)).std((0, 1)).clamp_min(1e-3)
    assert torch.allclose((torch.tensor(rte.mean(2)) - mu) / sd, xte, atol=1e-4), "split / statistics do not match train_ms2.prep"
    dev = rte - rte.mean(2, keepdims=True)  # per-stream deviations the quantizers never see
    ztr, zte = xtr @ P, xte @ P
    destd = lambda x: (x * sd + mu).numpy()
    variants = {"true_R": rte,
                "bcast_train_mean": np.broadcast_to(mu.numpy(), rte.shape),
                "bcast_true_mean": np.broadcast_to(rte.mean(2, keepdims=True), rte.shape),
                "dev+train_mean": dev + mu.numpy(),
                "dev+pca_only": dev + destd(zte @ P.T)[:, :, None]}
    for m, k1, k2 in VARIANTS:
        zh = t.run(m, k1, k2, 0, ztr, zte, P, xtr, xte, return_zte=True)
        variants[f"dev+{m}{k1}" + (f"+{k2}" if k2 else "")] = dev + destd(zh @ P.T)[:, :, None]
    with open(f"{work}/rows.f32", "wb") as f:
        for v in variants.values():
            f.write(np.ascontiguousarray(v, dtype=np.float32).tobytes())
    np.savez(f"{work}/meta.npz", nxt=nxt.reshape(-1), names=np.array(list(variants)))
    print(f"{len(variants)} variants x {nxt.size} rows -> {work}/rows.f32")


def score(work, out):
    m = np.load(f"{work}/meta.npz")
    nxt, names = m["nxt"], [str(n) for n in m["names"]]
    am = np.fromfile(f"{work}/argmax.i32", dtype=np.int32)
    assert len(am) == len(nxt) * len(names), f"engine wrote {len(am)} rows, expected {len(nxt) * len(names)}"
    res = {n: float((am[i * len(nxt):(i + 1) * len(nxt)] == nxt).mean()) for i, n in enumerate(names)}
    for n, a in res.items():
        print(f"{n:20s} top-1 agreement {a:.4f}")
    json.dump(dict(positions=int(len(nxt)), agreement=res), open(out, "w"), indent=1)


if __name__ == "__main__":
    prepare(*args[1:3]) if args[0] == "prepare" else score(*args[1:3])
