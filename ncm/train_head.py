"""Head-aware training of the stage-2 codecs: the reconstruction goes through the frozen teacher LM head (ncm/head.py) and the loss is the
KL from the teacher's next-token distribution (computed from the TRUE residual) to the reconstruction's.

Same protocol as head_agreement.py: the codec sees the standardised stream mean (PCA-128 coded); the 4-stream row is
R_hat[s] = R[s] - mean_s(R) + destandardise(x_hat), i.e. the true per-stream deviations are kept. Score = top-1 agreement with the teacher's
recorded next token on the held-out windows (+ NMSE in the standardised 2560-d space, as before).

modes: nmse = mse + commitment (the earlier objective, same batches/steps)   head = KL + commitment   mix = KL + mse + commitment
methods: A K (rvq)   B K (filter bank + rvq, K+K levels, two codebook sets)   lin (unquantised linear 128 -> 2560 decoder: the bottleneck without a quantiser)

usage: NCM_DEVICE=cuda:1 [NCM_SPLIT_SEED=n] python ncm/train_head.py CAP_DIR OUT.json MODES [STEPS=800] [BATCH_WINDOWS=2]   (MODES e.g. nmse,head,mix)
"""
import json, os, sys
import numpy as np, torch
import torch.nn.functional as F
import head_agreement as ha  # trims sys.argv; its parsed arguments are ha.args
from head import TeacherHead
from multiscale import LiftingFilterBank, ResidualVQ

t = ha.t
DEV = t.DEV
if DEV.type != "cuda" or not torch.cuda.is_available():  # standing rule: training runs on the GPU only
    sys.exit("train_head.py needs a CUDA device (set NCM_DEVICE=cuda:N); it will not train on the CPU")
CAP, OUT, MODES = ha.args[0], ha.args[1], ha.args[2].split(",")
STEPS = int(ha.args[3]) if len(ha.args) > 3 else 800
BATCH = int(ha.args[4]) if len(ha.args) > 4 else 2
METHODS = [("A", 1), ("A", 4), ("B", 1), ("B", 4), ("lin", 0)]
if os.environ.get("NCM_METHODS"):  # e.g. "A4,B4,lin" -- the full list is slow on a CPU
    METHODS = [(m.rstrip("0123456789"), int(m[len(m.rstrip("0123456789")):] or 0)) for m in os.environ["NCM_METHODS"].split(",")]
SEEDS = range(int(os.environ.get("NCM_NSEEDS", 3)))


def build(method, K, ztr, P):
    """-> (recon: z [N,T,R] -> (x_hat [N,T,2560] standardised, commitment loss), parameters)"""
    D = ztr.shape[-1]
    if method == "lin":
        dec = torch.nn.Linear(D, P.shape[0], bias=False).to(DEV)
        with torch.no_grad():
            dec.weight.copy_(P)
        return (lambda z: (dec(z), 0.0)), list(dec.parameters())
    def quantize(rvq, z):
        """Value = the quantised z. ResidualVQ.forward alone is straight-through, so no loss except its own quantisation loss would ever reach the
        codebooks (and train_ms2's 'nmse' training was exactly that). Adding decode() - decode().detach() sends the task gradient to the codebooks too."""
        zq, codes, ql = rvq(z)
        dec = rvq.decode(codes)
        return zq + dec - dec.detach(), ql

    if method == "A":
        rvq = ResidualVQ(D, K, t.CODES).to(DEV)
        t.init_books(rvq, ztr)
        return (lambda z: ((lambda o: (o[0] @ P.T, o[1]))(quantize(rvq, z)))), list(rvq.parameters())
    fb, rs, rd = LiftingFilterBank(D).to(DEV), ResidualVQ(D, K, t.CODES).to(DEV), ResidualVQ(D, K, t.CODES).to(DEV)
    with torch.no_grad():
        s, d = fb.analyze(ztr)
        t.init_books(rs, s); t.init_books(rd, d)

    def recon(z):
        s, d = fb.analyze(z)
        sh, qs = quantize(rs, s)
        dh, qd = quantize(rd, d)
        return fb.synthesize(sh, dh, z.shape[1]) @ P.T, qs + qd
    return recon, list(fb.parameters()) + list(rs.parameters()) + list(rd.parameters())


def main():
    torch.manual_seed(0)
    xtr, xte, P = t.prep(CAP)
    rtr, rte, nxt = ha.test_windows(CAP)
    rtr, rte, nxt = torch.tensor(rtr), torch.tensor(rte), torch.tensor(nxt.reshape(-1), device=DEV)
    sm = rtr.mean(2)
    mu, sd = sm.mean((0, 1)).to(DEV), sm.std((0, 1)).clamp_min(1e-3).to(DEV)
    assert torch.allclose((rte.mean(2).to(DEV) - mu) / sd, xte, atol=1e-4), "split / statistics do not match train_ms2.prep"
    ztr, zte = xtr @ P, xte @ P
    head = TeacherHead(DEV)
    print(f"split seed {os.environ.get('NCM_SPLIT_SEED', '(manifest)')}: train windows {len(xtr)}, test windows {len(xte)}, steps {STEPS}, batch {BATCH} windows", flush=True)

    def rows(xh, r):  # r: true rows [B,T,hc,n_embd] on DEV -> reconstructed rows [B*T,hc,n_embd]
        return (r - r.mean(2, keepdim=True) + (xh * sd + mu)[:, :, None]).flatten(0, 1)

    @torch.no_grad()
    def evaluate(recon):
        xh, _ = recon(zte)
        ok = 0
        for i in range(0, len(xte), 4):
            am = head(rows(xh[i:i + 4], rte[i:i + 4].to(DEV))).argmax(-1)
            ok += int((am == nxt[i * t.T:(i + 4) * t.T]).sum())
        return ok / nxt.numel(), t.nmse(xh, xte)

    results = []
    for mode in MODES:
        for method, K in METHODS:
            res = []
            for seed in SEEDS:
                torch.manual_seed(seed)
                recon, params = build(method, K, ztr, P)
                opt = torch.optim.Adam(params, 1e-3)
                for _ in range(STEPS):
                    idx = torch.randint(0, len(xtr), (BATCH,))
                    xh, ql = recon(ztr[idx])
                    loss = ql + (((xh - xtr[idx]) ** 2).mean() if mode != "head" else 0.0)
                    if mode != "nmse":
                        r = rtr[idx].to(DEV)
                        with torch.no_grad():
                            tl = F.log_softmax(head(r.flatten(0, 1)), -1)
                        loss = loss + F.kl_div(F.log_softmax(head(rows(xh, r)), -1), tl, log_target=True, reduction="batchmean")
                    opt.zero_grad(); loss.backward(); opt.step()
                res.append(evaluate(recon))
            a = np.array(res)
            results.append(dict(mode=mode, method=method, K=K, agree=[float(a[:, 0].mean()), float(a[:, 0].std())],
                                test_nmse=[float(a[:, 1].mean()), float(a[:, 1].std())]))
            print(f"{mode:5s} {method:3s} K={K}  top-1 agreement {a[:,0].mean():.4f}±{a[:,0].std():.4f}  test NMSE {a[:,1].mean():.3f}±{a[:,1].std():.3f}", flush=True)
            json.dump(dict(steps=STEPS, batch_windows=BATCH, split=os.environ.get("NCM_SPLIT_SEED", "manifest"), results=results), open(OUT, "w"), indent=1)


if __name__ == "__main__":
    main()
