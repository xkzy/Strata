#!/usr/bin/env python
"""NCM stage-1 tokenizer on TEACHER tokens (paper arXiv 2610.14162, as summarised in research/ncm_algorithm_analysis.md), teacher-derived init.
  z      = Ez[tok] + wpe             Ez = PCA(d) of the teacher's token_embd (IQ4_XS), learnable afterwards; wpe learned (zeros at init)
  level k: res = z - zhat^(k-1);  u = avgpool(res, l_k);  e = nearest codebook_k entry (EMA codebook, dead-code restart);
           zhat^(k) = zhat^(k-1) + linear_upsample(u + (e-u).detach(), L)           (straight-through)
  decoder: learned pos + 1 transformer layer (bidirectional) + Linear -> logits = h @ Oz^T * s;  Oz = PCA(d) of the teacher's output.weight (Q6_K), frozen
  loss   : sum_k CE(decode(zhat^(k)))  + CE(decode(z)) (no-quant reference) + 0.25 * commitment.
Scored on held-out prompts, full 248,320-token vocabulary, argmax accuracy per level; bits/token = sum_{j<=k} l_j*log2(C) / L.
Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID ncm_tokenizer.py CAP_DIR OUT.json C [steps=3000] [device=cuda:1]"""
import sys, json, math, time, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here); sys.path.insert(0, here.rsplit('/', 1)[0] + '/ncm')
import wavelet_stats as ws          # generic GGUF tensor loader (reads headers only at import)
from capture import load_spec_residuals

cap, outp, C = sys.argv[1], sys.argv[2], int(sys.argv[3])
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 3000
dev = sys.argv[5] if len(sys.argv) > 5 else 'cuda:1'
L, D, LV, BETA, DECAY = 32, 128, [1, 2, 4, 8, 16, 32], 0.25, 0.99
torch.manual_seed(0); np.random.seed(0)

def pca_rows(name):
    W, qt = ws.load(name)                       # [vocab, 2560] float32 (CPU)
    X = torch.from_numpy(W).to(dev); del W
    X -= X.mean(0, keepdim=True)                # centre in place (no second copy: the card has 8 GB)
    tot = float(torch.linalg.norm(X)) ** 2
    U, S, V = torch.pca_lowrank(X, q=D, center=False, niter=3)
    P = X @ V; var = float((S[:D] ** 2).sum()) / tot
    del X, U; torch.cuda.empty_cache()
    return P / P.std(), qt, var                 # unit-variance projected rows [vocab, D]

t0 = time.time()
Ez, qe, ve = pca_rows('token_embd.weight'); Oz, qo, vo = pca_rows('output.weight')
print('PCA', qe, 'energy kept', round(ve, 3), qo, round(vo, 3), '%ds' % (time.time() - t0), flush=True)
V = Ez.shape[0]

# windows of L teacher tokens from the cap2 greedy continuations; split by prompt as in the manifest
man = json.load(open(cap + '/manifest.json')); W = {'train': [], 'test': []}
import glob
for f in sorted(glob.glob(cap + '/final_r_*.bin')):
    k = int(os.path.basename(f)[8:-4]); _, tok, nxt, _ = load_spec_residuals(f)
    s = np.concatenate([tok, nxt[-1:]]); stride = 16 if man[k]['split'] == 'train' else L
    W[man[k]['split']] += [s[i:i + L] for i in range(0, len(s) - L + 1, stride)]
W = {k: torch.from_numpy(np.stack(v).astype(np.int64)).to(dev) for k, v in W.items()}
print({k: tuple(v.shape) for k, v in W.items()}, flush=True)

class Quant(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('embed', torch.zeros(C, D)); self.register_buffer('size', torch.ones(C)); self.register_buffer('avg', torch.zeros(C, D))
        self.register_buffer('ready', torch.zeros(()))
    def forward(self, u):                        # u [B, l, D]
        flat = u.reshape(-1, D).detach()
        if self.training and not self.ready:
            idx = torch.randint(0, len(flat), (C,), device=u.device); self.embed.copy_(flat[idx] if len(flat) >= 1 else self.embed)
            self.avg.copy_(self.embed); self.ready.fill_(1)
        d = flat.pow(2).sum(1, keepdim=True) - 2 * flat @ self.embed.T + self.embed.pow(2).sum(1)
        i = d.argmin(1); e = self.embed[i].view_as(u)
        if self.training:
            with torch.no_grad():
                oh = F.one_hot(i, C).float(); n = oh.sum(0)
                self.size.mul_(DECAY).add_(n, alpha=1 - DECAY); self.avg.mul_(DECAY).add_(oh.T @ flat, alpha=1 - DECAY)
                self.embed.copy_(self.avg / self.size.clamp_min(1e-5).unsqueeze(1))
        return e, i.view(u.shape[:2]), BETA * F.mse_loss(u, e.detach())
    @torch.no_grad()
    def restart(self, u):
        dead = self.size < 0.3
        if dead.any():
            flat = u.reshape(-1, D); r = flat[torch.randint(0, len(flat), (int(dead.sum()),), device=u.device)]
            self.embed[dead] = r; self.avg[dead] = r; self.size[dead] = 1.0
        return int(dead.sum())

class Tok(nn.Module):
    def __init__(self):
        super().__init__()
        self.wte = nn.Embedding.from_pretrained(Ez.clone(), freeze=False); self.wpe = nn.Parameter(torch.zeros(L, D))
        self.q = nn.ModuleList([Quant() for _ in LV])
        self.dpos = nn.Parameter(torch.zeros(L, D)); self.blk = nn.TransformerEncoderLayer(D, 4, 4 * D, 0.0, batch_first=True, norm_first=True)
        self.us = [None] * len(LV); self.out = nn.Linear(D, D); self.s = nn.Parameter(torch.tensor(8.0)); self.register_buffer('Oz', Oz)
    def decode(self, zh):
        return self.out(self.blk(zh + self.dpos)) @ self.Oz.T * (self.s / math.sqrt(D))      # [B, L, V]
    def encode(self, tok):
        z = self.wte(tok) + self.wpe; zh = torch.zeros_like(z); cum = []; commit = 0; codes = []
        for k, l in enumerate(LV):
            u = F.adaptive_avg_pool1d((z - zh).transpose(1, 2), l).transpose(1, 2)
            e, i, c = self.q[k](u); commit = commit + c; codes.append(i); self.us[k] = u.detach()
            es = u + (e - u).detach()
            zh = zh + (es if l == L else F.interpolate(es.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2))
            cum.append(zh)
        return z, cum, commit, codes

m = Tok().to(dev)
learned = sum(p.numel() for n, p in m.named_parameters())
copied = {'wte (PCA of teacher token_embd, then trained)': Ez.numel(), 'Oz (PCA of teacher output.weight, frozen)': Oz.numel()}
opt = torch.optim.AdamW([p for p in m.parameters()], 1e-3, weight_decay=0.0, betas=(0.9, 0.98))
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=STEPS, pct_start=0.1)
B = 8
for step in range(STEPS):
    m.train(); tok = W['train'][torch.randint(0, len(W['train']), (B,), device=dev)]
    z, cum, commit, _ = m.encode(tok); opt.zero_grad()
    loss_tot = commit.item()
    (commit).backward(retain_graph=True)
    for zh in [z] + cum:
        l = F.cross_entropy(m.decode(zh).reshape(-1, V).float(), tok.reshape(-1)); l.backward(retain_graph=True); loss_tot += l.item()
    opt.step(); sched.step()
    if step % 100 == 99:
        with torch.no_grad():
            dead = [m.q[k].restart(m.us[k]) for k in range(len(LV))]
        print('step', step + 1, 'loss', round(loss_tot, 3), 'dead_restarted', dead, flush=True)

@torch.no_grad()
def evaluate(split):
    m.eval(); n = 0; ok = np.zeros(len(LV) + 1); X = W[split]
    for i in range(0, len(X), 16):
        tok = X[i:i + 16]; z, cum, _, _ = m.encode(tok)
        for j, zh in enumerate([z] + cum): ok[j] += (m.decode(zh).argmax(-1) == tok).sum().item()
        n += tok.numel()
    return ok / n
res = {'C': C, 'L': L, 'D': D, 'levels': LV, 'steps': STEPS, 'windows': {k: len(v) for k, v in W.items()}, 'learned_params_total': learned,
       'pca_energy_kept': {'token_embd': ve, 'output': vo}, 'copied_or_transformed': copied,
       'bits_per_token_cum': [sum(l * math.log2(C) for l in LV[:k + 1]) / L for k in range(len(LV))]}
for sp in ('train', 'test'):
    a = evaluate(sp); res[sp + '_acc_unquantised_z'] = float(a[0]); res[sp + '_acc_by_level'] = a[1:].tolist()
print(json.dumps(res, indent=1))
json.dump(res, open(outp, 'w'), indent=1)
