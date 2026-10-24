#!/usr/bin/env python
"""Ceiling pilot (heads-only block drafts). From the teacher's final state at position t, can a small learned map + the FROZEN teacher head
reproduce the teacher's own greedy token j steps later (nxt[t+j], j=0 is the head itself)?
  A: input = mix(h_t)                           (state only)
  B: input = [mix(h_t), out_row(nxt[t])]        (state + output-embedding row of the already-known next token; no new vocab weights)
Map = residual MLP (x + W2 gelu(W1 x)), W2 init 0, trained with CE through the frozen head on the train prompts; scored on held-out prompts.
Also: accepted-prefix length over j=1..J (greedy chain accepted while every step matches).
Usage: CUDA_DEVICE_ORDER=PCI_BUS_ID ceiling_pilot.py CAP_DIR OUT.json [J=8] [epochs=6] [device=cuda:1]"""
import sys, json, time
import numpy as np, torch
sys.path.insert(0, __file__.rsplit('/', 2)[0] + '/ncm')
from dataset import load_sequences
from head import TeacherHead

cap, outp = sys.argv[1], sys.argv[2]
J = int(sys.argv[3]) if len(sys.argv) > 3 else 8
EP = int(sys.argv[4]) if len(sys.argv) > 4 else 6
dev = sys.argv[5] if len(sys.argv) > 5 else 'cuda:1'
torch.manual_seed(0); torch.backends.cuda.matmul.allow_tf32 = True
man = json.load(open(cap + '/manifest.json')); seqs = load_sequences(cap)
head = TeacherHead(dev)
S = {'train': [], 'test': []}
with torch.no_grad():
    for k, s in sorted(seqs.items()):
        R = torch.from_numpy(s['rows']).to(dev)
        X = torch.cat([head.mix(R[i:i + 512]) for i in range(0, len(R), 512)])
        S[man[int(k)]['split']].append((X, torch.from_numpy(s['nxt'].astype(np.int64)).to(dev)))
print({k: sum(len(x) for x, _ in v) for k, v in S.items()}, flush=True)
D = 2560

def pairs(split, j):
    xs, ts, ys = [], [], []
    for X, n in S[split]:
        m = len(X) - j
        if m <= 0: continue
        xs.append(X[:m]); ts.append(n[:m]); ys.append(n[j:j + m])
    return torch.cat(xs), torch.cat(ts), torch.cat(ys)

class Map(torch.nn.Module):
    def __init__(self, din):
        super().__init__()
        self.inp = torch.nn.Linear(din, D, bias=False) if din != D else torch.nn.Identity()
        self.l1, self.l2 = torch.nn.Linear(D, D), torch.nn.Linear(D, D)
        torch.nn.init.zeros_(self.l2.weight); torch.nn.init.zeros_(self.l2.bias)
        if din != D:
            with torch.no_grad(): self.inp.weight.zero_(); self.inp.weight[:, :D] = torch.eye(D)
    def forward(self, x):
        x = self.inp(x); return x + self.l2(torch.nn.functional.gelu(self.l1(x)))

def feats(variant, x, t): return x if variant == 'A' else torch.cat([x, head.w_out[t]], 1)

def logits(h): return h @ head.w_out.T

res = {'J': J, 'epochs': EP, 'n_train_pairs': None, 'rows': []}
pred_test = {v: {} for v in 'AB'}
for variant in 'AB':
    for j in range(0, J + 1):
        xtr, ttr, ytr = pairs('train', j); xte, tte, yte = pairs('test', j)
        if j == 0 and variant == 'A':  # frozen head, no learning: the j=0 control
            with torch.no_grad(): p = torch.cat([logits(xte[i:i + 256]).argmax(1) for i in range(0, len(xte), 256)])
            acc = (p == yte).float().mean().item(); res['rows'].append({'variant': 'A', 'j': 0, 'test_acc': acc, 'train_acc': None, 'learned': False})
            pred_test['A'][0] = p.cpu(); print('A j=0 frozen head', round(acc, 4), flush=True); continue
        if j == 0:
            pred_test['B'][0] = pred_test['A'][0]; continue
        m = Map(D if variant == 'A' else 2 * D).to(dev); opt = torch.optim.AdamW(m.parameters(), 3e-4, weight_decay=0.01)
        n = len(xtr); steps = EP * (n // 256); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=steps)
        t0 = time.time()
        for st in range(steps):
            idx = torch.randint(0, n, (256,), device=dev)
            loss = torch.nn.functional.cross_entropy(logits(m(feats(variant, xtr[idx], ttr[idx]))), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        with torch.no_grad():
            def acc_of(x, t, y):
                p = torch.cat([logits(m(feats(variant, x[i:i + 256], t[i:i + 256]))).argmax(1) for i in range(0, len(x), 256)])
                return (p == y).float().mean().item(), p.cpu()
            tr, _ = acc_of(xtr[:4096], ttr[:4096], ytr[:4096]); te, p = acc_of(xte, tte, yte)
        pred_test[variant][j] = p
        res['rows'].append({'variant': variant, 'j': j, 'test_acc': te, 'train_acc_first4096': tr, 'final_loss': loss.item(), 'secs': time.time() - t0, 'learned_params': sum(q.numel() for q in m.parameters())})
        print(variant, 'j=%d' % j, 'test', round(te, 4), 'train', round(tr, 4), 'loss', round(loss.item(), 3), '%ds' % (time.time() - t0), flush=True)
        json.dump(res, open(outp, 'w'), indent=1)
# accepted-prefix length per test position t: how many consecutive j=1..J match (needs t+J in range; same positions per split sequence tails dropped)
for variant in 'AB':
    ok = []
    for j in range(1, J + 1):
        # correctness vector aligned to the t index of pairs(test, j); truncate to the common length of pairs(test, J)
        _, _, y = pairs('test', j); ok.append((pred_test[variant][j] == y.cpu()))
    # sequences have different tails; recompute alignment exactly per sequence
    L = []
    off = {j: 0 for j in range(1, J + 1)}
    for X, n in S['test']:
        cnt = {}
        for j in range(1, J + 1):
            m = len(X) - j
            cnt[j] = ok[j - 1][off[j]:off[j] + m]; off[j] += m
        mm = len(X) - J
        if mm <= 0: continue
        run = torch.ones(mm, dtype=torch.bool); plen = torch.zeros(mm)
        for j in range(1, J + 1):
            run = run & cnt[j][:mm]; plen += run.float()
        L.append(plen)
    L = torch.cat(L)
    res['prefix_' + variant] = {'mean_accepted_prefix_of_%d' % J: L.mean().item(), 'tokens_per_pass_if_verified': 1 + L.mean().item(),
                                'frac_full_block': (L == J).float().mean().item(), 'n': len(L)}
    print(variant, res['prefix_' + variant], flush=True)
json.dump(res, open(outp, 'w'), indent=1)
