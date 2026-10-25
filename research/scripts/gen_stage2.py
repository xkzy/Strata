# Stage-2 generator for a FROZEN NCM tokenizer. Executed inside ncm_tokenizer.py (load=CKPT gen=this_file [gensteps=N] [direct]): uses its m, V, D, L, LV, C, Oz, dev, outp.
# Task: given the 32 tokens before a window (prefix), generate the next 32 tokens.
#   code generator: for level k it predicts all l_k codes of the window in ONE pass, from [prefix token embeddings ; cumulative reconstruction of levels < k ; l_k learned queries]
#                   (bidirectional transformer, 6 layers, width 384); trained with teacher forcing on the true lower levels; generation feeds back its own greedy codes, then the
#                   tokenizer's decoder turns the final reconstruction into tokens.
#   direct (reference): same transformer, 32 queries, tokens read through the frozen Oz table: parallel next-32 prediction without any codes.
# Data: build_pairs.py (document-level split). Scored on 4,096 held-out-document pairs: token accuracy of the 32 generated tokens against the true continuation.
import time
GSTEPS = int(next((a[9:] for a in sys.argv if a.startswith('gensteps=')), 4000)); DIRECT = 'direct' in sys.argv; GD = 384
PD = '/home/khing/Downloads/Strata-data/ncm/pairs_w64_'
GP = torch.from_numpy(np.load(PD + 'train.npy').astype(np.int64)).to(dev); GT = torch.from_numpy(np.load(PD + 'test.npy').astype(np.int64)).to(dev)
for p in m.parameters(): p.requires_grad_(False)
m.eval()

def up(e, l): return e if l == L else F.interpolate(e.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
def recon(codes):                        # codes of levels 1..n ([B, l_k] each) -> cumulative reconstruction after each level, from the tokenizer's codebooks
    zh, out = 0, []
    for k, c in enumerate(codes): zh = zh + up(m.q[k].embed[c], LV[k]); out.append(zh)
    return out

class Gen(nn.Module):
    def __init__(s):
        super().__init__()
        s.tin = nn.Linear(D, GD); s.zin = nn.Linear(D, GD); s.pp = nn.Parameter(torch.zeros(L, GD)); s.zp = nn.Parameter(torch.zeros(L, GD))
        s.q = nn.ParameterList([nn.Parameter(torch.randn(l, GD) * 0.02) for l in ([L] if DIRECT else LV)])
        s.blk = nn.TransformerEncoder(nn.TransformerEncoderLayer(GD, 6, 4 * GD, 0.1, batch_first=True, norm_first=True), 6); s.ln = nn.LayerNorm(GD)
        if DIRECT: s.out = nn.Linear(GD, D); s.s = nn.Parameter(torch.tensor(8.0))
        else: s.head = nn.ModuleList([nn.Linear(GD, C) for _ in LV])
    def forward(s, pe, zh, k):           # pe [B, L, D] prefix embeddings, zh [B, L, D] reconstruction so far
        with torch.autocast('cuda', dtype=torch.bfloat16):          # fp32 ran at 0.85 s/step on the 4060
            x = torch.cat([s.tin(pe) + s.pp, s.zin(zh) + s.zp, s.q[0 if DIRECT else k].expand(len(pe), -1, -1)], 1)
            h = s.ln(s.blk(x))[:, 2 * L:]
            return (s.out(h) @ Oz.T * (s.s / math.sqrt(D)) if DIRECT else s.head[k](h)).float()

gen = Gen().to(dev); nparam = sum(p.numel() for p in gen.parameters())
opt = torch.optim.AdamW(gen.parameters(), 5e-4, weight_decay=0.01); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=GSTEPS, pct_start=0.1)
GBS, nl = (64 if DIRECT else 128), (1 if DIRECT else len(LV)); acc_l = np.zeros(nl); t0 = time.time()
with torch.no_grad():                    # sanity: the codebook reconstruction equals the tokenizer's own cumulative reconstruction
    _, cum, _, codes = m.encode(GT[:8, L:])
    if not DIRECT: assert max(float((a - b).abs().max()) for a, b in zip(recon(codes), cum)) < 1e-4
for step in range(GSTEPS):
    gen.train(); pair = GP[torch.randint(0, len(GP), (GBS,), device=dev)]; pre, tgt = pair[:, :L], pair[:, L:]
    with torch.no_grad():
        pe = m.wte(pre)
        if not DIRECT: _, cum, _, codes = m.encode(tgt)
    opt.zero_grad()
    if DIRECT:
        for i in range(0, GBS, 16):
            l = F.cross_entropy(gen(pe[i:i + 16], torch.zeros_like(pe[i:i + 16]), 0).reshape(-1, V).float(), tgt[i:i + 16].reshape(-1)) * 16 / GBS; l.backward(); acc_l[0] += l.item()
    else:
        for k in range(nl):
            l = F.cross_entropy(gen(pe, torch.zeros_like(pe) if k == 0 else cum[k - 1], k).reshape(-1, C), codes[k].reshape(-1)); (l / nl).backward(); acc_l[k] += l.item()
    torch.nn.utils.clip_grad_norm_(gen.parameters(), 1.0); opt.step(); sched.step()
    if step % 200 == 199: print('gen step', step + 1, 'CE per level (nats)', [round(x / 200, 3) for x in acc_l], '%ds' % (time.time() - t0), flush=True); acc_l[:] = 0

@torch.no_grad()
def score():
    gen.eval(); EB = 32; n = 0; tf_ok = np.zeros(nl); tf_ce = np.zeros(nl); e2e = np.zeros(nl); orc = 0
    for i in range(0, len(GT), EB):
        pair = GT[i:i + EB]; pre, tgt = pair[:, :L], pair[:, L:]; pe = m.wte(pre); b = tgt.numel()
        if DIRECT:
            e2e[0] += (gen(pe, torch.zeros_like(pe), 0).argmax(-1) == tgt).sum().item(); n += b; continue
        _, cum, _, codes = m.encode(tgt); orc += (m.decode(cum[-1]).argmax(-1) == tgt).sum().item(); hat = []; zh = torch.zeros_like(pe)
        for k in range(nl):
            lg = gen(pe, torch.zeros_like(pe) if k == 0 else cum[k - 1], k)                     # teacher-forced
            tf_ok[k] += (lg.argmax(-1) == codes[k]).sum().item() / codes[k].numel() * b; tf_ce[k] += F.cross_entropy(lg.reshape(-1, C), codes[k].reshape(-1), reduction='mean').item() * b
            hat.append(gen(pe, zh, k).argmax(-1)); zh = recon(hat)[-1]                         # free-running: own greedy codes fed back
            e2e[k] += (m.decode(zh).argmax(-1) == tgt).sum().item()
        n += b
    return {'tokens_scored': n, 'token_acc_after_level': (e2e / n).tolist(), 'oracle_tokenizer_acc': orc / n,
            'teacher_forced_code_acc_by_level': (tf_ok / n).tolist(), 'teacher_forced_code_bits_by_level': (tf_ce / n / math.log(2)).tolist()}
torch.save(gen.state_dict(), outp[:-5] + '_gen.pt')     # saved before scoring: a crash in the evaluation (GPU fault on 2026-10-25) cost a full 4000-step run
gres = {'mode': 'direct' if DIRECT else 'codes', 'levels': LV, 'C': C, 'gen_params': nparam, 'steps': GSTEPS, 'batch': GBS, 'train_pairs': len(GP), 'test_pairs': len(GT),
        'sequential_passes': 1 if DIRECT else len(LV), 'train_seconds': time.time() - t0, 'ckpt': LOAD}
gres.update(score()); print(json.dumps(gres, indent=1)); json.dump(gres, open(outp, 'w'), indent=1)
