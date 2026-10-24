#!/usr/bin/env python3
"""Layer/role-wise weight provenance of candidate teacher-derived students (research/weight_provenance.md).
Parameter counts come from the GGUF headers (shapes only). Provenance categories are DESIGN CLAIMS about what each
candidate would do to a tensor; nothing here was trained or run. High reuse % is not evidence of preserved knowledge."""
import json, re, collections
import numpy as np

H = json.load(open('research/data/teacher_gguf_headers.json'))
role_p = collections.OrderedDict()
for h in H:
    for t in h['tensors']:
        r = re.sub(r'blk\.\d+\.', '', t['name'])
        role_p[r] = role_p.get(r, 0) + int(np.prod(t['shape']))
total = sum(role_p.values())

def group(r):
    if r.startswith('per_layer_token_embd'): return 'PLE n-gram table'
    if 'exps' in r: return 'routed experts (256/layer)'
    if r.startswith('hc_') or r.startswith('output_hc'): return 'hyper-connection (BF16)'
    if 'ffn_gate_inp' in r and 'shexp' not in r: return 'MoE router'
    if 'shexp' in r: return 'shared expert (+gate)'
    if r.startswith('ssm') : return 'DeltaNet mixer'
    if r in ('attn_gate.weight', 'attn_qkv.weight'): return 'DeltaNet mixer'
    if r.startswith('attn_') or r.startswith('indexer'): return 'sparse attention + indexer'
    if r.startswith('ple_'): return 'PLE query/key/value projections'
    if r in ('token_embd.weight', 'output.weight'): return 'embedding / LM head'
    return 'other'

g = collections.OrderedDict()
for r, p in role_p.items(): g[group(r)] = g.get(group(r), 0) + p

# provenance per candidate: C=copied unchanged, D=derived deterministically (no training), M=mapped into changed shape, L=learned
CAND = {
 'S0 optimised teacher (placement/kernels only)': {k: 'C' for k in g},
 'S1 = S0 + hc re-quantised (Q8/Q6)':            {**{k: 'C' for k in g}, 'hyper-connection (BF16)': 'D'},
 'S2 = S1 + router & MTP-head fine-tune':        {**{k: 'C' for k in g}, 'hyper-connection (BF16)': 'D', 'MoE router': 'L'},
 'S3 = S2 + experts pruned 256->192 (structural)': {**{k: 'C' for k in g}, 'hyper-connection (BF16)': 'D', 'MoE router': 'L'},
}
print('%-34s %10s %7s' % ('group', 'params (B)', '%'))
for k, p in sorted(g.items(), key=lambda x: -x[1]): print('%-34s %10.3f %6.2f%%' % (k, p / 1e9, 100 * p / total))
print('%-34s %10.3f' % ('TOTAL', total / 1e9))
res = {'groups': g, 'total': total}
for name, m in CAND.items():
    tally = collections.Counter()
    for k, p in g.items():
        c = m[k]
        if name.startswith('S3') and k == 'routed experts (256/layer)':
            tally['C'] += p * 192 / 256; tally['pruned (removed)'] += p * 64 / 256; continue
        tally[c] += p
    print('\n' + name)
    for c, p in tally.items(): print('   %-18s %10.3f B  %6.2f%%' % (c, p / 1e9, 100 * p / total))
    res[name] = {c: p for c, p in tally.items()}
# MTP head is a separate artifact (not in the 116.5 B teacher file): dense 116 MB + experts 708 MB on disk.
json.dump(res, open('research/data/provenance.json', 'w'), indent=1)
