#!/usr/bin/env python3
"""Layer-by-layer weight provenance of candidate teacher-derived students (research/weight_provenance.md).
Parameter counts come from the GGUF headers (shapes only). Provenance categories are DESIGN CLAIMS about what each
candidate would do to a tensor; nothing here was trained or run. High reuse % is not evidence of preserved knowledge.
Categories: C copied unchanged, D derived deterministically, M mapped into a changed layout, L learned, X removed."""
import json, re, csv, collections
import numpy as np

H = json.load(open('research/data/teacher_gguf_headers.json'))

def group(r):
    if r.startswith('per_layer_token_embd'): return 'ple_table'
    if 'exps' in r: return 'experts'
    if r.startswith('hc_') or r.startswith('output_hc'): return 'hyper_conn'
    if 'ffn_gate_inp' in r and 'shexp' not in r: return 'router'
    if 'shexp' in r: return 'shared_expert'
    if r.startswith('ssm') or r in ('attn_gate.weight', 'attn_qkv.weight'): return 'mixer_deltanet'
    if r.startswith('attn_') or r.startswith('indexer'): return 'mixer_attention'
    if r.startswith('ple_'): return 'ple_proj'
    return 'embed_head'

# params[layer or 'global'][group]
P = collections.defaultdict(lambda: collections.defaultdict(int))
for h in H:
    for t in h['tensors']:
        m = re.match(r'blk\.(\d+)\.(.*)', t['name'])
        n = int(np.prod(t['shape']))
        if m and 'ple_' not in m.group(2):  # blk.N.ple_* tensors exist once (global), stored under a block prefix
            P[int(m.group(1))][group(m.group(2))] += n
        else:
            P['global'][group(m.group(2) if m else t['name'])] += n
total = sum(v for d in P.values() for v in d.values())

# category per candidate and group: (fraction_kept_category, ...) -> list of (category, share)
def cand(name):
    base = {g: [('C', 1.0)] for g in ('ple_table', 'experts', 'hyper_conn', 'router', 'shared_expert', 'mixer_deltanet', 'mixer_attention', 'ple_proj', 'embed_head')}
    if name in ('S1', 'S2', 'S3'): base['hyper_conn'] = [('D', 1.0)]
    if name in ('S2', 'S3'): base['router'] = [('L', 1.0)]
    if name == 'S3': base['experts'] = [('M', 192 / 256), ('X', 64 / 256)]  # survivors restacked into a 192-expert layout
    return base

NAMES = {'S0': 'teacher, placement/kernels only', 'S1': 'S0 + hc re-quantised', 'S2': 'S1 + router fine-tune',
         'S3': 'S2 + experts 256->192 per layer'}
rows, tot = [], {s: collections.Counter() for s in NAMES}
for layer in list(range(48)) + ['global']:
    for s in NAMES:
        c = collections.Counter()
        for g, n in P[layer].items():
            for cat, share in cand(s)[g]: c[cat] += n * share
        tot[s].update(c)
        rows.append([layer, s] + [int(c.get(k, 0)) for k in 'CDMLX'] + [int(sum(P[layer].values()))])
with open('research/data/provenance_by_layer.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['layer', 'candidate', 'C', 'D', 'M', 'L', 'X', 'teacher_params']); w.writerows(rows)

print('layer-kind parameter share (B): DeltaNet layer %.3f, attention layer %.3f, global %.3f' % (
    sum(P[0].values()) / 1e9, sum(P[3].values()) / 1e9, sum(P['global'].values()) / 1e9))
for k in (0, 3): print('  layer', k, {g: round(v / 1e9, 4) for g, v in P[k].items()})
print('  global', {g: round(v / 1e9, 3) for g, v in P['global'].items()})
print('TOTAL %.3f B' % (total / 1e9))
for s, c in tot.items():
    assert abs(sum(c.values()) - total) < 1, (s, sum(c.values()), total)
    print('\n%s  %s' % (s, NAMES[s]))
    for k in 'CDMLX':
        if c.get(k): print('   %s %10.3f B  %6.2f%%' % (k, c[k] / 1e9, 100 * c[k] / total))
    # share of the whole teacher that is neither learned nor removed: params kept bit-exact (C) or by value (M)
    print('   reuse by value (C+M): %.2f%%' % (100 * (c.get('C', 0) + c.get('M', 0)) / total))
json.dump({s: dict(c) for s, c in tot.items()} | {'total': total}, open('research/data/provenance.json', 'w'), indent=1)
