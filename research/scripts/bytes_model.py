#!/usr/bin/env python3
"""Exact bytes per tensor role from GGUF tensor types (bits/weight per ggml type), and bytes READ per decoded token.
Type table: standard ggml block formats; type 42 = Strata private Q2_0 (18 bytes / 64 values, import-ollama-coder.sh)."""
import json, math, re, collections
BITS = {0: 32, 1: 16, 8: 8.5, 12: 4.5, 13: 5.5, 14: 6.5625, 18: 3.0625, 20: 4.5, 21: 3.4375, 22: 2.5625, 23: 4.25, 30: 16, 42: 2.25}
H = json.load(open('research/data/teacher_gguf_headers.json'))
tens = [t for s in H for t in s['tensors']]
E, K = 256, 10
role = collections.defaultdict(lambda: dict(params=0, bytes=0.0, types=collections.Counter()))
for t in tens:
    n = math.prod(t['shape']); b = n * BITS[t['type']] / 8
    r = re.sub(r'^blk\.\d+\.', '', t['name'])
    d = role[r]; d['params'] += n; d['bytes'] += b; d['types'][t['type']] += 1
tot = sum(d['bytes'] for d in role.values())
print(f'total stored bytes (model) = {tot/1e9:.2f} GB ; shard-1 on disk 29.61 GB, shard-2 28.80 GB => sum {29.608+28.800:.2f} GB')
EXP = {r for r in role if r.endswith('_exps.weight')}
SKIP = {'per_layer_token_embd.weight', 'token_embd.weight'}   # gathers: a few rows per token
active = collections.OrderedDict()
for r, d in role.items():
    if r in SKIP: continue
    active[r] = d['bytes'] * (K / E if r in EXP else 1.0)
A = sum(active.values())
print(f'bytes read per decoded token (non-gather) = {A/1e9:.3f} GB ; of which routed experts {sum(v for r,v in active.items() if r in EXP)/1e9:.3f} GB, dense/other {sum(v for r,v in active.items() if r not in EXP)/1e9:.3f} GB')
print('\nTop dense (non-expert) byte streams per token:')
dense = sorted(((v, r) for r, v in active.items() if r not in EXP), reverse=True)
for v, r in dense[:12]:
    d = role[r]; print(f'  {v/1e6:9.1f} MB  {r:34s} params {d["params"]/1e9:7.4f} B  bits/w {d["bytes"]*8/d["params"]:5.2f}  types {dict(d["types"])}')
hc = sum(v for r, v in active.items() if r.startswith(('hc_', 'output_hc')))
print(f'\nhyper-connection tensors: {hc/1e6:.1f} MB/token = {hc/A:.1%} of all bytes read, {hc/sum(v for r,v in active.items() if r not in EXP):.1%} of dense bytes; params {sum(d["params"] for r,d in role.items() if r.startswith(("hc_","output_hc")))/1e9:.3f} B')
eb = sum(role[r]['bytes'] for r in EXP) / (48 * E)
print(f'average bytes per expert (gate+up+down) = {eb/1e6:.3f} MB ; per token routed reads = {K*48*eb/1e9:.3f} GB')
for bw, nm in ((448e9, '5060Ti 448GB/s'), (272e9, '4060 272GB/s')):
    dn = sum(v for r, v in active.items() if r not in EXP)
    print(f'GPU-side floor if dense bytes resident on {nm}: {dn/bw*1e3:5.2f} ms/pass ; if routed experts too: {A/bw*1e3:5.2f} ms')
json.dump({r: dict(params=d['params'], bytes=d['bytes'], types=dict(d['types'])) for r, d in role.items()}, open('research/data/bytes_by_role.json', 'w'), indent=1)
