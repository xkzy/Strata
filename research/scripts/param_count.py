#!/usr/bin/env python3
"""Count teacher parameters per role from the tensor shapes in teacher_gguf_headers.json.
Roles are grouped by stripping the layer index (blk.N.) from the tensor name."""
import json, re, sys, collections

d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else 'research/data/teacher_gguf_headers.json'))
tens = [t for shard in d for t in shard['tensors']]
role = collections.OrderedDict()
layers = collections.defaultdict(set)
for t in tens:
    n = 1
    for s in t['shape']:
        n *= s
    m = re.match(r'blk\.(\d+)\.(.*)', t['name'])
    r = m.group(2) if m else t['name']
    if m:
        layers[r].add(int(m.group(1)))
    e = role.setdefault(r, {'params': 0, 'count': 0, 'shapes': set(), 'types': set()})
    e['params'] += n
    e['count'] += 1
    e['shapes'].add(tuple(t['shape']))
    e['types'].add(t['type'])
tot = sum(e['params'] for e in role.values())
print(f'tensors={len(tens)} total_params={tot:,} ({tot/1e9:.3f} B)')
for r, e in sorted(role.items(), key=lambda x: -x[1]['params']):
    ls = sorted(layers.get(r, []))
    lay = f'{len(ls)} layers' if ls else 'global'
    print(f"{e['params']/1e9:9.4f}B {r:42s} x{e['count']:<3d} {lay:10s} types={sorted(e['types'])} shapes={sorted(e['shapes'])[:3]}")
