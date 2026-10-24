#!/usr/bin/env python3
"""Parse Strata server log lines from the E0 run into per-size statistics (server-side timings, not client timings).
Usage: e0_parse.py server.log out.json"""
import re, sys, json, statistics as st
P = re.compile(r'strata serve: prompt (\d+) tokens = \d+ reused \+ \d+ read in (\d+) ms \(([\d.]+) tok/s\), (\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)')
H = re.compile(r'decode expert cache hit rate: ([\d.]+)%')
rows, hit = [], None
for l in open(sys.argv[1]):
    m = P.search(l)
    if m: rows.append(dict(zip(('prompt', 'prefill_ms', 'prefill_tps', 'gen', 'gen_ms', 'decode_tps', 'acc', 'drafts'), map(float, m.groups())))); continue
    m = H.search(l)
    if m and rows and 'hit' not in rows[-1]: rows[-1]['hit'] = float(m.group(1))
rows = [r for r in rows if r['gen'] >= 200]          # drops the 84-token and 128-token smoke requests
groups = {}
for r in rows: groups.setdefault(round(r['prompt'], -2), []).append(r)
out = {}
for k, g in sorted(groups.items()):
    g = g[1:]                                         # first request of each size is the warm-up
    f = lambda key: [r[key] for r in g if key in r]
    d = f('decode_tps')
    out[int(k)] = {'n': len(g), 'prefill_tps_mean': st.mean(f('prefill_tps')), 'decode_tps_mean': st.mean(d),
                   'decode_tps_sd': st.stdev(d) if len(d) > 1 else None, 'decode_tps_min': min(d), 'decode_tps_max': max(d),
                   'accept_mean': st.mean(r['acc'] / r['drafts'] for r in g), 'hit_mean': st.mean(f('hit')), 'rows': g}
    o = out[int(k)]
    print(f"~{int(k)} tok  n={o['n']}  prefill {o['prefill_tps_mean']:.0f} tok/s  decode {o['decode_tps_mean']:.1f} (sd {o['decode_tps_sd']:.1f}, {o['decode_tps_min']:.1f}-{o['decode_tps_max']:.1f}, CV {100*o['decode_tps_sd']/o['decode_tps_mean']:.0f}%)  accept {o['accept_mean']:.2f}  hit {o['hit_mean']:.0f}%")
json.dump(out, open(sys.argv[2], 'w'), indent=1)
