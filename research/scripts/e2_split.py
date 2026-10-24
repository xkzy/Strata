#!/usr/bin/env python3
"""Wait split + kernel-time breakdown for an nsys sqlite export. Waits are labelled by the kernel that follows them
(A: copy_i32_from_mapped, B: fetch_blobs, C: copy_rows_from_mapped). Prefill/decode boundary = first kernel + the
request's server-side prefill_ms. Usage: e2_split.py trace.sqlite server.out"""
import re, sqlite3, sys, collections, statistics as st
db = sqlite3.connect(sys.argv[1])
rows = db.execute('''select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k
                     join StringIds s on s.id=k.demangledName order by k.start, k.end''').fetchall()
P = re.compile(r'prompt (\d+) tokens = \d+ reused \+ \d+ read in (\d+) ms .*?(\d+) generated in (\d+) ms \(([\d.]+) tok/s\)')
m = [P.search(l) for l in open(sys.argv[2])]; m = [x for x in m if x and int(x.group(3)) >= 200][-1]
prefill_ms = int(m.group(2)); print('request: prompt', m.group(1), 'prefill_ms', prefill_ms, 'generated', m.group(3), 'in', m.group(4), 'ms =', m.group(5), 'tok/s')
t0 = rows[0][0]; tb = t0 + prefill_ms * 1e6
def kind(nm):
    for key, lab in (('copy_i32_from_mapped', 'A'), ('fetch_blobs', 'B'), ('copy_rows_from_mapped', 'C'), ('copy_from_mapped', 'C')):
        if key in nm: return lab
waits = collections.defaultdict(list)
for i, r in enumerate(rows):
    if 'wait_flag_ge' not in r[2]: continue
    lab = None
    for j in range(i + 1, min(i + 7, len(rows))):
        if 'wait_flag_ge' in rows[j][2]: continue
        lab = kind(rows[j][2])
        if lab: break
    waits[('decode' if r[0] > tb else 'prefill', lab or '?')].append(r[1] - r[0])
def union(iv):          # GPU-busy time: union of kernel intervals
    tot, end = 0, 0
    for a, b in sorted(iv):
        if b > end: tot += b - max(a, end); end = b
    return tot
for ph in ('prefill', 'decode'):
    sel = [r for r in rows if (r[0] > tb) == (ph == 'decode')]
    span = sel[-1][1] - sel[0][0]; tot = sum(r[1] - r[0] for r in sel)
    nonwait = [(r[0], r[1]) for r in sel if 'wait_flag_ge' not in r[2]]
    print(f'{ph}: kernels {len(sel)} span {span/1e6:.0f} ms, sum kernel time {tot/1e6:.0f} ms, GPU busy excluding waits {union(nonwait)/1e6:.0f} ms ({100*union(nonwait)/span:.0f}% of span)')
    for lab in 'ABC?':
        d = waits.get((ph, lab), [])
        if d: print(f'  wait {lab}: n={len(d)} sum={sum(d)/1e6:.0f} ms ({100*sum(d)/tot:.1f}% of kernel time, {100*sum(d)/span:.0f}% of span) '
                    f'median={st.median(d)/1e3:.1f} us p90={sorted(d)[int(.9*len(d))]/1e3:.1f} us max={max(d)/1e3:.0f} us')
    by = collections.Counter()
    for r in sel: by[re.sub(r'\(.*', '', r[2]).split('::')[-1][:40]] += r[1] - r[0]
    print('  top kernels (% of kernel time):', ', '.join(f'{k} {100*v/tot:.1f}' for k, v in by.most_common(6)))
