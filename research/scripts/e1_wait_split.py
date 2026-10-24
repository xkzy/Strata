import sqlite3, collections, statistics as st
db = sqlite3.connect('/home/khing/.claude/jobs/189fb9b9/tmp/e1_decode.sqlite')
rows = db.execute('''select k.start, k.end, s.value from CUPTI_ACTIVITY_KIND_KERNEL k
                     join StringIds s on s.id=k.demangledName order by k.start, k.end''').fetchall()
t0 = rows[0][0]
def kind(nm):
    for key, lab in (('copy_i32_from_mapped', 'A'), ('fetch_blobs', 'B'), ('copy_rows_from_mapped', 'C'),
                     ('copy_from_mapped', 'C'), ('copy_or_zero', 'C')):
        if key in nm: return lab
    return None
# label each wait by the nearest following copy/fetch kernel that is not itself a wait (<=6 kernels ahead)
waits = collections.defaultdict(list)   # (phase, label) -> durations ns
for i, r in enumerate(rows):
    if 'wait_flag_ge' not in r[2]: continue
    lab = None
    for j in range(i + 1, min(i + 7, len(rows))):
        if 'wait_flag_ge' in rows[j][2]: continue
        lab = kind(rows[j][2])
        if lab: break
    ph = 'decode' if (r[0] - t0) > 2.3e9 else 'prefill'
    waits[(ph, lab or '?')].append(r[1] - r[0])
tot = {ph: sum(r[1] - r[0] for r in rows if (r[0] - t0 > 2.3e9) == (ph == 'decode')) for ph in ('decode', 'prefill')}
for ph in ('prefill', 'decode'):
    print(ph, 'all-kernel ms', tot[ph] / 1e6)
    for lab in ('A', 'B', 'C', '?'):
        d = waits.get((ph, lab), [])
        if not d: continue
        print(f'  {lab}: n={len(d)} sum={sum(d)/1e6:.0f} ms ({100*sum(d)/tot[ph]:.1f}% of kernel time) '
              f'median={st.median(d)/1e3:.1f} us p90={sorted(d)[int(.9*len(d))]/1e3:.1f} us max={max(d)/1e3:.0f} us')
