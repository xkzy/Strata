#!/usr/bin/env python3
"""E3: measured decode pass time vs window size. Sweeps --spec 2..6 with --spec-min-p 0 (every round uses the full window),
4 pool workers, -no-autotune (engine flags asserted from /proc). From the server log line of each request:
    rounds = generated - accepted;  t_round = gen_ms / rounds;  tokens/round = generated / rounds.
Usage: e3_pass_time.py out.json [repeats]"""
import json, os, re, signal, subprocess, sys, time, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e0_bench, e1_sweep

TMP = e1_sweep.TMP; BASE = e1_sweep.BASE
P = e1_sweep.P

def one(spec, reps):
    cfg = json.loads(json.dumps(BASE)); a = cfg['args']
    a[a.index('--spec') + 1] = str(spec); a[a.index('--spec-min-p') + 1] = '0'; a[a.index('--pool-workers') + 1] = '4'
    log = f'{TMP}e3-spec{spec}.log'; cfg['log'] = log
    json.dump(cfg, open(f'{TMP}e3-cfg.json', 'w'))
    since = time.strftime('%Y-%m-%d %H:%M:%S')
    proc = subprocess.Popen(['/home/khing/Downloads/Strata copy/bin/strata', 'serve', '-binary-path', '/home/khing/Downloads/Strata copy/engine/strata',
                             '-config', f'{TMP}e3-cfg.json', '-port', '8091', '-no-autotune'], cwd='/home/khing/Downloads/Strata copy',
                            start_new_session=True, stdout=open(log + '.out', 'w'), stderr=subprocess.STDOUT)
    res = {'spec': spec}
    try:
        if not e1_sweep.ready(proc): res['error'] = 'server did not come up'; return res
        for d in os.listdir('/proc'):
            if d.isdigit():
                try: c = open(f'/proc/{d}/cmdline', 'rb').read().decode().split('\0')
                except OSError: continue
                if c and c[0].endswith('engine/strata') and '--serve' in c:
                    res['engine'] = {k: c[c.index(k) + 1] for k in ('--spec', '--spec-min-p', '--pool-workers') if k in c}
        if res.get('engine', {}).get('--spec') != str(spec) or res['engine'].get('--pool-workers') != '4':
            res['error'] = f"engine flags {res.get('engine')}"; return res
        for i in range(reps + 1): e0_bench.run(4000, i, 256)
        rows = [dict(zip(('prompt', 'prefill_ms', 'prefill_tps', 'gen', 'gen_ms', 'decode_tps', 'acc', 'drafts'), map(float, m.groups())))
                for m in map(P.search, open(log + '.out')) if m]
        rows = [r for r in rows if r['gen'] >= 200][1:]
        out = []
        for r in rows:
            rounds = r['gen'] - r['acc']  # each pass emits its accepted drafts + 1 token
            out.append({'rounds': rounds, 't_round_ms': r['gen_ms'] / rounds, 'tok_per_round': r['gen'] / rounds, 'decode_tps': r['decode_tps'],
                        'accept': (r['acc'] / r['drafts']) if r['drafts'] else None})
        res['runs'] = out
        for k in ('t_round_ms', 'tok_per_round', 'decode_tps'):
            v = [o[k] for o in out]; res[k + '_mean'] = st.mean(v); res[k + '_sd'] = st.stdev(v) if len(v) > 1 else None
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try: proc.wait(60)
        except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL)
        time.sleep(8)
    res['xid'] = e1_sweep.xid(since)
    return res

if __name__ == '__main__':
    out, reps = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5
    results = []
    for spec in (2, 3, 4, 5, 6):  # engine needs --spec >= 2 for the native pack
        r = one(spec, reps); results.append(r); json.dump(results, open(out, 'w'), indent=1)
        print(spec, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k not in ('runs',)}, flush=True)
        if r.get('xid') or r.get('error'): print('ABORT'); break
