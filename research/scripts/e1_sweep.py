#!/usr/bin/env python3
"""E1 sweep: --pool-workers x --pcie-frac, one server per config, server-side timings from the log, Xid check after each.
Aborts on the first Xid or a server that does not come up. Usage: e1_sweep.py out.json [repeats]"""
import json, os, re, signal, subprocess, sys, time, statistics as st, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e0_bench

TMP = '/home/khing/.claude/jobs/189fb9b9/tmp/'
BASE = json.load(open(TMP + 'e0-clean.json'))
CTX_CHARS = 4000
# (workers, pcie_frac); base first and last to expose drift
CONFIGS = [(4, .25), (4, 0), (4, .15), (4, .4), (4, .5), (2, .25), (3, .25), (6, .25), (8, .25), (4, .25)]
P = re.compile(r'strata serve: prompt (\d+) tokens = \d+ reused \+ \d+ read in (\d+) ms \(([\d.]+) tok/s\), (\d+) generated in (\d+) ms \(([\d.]+) tok/s\), drafts accepted (\d+) of (\d+)')

def xid(since):
    r = subprocess.run(['journalctl', '-k', '--since', since, '-g', 'Xid', '--no-pager'], capture_output=True, text=True)
    return [l for l in r.stdout.splitlines() if 'Xid' in l]

def ready(proc):
    for _ in range(600):
        if proc.poll() is not None: return False
        try:   # /v1/models answers before the engine is loaded (503 on chat), so probe with a 1-token chat
            req = urllib.request.Request(e0_bench.URL, json.dumps({'model': BASE['model_name'], 'max_tokens': 1, 'temperature': 0,
                                         'messages': [{'role': 'user', 'content': 'hi'}]}).encode(), {'Content-Type': 'application/json'})
            urllib.request.urlopen(req, timeout=300).read(); return True
        except Exception: time.sleep(3)
    return False

def one(w, f, reps, idx):
    cfg = json.loads(json.dumps(BASE)); a = cfg['args']
    a[a.index('--pool-workers') + 1] = str(w); a[a.index('--pcie-frac') + 1] = str(f)
    log = f'{TMP}sweep-{idx}-w{w}-f{f}.log'; cfg['log'] = log
    json.dump(cfg, open(f'{TMP}sweep-cfg.json', 'w'))
    since = time.strftime('%Y-%m-%d %H:%M:%S')
    proc = subprocess.Popen(['/home/khing/Downloads/Strata copy/bin/strata', 'serve', '-binary-path',
                             '/home/khing/Downloads/Strata copy/engine/strata', '-config', f'{TMP}sweep-cfg.json', '-port', '8091'],
                            cwd='/home/khing/Downloads/Strata copy', start_new_session=True,
                            stdout=open(log + '.out', 'w'), stderr=subprocess.STDOUT)
    res = {'workers': w, 'pcie_frac': f, 'log': log}
    try:
        if not ready(proc): res['error'] = 'server did not come up'; return res
        for i in range(reps + 1):
            e0_bench.run(CTX_CHARS, i, 256)
        rows = [dict(zip(('prompt', 'prefill_ms', 'prefill_tps', 'gen', 'gen_ms', 'decode_tps', 'acc', 'drafts'), map(float, m.groups())))
                for m in map(P.search, open(log + '.out')) if m]
        rows = [r for r in rows if r['gen'] >= 200][1:]          # drop warm-up
        d = [r['decode_tps'] for r in rows]
        res |= {'n': len(d), 'decode_tps': d, 'decode_mean': st.mean(d), 'decode_sd': st.stdev(d) if len(d) > 1 else None,
                'prefill_tps_mean': st.mean(r['prefill_tps'] for r in rows), 'prompt_tokens': rows[0]['prompt'],
                'accept_mean': st.mean(r['acc'] / r['drafts'] for r in rows)}
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try: proc.wait(60)
        except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL)
        time.sleep(8)
    res['xid'] = xid(since)
    return res

if __name__ == '__main__':
    out, reps = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 5
    results = []
    for i, (w, f) in enumerate(CONFIGS):
        r = one(w, f, reps, i); results.append(r)
        json.dump(results, open(out, 'w'), indent=1)
        print(i, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != 'decode_tps'}, flush=True)
        if r.get('xid'): print('XID - aborting'); break
        if r.get('error'): print('ERROR - aborting'); break
