#!/usr/bin/env python3
"""E1 perf: start the E0 clean server, warm it, then perf-record the engine process (all threads) during one short-prompt
decode request, and diff per-thread user/sys CPU ticks from /proc over the same window. Aborts on Xid.
Usage: e1_perf.py outdir"""
import json, os, signal, subprocess, sys, threading, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e0_bench, e1_sweep

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
TMP = e1_sweep.TMP
cfg = json.load(open(TMP + 'e0-clean.json')); cfg['log'] = out + '/server.log'
json.dump(cfg, open(out + '/cfg.json', 'w'))
since = time.strftime('%Y-%m-%d %H:%M:%S')
proc = subprocess.Popen(['/home/khing/Downloads/Strata copy/bin/strata', 'serve', '-binary-path',
                         '/home/khing/Downloads/Strata copy/engine/strata', '-config', out + '/cfg.json', '-port', '8091'],
                        cwd='/home/khing/Downloads/Strata copy', start_new_session=True,
                        stdout=open(out + '/server.out', 'w'), stderr=subprocess.STDOUT)

def engine_pid():
    for d in os.listdir('/proc'):
        if d.isdigit():
            try:
                c = open(f'/proc/{d}/cmdline', 'rb').read().split(b'\0')
            except OSError: continue
            if c and c[0].endswith(b'engine/strata') and b'serve' not in c[:2]: return int(d)

def ticks(pid):
    t = {}
    for tid in os.listdir(f'/proc/{pid}/task'):
        try:
            s = open(f'/proc/{pid}/task/{tid}/stat').read()
            nm = s[s.index('(') + 1:s.rindex(')')]; f = s[s.rindex(')') + 2:].split()
            t[tid] = (nm, int(f[11]), int(f[12]))          # utime, stime in clock ticks
        except OSError: pass
    return t

try:
    assert e1_sweep.ready(proc), 'server did not come up'
    e0_bench.run(400, 0, 64)                                # warm-up
    pid = engine_pid(); assert pid, 'engine pid not found'
    before = ticks(pid)
    res = {}
    th = threading.Thread(target=lambda: res.update(e0_bench.run(400, 1, 512))); th.start()
    time.sleep(1.5)                                         # let prefill of the short prompt finish
    p = subprocess.run(['perf', 'record', '-F', '499', '-g', '-p', str(pid), '-o', out + '/perf.data', '--', 'sleep', '8'],
                       capture_output=True, text=True)
    open(out + '/perf_record.txt', 'w').write(p.stdout + p.stderr)
    th.join(); after = ticks(pid)
    rows = [(after[t][0], before.get(t, (0, 0, 0))[1], after[t][1] - before.get(t, (0, 0, 0))[1], after[t][2] - before.get(t, (0, 0, 0))[2], t)
            for t in after]
    rows = [(n, du, ds, t) for n, _, du, ds, t in rows if du + ds > 0]
    json.dump({'request': res, 'clk_tck': os.sysconf('SC_CLK_TCK'), 'threads': rows, 'pid': pid}, open(out + '/ticks.json', 'w'), indent=1)
    print('perf rc', p.returncode, 'request', {k: round(v, 2) for k, v in res.items() if isinstance(v, float)})
finally:
    os.killpg(proc.pid, signal.SIGTERM)
    try: proc.wait(60)
    except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL)
    time.sleep(8)
    x = e1_sweep.xid(since); open(out + '/xid.txt', 'w').write('\n'.join(x)); print('xid', x)
