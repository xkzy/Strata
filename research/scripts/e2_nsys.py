#!/usr/bin/env python3
"""E1 re-profile at 4 pool workers: server under `nsys launch` with -no-autotune, warm-up, then one profiled request
(~913-token prompt, 256 tokens). Aborts if the engine did not get the requested --pool-workers. Checks Xid at the end.
Usage: e2_nsys.py outdir [workers]"""
import json, os, signal, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e0_bench, e1_sweep

out = sys.argv[1]; w = int(sys.argv[2]) if len(sys.argv) > 2 else 4
os.makedirs(out, exist_ok=True)
NSYS = '/usr/local/cuda-13.2/bin/nsys'; R = '/home/khing/Downloads/Strata copy'
cfg = json.load(open(e1_sweep.TMP + 'e0-clean.json')); a = cfg['args']
a[a.index('--pool-workers') + 1] = str(w); cfg['log'] = out + '/server.log'
json.dump(cfg, open(out + '/cfg.json', 'w'))
since = time.strftime('%Y-%m-%d %H:%M:%S')
proc = subprocess.Popen([NSYS, 'launch', '--session-new=e2', '-t', 'cuda,nvtx', '--cuda-graph-trace=node',
                         R + '/bin/strata', 'serve', '-binary-path', R + '/engine/strata', '-config', out + '/cfg.json',
                         '-port', '8091', '-no-autotune'], cwd=R, start_new_session=True,
                        stdout=open(out + '/server.out', 'w'), stderr=subprocess.STDOUT)
try:
    assert e1_sweep.ready(proc), 'server did not come up'
    flags = None
    for d in os.listdir('/proc'):
        if d.isdigit():
            try: c = open(f'/proc/{d}/cmdline', 'rb').read().decode().split('\0')
            except OSError: continue
            if c and c[0].endswith('engine/strata') and '--serve' in c: flags = c
    assert flags and int(flags[flags.index('--pool-workers') + 1]) == w, f'engine flags: {flags}'
    e0_bench.run(4000, 0, 64)                                # warm-up (also fills caches as in E0)
    subprocess.run([NSYS, 'start', '--session=e2', '-o', out + '/e2', '--force-overwrite=true'], check=True)
    res = e0_bench.run(4000, 1, 256)
    subprocess.run([NSYS, 'stop', '--session=e2'], check=True)
    json.dump(res, open(out + '/request.json', 'w'), indent=1)
    print('request', {k: round(v, 2) for k, v in res.items() if isinstance(v, float)})
finally:
    os.killpg(proc.pid, signal.SIGTERM)
    try: proc.wait(90)
    except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL)
    time.sleep(8)
    x = e1_sweep.xid(since); open(out + '/xid.txt', 'w').write('\n'.join(x)); print('xid', x)
