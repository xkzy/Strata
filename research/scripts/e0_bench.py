#!/usr/bin/env python3
"""E0 clean live baseline client. Streams /v1/chat/completions, temperature 0, unique prompt per repeat (the server has
--prompt-cache on). Reports TTFT, prefill tok/s (prompt_tokens / TTFT), decode tok/s (tokens after the first / time after
the first), end-to-end tok/s (completion tokens / total time). Usage: e0_bench.py out.json ctx_chars,... repeats [max_tokens]"""
import json, sys, time, random, urllib.request, statistics as st

URL = 'http://127.0.0.1:8091/v1/chat/completions'
SEED_TEXT = ('The quick brown fox jumps over the lazy dog while a scheduler assigns tasks to workers; '
             'each worker reads a record, validates it, and writes a summary line to the log. ')

def run(ctx_chars, rep, max_tokens):
    rnd = random.Random(1000 + rep)
    tag = ''.join(rnd.choice('abcdefghijklmnop') for _ in range(24))   # unique prefix defeats the prompt cache
    body = (f'[doc {tag}] ' + SEED_TEXT * (ctx_chars // len(SEED_TEXT) + 1))[:ctx_chars]
    msgs = [{'role': 'user', 'content': body + '\n\nWrite a Python function that parses a CSV line with quoted fields, with a docstring and two tests.'}]
    req = urllib.request.Request(URL, json.dumps({'model': 'qwen3.8-flash-next-coder-iq1_m', 'messages': msgs, 'temperature': 0,
                                                  'max_tokens': max_tokens, 'stream': True,
                                                  'stream_options': {'include_usage': True}}).encode(),
                                 {'Content-Type': 'application/json'})
    t0 = time.time(); t_first = None; n = 0; usage = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith('data:') or line.endswith('[DONE]'): continue
            d = json.loads(line[5:])
            if d.get('usage'): usage = d['usage']
            ch = d.get('choices') or []
            if ch and (ch[0].get('delta', {}).get('content') or ch[0].get('delta', {}).get('reasoning_content')):
                n += 1
                if t_first is None: t_first = time.time()
    t1 = time.time()
    pt = (usage or {}).get('prompt_tokens'); ct = (usage or {}).get('completion_tokens', n)
    ttft = t_first - t0
    return {'prompt_tokens': pt, 'completion_tokens': ct, 'chunks': n, 'ttft_s': ttft,
            'prefill_tok_s': (pt / ttft) if pt else None,
            'decode_tok_s': ((ct - 1) / (t1 - t_first)) if ct > 1 and t1 > t_first else None,
            'e2e_tok_s': ct / (t1 - t0), 'total_s': t1 - t0}

if __name__ == '__main__':
    out, ctxs, reps = sys.argv[1], [int(x) for x in sys.argv[2].split(',')], int(sys.argv[3])
    mt = int(sys.argv[4]) if len(sys.argv) > 4 else 256
    res = {}
    for c in ctxs:
        rows = []
        for i in range(reps + 1):                       # first run of each size is a warm-up, kept but flagged
            r = run(c, i, mt); r['warmup'] = (i == 0); rows.append(r)
            print(c, i, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
            json.dump(res | {str(c): rows}, open(out, 'w'), indent=1)
        res[str(c)] = rows
        for k in ('decode_tok_s', 'prefill_tok_s', 'e2e_tok_s'):
            v = [r[k] for r in rows if not r['warmup'] and r[k]]
            if len(v) > 1: print(f'  ctx {c} {k}: mean {st.mean(v):.2f} sd {st.stdev(v):.2f} cv {100*st.stdev(v)/st.mean(v):.1f}% n={len(v)}', flush=True)
    json.dump(res, open(out, 'w'), indent=1)
