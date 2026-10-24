#!/usr/bin/env python3
"""Weight-space error of re-quantising the BF16 hyper-connection / router tensors (exact at source) to block formats.
Relative Frobenius error ||W - W'|| / ||W||. Weight-space only: NOT a function-space or quality measurement."""
import json
import numpy as np

H = json.load(open('research/data/teacher_gguf_headers.json'))[0]
path, base = H['file'], H['data_start']
T = {t['name']: t for t in H['tensors']}

def bf16(name):
    t = T[name]; n = int(np.prod(t['shape']))
    raw = np.fromfile(path, dtype=np.uint16, count=n, offset=base + t['off'])
    return (raw.astype(np.uint32) << 16).view(np.float32)

def blockq(x, bits, block=32):
    x = x[: x.size // block * block].reshape(-1, block)
    qmax = 2 ** (bits - 1) - 1
    d = np.abs(x).max(1, keepdims=True) / qmax
    d = np.where(d == 0, 1, d).astype(np.float16).astype(np.float32)   # scale stored as fp16, as in Q8_0 / Q4_0
    q = np.clip(np.round(x / d), -qmax - 1, qmax)
    return x, q * d

def rel(x, y): return float(np.linalg.norm(x - y) / np.linalg.norm(x))

print('relative Frobenius error vs exact BF16 weights, 32-value blocks with fp16 scale (Q8_0/Q4_0-style, symmetric absmax)')
print('%-22s %10s %10s %10s' % ('tensor role (48 layers)', 'Q8 (8b)', 'Q6-like', 'Q4 (4b)'))
res = {}
for role in ('hc_attn_down', 'hc_attn_up', 'hc_ffn_down', 'hc_ffn_up', 'ffn_gate_inp'):
    e = {8: [], 6: [], 4: []}
    for l in range(0, 48, 4):                      # every 4th layer (12 of 48) keeps runtime small
        w = bf16(f'blk.{l}.{role}.weight')
        for b in e:
            x, y = blockq(w, b); e[b].append(rel(x, y))
    res[role] = {b: float(np.mean(v)) for b, v in e.items()}
    print('%-22s %10.5f %10.5f %10.5f' % (role, res[role][8], res[role][6], res[role][4]))
# reference: the rounding error BF16 itself introduced relative to float32 (8-bit significand) = ~2^-9 per element (uniform)
print('\nreference: BF16 round-to-nearest relative element error bound 2^-9 = %.5f (rms ~ %.5f)' % (2 ** -9, 2 ** -9 / np.sqrt(3)))
b8 = 8.5 / 16; b6 = 6.5625 / 16; b4 = 4.5 / 16
print('bytes vs BF16: Q8_0 %.0f%%, Q6_K %.0f%%, Q4_0/Q4_K %.0f%%  -> hc bytes/token 1283 MB -> %.0f / %.0f / %.0f MB' % (b8 * 100, b6 * 100, b4 * 100, 1283 * b8, 1283 * b6, 1283 * b4))
json.dump(res, open('research/data/hc_requant_probe.json', 'w'))
