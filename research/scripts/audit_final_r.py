#!/usr/bin/env python3
"""Audit --dump-final-r captures: records are {int64 pos, int64 tok, f32[hc*n_embd]}. Reports per file the record count, whether pos
is strictly increasing from the first record (rejected drafts would repeat or go back), and dedupes identical token sequences.
Usage: audit_final_r.py captures_dir [hc n_embd]"""
import os, sys, struct, hashlib, collections
d = sys.argv[1]; hc = int(sys.argv[2]) if len(sys.argv) > 2 else 4; ne = int(sys.argv[3]) if len(sys.argv) > 3 else 2560
rec = 16 + hc * ne * 4
tot, bad, seqs, sizes = 0, [], collections.defaultdict(list), []
for dp, _, fs in os.walk(d):
    for f in sorted(fs):
        if not (f.startswith('final_r') and f.endswith('.bin')): continue
        p = os.path.join(dp, f); sz = os.path.getsize(p)
        if sz % rec: bad.append((p, 'size not a multiple of record', sz % rec)); continue
        n = sz // rec; toks, last, mono = [], -1, True
        with open(p, 'rb') as fh:
            for i in range(n):
                pos, tok = struct.unpack('<qq', fh.read(16)); fh.seek(hc * ne * 4, 1)
                if pos != last + 1: mono = False
                last = pos; toks.append(tok)
        if not mono: bad.append((p, 'pos not consecutive from first record', n))
        h = hashlib.sha1(repr(toks).encode()).hexdigest()[:12]
        seqs[h].append(p); tot += n; sizes.append(n)
print('files', len(sizes), 'records', tot, 'unique token sequences', len(seqs), 'duplicates', len(sizes) - len(seqs))
if sizes: print('len min/median/max', min(sizes), sorted(sizes)[len(sizes) // 2], max(sizes))
print('problems', bad[:10], len(bad))
