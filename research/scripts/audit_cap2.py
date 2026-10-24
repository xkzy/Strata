#!/usr/bin/env python3
"""Audit the earlier session's cap2 captures with its own reader: sequences, positions per split, duplicate token sequences and
duplicate prompts, how many positions are prompt vs generated, and whether tok[t+1] == nxt[t] (teacher-forced greedy continuation).
Usage: audit_cap2.py CAP_DIR"""
import sys, json, hashlib, collections
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 2)[0] + '/ncm')
from dataset import load_sequences
cap = sys.argv[1]
man = json.load(open(cap + '/manifest.json'))
seqs = load_sequences(cap)
tot = collections.Counter(); dup = collections.defaultdict(list); chain = []; lens = []
for k, s in sorted(seqs.items()):
    split = man[int(k)]['split']; n = len(s['tok'])
    tot[split] += n; lens.append(n)
    dup[hashlib.sha1(s['tok'].tobytes()).hexdigest()].append(k)
    chain.append(float((s['tok'][1:] == s['nxt'][:-1]).mean()) if n > 1 else 1.0)
txt = collections.Counter(m['text'] for m in man)
print('sequences', len(seqs), 'positions by split', dict(tot), 'total', sum(tot.values()))
print('length min/median/max', min(lens), sorted(lens)[len(lens) // 2], max(lens))
print('identical token sequences:', sum(len(v) - 1 for v in dup.values()), ' duplicate prompt texts:', sum(c - 1 for c in txt.values()))
print('unique prompts by split:', collections.Counter(m['split'] for m in {m['text']: m for m in man}.values()))
print('share of positions where tok[t+1]==nxt[t] (mean over sequences):', round(float(np.mean(chain)), 4))
split_of_text = collections.defaultdict(set)
for m in man: split_of_text[m['text']].add(m['split'])
print('prompt texts appearing in both train and test:', sum(1 for v in split_of_text.values() if len(v) > 1))
