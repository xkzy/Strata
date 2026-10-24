#!/usr/bin/env python
"""Tokenise cached English text with the TEACHER's tokenizer (BPE from the pack's vocab.json + merges.txt, qwen35 pre-pattern) into
non-overlapping 32-token windows for NCM tokenizer training. Sources (HF cache, offline): Anthropic/hh-rlhf 'chosen', OpenAssistant/oasst1 'text',
wikitext-2-raw-v1 train. Dolly is excluded on purpose: the held-out cap2 test windows come from Dolly prompts (no leakage).
First checks the tokenizer: decode(cap2 tokens) -> encode must give the same ids (greedy teacher output is mostly canonical BPE).
Usage: build_corpus.py CAP_DIR OUT.npy [max_windows=600000]"""
import sys, os, json, glob
os.environ['HF_DATASETS_OFFLINE'] = '1'
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, Regex
sys.path.insert(0, __file__.rsplit('/', 2)[0] + '/ncm')
from capture import load_spec_residuals

PACK = '/home/khing/Downloads/Strata-data/packs/coder-iq1_m/tokenizer/'
PAT = json.load(open(PACK + 'tokenizer.json'))['pre_pattern']
tk = Tokenizer(models.BPE.from_file(PACK + 'vocab.json', PACK + 'merges.txt'))
tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(PAT), 'isolated'), pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
tk.decoder = decoders.ByteLevel()
cap, outp = sys.argv[1], sys.argv[2]
MAXW = int(sys.argv[3]) if len(sys.argv) > 3 else 600000
L = 32

same = tot = 0
for f in sorted(glob.glob(cap + '/final_r_*.bin'))[:20]:
    _, tok, nxt, _ = load_spec_residuals(f); ids = tok.tolist()
    runs, cur = [], []                                  # split at special/control ids (>= 248044): each run of ordinary tokens is round-tripped alone
    for i in ids:
        if i >= 248044:
            runs.append(cur); cur = []
        else: cur.append(i)
    runs.append(cur)
    for r in runs:
        if len(r) < 2: continue
        tot += 1; same += tk.encode(tk.decode(r)).ids == r
print('tokenizer round trip on cap2 (20 sequences): runs of ordinary tokens re-encoding to identical ids', same, 'of', tot, flush=True)

from datasets import load_dataset
texts = {}
texts['hh-rlhf'] = [r['chosen'] for r in load_dataset('Anthropic/hh-rlhf', split='train')]
texts['oasst1'] = [r['text'] for r in load_dataset('OpenAssistant/oasst1', split='train')]
texts['wikitext2'] = [r['text'] for r in load_dataset('wikitext', 'wikitext-2-raw-v1', split='train') if len(r['text']) > 200]
wins, src = [], []
for name, tx in texts.items():
    n0 = len(wins)
    for e in tk.encode_batch(tx):
        ids = e.ids
        wins += [ids[i:i + L] for i in range(0, len(ids) - L + 1, L)]
    print(name, len(tx), 'docs ->', len(wins) - n0, 'windows', flush=True); src += [name] * (len(wins) - n0)
A = np.array(wins, dtype=np.int32); assert A.max() < 248320
rng = np.random.default_rng(0); idx = rng.permutation(len(A))[:MAXW]; A = A[idx]
np.save(outp, A)
print('saved', A.shape, 'tokens', A.size, 'unique token ids', len(np.unique(A)), flush=True)
