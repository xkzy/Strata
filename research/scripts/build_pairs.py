#!/usr/bin/env python
"""Prefix/continuation pairs for the stage-2 generator: non-overlapping 64-token chunks (32 prefix + 32 target) of the same cached text as build_corpus.py,
tokenised with the teacher's BPE. Split BY DOCUMENT (every 40th document is test) so no test document contributes a training pair.
Note: the frozen tokenizers were trained on shuffled 32-token windows of these same sources, so test documents may have been seen by the tokenizer (not by the generator).
Usage: build_pairs.py OUT_PREFIX [max_train=400000]  -> OUT_PREFIX_train.npy, OUT_PREFIX_test.npy  ([N, 64] int32)"""
import sys, os, json
os.environ['HF_DATASETS_OFFLINE'] = '1'
import numpy as np
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, Regex

PACK = '/home/khing/Downloads/Strata-data/packs/coder-iq1_m/tokenizer/'
PAT = json.load(open(PACK + 'tokenizer.json'))['pre_pattern']
tk = Tokenizer(models.BPE.from_file(PACK + 'vocab.json', PACK + 'merges.txt'))
tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.Split(Regex(PAT), 'isolated'), pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
tk.decoder = decoders.ByteLevel()
outp, MAXW = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 400000
from datasets import load_dataset
texts = {'hh-rlhf': [r['chosen'] for r in load_dataset('Anthropic/hh-rlhf', split='train')],
         'oasst1': [r['text'] for r in load_dataset('OpenAssistant/oasst1', split='train')],
         'wikitext2': [r['text'] for r in load_dataset('wikitext', 'wikitext-2-raw-v1', split='train') if len(r['text']) > 200]}
tr, te = [], []
for name, tx in texts.items():
    for d, e in enumerate(tk.encode_batch(tx)):
        ids = e.ids; (te if d % 40 == 0 else tr).extend(ids[i:i + 64] for i in range(0, len(ids) - 63, 64))
    print(name, len(tx), 'docs; pairs so far train', len(tr), 'test', len(te), flush=True)
rng = np.random.default_rng(0); tr = np.array(tr, dtype=np.int32); te = np.array(te, dtype=np.int32)
tr = tr[rng.permutation(len(tr))[:MAXW]]; te = te[rng.permutation(len(te))[:4096]]
assert tr.max() < 248320 and te.max() < 248320
np.save(outp + '_train.npy', tr); np.save(outp + '_test.npy', te); print('saved', tr.shape, te.shape)
