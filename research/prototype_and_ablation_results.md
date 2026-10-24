# Prototype and ablation results

All runs: production engine untouched; GPU runs on the RTX 4060 (training) or 5060 Ti (engine timing); Xid checked after each run, none seen. Raw data in `research/data/`, scripts in `research/scripts/`. Data: 160 greedy teacher continuations (Dolly prompts), 41,028 positions, split by prompt: 32,831 train / 8,197 test; no duplicate prompts or sequences. [M]

## 1. Head sanity
A PyTorch transcription of the teacher's output path (hc norm → silu down → sigmoid up → mean → Q6_K head) matches the recorded next token 0.9985 over 4,096 positions. [M]

## 2. Heads-only multi-token drafts (`ceiling_pilot.py`)
Residual-MLP map from the final state `h_t` (A) or `h_t` plus the known next token's output row (B), trained with CE through the frozen head, scored on held-out prompts.

| j tokens ahead | 0 (head) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| A test | 0.997 | 0.402 | 0.204 | 0.130 | 0.109 | 0.091 | 0.084 | 0.072 | 0.066 |
| B test | – | 0.401 | 0.208 | 0.139 | 0.108 | 0.092 | 0.083 | 0.071 | 0.069 |
Accepted prefix over 8: A 0.82, B 0.80 → 1.82 / 1.80 tokens per pass; fully correct 8-token blocks 2.2% / 1.9%. Train agreement was 0.83–0.95 against test 0.07–0.40: the 13M-parameter maps memorised 33k positions. **Limit:** more data or smaller maps untested [U]; the next-token conditioning used only an output row.

## 3. NCM tokenizer on teacher tokens (`ncm_tokenizer.py`)
32-token windows, levels 1, 2, 4, 8, 16, 32, d = 128, 3,000 steps, 1,920 train / 256 test windows, full 248,320 vocabulary.

| Level (codes) | C = 256 test | C = 4096 test |
|---|---|---|
| 1 | 0.058 | 0.055 |
| 2 | 0.077 | 0.073 |
| 3 (4) | 0.108 | 0.101 |
| 4 (8) | 0.181 | 0.180 |
| 5 (16) | 0.350 | 0.351 |
| 6 (32) | 0.606 | 0.623 |
| unquantised z | 0.879 | 0.885 |
Observations [M]: codebook size changes nothing; ~249/256 and ~4,090/4,096 first-level codes are dead at alternate restart checks (the quantiser uses tens of codes); the decoder alone reaches 88%. **This is a failed reproduction of the paper's 98.8–99.99%, attributable to this implementation (EMA collapse, 128-d PCA, 1,920 windows), not evidence against NCM.** The hierarchical-vs-flat ablation was **not run**: it needs a working tokenizer. [U]

## 4. Weight-structure nulls
- **Wavelets** (12 dense matrices, layers 0/3/7/47; `wavelet_stats.py`): fraction of coefficients holding 99% of the energy: raw 0.58–0.72, Haar 0.65–0.73, db4 0.65–0.73, random-sign Haar control 0.67–0.73. No compaction beyond a random basis. SVD storage for 99% energy is 1.03–1.38× the matrix, so no low-rank structure. [M]
- **Experts** (layers 3, 20, 47, gate and up; `expert_share.py`): mean expert 0.4–0.7% of energy; top-8 components 3.5–6.4% (random 3.1%); cosine ≤ 0.027; k-means cluster residual equals random-label residual (0.84–0.87 vs 0.87–0.88 at k = 32). One-expert Haar: 0.73 vs raw 0.86 (a 2-bit-level artefact; no random control run). Weight-space only: hidden-unit permutations untested. [M]

## 5. Pass time
See mathematical_cost_model.md §1 (15.4 + 13.1·T ms). [M]

## 6. Not done
Engine-side NCM prototype (needs approval), the flat-vs-hierarchical ablation, long-context scaling, sustained-stability runs of an NCM variant.
