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
32-token windows, levels 1, 2, 4, 8, 16, 32, 5,000 steps, full 248,320 vocabulary, test = 256 windows from held-out prompts.

**v1 (3,000 steps, 1,920 train windows, restarts drawn from the current batch).** Finest level 0.606 (C = 256), 0.623 (C = 4096). Cause found: restarts copied the ≤ 8 level-1 vectors of a batch, so ~250 of 256 codes were duplicates that nearest-code search never picked. Data in `ncm_tokenizer_C*.json`.

**v2 (fixed).** Codebooks seeded from a no-grad pass over 16k windows (6k at d = 256); unused codes re-seeded every 500 steps from a per-level reservoir of 8,192 recent vectors; 28,991 stride-1 train windows. Data in `ncm_tokenizer_v2_*.json`.

| Level (codes) | C=256, d=128 | C=4096, d=128 | C=4096, d=256 |
|---|---|---|---|
| 1 | 0.064 | 0.071 | 0.073 |
| 2 | 0.086 | 0.082 | 0.085 |
| 3 (4) | 0.121 | 0.115 | 0.120 |
| 4 (8) | 0.204 | 0.206 | 0.211 |
| 5 (16) | 0.376 | 0.412 | 0.404 |
| 6 (32) | 0.618 | 0.753 | 0.746 |
| unquantised z | 0.881 | 0.896 | 0.914 |
| train acc, level 6 | 0.752 | 0.944 | 0.967 |
| codes used on test, level 6 | 198/256 | 1592/4096 | 1387/4096 |
| bits/token at level 6 | 15.75 | 23.63 | 23.63 |
**v3 (more text).** `build_corpus.py` tokenises hh-rlhf, oasst1 and wikitext-2 (cached, Dolly excluded) with the teacher's own BPE (55 of 56 captured runs of ordinary tokens re-encode to identical ids) into 600,000 windows (19.2M tokens; 125,666 distinct ids); train = 626,943 windows, 15,000 steps at batch 8 (≈ 0.19 epoch), C = 4096, d = 128. Data: `ncm_tokenizer_v3_corpus_C4096_D128.json`.

| Level (codes) | cap2 test (256 windows) | held-out corpus (2,048 windows) |
|---|---|---|
| 1 | 0.075 | 0.072 |
| 2 | 0.089 | 0.093 |
| 3 (4) | 0.122 | 0.139 |
| 4 (8) | 0.215 | 0.267 |
| 5 (16) | 0.443 | 0.497 |
| 6 (32) | **0.721** | **0.751** |
| unquantised z | 0.976 | 0.969 |
Train accuracy at level 6 (cap2 train windows) is 0.671, so the train–test gap closed (v2 0.94 vs 0.75) and the memorisation is gone. The unquantised decoder now reaches 97–98% (v2: 0.896), so the embedding transform and decoder generalise; what holds the result at 0.72–0.75 is the quantised path, which is under-trained (≈ 0.19 epoch, loss flat near 35) and, at level 1, sees only 8 vectors per step for 4,096 codes (≈ 3,000 codes re-seeded at every restart). **Result vs v2 on the same cap2 test: 0.721 vs 0.753, no gain from 20× the text at this training budget** [M]; whether a longer or larger-batch run helps is untested [U].

**v4 (longer, bigger batch).** Same C = 4096, d = 128, same 626,943-window corpus; 12,000 steps; codebooks see 128 windows per step, CE on 32 windows (chunks of 8), finest + 2 sampled coarser levels per step (weights keep the expected loss at 6 levels) → ≈ 384k windows seen (0.6 epoch) against 120k in v3. Data: `ncm_tokenizer_v4_big_C4096_D128.json`.

| Level (codes) | cap2 test | held-out corpus |
|---|---|---|
| 1 | 0.077 | 0.079 |
| 2 | 0.096 | 0.103 |
| 3 (4) | 0.138 | 0.156 |
| 4 (8) | 0.243 | 0.297 |
| 5 (16) | 0.496 | 0.545 |
| 6 (32) | **0.784** | **0.801** |
| unquantised z | 0.976 | 0.974 |
| train acc, level 6 | 0.768 | – |
Mid-run cap2 readings at the finest level: 0.776 (step 2,500), 0.783 (5,000), 0.785 (7,500), 0.785 (10,000): it plateaued by step 5,000 while level 5 kept creeping up (0.396 → 0.493). Versus v3 (0.721 / 0.751): +6.3 and +5.0 points at the finest level, with 3× the windows seen and a changed loss (sampled levels), so the gain is not attributable to batch size alone [M]. Still far below the paper's 98.8–99.99%; coarse levels (1–3 codes) stay at 8–16%; 23.6 bits per token at level 6 is above a raw token id (~17.9). First-level idle codes at restart stayed ~1,700 of 4,096, at the finest level ~0 [M].

**v5 (finite scalar quantisation instead of EMA-VQ).** Same corpus, d = 128, big-batch loss; 8,000 steps; each level's latent is squashed and rounded to a fixed grid (no learned codebook, so no dead codes by construction: restart column 0). Data: `ncm_tokenizer_v5_fsq4096.json` (grid 8·8·8·8), `ncm_tokenizer_v5_fsq64000.json` (grid 8·8·8·5·5·5).

| Run | cap2 test, level 6 | held-out corpus, level 6 | bits/token | levels 1–5 (cap2) |
|---|---|---|---|---|
| VQ v4 (4,096 codes, 12k steps) | 0.784 | 0.801 | 23.6 | 0.077 / 0.096 / 0.138 / 0.243 / 0.496 |
| FSQ 4,096 (8k steps) | 0.735 | 0.751 | 23.6 | 0.061 / 0.073 / 0.088 / 0.130 / 0.226 |
| FSQ 64,000 (8k steps) | **0.799** | **0.804** | 31.4 | 0.058 / 0.074 / 0.093 / 0.143 / 0.266 |

At equal bits (23.6) FSQ is 4.9 points below VQ (0.735 vs 0.784, 8k vs 12k steps, so not a clean comparison). The 64,000-code grid reaches 0.799, +1.5 points on cap2 and +0.3 on new text over VQ v4, which is within what two single-seed runs can distinguish, and costs 7.8 more bits per token (31.4, nearly twice a raw token id). Coarse levels are weaker than VQ's. Mid-run cap2 level 6: FSQ-4096 0.646 (step 2,500) → 0.735 (7,500); FSQ-64000 0.705 → 0.776 (5,000) → 0.799. **Result: FSQ does not fix the weak coarse levels or beat VQ at equal bits** [M]; single seed each [U].

**v6 (VQ-4096, 5,000 steps, recall@K diagnostic).** Same setup as v4 with the evaluation extended: for every level, the share of true tokens whose rank in that level's decoder logits is below K. Data: `ncm_tokenizer_v6_vq4096_recall.json`; checkpoint kept outside the repo.

| Level (codes) | top-1 | top-16 | top-256 | top-4096 |
|---|---|---|---|---|
| 1 | 0.075 | 0.348 | 0.592 | 0.812 |
| 2 | 0.092 | 0.383 | 0.610 | 0.825 |
| 3 (4) | 0.132 | 0.451 | 0.639 | 0.835 |
| 4 (8) | 0.228 | 0.555 | 0.685 | 0.850 |
| 5 (16) | 0.438 | 0.703 | 0.785 | 0.891 |
| 6 (32) | 0.788 | 0.878 | 0.934 | 0.981 |
(cap2 test, 256 windows; held-out corpus level 6: 0.806 / 0.882 / 0.931 / 0.975.) A 5,000-step rerun of v4 gave 0.788 at level 6 (v4 final 0.784), so the result is repeatable to about ±0.01 [M]. What this says: **the level-5 decoder already puts the true token in its top 4,096 for 89% of positions (12 bits)**, and the level-6 decoder does so for 98%, so replacing the last level's learned 12-bit code with the rank of the true token (escape otherwise) would make 89% of positions lossless at the same 12 bits for that level (23.6 cumulative, unchanged), against 79% exact now. The saving is quality, not bits.

**v7 (rank-coded level, measured on the saved v6 model, no retraining).** `ncm_tokenizer.py ... load=CKPT`: the encoder sends the rank of the true token in the level-j decoder's logits; the decoder returns the token at that rank (stable sort; round trip checked exact on all 8,192 + 65,536 test tokens). Cost = bits of levels ≤ j + the rank code. Data: `ncm_rankcode_v6.json`. Baselines on the 19.2M-token corpus: raw id 17.92 bits, unigram entropy 10.65 bits.

| Scheme (rank from level-5 decoder, K = 4,096) | bits/token | exact: cap2 test | exact: new text |
|---|---|---|---|
| learned 12-bit level 6 (v6, as before) | 23.6 | 0.788 | 0.806 |
| rank code, rank ≥ 4,096 left wrong (lossy) | 23.6 | **0.891** | **0.880** |
| rank code + escape (rank 4,096 = "raw id follows") | 25.6 | 1.000 | 1.000 |
| rank code, Elias-gamma, no escape | 19.4 (cap2) / 19.2 (new text) | 1.000 | 1.000 |
| level 6 learned code + rank code, K = 4,096, lossy | 35.6 | 0.981 | 0.975 |
| level 6 learned code + Elias-gamma rank | 26.9 | 1.000 | 1.000 |

At the same 23.6 bits, swapping the learned last level for its rank code lifts exact reconstruction by +10.3 points (cap2) and +7.4 points (new text) [M]. Every lossless variant costs more than a plain token id (19.2 bits at best against 17.92) and far more than an entropy-coded token stream (10.65) [M]. Median rank at level 5 is 1 (rank 0: 44–49%); at levels 1–3 it is 31–106, i.e. the coarse codes narrow the candidates very little. **Limits:** one model trained for its own learned codes, not for ranking [U]; 256 + 2,048 test windows, single seed; "lossless" means the round trip, not a better compressor.

Observations [M]: the collapse is fixed (all codes used on train at every level); capacity now matters (C = 4096 beats 256 by 13.5 points); d = 256 does not help (0.746 vs 0.753), so the 128-d PCA is not the limit; coarse levels carry little (7–21% through level 4); the train–test gap (0.94–0.97 vs 0.75) and 23.6 bits per token (a raw token id is ~17.9) say the finest level memorises a small corpus. **Still far from the paper's 98.8–99.99%; the paper's data, vocabulary and ~100M-parameter scale differ, so this is not evidence against NCM.** Whether more data closes the gap is untested [U]. The hierarchical-vs-flat ablation was **not run**: coarse levels are too weak to compare.

## 4. Weight-structure nulls
- **Wavelets** (12 dense matrices, layers 0/3/7/47; `wavelet_stats.py`): fraction of coefficients holding 99% of the energy: raw 0.58–0.72, Haar 0.65–0.73, db4 0.65–0.73, random-sign Haar control 0.67–0.73. No compaction beyond a random basis. SVD storage for 99% energy is 1.03–1.38× the matrix, so no low-rank structure. [M]
- **Experts** (layers 3, 20, 47, gate and up; `expert_share.py`): mean expert 0.4–0.7% of energy; top-8 components 3.5–6.4% (random 3.1%); cosine ≤ 0.027; k-means cluster residual equals random-label residual (0.84–0.87 vs 0.87–0.88 at k = 32). One-expert Haar: 0.73 vs raw 0.86 (a 2-bit-level artefact; no random control run). Weight-space only: hidden-unit permutations untested. [M]

## 5. Pass time
See mathematical_cost_model.md §1 (15.4 + 13.1·T ms). [M]

## 6. Not done
Engine-side NCM prototype (needs approval), the flat-vs-hierarchical ablation, long-context scaling, sustained-stability runs of an NCM variant.
