# Dense models (Gemma 4 31B, Qwen3.8-27B)

Status: **the CUDA engine cannot run these yet.** What exists is a checked reading of the files and a plain-Go CPU
reference forward pass (`pkg/densecpu`) that runs both models correctly. There is no GPU path and no speed work yet.

## CPU reference (`pkg/densecpu`)

A slow, no-cache Go forward pass over the memory-mapped GGUF, written from the vendored llama.cpp graphs. It
is the thing the CUDA kernels will be checked against. Measured on this PC (32 threads, CPU only, the 6-token prompt
`The capital of France is`, one full pass, no KV cache):

| | top-1 next token (logit) | next best | one pass | 12-token greedy continuation |
|---|---|---|---|---|
| Gemma 4 31B (Q4_K_XL) | `▁Paris` (27.5) | `▁**` (23.5) | 6.3 s | ` Paris.\n\nThe capital of France is Paris.\n\nThe` |
| Qwen3.8-27B (Q4_K_M) | `ĠParis` (16.4) | `Ġnot` (13.4) | 8.0 s | ` Paris.` / `The capital of Germany is Berlin.` / `The` |

The model is Qwen3.8; `qwen35` is only the architecture id inside its GGUF (and in llama.cpp), not a model version.

The Qwen file's MTP block (`blk.64`) runs too (`Qwen.MTP`): fed the main model's final-norm hidden state at position i
and the token at i+1, it predicts token i+2. On the 18-token sequence above (prompt plus the 12 generated tokens) its
draft matched the main model's own next-next token in 12 of 15 positions (80%). That is one prompt, so it is a
sanity check of the block, not an acceptance rate for real workloads. The block's rope position offset made no
difference (rope is relative), so it is not a tuning knob.

The 12-token runs took 100 s (Gemma) and 80 s (Qwen) because every step re-runs the whole sequence. Run with
`go test ./pkg/densecpu -v` (skips when the models are not in `Strata-data/models`; `-short` skips the slow tests).
The tokenizer in the tests is a greedy longest-match, not the real one, so only plain English prompts are used.
What this does not show: agreement with llama.cpp on the logits of other prompts, long contexts (the sliding window
was never exceeded).

## GPU matmul (`pkg/densecpu/cuda`, build tag `cuda`)

The same forward pass with the weight matrices on the GPUs. `cuda/build.sh` builds `libdensegpu.so` (nvcc 13.2 with
g++-15 as host compiler, sm_89 and sm_120), `EnableGPU()` copies every supported 2-D weight to the card with the most
free memory (2 GiB kept free per card) and anything that does not fit stays on the CPU. Only the quantized matrix
products run on the GPU; norms, rope, attention, the GDN recurrence and sampling are still Go code on the CPU, and
every call copies the activations to the card and the result back. Run: `sh pkg/densecpu/cuda/build.sh && go test -tags cuda ./pkg/densecpu -v`.

Measured on the RTX 5060 Ti (16 GB) + RTX 4060 (8 GB), one pass over a single token (about one decode step without
attention over a long cache), after warm-up:

| | CPU, dequantize then dot | CPU, fused | GPU, first kernels | GPU, 8-weights-per-thread kernels |
|---|---|---|---|---|
| Gemma 4 31B | 2.37 s | 1.45 s | 0.481 s | **0.155 s** |
| Qwen3.8-27B | 1.72 s | 1.01 s | 0.454 s | **0.189 s** (0.100 s in GPU calls) |

All 411 (Gemma) and 506 (Qwen) weight tensors fit: 17.5 and 17.2 GiB, 12.7 GiB on the 16 GB card. The logit of the
top token is unchanged (27.484 and 16.356). The CUDA kernels are also checked against the CPU kernels on random
matrices for every type, on both cards (`TestCUDAKernelsMatchCPU`). What is not done: the remaining Go code
(0.088 s of Qwen's 0.189 s), keeping activations on the card between matrix products, a KV cache, and batching of
decode steps; so these are single-token pass times, not a tokens-per-second figure for a running server.

## Long context: KV cache and incremental decode

`NewState()` / `Step(state, newTokens)` (both models) feed only the new tokens, so a decode step no longer re-runs the
whole sequence. Qwen keeps the delta-net state and the last 3 conv rows per GDN block (constant size) plus K/V rows for
its 16 full-attention blocks. Gemma keeps K/V rows per layer and drops rows older than the 1024-token window on its
50 sliding layers; only its 10 global layers grow with the context. Attention and the delta-net heads run on all CPU
cores. Checked: stepping token by token gives the same logits as one full pass (`TestQwenStepMatchesForward`,
`TestGemmaStepMatchesForward`, the latter with the window cut to 4 so the trimming is exercised).

The cache is stored as half floats (`TestHalfRoundTrip`; the Paris checks and the step-versus-full checks still
pass on it) and grows with the context: Qwen 64 KiB per token, Gemma global layers 80 KiB per token (computed from
the head sizes, not measured). At a 262,144-token context that is about 16 GiB and 20 GiB, next to 17 GiB of weights,
so on this PC (24 GB of VRAM in total) the maximum context still does not fit on the GPUs; the cache lives in system
RAM (60 GB), which fits. An 8-bit cache would halve it again (not done).
Decode time against context is in the prefill section below (`TestDecodeVsContext`, build tag `cuda`, up to 4096 tokens; longer is not measured).

### Prefill (measured: Qwen3.8-27B, 512 filler tokens in one session, both GPUs, 32 CPU threads)

| change | prefill, 512 tokens | of that, in GPU calls |
|---|---|---|
| float kernel, one input per weight read | 59.9 s | 44.3 s |
| 8 inputs per weight read (still float) | 60.0 s | not split |
| int8 inputs + `dp4a` for batches of 4 or more | 25.5 s | 10.0 s |
| row-major delta-net state update, 4-accumulator attention over the half cache | 23.8 s (21 tokens/s) | 10.0 s |
| attention for batches of 4 or more on the GPU (`dg_attn`, online softmax, cache copied to the card per call) | 20.4 s | 10.3 s |
| delta-net recurrence on the GPU (`dg_gdn`, one block per value head, state column in registers) | 18.4 s (28 tokens/s) | 11.2 s |
| each layer's weights on one card (placement was tensor by tensor, 42 of 64 FFN blocks were split) and the feed-forward block run on the card (`dg_ffn`: x up once, gate/up/hidden stay there, one result back) | 16.9 s, then 13.2 s after the placement fix | 9.8 s |
| q/k/v and qkv/gate/beta/alpha share one upload of their input (`dg_multi`) | 12.9 s | 9.5 s |
| int8 tensor-core matmul (`mma.sync m16n8k32`, 64 rows x 32 inputs per block, weights decoded once per tile into shared memory) | **7.4 s** (69 tokens/s) | 4.1 s |

Data crossing the bus in that run (`MatvecStats`): 15 GB (7 GB up, 8 GB down) at the start, 5.6 GB at the end. Matmul kernel time: 7.4 s, then 1.9 s. Gemma 4 31B, same test: 19.3 s (27 tokens/s), of which only 4.8 s was in GPU calls; the profile showed 10.9 s in `ropeNeox`, which recomputed `Pow` and `Sincos` for every head of every token. Computing the angles once per position (`ropeAngles`, same arithmetic, same logits) brought it to **8.6 s (60 tokens/s)**, 4.7 s of it in GPU calls, and Gemma's decode step to 0.226 s. A decode step (one token, 512 tokens of context) was 0.148 s for Qwen and 0.267 s for Gemma before the next change: measured on real weights, the one-token k-quant matmuls read only 112 to 225 GB/s (the BF16 output matrix reached 427 GB/s, the 5060 Ti's limit). Sending 1 to 3 inputs through the int8 `dp4a` kernel (one block per row, input quantized once) raised that to 210 to 390 GB/s and the step to **0.104 s for Qwen (about 10 tokens/s) and 0.182 s for Gemma**; 0.056 s of Qwen's is the matmul kernels, 0.042 s Go code (decode attention and the delta-net still run on the CPU, with the cache in host memory). The next step moved the rest onto the card: the K/V cache of every unwindowed attention layer (Qwen's 16, Gemma's global ones) and each delta-net block's state now live in VRAM next to that layer's weights (`devKV`, `devS`; the cache only uploads its new rows and grows by doubling, the state is read back only if the CPU has to continue it), and the attention and delta-net kernels serve single tokens. Measured: Qwen decode step **0.083 s at 512 tokens of context (12 tokens/s) and 87 ms at 2048** (143 ms before, when the CPU read the whole cache each step); Gemma 0.156 s (its 50 sliding layers still attended on the CPU for single tokens, 1024 rows each). Those layers now keep a ring of window + 512 slots on the card (`attn_kernel` takes the ring base and capacity; the cache re-uploads only new rows, with wraparound), and Gemma's decode step is **0.089 s at 512 tokens of context, 0.103 s at 2048 and 0.115 s at 4096** (Qwen: 0.085, 0.088, 0.097 s; `TestDecodeVsContext`). The ring costs about 600 MB for the 50 sliding layers; batches over 512 tokens use a temporary upload or the CPU. Card memory is not seen by the garbage collector: call `Free()` on a `QwenState`/`GemmaState` when done (finalizers are only a backstop; without it a second model loaded in the same process ran out of memory). Resident caches cost VRAM (Qwen 64 KiB per token); if an allocation fails the call falls back to the CPU. Decode inputs are now int8-quantized like the batched ones; `TestCUDAKernelsMatchCPU` checks sizes 1, 2, 3, 5, 8, 19 against the CPU kernels on the same quantized inputs, and `TestDecodeBandwidth` prints the per-matrix rates.

The int8 path quantizes each input in groups of 32 (scale max|x|/127), so batched results differ slightly from the
float path; `TestCUDAKernelsMatchCPU` compares the GPU against the CPU kernels run on the same quantized inputs. Single
tokens (decode) still use the exact float kernels. One decode step with 512 tokens of context: 0.167 s; with 2048 tokens
of context 0.191 s (measured before the prefill changes). Gemma 4 31B, same 512-token prefill (`TestPrefill512Gemma`): 47.8 s (11 tokens/s) after the int8 changes, 34.4 s (15 tokens/s, 13.0 s in GPU calls) with attention on the GPU too; one decode step afterwards 0.277 s. Not measured: 4096 tokens.
The two GPU kernels (the delta-net one is Qwen only; the attention one also serves Gemma, with its sliding window and attention scale 1) agree with the CPU loops: `TestCUDAAttnMatchesCPU`
(2e-4, including window cases) and, for the recurrence, the model-level checks (`TestQwenOnGPU`: Paris logit 16.323 against 16.356 on the CPU;
`TestQwenStepMatchesForward`). What is left outside GPU calls is about 7 s, mostly `silu`/`sigmoid` of the causal conv and
the per-call copies of the activations, the state (3 MiB per layer) and the K/V cache.

## Qwen3.8-27B with the whole forward pass on the GPUs (`EnableResident`, build tag `cuda`)

`Qwen.EnableResident(maxCtx)` puts every layer on the GPUs: norms, the gated delta-net (conv, the recurrence with its 128 x 128 state per
head kept in registers, gated norm), attention with a half-float K/V cache on the card, the FFN and the residuals. Layers are split over the
cards in proportion to their measured copy bandwidth (here 29 and 35 layers; the 2.4 GB output matrix goes on the last card), so only the
hidden state (n x 5120 floats) crosses the bus, twice per call. Embedding lookup and the final norm stay in Go. Same prompts as above, RTX 5060 Ti +
RTX 4060, `TestQwenResident`:

| | before (Go around GPU matmuls) | resident, dp4a matmuls | resident, int8 tensor-core matmuls (16+ tokens) |
|---|---|---|---|
| decode, 1 token | 167 ms | 110 ms | 107-112 ms (unchanged: 1 column); **62 ms (16 tok/s)** once single tokens also use the int8 `dp4a` kernels instead of the float ones (same 16 greedy tokens) |
| prefill, 512 tokens | 23.8 s (21 tok/s) | 8.5 s (60 tok/s) | **2.74 s (187 tok/s)** |

Checked against the Go CPU path: top token and logits of a 6-token prompt (`Paris`, 16.357 vs 16.329, max difference 0.11) and a 23-token prompt
that takes the tensor-core path (19.720 vs 19.727, max difference 0.14); the differences come from the int8 inputs. Where the time went before the
tensor-core step (512 tokens, `DG_PROF=1` prints the per-section GPU time): FFN 5.9 s of 8.5, gated delta-net projections 1.5 s, attention
projections 0.4 s. Limits: one session at a time, K/V cache for `maxCtx` tokens on the cards, head size 256 and a 128-wide delta-net head are
assumed.

Later steps (same machine): 128-column tiles for the tensor-core matmuls took the 512-token prefill from 2.74 s to **1.56 s (326 tok/s)**; the chunk
size no longer matters (2048 tokens: 301 / 314 / 315 tok/s at chunks of 256 / 512 / 1024), so prefill is bound by the matmul kernels, not by
weight reads. A warp-per-row kernel and a precomputed activation-sum for the Q4_K minimum term were tried and were not faster (66 vs 62 ms;
1.82 vs 1.57 s) and were dropped. The copy bandwidth of the two cards is 252 and 387 GB/s (read plus write), which puts the floor of one
decode step at about 54 ms with the cards working one after the other, so 62 ms is near it. Attention for 1-8 new tokens on a long context is split
over up to 32 blocks per head and combined (`attn_combine_k`): decode at 15,000 tokens of context 79 -> 66.5 ms (63 / 63.5 / 64.6 / 66.5 ms
at 1k / 4k / 8k / 15k). Prefill still slows with depth (324 / 295 / 264 / 221 tok/s at 1k / 4k / 8k / 15k).
The split gives the same top token as the single-block kernel but not the same logits (largest difference 0.39 on a 1500-token context, the same
size as the noise between a 10-token call and a 1-token call of the same text, 0.13 at shallow depth).

### Later prefill work (same machine, 512-token prefill unless stated)

| change | prefill |
|---|---|
| 128-column tiles (above) | 1.56 s |
| Q4_K matmul takes 8 consecutive k values per thread in the `mma` (the k order inside an `mma` is free if weights and activations agree), one 64-bit load for each operand | 1.38 s |
| Q6_K matmul reads each weight word once for the four k groups it serves | 1.32 s (388 tok/s) |

Chunks of one call are now pipelined over the two cards (the first card runs chunk k+1 while the second runs chunk k; results are bit-identical
to running the chunks one call at a time, `TestQwenPipelineEqual`), and prefill attention runs on the tensor cores (f16 `mma`, fp32
accumulation, 16 tokens x the 6 query heads of a kv head per block, so each K/V tile is read once for 6 heads). Prefill speed of 1,024-token
calls at depth, tokens per second:

| context already in the cache | 0-1k | 3-4k | 7-8k | 14-15k |
|---|---|---|---|---|
| start of this round (scalar attention, one card at a time) | 324 | 294 | 263 | 221 |
| pipelined over the cards | 494 | 446 | 393 | 323 |
| + tensor-core attention | **508** | **497** | **482** | **458** |

The tensor-core attention differs from the scalar one in the largest logit by 0.44 on a 1,300-token text (same top token); two runs of one
1,500-token text that only differ in how it is cut into chunks differ by up to 2.1 in the largest logit (same top token) - the f16 attention
probabilities and the int8 matmul inputs make the exact values depend on the chunking. Not done: shared-memory staging of the activations in the
matmul kernels (the FFN is still about half of prefill time at about 25 TOPS for Q4_K and 14 for Q6_K, a small part of the int8 peak of these cards).

### After merging `main` (second tensor-core matmul)

`main` has its own dense GPU path (`mma_gemm`, shared-memory tiles, activation sums, used from 4 columns). Both paths now share `matvec_core`: it keeps
`main`'s dp4a kernel below 4 columns and `main`'s `mma_gemm` for 4-15 columns and for Q5_K / Q8_0, and uses `mma_q` for Q4_K / Q6_K from 16 columns. Same
machine, `TestQwenDecodeDepth`, prefill at 1k / 4k / 8k / 15k context: `mma_gemm` alone 314 / 310 / 304 / 294 tok/s, with `mma_q` 504 / 493 / 480 / 455 tok/s.
`mma_gemm` alone also made the split-attention check noisier (largest logit difference 2.2 against 0.5 with `mma_q`, same top token). Decode (62 ms) and speculative
decoding (22-24 tok/s) are unchanged. `main`'s non-resident 512-token prefill (`TestPrefill512`) went from 6.8 s to 6.1 s.

### Speculative decoding with the model's own MTP block (`EnableMTP`, `Generate(..., spec=true)`)

The MTP block (Q8_0, 0.45 GB) runs on the last card with its own attention cache. After each accepted token it drafts the token after the
next one; one 2-token pass of the main model checks the draft (the second token costs almost nothing because a pass is bound by reading
the weights). A right draft yields two tokens per pass. A wrong one rolls the gated delta-net state back to a copy taken after the first
token of the pass (`dg_qwen_restore`; the attention cache needs no rollback because later calls overwrite those rows). Greedy decoding, 64
new tokens after a short prompt, `TestQwenSpecDecode`, plain decode against speculative, same session:

| prompt | plain | speculative | drafts right | tokens identical to plain |
|---|---|---|---|---|
| capitals (3 shots) | 15.5 tok/s | **23.9 tok/s** | 32 of 32 | 64 of 64 |
| short story | 15.5 | **21.9** | 29 of 35 | 64 of 64 |
| Python function | 15.4 | **23.1** | 31 of 33 | 64 of 64 |

Three prompts and 64 tokens each, so this is a demonstration, not an acceptance rate for real workloads. The tokens agree here, but a
2-token pass rounds its int8 inputs the same way a 1-token pass does only up to sum order, so on other text a late token can differ from plain decode.
Per cycle the MTP step costs about 15 ms outside the main pass (its 5120 x 10240 input projection and a second read of the 2.5 GB output matrix
for its logits, both through host round trips).

## What is in the repo

- `pkg/modelschema/dense_gguf.go` - `ReadDensePlan(path)` reads a `qwen35` or `gemma4` GGUF and returns the
  per-layer kinds and head geometry, or an error naming the first missing tensor. Tested on the two real files
  (`go test ./pkg/modelschema -run DensePlan`; the tests skip when the files are not in `Strata-data/models`).
- `third_party/llama.cpp-dense/` - `gemma4.cpp`, `qwen35.cpp` and `delta-net-base.cpp` from llama.cpp
  (MIT, commit in `VERSION.txt`). They are the reference the port follows, not code that is compiled.

## Measured from the files

| | Qwen3.8-27B (`qwen35`) | Gemma 4 31B (`gemma4`) |
|---|---|---|
| decoded blocks | 64, plus 1 trailing MTP (`nextn`) block | 60 |
| layer pattern | GDN, GDN, GDN, full attention (every 4th) | 5 sliding, 1 global (every 6th) |
| attention (heads / KV heads / head dim) | 24 / 4 / 256, query is gated (`attn_q` is 2x wide) | sliding 32 / 16 / 256; global 32 / 4 / 512, no `attn_v` (V = K) |
| n_embd / n_ff / vocab | 5120 / 17408 / 248320 | 5376 / 21504 / 262144 |
| norms per block | attn_norm, post_attention_norm | attn_norm, post_attention_norm, ffn_norm, post_ffw_norm, q/k norm, `layer_output_scale` |

## Layer recipe (from the vendored reference)

Gemma 4, per block: `x = rms(h)*attn_norm`; Q = `wq x`, per-head rms-norm with `attn_q_norm`, rope;
K = `wk x` (V = `wv x`, or K when `attn_v` is absent), K per-head rms-norm with `attn_k_norm`, V a plain rms-norm
without weight, K rope; global layers use `rope_freqs` as the frequency factors; attention scale is
`f_attention_scale`; `o = wo attn`, then `o = rms(o)*post_attention_norm`, `a = h + o`; FFN is
`down(gelu(gate x) * up x)` on `rms(a)*ffn_norm`, then `*post_ffw_norm`, `h = a + ffn`, then `h *= layer_output_scale`.
The input embedding is scaled by `sqrt(n_embd)`. Logits are soft-capped: `30 * tanh(logit / 30)`.

Qwen3.8-27B GDN block: `beta = sigmoid(ssm_beta x)`; `g = ssm_a * softplus(ssm_alpha x + ssm_dt)`; causal
conv over `attn_qkv x` with `ssm_conv1d` (kernel 4); q and k are L2-normalised; the delta-rule recurrence updates
a per-head state; the output is gated-normed with `ssm_norm` and the `attn_gate` branch, then `ssm_out`.
Full-attention blocks multiply the attention output by `sigmoid(gate)` from the doubled query projection.

## Order of work

1. ~~CPU reference forward~~ (done, above; checked by the model's own answer, not against llama.cpp).
2. CUDA kernels only for what is not already in the engine (Gemma soft-cap, dual-head attention, K=V global layers).
3. A dense decode loop with layer offload, since each file is about 17.6 GB and the target card has 16 GB.
4. Measure, then tune.

Open points: the installed llama.cpp (build 8370) loads neither file, so there is still no logit-level comparison
with llama.cpp (needs b8637 or later).
