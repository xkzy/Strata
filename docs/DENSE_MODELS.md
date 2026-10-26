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
Decode-time-versus-context numbers are not measured yet (`TestDecodeVsContext`, build tag `cuda`).

### Prefill (measured: Qwen3.8-27B, 512 filler tokens in one session, both GPUs, 32 CPU threads)

| change | prefill, 512 tokens | of that, in GPU calls |
|---|---|---|
| float kernel, one input per weight read | 59.9 s | 44.3 s |
| 8 inputs per weight read (still float) | 60.0 s | not split |
| int8 inputs + `dp4a` for batches of 4 or more | 25.5 s | 10.0 s |
| row-major delta-net state update, 4-accumulator attention over the half cache | **23.8 s** (21 tokens/s) | 10.0 s |

The int8 path quantizes each input in groups of 32 (scale max|x|/127), so batched results differ slightly from the
float path; `TestCUDAKernelsMatchCPU` compares the GPU against the CPU kernels run on the same quantized inputs. Single
tokens (decode) still use the exact float kernels. One decode step with 512 tokens of context: 0.167 s; with 2048 tokens
of context 0.191 s (measured before the prefill changes). Gemma 4 31B, same 512-token prefill after the int8 changes (`TestPrefill512Gemma`): 47.8 s (11 tokens/s), 11.3 s of it in GPU calls, one decode step afterwards 0.273 s. Not measured: 4096 tokens.
The remaining 13.8 s of the prefill is Go code on the CPU: the attention loops and the delta-net recurrence, which only
GPU kernels will speed up.

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
