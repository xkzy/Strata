# Plan: online uncensoring (abliteration) in Strata

Source: Maxime Labonne, *Uncensor any LLM with abliteration* (huggingface.co/blog/mlabonne/abliteration), based on
Arditi et al., *Refusal in LLMs is mediated by a single direction*. Goal: an **uncensor mode** the user switches on
per request, with the direction **computed once at load and applied only while the mode is on** — no weight edits, no
retraining, no cost when it is off.

**Status**

- **Phase 0 — implemented (2026-10-11).** `--abliterate` with no vector file now fails fast with an actionable
  message instead of steering by a fabricated "canonical" direction; `generate_canonical_refusal_direction` was
  removed from `include/strata/kernels/fast_abliteration.hpp` and its test case replaced by a serialization-only
  round trip. `--abliteration-vector` rows are now unit-normalized at load so `--abliteration-strength` means the
  same thing for every file (the strength plumbing itself is Phase 1). Verified here: `test_fast_abliteration` 6/6,
  engine target `strata` builds, `go test ./pkg/runconfig/` ok. Not yet done: a runtime exercise of the new error
  path (needs a model pack this checkout does not have).

## 1. What the technique does (from the blog)

- Refusal is mediated by one direction in the residual stream. Collect activations on harmful vs harmless prompts at
  the last token, per layer: `r = normalize(mean(harmful) - mean(harmless))`, then rank layers and keep the best.
- **Online (inference-time) intervention** — this is what Strata needs: after each layer, for every token,
  `h <- h - alpha * (h . r) r`. Nothing in the weights changes; the model simply cannot represent the refusal
  direction while the intervention is on.
- **Offline alternative**: orthogonalize the weights that write to the residual (`embed_tokens`, attention `o_proj`,
  MLP `down_proj`). Permanent, zero runtime cost, but it edits the model file.
- The blog measures a quality drop from abliteration (benchmarks fall across the board) and heals most of it with DPO.
  Strata ships no training path, so we accept a small measured shift on neutral text instead (see §2) and must keep
  measuring it.

## 2. Where Strata already stands (audited, not a plan — current state)

Strata already has the blog's online intervention, under the name **experimental speed projection (ESP)**:

| Piece | Location | State |
| --- | --- | --- |
| Real refusal-direction vector: 47 unit directions of 2,560, applied to layers 4-44, `mode: "project"` | `data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf` + `.json` | Ships; its README: declines 1 of 50 vs 50 of 50 on its test set |
| The intervention kernel `h <- h - s (h . v) v` on every hyper-connection stream, after each steered layer's FFN write | `include/strata/kernels/cvec.hpp`, `src/kernels/cuda/cvec.cu` (SYCL mirror in `sycl/`), applied in `src/core/layer.cpp`, `src/prefill/prefill.cpp` (fused `gr_write_cvec_norm_rs`), `src/core/verify.cpp` | Done, with `src/kernels/cvec_parity.cpp` vs a double-precision reference |
| Direction tables **computed once at load**: GGUF `direction.<l>` read, unit-normalized per layer, `dir`/`s` uploaded, session files bound to the digest | `load_control_vectors()` in `src/program/generate.cpp` | Done — this is the "calculate once" the design calls for |
| Per-request switch, graphs identical either way, off = bit-identical to stock | `cvec_set_enabled` (`src/kernels/cuda/cvec.cu`), request keys `cvec=0|1` and `abliterate=0|1` (`src/program/generate.cpp`), conversation/prompt caches keyed on the flag (`want_cvec`, `sl.cvec`, `conversations.best(..., want_cvec)`) | Done (measured in `bench/results/2026-09-27-esp/`: 65/65 first tokens and three 256-token generations token-identical with the switch off) |
| API surface | body `experimental_speed_projection` (`pkg/server/webui.go`), body `abliterate` + `abliteration_strength` (`pkg/server/api_openai.go`, `api_anthropic.go`), headers `X-Strata-Abliterate` / `X-Strata-Abliteration-Strength` (`pkg/server/generate.go`), engine request `abliterate=1` (`pkg/engineipc/orchestrator.go`) | Wired, but two names for one feature |
| Run config / setup | keys `abliterate`, `abliteration_strength` -> `--abliterate`, `--abliteration-strength` (`pkg/runconfig/editor.go`); `--experimental-speed-projection` in `setup.py` | Run-config editor only; setup asks only about ESP |
| Engine flags | `--abliterate` (forces `--cvec-mode project`), `--abliteration-strength S`, `--abliteration-vector FILE` (`src/program/generate.cpp`) | Present, see gap G1 |
| CPU helpers + offline math (normalize, project-and-subtract, **contrastive mean-difference**, `.ablr` file) | `include/strata/kernels/fast_abliteration.hpp`, tests `tests/kernels/test_fast_abliteration.cpp` (CMake `test_fast_abliteration`) | Built and run here: 6/6 tests pass. Used by the loader and tests only — the GPU path is the cvec kernel |

Measured cost of the projection on the same text: 0.2-0.4% per token; mean KL 0.063 nats (max 4.1), top-1 changes at
10% of positions, perplexity +15% on code, +2.3% on a document, +0.4% on chat (Q2_0, RTX 5070,
`bench/results/2026-09-27-esp/`).

## 3. Gaps (why an "integrate" plan is still needed)

- **G1 — the synthetic default direction is not a refusal direction.** With `--abliterate` and no vector file, the
  loader fills the tables from `generate_canonical_refusal_direction()` — a deterministic sine/cosine basis weighted
  by depth (`src/program/generate.cpp`). Nothing derives that from activations, so it cannot mediate refusal:
  projecting it out perturbs the model (quality loss) without uncensoring it. This path must not be a default.
- **G2 — per-request strength is dropped.** Go writes `abliteration_strength=<f>` into the engine request line, but
  the engine only parses `cvec`/`abliterate`; strength is read only from startup flags (baked into the tables at
  load). Also, strength changes the model state, so it must key the conversation cache like the on/off flag does.
- **G3 — two names, one feature.** `experimental_speed_projection` and `abliterate` are the same engine mechanism.
  Docs and the web UI expose only the ESP name; setup has no uncensor prompt; the API accepts both fields.
- **G4 — one model only.** The shipped vector is for the original Qwen3.8-Flash-Next (its README says so). Swift 1.5,
  the Coder and Unsloth variants have no derivation tooling; the blog's steps 1-3 (collect, mean-difference, rank)
  are not automated anywhere in this repo, and TransformerLens cannot load this custom MoE + hyper-connection
  architecture anyway.
- **G5 — no refusal measurement of our own.** `bench/results/2026-09-27-esp/` measured KL, perplexity and speed, not
  refusals; the "1 of 50" number is the vector package's claim. Repo rule: no claims without a measurement.
- **G6 — no tests for the API surface.** No Go test covers the `abliterate` body fields/headers, and nothing asserts
  `abliterate=1` behaves identically to `cvec=1`.

## 4. Architecture decision: compute once, apply on enable

This is the design the engine already implements for ESP, and the plan standardizes on it:

1. **At load (once):** read the refusal direction (GGUF or `.ablr`), unit-normalize per layer, build the `dir`/`s`
   tables, `cvec_upload`, and decide *at capture time* which layers carry the kernel (`Cvec::covers()` — it changes
   where the residual write happens, so it can never be per request). Bind the session-file identity digest to the
   tables so a cached conversation is never reused under a different vector.
2. **Per request (cheap):** `abliterate=1|0` flips one device int (`cvec_set_enabled`) between requests; captured
   graphs are the same either way; prompt/conversation/prompt-cache resumes only reuse state computed with the same
   flag. Off is bit-identical to the stock engine (already verified).
3. **Never** recompute directions in serve mode, never touch weights, never fork the forward pass per request.

`fast_abliteration.hpp` stays the offline/tooling implementation (load, contrastive math, files, tests); the single
runtime implementation remains the CUDA/SYCL cvec kernel.

## 5. Phases

### Phase 0 — Correctness and naming (no new capability)

- Make `--abliterate` without a vector **fail fast** with an actionable message, or default to the packaged
  ESP GGUF when the model matches; delete `generate_canonical_refusal_direction` from the load path (and its test
  case) so no configuration can steer by a fabricated direction. Keep `--abliteration-vector` for real files
  (`.ablr` or the ESP GGUF via `--control-vector-scaled`).
- Pick one user-facing name: **uncensor mode (abliteration)**; keep `experimental_speed_projection` and
  `abliterate` as accepted API aliases mapping to the same field.
- Tests: `test_fast_abliteration` green; new check that `--abliterate` with no vector exits 2 with the message.
- Acceptance: no path in the tree can load a direction that was not read from a file.

### Phase 1 — Per-request strength (the missing half of "apply when enabled")

- Engine: parse `abliteration_strength=<f>` in the request loop (clamp 0..2, same as ESP's on/off handling). Keep a
  host staging copy of the `s` table; when the value changes, rescale and upload it **between requests** with the
  same synchronization `cvec_set_enabled` uses; fold the value into the conversation-cache identity so state computed
  at one strength is never reused at another. Default remains the load-time `--abliteration-strength`.
- Server: validate `abliteration_strength` (400 outside 0..2) in both body fields and headers; the orchestrator line
  already writes the key, so no protocol change.
- Tests: strength change drops the cache exactly once (pattern: `esp_serve.py` on/off/on); off at any strength is
  token-identical to stock; re-run `esp_kl.py` to confirm the 0.2-0.4%/token band still holds with strength scaled.

### Phase 2 — Derivation tooling for every shipped variant

- Engine: an offline capture mode (e.g. `--dump-resid PREFIX`) that runs a batch of token prompts and writes the
  last-token residual per layer per hyper-connection stream as f32. Not on the serve path.
- Tool: `tools/refusal_direction.py` — numpy only (requirements.txt pins numpy; no torch/TransformerLens in this
  repo): harmful + harmless prompt sets -> captured activations -> per-layer `normalize(mean_h - mean_u)` -> rank by
  `|mean|` (the blog's selection) -> pick layer range -> write a `controlvector` GGUF with `direction.<l>` via
  `tools/gguf_writer.py`, plus a sidecar JSON of scores in the shape of the ESP JSON.
- Calibration: derive the vector for the original Qwen3.8-Flash-Next and compare per-layer cosine similarity with
  the shipped ESP vector before trusting the pipeline on other variants (Swift 1.5, Coder, Unsloth).
- Acceptance: a derived vector for a variant exists only with its own measured refusal numbers (Phase 3).

### Phase 3 — Evaluation harness (produce our own numbers)

- `tools/uncensor_eval.py`: N harmful + N harmless prompts through the served API with the mode on and off;
  count refusals (refusal-phrase match, same idea as the blog's `I cannot`/`I can't` blacklist); report decline rate,
  neutral-text KL/perplexity (reuse `esp_kl.py`'s teacher-forced path) and decode tok/s.
- Record results as `bench/results/<date>-uncensor/README.md` in the existing bench style (hardware, quant, flags).
- Acceptance targets (state only what is measured): refusal rate drops sharply on vs off; neutral-text shift and
  per-token cost stay in the band already documented for ESP.

### Phase 4 — Setup, web UI, docs

- `setup.py`: an `--uncensor on|off` question (default off) that writes `--abliterate` plus the vector path into
  `strata-<model>.json`, mirroring `--experimental-speed-projection`; `tools/strata_mcp.py` gets the same flag.
  Run `python tools/test_setup_*.py` after touching setup.py.
- Web app: rename the Sampling-drawer switch to "Uncensor mode (abliteration)" on the existing `projectionLoaded()`
  logic; the Monitor's ESP/stock badge and the About line follow.
- Docs: a `docs/UNCENSOR.md` section in the repo's style — plain words, what it does, how to turn it on, the
  measured numbers with the machine they came from, and the existing warning that removing refusals removes a safety
  behaviour and the user is responsible for the output. Cross-link from `data/experimental-speed-projection/README.md`
  and `docs/DETAILS.md`.

### Phase 5 — Optional: offline weight orthogonalization (the blog's other method)

- `tools/abliterate_weights.py`: orthogonalize `embed_tokens`, attention `o_proj` and `down_proj` against the chosen
  direction and write a derived GGUF, for users who do not want the 0.2-0.4% runtime cost. Caveat to document:
  the shipped shards are quantized (Q2_0/IQ*), so this needs dequant -> edit -> requant (or an f16 source), and the
  result must be re-measured (perplexity + refusal rate) before it is offered.
- Not required for the online mode; keep it clearly separate so "online" stays the default recommendation.

## 6. Guardrails

- Off by default everywhere (setup, API, web app); per-request switch only changes the device flag, never the model
  files. Keep the responsibility warning in every surface (repo already words it this way in the ESP README).
- Never expose the server beyond `127.0.0.1` without `--api-key` (AGENTS.md) — an uncensor toggle makes that rule
  more important, not less.
- Derived vectors carry the model's license (Qwen Community License 1.0, as the ESP README states).
- Every phase lands with its measurement; no speed or refusal claim without a bench result in `bench/results/`.

## 7. Suggested order and effort

| Phase | Depends on | Main risk |
| --- | --- | --- |
| 0 correctness/naming | — | small, removes a bad default |
| 1 per-request strength | 0 | cache-identity mistakes (mitigated: same pattern as the on/off flag) |
| 2 derivation tool | 0 | capture hook touches the hot engine (keep it offline-only) |
| 3 eval harness | 1 | nothing ships without numbers |
| 4 setup/UI/docs | 0-3 | setup tests must stay green |
| 5 weight tool | 2-3 | quantization round-trip quality |
