# Distillation plan (option b: Strata coder as frozen teacher, small replacement modules)

Scope after the teacher-size finding ([parameter_budget.md](parameter_budget.md)): the 116.5 B teacher is never trained or copied.
Only small modules are trained against activations captured from the unmodified engine. A same-budget standalone student is **out of
scope** on this hardware; the ±2% criterion does not apply to these modules and is reported as unmet by design.

## Capture path (exists; no engine changes)

`build-fat/strata generate ... --dump-layers L.bin --dump-logits G.bin --dump-routing R.bin` (flags at `src/program/generate.cpp:721-726`).
- Layers: no header, per position `(48+1) x 4 x 2560` float32 = 1,998,848 B/position. Read with `ncm/capture.py` (tested on a synthetic file only).
- Logits: header then rows of `n_vocab` float32 (`:4799-4820`); *header exact layout not yet parsed here*.
- Routing: records of `layer, k, ids, weights`; *record layout not yet parsed here*.
- Constraint from source: `--dump-layers` is written by `session_loop`, so `--no-capture` cannot be used (`:4719`); `--dump-routing` needs the expert pool (`:4650`); `--logits-stride >1` needs `--max-new 1`.
- Cost: ~2 MB/position, so 1 M positions of layer dumps is ~2 TB. Capture a chosen subset of layers (e.g. 12, 24, 36, 48) by post-filtering, and use sequences of a few thousand tokens.

**First real capture, 2026-10-24 (RTX 4060 8 GB, after the driver recovered).** `ncm/capture_short.sh`: prompt `9707,11,1879,0`, 2 new
tokens (`1892 1042`), 3.2 tok/s decode, 1.9 s to first token, `--expert-cache 256 --prefill 16` (the defaults left 2 MiB free and the
8192-token prefill buffers failed to allocate). What each dump gave:

| Dump | Result |
|---|---|
| `--dump-routing` | **Usable ids.** 144 records = 48 layers x 3 passes, k=10, ids in 0..255 (matches the pruned 256-expert model). Weights are **all zero**: the verify-window path fuses router weights into the kernel and writes zeros (`generate.cpp:973`). |
| `--dump-residual` | **Empty of data.** Header 4 x 2560, every float is 0 on the `--spec` path. |
| `--dump-logits` | **Empty file** (0 bytes) on the `--spec` path, although the run prints "logits dumped". |
| `--dump-layers` | **Cannot be combined** with this pack: the native pack requires `--spec`, which needs the single-graph token path, and `--dump-layers`/`--dump-halves` disable it (`generate.cpp:5141`). |

So the engine as shipped yields **no usable hidden states or logits** for this pack, only expert ids.

**Engine patch, same day.** `nvcc` 13.2 is at `/usr/local/cuda-13.2` (not on PATH; needs `g++-15` as host compiler, as the existing
`build-fat` used). `src/program/generate.cpp` now writes the verify window's final residual rows for `--dump-final-r` on the `--spec`
path (record layout in `ncm/capture.py:load_spec_residuals`; row validity tracked with `n_valid` so rows after a rejected draft are
dropped). Built to `build-ncm/` with `-DCMAKE_CUDA_ARCHITECTURES=89` (the 4060 only; not tested on other GPUs). Re-capture with the
patched binary (`BIN=build-ncm/strata bash ncm/capture_short.sh`):

- 2 decode positions, rows `[2, 4, 2560]` float32, all finite and nonzero, mean |x| 0.26, max 2.87.
- Generated tokens `1892 1042` are the same as with the unpatched binary, so the dump does not change the teacher's output
  (2 tokens only: weak evidence).
- Prompt positions are **not** captured (they run through the batched prefill path, which has no dump); only positions produced
  by verify windows. A dataset therefore comes from the teacher's own continuation (`--max-new N`) and each sequence's prompt rows are lost.
- Logits are still not dumped on this path (only the argmax token per row). Distillation to teacher *distributions* is not possible yet;
  hidden-state and argmax-token targets are.
- Untouched: `--dump-layers` (per-layer residuals) remains incompatible with this pack.

The `ncm/capture.py` readers are tested on synthetic files; the routing reader and the new residual reader also parsed real captures.

**Earlier attempt 2026-10-23, blocked by a GPU/driver fault.** `ncm/capture_short.sh` (unmodified teacher, 4-token prompt, `--spec 2 --prefill auto`
because the native pack requires both) got past argument checks and failed at start-up with
`native embedding: cannot pin 322 MiB: unknown error, nor place it in VRAM`. The same machine state fails outside Strata:
`torch.cuda` init raises `CUDA unknown error`, and `nvidia-smi` prints `Unable to determine the device handle for GPU0`. A second attempt on the other card
(`--gpu 1`, the RTX 4060; the faulted one is the 5060 Ti at `01:00.0`) failed identically, and PyTorch with only the 4060's UUID in
`CUDA_VISIBLE_DEVICES` also fails CUDA init, so the fault is in the shared driver state, not one card. So no
activations exist yet. Not retried, and the repo's GPU reset scripts were not run; the fault needs the owner (driver reload, reboot or
reset) before any capture. Original text follows. Loading the 58 GB teacher needs the engine's expert tiering. This machine's GPU reported
`Unable to determine the device handle for GPU0` and the engine config in the repo was tuned for a 5060 Ti + 4060 pair. A first capture
should be a short prompt (tens of tokens) to validate formats before any dataset run, and should be started by the owner, watching for GPU errors.

**Recurrence 2026-10-24 19:26 (+07), after hours of working GPU use.** While `ncm/train_head.py` was running on the 5060 Ti (PyTorch, a 248,320-wide
matmul per step), the kernel logged `NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus`, then a stream of `rpcSendMessage failed`.
`nvidia-smi` again cannot get a handle for `0000:01:00.0`; `torch.cuda` init now fails with `CUDA unknown error` even with only the 4060's UUID in
`CUDA_VISIBLE_DEVICES`, as on 2026-10-23. The 4060 itself still answers `nvidia-smi`. Earlier the same day the card ran the engine and two multi-minute
PyTorch jobs without trouble, so it faults intermittently, not on every use. Not retried, no reset or reboot attempted; anything after this point that needs CUDA waits
for the owner (driver reload or reboot; PCIe slot, riser and power are the usual suspects for Xid 79). Work that needs no GPU continues on the CPU.

## Stages for the modules

1. **Format check.** One short capture; parse logits/routing headers; assert layer-dump size = positions x 1,998,848 B.
2. **Multiscale tokenizer on residuals.** Input: the hc-stream residual (mean over 4 streams first, or a learned stream projection) at one layer, `[T, 2560]`.
   Train `LiftingFilterBank` + `ResidualVQ` for reconstruction + commitment loss only. Metric: relative reconstruction error per level and bits per position vs RVQ-only (the required ablation).
3. **Coarse-to-fine predictor.** Predict next-block codes from the prompt state. Metric: top-1 code accuracy, and agreement of the decoded block with the teacher's greedy tokens. Only this agreement, not code accuracy, says anything about language quality.
4. **MoE head** (`ncm/moe.py`) vs a dense head at equal total and equal active parameters, trained on the same targets.
5. **Verification.** Teacher greedy-agreement of drafted blocks via the existing greedy-exact verify window (`core/verify.cpp`). This is greedy agreement, **not** exact speculative sampling.

Losses in use: reconstruction (MSE), commitment/codebook (in `ResidualVQ`), load balance (in `SparseMoE`), and KL to teacher logits only for stages that output token distributions. No placeholder terms.

## Known limitations

- The teacher is a 1–3 bit quantized checkpoint: captured activations carry quantization error and are not those of the original bf16 model.
- Activations come from a model with hyper-connection residual streams and GDN layers; a module trained on layer-L residuals only works when plugged in at that layer.
- Nothing here yields a standalone model. A real student that runs without the teacher requires a full architecture and a new engine forward graph (see the audit).
