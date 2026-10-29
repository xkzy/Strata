# Flint (preview)

`flint-UD-Q4_K_XL.gguf` (arch `flint`: 49 blocks, 256 experts with 6 used, sliding-window and full-attention layers with
per-head sinks) runs on one NVIDIA card with the experts that do not fit read from pinned host memory.

    cmake --build build --target flint_engine
    bin/strata serve --config strata-flint.json        # edit "--native" to point at the GGUF

`strata serve` starts `build/flint_engine`, which speaks the same line protocol as `bin/densengine`. On the first start
the server unpacks the GGUF's tokenizer and chat template into `<gguf>.tokenizer/`.

## Measured

On an RTX 5060 Ti (16 GB) with `--max-context 8192` (fp32 KV cache, 1.6 GB):

- load: 4 s with a warm file cache, 38 s cold; all 21.5 GiB of experts in pinned host memory, plus a 10.4 GiB pool of expert slots on the card (4390 of 12032 experts)
- decode: about 32 ms per token (31 tok/s) over four 300-token replies; 52 ms before the top-k kernel was parallelised, 41 ms before the expert pool. The pool is filled by use: an expert a layer routes to that is not in the pool is copied into the least-used slot, so the hot experts of every layer stay. Hit rate in those replies: 57% (an fp16 KV cache gave 4711 slots and the same decode time)
- prefill: about 100 tok/s (3298-token prompt answered in 31 s with 60 new tokens; 2 minutes when every token ran its experts alone); the same prompt again reuses the cached prefix (3.7 s for 60 new tokens)
- prefill chunk (`--chunk`, default 2048): every chunk copies the experts it needs from host RAM, so a bigger chunk spreads that cost over more tokens. 8000 fresh tokens (RTX 5060 Ti, 0.5 GiB of K/V on the card): 59.5 s at 256 (134 tok/s), 46 s at 512, 38 s at 1024, 34 s at 2048 (235 tok/s). At 2048 the next 8000 tokens (positions 8000-16000) took 48 s (165 tok/s) because attention grew with depth. The attention kernel now serves the 8 query heads of a kv head from one read of each K/V row (tiles of 32 positions in shared memory): the same 8000 tokens took 27.3 s (293 tok/s) and the next 8000 took 30.9 s (259 tok/s), same card and settings. Q8_0 matrices (the dense projections and shared expert) now run on the tensor cores (int8 `mma`) when a call has 16 or more columns: the first 8000 tokens took 18.4 s (400-470 tok/s) and the next 8000 took 22.0 s (334-375 tok/s). The result matches the old kernel to float rounding (relative difference 3e-7, 4100 rows x 77 columns, random weights). The Q5_K expert matrices do the same (5-bit values as int8 `mma`, the min term from an `mma` against ones): the first 8000 tokens took 14.6 s (490-600 tok/s) and the next 8000 took 18.1 s (400-460 tok/s); relative difference to the old kernel 3e-7. Copying the missing experts now runs on a second stream while the experts already on the card compute (a copy used to wait for the previous expert's work and the next expert's work waited for the copy): 11.1 s and 14.6 s for the same two runs (630-800 tok/s and 490-570 tok/s). Each token's experts are summed in expert order whichever ran first, so the result does not depend on what the cache held; per-layer outputs of a 3000-token prompt were bit-identical between the overlapped and the serial run. A STOP is read between chunks, so a cancel waits for the chunk in flight (about 10 s at depth)
- prefix reuse: the engine keeps the tokens of the last request and runs only what differs. The runtime's notes (retrieved context, verified facts) are placed after the conversation, in the last user turn, so they no longer change the tokens before them: the same 16370-token document asked a second question took 9 s, against 216 s when the notes sat in the leading system block
- answers: the capital of France, and the value of item 77 in a 3298-token list, both correct

The GPU forward agrees with the Go reference (`pkg/densecpu/flint.go`) on the top token at 9 of 9 steps; its logits differ
by what q8_1 rounding of the activations alone does to the reference (`src/kernels/flint_model_parity.cpp` has the numbers).

## Limits

- Context: the shipped config says `--max-context 0`, which is the model's limit (262144) or what RAM holds, whichever is less.
  Of the 49 layers 35 slide a 513-token window and keep a ring of window + chunk positions; the other 14 keep K/V in fp16
  (`--kv-fp32` for fp32): the first positions on the card (`--kv-device-gib`, default 1.5 GiB = 56172 positions) and the rest
  in pinned host RAM, which the attention kernel reads over the bus. Attention is split over positions (flash-decoding), so
  the span is not bound by shared memory. At 262144 positions the K/V cache is 7 GiB (14 GiB in fp32). fp16 vs fp32 K/V:
  `flint_model_parity` passes 9/9 with TV 0.056 0.002 0.359 0.005 0.051 0.000 0.008 0.006 0.050 (fp32: 0.090 0.001 0.343
  0.005 0.004 0.000 0.010 0.004 0.022; that run's TV floor from q8_1 alone is in the harness header).
- Decode time is now the misses: about 120 experts (300 MB) copied over PCIe per token. A larger pool or a
  stored expert profile to warm it would raise the hit rate.
- The pre-tokenizer pattern (digits in groups of three, letters only) is the GLM4 one, not read from the file; checked
  against the Python `regex` module, not against the model's own tokenizer.
- Tool calls come back as text: the output parser does not know flint's `<tool_call>name<arg_key>` format yet.
- Only the card with the most memory is used.

## Several models

Every `strata-*.json` engine config in the working directory is a model, named by its `model_name`. A request whose
`model` names another config's model stops the running engine and starts that one (the tokenizer and chat template
switch with it); a name no config knows goes to the loaded model. A swap waits for the running requests to finish and holds back new ones until the new model is ready.

Model `"auto"` answers on `flint-preview` first; if the reply failed (an error, nothing in it, or cut off at `max_tokens`) it
is redone on `qwen3.8-flash-next-coder-iq1_m`, and the dense `qwen3.8-27b-dense` is the last resort. Models without a
config are skipped. A streamed request stays on the first model, since the client has already seen its start. After an escalated reply
the server swaps back to `flint-preview` in the background.
Measured: `flint-preview` to `qwen3.8-27b-dense` and back, 10 s for the first request after the swap to the dense model
(warm file cache).
