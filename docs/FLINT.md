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
- decode: about 32 ms per token (31 tok/s) over four 300-token replies; 52 ms before the top-k kernel was parallelised, 41 ms before the expert pool. The pool is filled by use: an expert a layer routes to that is not in the pool is copied into the least-used slot, so the hot experts of every layer stay. Hit rate in those replies: 57% (an fp16 KV cache gave 4711 slots and the same decode time, and flipped the top token at parity step 2, so the cache is fp32)
- prefill: about 100 tok/s (3298-token prompt answered in 31 s with 60 new tokens; 2 minutes when every token ran its experts alone); the same prompt again reuses the cached prefix (3.7 s for 60 new tokens)
- answers: the capital of France, and the value of item 77 in a 3298-token list, both correct

The GPU forward agrees with the Go reference (`pkg/densecpu/flint.go`) on the top token at 9 of 9 steps; its logits differ
by what q8_1 rounding of the activations alone does to the reference (`src/kernels/flint_model_parity.cpp` has the numbers).

## Limits

- Context is capped at 24000 tokens (attention scores sit in shared memory) and 8192 in the shipped config (fp32 KV, 200 KB per token).
- Decode time is now the misses: about 120 experts (300 MB) copied over PCIe per token. A larger pool or a
  stored expert profile to warm it would raise the hit rate.
- The pre-tokenizer pattern (digits in groups of three, letters only) is the GLM4 one, not read from the file; checked
  against the Python `regex` module, not against the model's own tokenizer.
- Tool calls come back as text: the output parser does not know flint's `<tool_call>name<arg_key>` format yet.
- Only the card with the most memory is used.

## Several models

Every `strata-*.json` engine config in the working directory is a model, named by its `model_name`. A request whose
`model` names another config's model stops the running engine and starts that one (the tokenizer and chat template
switch with it); a name no config knows goes to the loaded model. The swap is refused while a request is running.
Measured: `flint-preview` to `qwen3.8-27b-dense` and back, 10 s for the first request after the swap to the dense model
(warm file cache).
