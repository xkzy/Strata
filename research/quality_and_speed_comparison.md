# Quality and speed comparison

No NCM variant ran end to end, so there is **no measured tok/s for any NCM variant**. The table separates what was measured from what is a bound.

| System | tok/s decode | Quality vs teacher | Output distribution | Basis |
|---|---|---|---|---|
| Teacher, default `strata serve` (autotune → 1 pool worker) | 31.7 | reference | exact | [M] |
| Teacher, `-no-autotune`, 4 workers, adaptive speculation | **~44** | reference | exact (verified) | [M], strongest baseline |
| Teacher, forced full windows T = 2 / 4 / 6 | 39.7 / 38.5 / 30.4 | reference | exact | [M] |
| Heads-only drafts, 8 ahead, verified by the trunk | ≈ 13.7 | exact | exact | [D] from 1.82 tokens per pass at 133 ms; not run end to end |
| NCM levels 1, 2, 4, 8, 16, 32, zero quality loss | ≤ 34.9 | – | – | [D] upper bound |
| NCM two levels (4 then 32), zero quality loss | ≤ 63.7 | – | – | [D] upper bound; quality at this setting unmeasured [U] |
| NCM tokenizer v4, finest level, C = 4096 | – | 78.4% (cap2 test) / 80.1% (new-text test) token reconstruction; 23.6 bits/token | defines a different distribution | [M] |

## Distribution statement
- Verified drafts (heads checked by the same trunk): **preserves** the teacher's greedy output.
- Standalone NCM (codes decoded without verification): **approximates or redefines** it; the measured tokenizer reconstructs 78–80% of held-out tokens at its finest level (62% at C = 256), so it does not preserve it.

## Other measured costs
- Extra memory for the tokenizer prototype: 32.0 M trainable parameters (~128 MB fp32, of which 31.8 M is the embedding table) plus the frozen 31.8 M table. No engine memory measured.
- TTFT/prefill: ~448 tok/s on the baseline; unchanged because the engine is untouched.
- Long-context scaling and sustained GPU stability of any NCM variant: not measured.

## Reading
Every speed figure for NCM above is an upper bound that assumes zero quality loss at each level. The measured quality (heads-only 0.40/0.20/0.13 agreement; tokenizer 78–80%) is far from that, so none of the bounds applies as a prediction.
