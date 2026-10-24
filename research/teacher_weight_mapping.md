# Teacher weight mapping

**Reading of constraint 3 ("teacher need not stay loaded"):** no separate teacher or verifier process is needed in production. A student whose trunk weights are copied from the teacher plus learned NCM parts counts as weight-derived. Two modes are kept apart:
- **Standalone:** no verification; the student defines its own output distribution.
- **Drafts verified by the same trunk:** exact (the output equals the teacher's greedy output), hybrid, reference only.

## 1. What can be reused, by measurement
| Teacher part | Shape / type | Reuse in an NCM student | Evidence |
|---|---|---|---|
| Token embedding | 248,320 × 2,560, IQ4_XS | Encoder `wte` via PCA to d = 128 (keeps 18.9% of energy), then trained | [M] ncm_tokenizer |
| Output head | 248,320 × 2,560, Q6_K | Tied decoder rows via PCA to d = 128 (keeps 11.8%), frozen | [M] |
| Head collapse path (hc norm, silu down, sigmoid up, mean) | BF16 | Reproduced exactly in PyTorch: 0.9985 argmax agreement over 4,096 positions | [M] head.py |
| Trunk (48 layers, 65.3B ex-PLE) | mixed IQ/Q | Copied unchanged if used as the generator | [M] sizes |
| Dense projections (attention, SSM-out, gate) | Q4_K–Q6_K | **No wavelet structure**: 99% energy needs ~71% of coefficients in Haar/db4, same as a random basis | [M] wavelet_stats |
| Routed experts | 256 × (640 × 2,560) per layer | **No shared structure** in layers 3, 20, 47: mean expert 0.4–0.7% of energy, cluster residual equals random labels | [M] expert_share |

## 2. Parameter accounting (tokenizer prototype, C = 256)
| Class | Count |
|---|---|
| Transformed from teacher, then trained (`wte`) | 31.78 M |
| Transformed from teacher, frozen (`Oz`) | 31.78 M |
| Learned from scratch: decoder block, position tables, output projection (32,007,937 − 31,784,960) | 0.22 M |
| Codebooks 6 × 256 × 128, EMA buffers (not gradient-trained, not in the count above) | 0.20 M |
| Reported total trainable (includes `wte`) | 32.0 M |
At C = 4096 the codebooks are 3.1 M. These are the tokenizer's parts only; no generator was built.

## 3. What does not transfer [M]
- Wavelet bases give no weight compression on these matrices, so they offer nothing to initialise a coarse-to-fine hierarchy from.
- Experts do not share weights, so a shared-plus-residual split saves nothing in weight space. Hidden-unit permutations are untested [U].
- The 9 layers whose down-projection is custom type 42 cannot be read from Python [M].

## 4. Quality note
Mapping through a 128-dimension PCA keeps 19% and 12% of the embedding and head energy. The decoder still reaches 88% on unquantised input, so the PCA is not the dominant loss, but it caps what any such student can reproduce. [M]
