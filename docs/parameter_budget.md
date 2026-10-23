# Parameter budget

Measured with `python ncm/param_count.py <shards...>` (reads GGUF headers; counts each tensor name once).

| Quantity | Value | How |
|---|---|---|
| Tensors | 1,224 | both shards |
| **P_teacher (unique stored elements)** | **116,514,464,640** | sum of shape products |
| of which PLE n-gram table (`per_layer_token_embd.weight`, 160 x 320,001,536) | 51,200,245,760 | shard 2 |
| of which transformer + embeddings + output | 65,314,218,880 | shard 1 |
| ±2% target band on P_teacher | 114.18 B to 118.85 B | |

Caveats that change what "equal budget" means:

- The PLE table is a hash lookup (one row read per token per head), not matmul weight. Counting it as "learned parameters" makes a
  ±2% student look cheaper than it is. Report both **P_teacher = 116.5 B** and **P_teacher_excl_PLE = 65.3 B**; a fair comparison needs the second.
- Token embedding and `output.weight` are separate tensors (both 2560 x 248,320), so they are not tied; both are counted.
- Routed experts here are the pruned set (256 per layer, not 512).
- P_active, P_aux were **not computed**: the active count depends on top-10 of 256 routing plus shared experts plus the 36 GDN / 12 QSA
  layer mix, and has not been derived or measured. It is left blank rather than estimated.

**Feasibility:** a student of ~116 B (or 65 B) parameters cannot be trained or even held in bf16 on the available 8 GB GPU / 60 GB RAM
(65 B in bf16 is ~130 GB). A same-budget student is therefore out of reach here; see [ncm_status.md](ncm_status.md).
