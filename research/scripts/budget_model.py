"""Parameter / memory model of the teacher, computed from the GGUF headers (no weights are read).
usage: python research/scripts/budget_model.py   (needs ncm/param_count.py; shard dir is hard-coded below)
Everything marked ASSUMPTION is an inference, not a fact read from the file.
"""
import glob, re, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "ncm"))
from param_count import read_header

M = "/home/khing/Downloads/Strata-data/models/coder-IQ1_M/"
kv, T = {}, {}
for f in sorted(glob.glob(M + "*.gguf")):
    k, ts = read_header(f)
    kv.update(k)
    for n, d, t, c in ts:
        T[n] = c
g = lambda key: kv["qwen4exp." + key]

CLASSES = [  # (name, regex, active_fraction_per_token or None = lookup only)
    ("routed experts", r"ffn_(gate|up|down)_exps", g("expert_used_count") / g("expert_count")),
    ("shared expert", r"ffn_\w+_shexp", 1.0),
    ("router", r"ffn_gate_inp", 1.0),
    ("hyper-connections (layers + head)", r"(hc_|output_hc)", 1.0),
    ("GatedDeltaNet projections + state params", r"(attn_qkv|attn_gate|ssm_)", 1.0),
    ("sparse-indexer attention (q,k,v,o,indexer)", r"(attn_q|attn_k|attn_v|attn_output|indexer)", 1.0),
    ("PLE small tensors", r"^blk\.\d+\.ple_", 1.0),
    ("PLE n-gram table (hash lookup)", r"per_layer_token_embd", None),
    ("token embedding (row lookup)", r"token_embd", None),
    ("output.weight (LM head matmul)", r"^output\.weight", 1.0),
]
seen, rows = set(), []
for name, rx, frac in CLASSES:
    n = sum(c for t, c in T.items() if t not in seen and re.search(rx, t))
    seen |= {t for t in T if re.search(rx, t)}
    rows.append((name, n, frac))
rest = sum(c for t, c in T.items() if t not in seen)
rows.append(("unclassified (norms etc.)", rest, 1.0))
P_total = sum(T.values())
assert sum(n for _, n, _ in rows) == P_total
P_active = sum(n * f for _, n, f in rows if f is not None)
P_nolookup = sum(n for _, n, f in rows if f is not None)

print(f"{'class':46s}{'unique params':>18s}{'share':>8s}{'active/token':>18s}")
for name, n, f in rows:
    print(f"{name:46s}{n:>18,}{n / P_total:>8.1%}{(n * f if f is not None else 0):>18,.0f}")
print(f"{'TOTAL':46s}{P_total:>18,}{'':8s}{P_active:>18,.0f}")
print(f"\nP_T (unique stored)             = {P_total:,}   (+-2% band {0.98 * P_total / 1e9:.2f}..{1.02 * P_total / 1e9:.2f} B)")
print(f"P_T without lookup tables       = {P_nolookup:,}   (the weights that are matmul'd)")
print(f"P_A (matmul-active per token)   = {P_active:,.0f}   = {P_active / P_total:.2%} of P_T, {P_active / P_nolookup:.2%} of P_T w/o lookups")

# ---- decode-state memory model (per sequence) ----
L, Hkv, dk, dv = g("block_count"), g("attention.head_count_kv"), g("attention.key_length"), g("attention.value_length")
n_qsa = sum(1 for r in g("attention.compress_ratios") if r)          # layers with a non-zero compress ratio
n_gdn = L - n_qsa
kv_tok = n_qsa * Hkv * (dk + dv)                                       # elements per token, all attention layers
print(f"\nattention layers (compress_ratio != 0): {n_qsa}, linear-recurrent layers: {n_gdn}; full_attention_interval = {g('full_attention_interval')}")
for b, nm in ((2, "fp16/bf16"), (1, "8-bit")):
    print(f"KV cache, no compression, {nm}: {kv_tok * b:,} B/token -> {kv_tok * b * g('context_length') / 2**30:.2f} GiB at {g('context_length'):,} tokens")
print(f"  (compress_ratio is {sorted(set(g('attention.compress_ratios')))}; what it does to the stored KV is NOT verified here, "
      f"so the 1/4 case = {kv_tok * 2 * g('context_length') / 4 / 2**30:.2f} GiB is a hypothetical)")
print(f"  indexer top_k = {g('attention.indexer.top_k')}: attention reads at most that many keys per query per attention layer")
H, dstate = g("ssm.time_step_rank"), g("ssm.state_size")
dval = g("ssm.inner_size") // H
gdn_state = n_gdn * H * dstate * dval * 4
print(f"GDN recurrent state, ASSUMPTION {H} heads x {dstate} x {dval} fp32 per layer: {gdn_state / 2**20:.1f} MiB total, independent of context")

# ---- training-feasibility arithmetic for a same-budget student (bounded, assumption-labelled) ----
for P, label in ((P_total, "P_T (116.5 B incl. lookup)"), (P_nolookup, "P_T w/o lookup (65.3 B)")):
    opt = P * 16  # ASSUMPTION: bf16 weight 2 + bf16 grad 2 + fp32 master 4 + Adam m,v 8
    flops = 6 * P * 20 * P  # ASSUMPTION: dense, 20 tokens/param (Chinchilla-style), 6*N*D
    sec = flops / (15e12 * 0.4)  # ASSUMPTION: RTX 4060-class 15 TFLOP/s fp32-equivalent peak, 40% utilisation
    print(f"student of {label}: mixed-precision Adam state ~{opt / 1e12:.2f} TB; dense 20 tok/param training ~{flops:.2e} FLOP "
          f"= {sec / 3.15e7:,.0f} GPU-years on one 4060-class GPU")
