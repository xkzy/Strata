#!/usr/bin/env python3
"""Numerical model behind research/02-04. Reads only teacher_gguf_headers.json (verified tensor shapes).
Every assumption is a named constant at the top; nothing here is a measurement of student speed."""
import json, math, collections

H = json.load(open('research/data/teacher_gguf_headers.json'))
kv = H[0]['kv']
tens = [t for s in H for t in s['tensors']]
prod = lambda sh: math.prod(sh)

# ---- verified geometry (GGUF metadata) ----
L, D, E, K = kv['qwen4exp.block_count'], kv['qwen4exp.embedding_length'], kv['qwen4exp.expert_count'], kv['qwen4exp.expert_used_count']
HKV, HD = kv['qwen4exp.attention.head_count_kv'], kv['qwen4exp.attention.key_length']
CR = kv['qwen4exp.attention.compress_ratios']['head']
attn_layers = [i for i, r in enumerate(CR) if r] if len(CR) == L else None
n_attn = len([t for t in tens if t['name'].endswith('attn_q.weight')])
n_ssm = len([t for t in tens if t['name'].endswith('attn_qkv.weight')])

def params(pred):
    return sum(prod(t['shape']) for t in tens if pred(t['name']))

P_total = params(lambda n: True)
P_ple_table = params(lambda n: n == 'per_layer_token_embd.weight')
P_routed = params(lambda n: '_exps.weight' in n)
P_nontable = P_total - P_ple_table
# active per decoded token: everything except routed experts and PLE table, + K/E of routed experts.
# token_embd is a row gather (not a matmul) so it is excluded; output.weight is a full matmul.
P_active = (P_total - P_ple_table - P_routed - params(lambda n: n == 'token_embd.weight')) + P_routed * K / E
print(f'layers={L} attn_layers={n_attn} deltanet_layers={n_ssm} attn_layer_idx={attn_layers}')
print(f'P_total              = {P_total/1e9:8.3f} B  (ε=2% band {0.98*P_total/1e9:.1f}..{1.02*P_total/1e9:.1f} B)')
print(f'P_ple_table          = {P_ple_table/1e9:8.3f} B  ({P_ple_table/P_total:.1%}) lookup, ~0 FLOPs')
print(f'P_routed_experts     = {P_routed/1e9:8.3f} B  ({P_routed/P_total:.1%})')
print(f'P_total w/o PLE      = {P_nontable/1e9:8.3f} B')
print(f'P_active/token       = {P_active/1e9:8.3f} B  (upstream card says ~6B)')
print(f'IQ4_NL table size    = {P_ple_table*4.5/8/1e9:8.2f} GB (shard 2 on disk: 28.80 GB)')

# ---- KV cache: only full-attention layers hold per-token K/V ----
kv_tok = n_attn * HKV * HD * 2 * 2  # layers*heads*dim*(K+V)*fp16 bytes
repo_tok = L * HKV * HD * 2 * 2     # pkg/autotune KVBytesPerToken applied to all 48 layers
for ctx in (32768, 262144):
    print(f'KV @{ctx:>6}: hybrid-correct {kv_tok*ctx/2**30:6.2f} GiB | repo formula x48 layers {repo_tok*ctx/2**30:6.2f} GiB')
print(f'KV bytes/token: {kv_tok} (= {kv_tok/1024:.0f} KiB) vs repo-formula {repo_tok}')
# DeltaNet recurrent state (INFERRED shape: heads(v)=48 x dk=128 x dv=128, fp32), constant in context
ds = n_ssm * 48 * 128 * 128 * 4
print(f'DeltaNet state (inferred, fp32): {ds/2**20:.0f} MiB per sequence, independent of context')

# ---- MoE coverage: fraction of experts touched by n tokens under UNIFORM independent routing ----
print('expert coverage 1-(1-k/E)^n  (uniform-routing upper bound on sparsity benefit):')
for n in (1, 4, 16, 64, 128, 256, 1024):
    print(f'   n={n:5d}  coverage={1-(1-K/E)**n:6.1%}  routed-bytes-read/pass={(1-(1-K/E)**n)*P_routed/1e9:6.1f} B-params')

# ---- Training-compute blocker: 6*N_active*D (dense-equivalent rule of thumb; MoE fwd/bwd ignoring router overhead) ----
NCM_TOKENS = {'LM1B(100M)': 13.36e9, 'OWT(100M)': 127.22e9}  # NCM paper App. C.2 "training-token accounting"
for name, d in NCM_TOKENS.items():
    print(f'NCM budget {name}: {d/1e9:.1f}B positions = {d/100e6:.0f} positions per param')
SUSTAINED_FLOPS = 40e12   # ASSUMPTION (not measured): ~40 TFLOP/s sustained bf16 on one consumer GPU; replace with a measured number
for tokens in (1e9, 1e10, 1.0e11):
    f = 6 * P_active * tokens
    print(f'train tokens={tokens:.0e}: 6*N_act*D = {f:.2e} FLOP -> {f/SUSTAINED_FLOPS/86400:10.1f} GPU-days at {SUSTAINED_FLOPS/1e12:.0f} TFLOP/s')
# optimizer-state memory for training ALL student parameters with AdamW (bf16 weights + fp32 master + 2 fp32 moments + bf16 grad)
bytes_per_param = 2 + 4 + 8 + 2
print(f'AdamW full training state = {bytes_per_param} B/param -> {P_total*bytes_per_param/1e12:.2f} TB for {P_total/1e9:.1f}B params (vs 16+8 GB VRAM)')
print(f'  same, excluding PLE table (frozen/sparse-updated) = {P_nontable*bytes_per_param/1e12:.2f} TB')
