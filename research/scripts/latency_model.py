#!/usr/bin/env python3
"""Expected-latency model for draft-and-verify decoding (research/mathematical_analysis.md §5.1, §5.4).
Inputs marked MEASURED come from repo files cited in the docs; ASSUMED are sweep variables, not data."""
import math

E, K = 256, 10
P_ROUTED, P_ACTIVE_NONEXPERT = 60.398e9, 6.640e9 - 60.398e9 * 10 / 256   # from budget_model.py
ACTIVE_BYTES = 5.096e9             # exact, from bytes_model.py (per-tensor ggml types): 4.114 GB dense + 0.982 GB routed
print(f'active bytes/token = {ACTIVE_BYTES/1e9:.3f} GB (exact); non-expert active params = {P_ACTIVE_NONEXPERT/1e9:.2f} B')

def e_tokens(alpha, g):
    """Expected tokens committed per verify pass with g drafted tokens, i.i.d. per-position acceptance alpha
    (Leviathan et al. 2211.17192, §3 expected-tokens form). The +1 is the bonus/correction token."""
    return (1 - alpha ** (g + 1)) / (1 - alpha) if alpha < 1 else g + 1

def coverage(n, k=K, e=E):
    return 1 - (1 - k / e) ** n   # expected fraction of experts touched by n tokens, uniform independent routing

print('\nE[tokens/pass] (iid alpha):')
print('alpha  ' + ' '.join(f'g={g}' .rjust(6) for g in (1, 2, 3, 4, 6, 8)))
for a in (0.5, 0.6, 0.7, 0.77, 0.85, 0.95):
    print(f'{a:5.2f}  ' + ' '.join(f'{e_tokens(a, g):6.2f}' for g in (1, 2, 3, 4, 6, 8)))
print('check: alpha=0.77,g=3 ->', round(e_tokens(.77, 3), 2), '(docs/DETAILS.md:1294 measured 2.4-3.2 tokens/pass, MTP up to 3 drafts)')

def rho(n, expert_share, overlap=0.0):
    """Pass-time ratio T(n)/T(1) when a fraction expert_share of single-token pass time is expert traffic and the
    expert traffic scales with distinct experts touched; overlap in [0,1] interpolates between 'no sharing across
    positions' (0 -> n*k experts, capped by E) and 'perfect sharing' (1 -> k experts)."""
    distinct = (1 - overlap) * min(E, n * K) + overlap * K
    return (1 - expert_share) + expert_share * distinct / K

print('\nbreak-even: speculation wins iff E[tok](alpha,g) > rho(g+1). rho for fully expert-bound pass, by overlap:')
print('n(=g+1)  ' + ' '.join(f'ov={o:.1f}'.rjust(8) for o in (0, 0.5, 0.8, 0.95)))
for n in (2, 3, 4, 5, 7, 9):
    print(f'{n:6d}   ' + ' '.join(f'{rho(n, 1.0, o):8.2f}' for o in (0, 0.5, 0.8, 0.95)))
print('\nspeedup = E[tok]/rho at alpha=0.77, g=3 (n=4), by expert_share x overlap:')
for s in (0.3, 0.6, 0.9):
    print(f'expert_share={s}: ' + ' '.join(f'ov={o}:{e_tokens(.77,3)/rho(4,s,o):5.2f}x' for o in (0, 0.5, 0.8, 0.95)))

# Roofline for plain AR decode if ALL active weights were VRAM-resident (upper bound, ignores KV and launch limits)
for name, bw in (('RTX 5060 Ti 16GB (vendor spec 448 GB/s, ASSUMED)', 448e9), ('RTX 4060 8GB (vendor spec 272 GB/s, ASSUMED)', 272e9)):
    print(f'\nroofline {name}: {bw/ACTIVE_BYTES:6.1f} tok/s per sequence, no speculation, all-active-bytes-resident')
print('MEASURED same machine (strata-coder-iq1_m.json.bak4, tuned 2026-10-18): decode 44.8 tok/s (spec on), prefill 335 tok/s')
print('=> implied passes/s at 2.4-3.2 tok/pass: %.1f-%.1f ; pass time %.0f-%.0f ms' % (44.8/3.2, 44.8/2.4, 1000*3.2/44.8, 1000*2.4/44.8))

# KV / state read per decode step at ctx T (hybrid: 12 attention layers, indexer top-2048)
for T in (32768, 262144):
    kv_read_sel = 12 * 2 * 256 * 2 * 2 * min(T, 2048)   # selected K,V bytes read by attention (fp16), per token
    print(f'ctx={T}: attention K/V bytes read/step (top-2048, fp16) = {kv_read_sel/1e6:.1f} MB ; full cache = {12*2*256*2*2*T/2**30:.2f} GiB')

# Teacher-label generation cost for distillation (prefill-rate scoring)
PREFILL = 335.0   # MEASURED tok/s, same file above
for D in (1e7, 1e8, 1e9):
    print(f'teacher scoring {D:.0e} tokens at {PREFILL:.0f} tok/s = {D/PREFILL/86400:8.1f} days')
