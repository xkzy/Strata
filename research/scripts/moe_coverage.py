"""Expected distinct experts touched per layer by B tokens, and the weight-sweep time over PCIe.
usage: python research/scripts/moe_coverage.py
ASSUMPTIONS: routing is uniform and independent across tokens (real routing is skewed, which lowers coverage);
host->device 2.9 GB/s was measured on this machine; shard1 29.6 GB / shard2 28.8 GB from the model card.
"""
E, k = 256, 10
for B in (1, 2, 4, 8, 16, 32, 64, 128, 256, 1024):
    d = E * (1 - (1 - k / E) ** B)
    print(f"B={B:5d}  distinct experts/layer = {d:6.1f}  ({d / E:5.1%} of {E})")
# bytes per routed expert element are mixed-quant; use file-level ratio instead: routed share 51.8% of 58.4 GB
file_gb, routed_share, bw = 58.4, 0.518, 2.9
for frac in (10 / 256, 0.5, 0.9):
    t = file_gb * routed_share * frac / bw
    print(f"streaming {frac:5.1%} of routed experts at {bw} GB/s: {t:5.1f} s per full pass")
print(f"K=32 level passes at 90% coverage: {32 * file_gb * routed_share * 0.9 / bw / 60:.1f} min per 1024-token block (lower bound, ignores compute)")
