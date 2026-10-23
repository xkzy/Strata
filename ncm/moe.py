"""Sparse top-k MoE with hard expert capacity. Overflowing assignments are dropped (that token's contribution
from that expert is zero, the residual path carries it) and counted, never silently reassigned."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseMoE(nn.Module):
    def __init__(self, dim, hidden, n_experts, k=2, capacity_factor=1.25):
        super().__init__()
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.w1 = nn.Parameter(torch.randn(n_experts, dim, hidden) * dim**-0.5)
        self.w2 = nn.Parameter(torch.randn(n_experts, hidden, dim) * hidden**-0.5)
        self.n, self.k, self.cf = n_experts, k, capacity_factor
        self.stats = {}

    def forward(self, x):  # x [N, D] (flatten batch*time first)
        N = x.shape[0]
        probs = F.softmax(self.router(x), -1)
        w, ix = probs.topk(self.k, -1)
        w = w / w.sum(-1, keepdim=True)  # normalize over selected experts
        cap = max(1, int(self.cf * N * self.k / self.n))
        out, load, dropped = torch.zeros_like(x), torch.zeros(self.n), 0
        for e in range(self.n):
            tok, slot = (ix == e).nonzero(as_tuple=True)
            if len(tok) > cap:  # deterministic: keep highest routing weight first
                keep = w[tok, slot].argsort(descending=True)[:cap]
                dropped += len(tok) - cap
                tok, slot = tok[keep], slot[keep]
            load[e] = len(tok)
            if len(tok):
                y = F.gelu(x[tok] @ self.w1[e]) @ self.w2[e]
                out.index_add_(0, tok, y * w[tok, slot, None])
        frac = F.one_hot(ix[:, 0], self.n).float().mean(0)
        self.aux_loss = self.n * (frac * probs.mean(0)).sum()  # Switch-style load balance
        ent = -(probs * probs.clamp_min(1e-9).log()).sum(-1).mean()
        self.stats = {"capacity": cap, "load": load, "dropped": dropped, "router_entropy": float(ent),
                      "utilization": float((load > 0).float().mean())}
        return out
