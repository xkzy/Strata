"""Multiscale representations over continuous sequences [B, T, D] (hidden states / embeddings, never raw token ids).

LiftingFilterBank: learned analysis/synthesis via the lifting scheme, so reconstruction is exact for ANY learned
predict/update weights (up to float error). ResidualVQ: z_hat = sum_k codebook_k[c_k].
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LiftingFilterBank(nn.Module):
    """One level: even/odd split, detail d = odd - P(even), coarse s = even + U(d). Odd T is padded by edge
    replication; pass the original length to synthesize() to crop. Coarse has ceil(T/2) steps."""

    def __init__(self, dim, taps=3):
        super().__init__()
        self.pred = nn.Conv1d(dim, dim, taps, padding=taps // 2, groups=dim, bias=False)
        self.upd = nn.Conv1d(dim, dim, taps, padding=taps // 2, groups=dim, bias=False)
        for c in (self.pred, self.upd):  # init near Haar: P = identity tap, U = half
            nn.init.zeros_(c.weight)
        self.pred.weight.data[:, :, taps // 2] = 1.0
        self.upd.weight.data[:, :, taps // 2] = 0.5

    def analyze(self, x):
        if x.shape[1] % 2:
            x = torch.cat([x, x[:, -1:]], 1)
        even, odd = x[:, 0::2].transpose(1, 2), x[:, 1::2].transpose(1, 2)
        d = odd - self.pred(even)
        s = even + self.upd(d)
        return s.transpose(1, 2), d.transpose(1, 2)

    def synthesize(self, s, d, length=None):
        s, d = s.transpose(1, 2), d.transpose(1, 2)
        even = s - self.upd(d)
        odd = d + self.pred(even)
        x = torch.stack([even, odd], -1).flatten(2).transpose(1, 2)  # interleave
        return x if length is None else x[:, :length]


class ResidualVQ(nn.Module):
    def __init__(self, dim, levels=4, codes=256, beta=0.25):
        super().__init__()
        self.books = nn.ModuleList(nn.Embedding(codes, dim) for _ in range(levels))
        self.beta = beta

    def forward(self, z, n_levels=None):
        """Returns (z_hat, codes [B,T,K], quantization loss). Straight-through gradient to z."""
        res, out, codes, loss = z, torch.zeros_like(z), [], z.new_zeros(())
        for book in list(self.books)[: n_levels or len(self.books)]:
            idx = torch.cdist(res.reshape(-1, res.shape[-1]), book.weight).argmin(-1).view(res.shape[:-1])
            q = book(idx)
            loss = loss + F.mse_loss(q, res.detach()) + self.beta * F.mse_loss(res, q.detach())
            out, res = out + q, res - q.detach()
            codes.append(idx)
        return z + (out - z).detach(), torch.stack(codes, -1), loss

    def decode(self, codes):
        return sum(b(codes[..., k]) for k, b in enumerate(self.books[: codes.shape[-1]]))
