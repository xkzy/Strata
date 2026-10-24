# Copied unchanged from the strata-ncm worktree (earlier session, job a719c9cc), branch of the same repo: ncm/head.py
"""The teacher's LM head in PyTorch, frozen, transcribed from src/kernels/cuda/gr.cu (gr_read, FP32 path) and src/core/layer.cpp (lm_head):

    xn[c]  = R[c] * rsqrt(mean(R[c]^2) + eps) * w_norm[c]          c = 0..hc-1
    lo     = silu((xn_flat @ w_down.T) / hc)                       w_down [hc_lr, hc*n_embd]
    gated  = xn * sigmoid(lo @ w_up.T)                             w_up   [hc*n_embd, hc_lr]
    mixed  = mean_c gated[c]
    logits = output.weight @ mixed                                 Q6_K, dequantised to float32 here

The engine quantises `mixed` to Q8_K before the matmul; this does not, so logits differ by rounding. `check()` measures how often that flips the argmax.
Weights are read from the GGUF shard by byte offset (ncm/param_count.read_header), because the Python `gguf` reader rejects this file's type 42.
"""
import numpy as np, torch
from gguf import GGMLQuantizationType as Q
from gguf.quants import dequantize
from param_count import read_header

SHARD = "/home/khing/Downloads/Strata-data/models/coder-IQ1_M/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"
HC, N_EMBD, HC_LR, EPS = 4, 2560, 320, 9.999999974752427e-07


def _read(f, ty, dims, off):
    """dims are GGUF order (ne0 fastest), so the numpy shape is reversed."""
    n = int(np.prod(dims))
    q = {0: (Q.F32, 4 * n), 30: (Q.BF16, 2 * n), 14: (Q.Q6_K, n // 256 * 210)}[ty]
    f.seek(off)
    raw = np.frombuffer(f.read(q[1]), dtype=np.uint8)
    if q[0] == Q.F32:
        return torch.from_numpy(raw.view(np.float32).copy()).reshape(dims[::-1])
    if q[0] == Q.BF16:
        return torch.from_numpy(raw.view(np.int16).copy()).view(torch.bfloat16).float().reshape(dims[::-1])
    return torch.from_numpy(dequantize(raw.reshape(-1, 210), q[0])).reshape(dims[::-1])


class TeacherHead(torch.nn.Module):
    def __init__(self, device="cpu", shard=SHARD):
        super().__init__()
        offs = {}
        _, ts = read_header(shard, offs)
        info = {n: (d, t) for n, d, t, _ in ts}
        with open(shard, "rb") as f:
            for key, name in (("w_norm", "output_hc_norm.weight"), ("w_down", "output_hc_down.weight"),
                              ("w_up", "output_hc_up.weight"), ("w_out", "output.weight")):
                d, t = info[name]
                self.register_buffer(key, _read(f, t, d, offs[name]).to(device))
        self.w_norm = self.w_norm.reshape(HC, N_EMBD)
        for p in self.buffers():
            p.requires_grad_(False)

    def mix(self, R):  # [N, hc, n_embd] -> [N, n_embd]
        xn = R * torch.rsqrt(R.pow(2).mean(-1, keepdim=True) + EPS) * self.w_norm
        lo = torch.nn.functional.silu(xn.flatten(1) @ self.w_down.T / HC)
        return (xn * torch.sigmoid(lo @ self.w_up.T).reshape(-1, HC, N_EMBD)).mean(1)

    def forward(self, R):
        return self.mix(R) @ self.w_out.T  # [N, n_vocab]


@torch.no_grad()
def argmax(head, R, chunk=256):
    return torch.cat([head(R[i:i + chunk]).argmax(-1) for i in range(0, len(R), chunk)])


if __name__ == "__main__":  # python ncm/head.py CAP_DIR [N=4096] [device=cuda:1] : argmax vs the teacher's recorded next token
    import sys
    from dataset import load_sequences
    cap, n, dev = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 4096, sys.argv[3] if len(sys.argv) > 3 else "cuda:1"
    rows, nxt = [], []
    for s in load_sequences(cap).values():
        rows.append(s["rows"]); nxt.append(s["nxt"])
    rows, nxt = np.concatenate(rows)[::len(np.concatenate(nxt)) // n][:n], np.concatenate(nxt)[::len(np.concatenate(nxt)) // n][:n]
    head = TeacherHead(dev)
    am = argmax(head, torch.tensor(rows, device=dev)).cpu().numpy()
    print(f"python head vs recorded next token: {(am == nxt).mean():.4f} over {len(nxt)} positions")
