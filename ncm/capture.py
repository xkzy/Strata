"""Readers for the teacher dumps that `strata` (generate mode) writes. Formats verified from src/program/generate.cpp
and, for routing and the residual, against one real capture (docs/distillation_plan.md).

- layers  (--dump-layers):  no header; per position (n_layers+1) x hc x n_embd float32 (index 0 = input x). NOT usable with
  the native pack: it needs --spec, and --dump-layers disables the token graph --spec requires (generate.cpp:5141).
- routing (--dump-routing): records of int32 layer, int32 k, k x int32 expert ids, k x float32 weights (generate.cpp:926-976).
  The verify-window path (used with --spec) writes ZERO weights; only ids are real there.
- residual (--dump-residual): int32 hc, int32 n_embd header then hc x n_embd float32.
"""
import numpy as np

N_LAYERS, HC, N_EMBD = 48, 4, 2560


def load_layers(path, n_layers=N_LAYERS, hc=HC, n_embd=N_EMBD, layers=None):
    """Returns float32 [positions, len(layers), hc, n_embd]; `layers` selects indices into 0..n_layers (memory-mapped)."""
    per = (n_layers + 1) * hc * n_embd
    a = np.memmap(path, dtype="<f4", mode="r")
    if a.size % per:
        raise ValueError(f"{path}: {a.size} floats is not a multiple of {per} per position (wrong dims or truncated)")
    a = a.reshape(-1, n_layers + 1, hc, n_embd)
    return np.asarray(a if layers is None else a[:, list(layers)])


def load_routing(path, n_layers=N_LAYERS):
    """Returns (layer [N], ids [N,k], weights [N,k]); validates layer range and that the file is whole records."""
    raw = np.fromfile(path, dtype="<i4")
    if raw.size < 2:
        raise ValueError(f"{path}: empty")
    k = int(raw[1])
    rec = 2 + 2 * k
    if k <= 0 or raw.size % rec:
        raise ValueError(f"{path}: {raw.size} words is not a multiple of the {rec}-word record for k={k}")
    r = raw.reshape(-1, rec)
    if (r[:, 1] != k).any() or r[:, 0].min() < 0 or r[:, 0].max() >= n_layers:
        raise ValueError(f"{path}: records disagree on k or have a layer outside 0..{n_layers - 1}")
    return r[:, 0].copy(), r[:, 2:2 + k].copy(), r[:, 2 + k:].view("<f4").copy()


def load_residual(path):
    raw = open(path, "rb").read()
    hc, n_embd = np.frombuffer(raw[:8], "<i4")
    x = np.frombuffer(raw[8:], "<f4")
    if x.size != hc * n_embd:
        raise ValueError(f"{path}: header says {hc}x{n_embd}, file holds {x.size} floats")
    return x.reshape(hc, n_embd)
