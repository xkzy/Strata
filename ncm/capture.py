"""Reader for the teacher activations that `strata generate --dump-layers PATH` writes (src/program/generate.cpp:4829,11226).

Layout, verified from source: no header; per position, (n_layers+1) x hc x n_embd float32 (index 0 is the input x,
then R after each layer). For the Coder GGUF: 48 layers, hc 4, n_embd 2560 => 49*4*2560*4 B = 1,998,848 B per position.
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
