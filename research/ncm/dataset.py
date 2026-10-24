# Copied unchanged from the strata-ncm worktree (earlier session, job a719c9cc), branch of the same repo: ncm/dataset.py
"""Load a directory of per-prompt `--dump-final-r` captures (ncm/capture_batch.sh) into sequences.
Each sequence: positions are contiguous (asserted), rows finite (asserted)."""
import glob, os
import numpy as np
from capture import load_spec_residuals


def load_sequences(cap_dir):
    out = {}
    for f in sorted(glob.glob(os.path.join(cap_dir, "final_r_*.bin"))):
        pos, tok, nxt, rows = load_spec_residuals(f)
        if len(pos) == 0:
            continue
        if not (np.diff(pos) == 1).all():
            raise ValueError(f"{f}: positions are not contiguous: {pos[:12].tolist()}...")
        if not np.isfinite(rows).all():
            raise ValueError(f"{f}: non-finite residuals")
        out[os.path.basename(f)[8:-4]] = dict(pos=pos, tok=tok, nxt=nxt, rows=rows)  # rows [T, hc, n_embd]
    return out
