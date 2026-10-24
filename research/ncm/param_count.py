# Copied unchanged from the strata-ncm worktree (earlier session, job a719c9cc), branch of the same repo: ncm/param_count.py
"""Count unique learned parameters in GGUF shards by reading headers only (no quant-type enum needed).

usage: python ncm/param_count.py shard1.gguf [shard2.gguf ...]
"""
import struct, sys, collections

_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def _val(f, t):
    if t in _SCALAR:
        fmt = _SCALAR[t]
        return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
    if t == 8:
        return f.read(struct.unpack("<Q", f.read(8))[0]).decode("utf-8", "replace")
    if t == 9:
        et, n = struct.unpack("<IQ", f.read(12))
        return [_val(f, et) for _ in range(n)]
    raise ValueError(f"bad kv type {t}")


def read_header(path, offsets=None):
    """-> (kv, tensors). If `offsets` is a dict it is filled with name -> byte offset in the file (data start already added)."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path}: not GGUF")
        _, nt, nkv = struct.unpack("<IQQ", f.read(20))
        kv = {}
        for _ in range(nkv):
            k = f.read(struct.unpack("<Q", f.read(8))[0]).decode()
            t = struct.unpack("<I", f.read(4))[0]
            v = _val(f, t)
            if not (isinstance(v, list) and len(v) > 64):  # skip huge vocab arrays
                kv[k] = v
        tensors = []
        for _ in range(nt):
            name = f.read(struct.unpack("<Q", f.read(8))[0]).decode()
            nd = struct.unpack("<I", f.read(4))[0]
            dims = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            ty, off = struct.unpack("<IQ", f.read(12))
            n = 1
            for d in dims:
                n *= d
            tensors.append((name, dims, ty, n))
            if offsets is not None:
                offsets[name] = off
        if offsets is not None:
            align = kv.get("general.alignment", 32)
            start = (f.tell() + align - 1) // align * align
            for k in offsets:
                offsets[k] += start
    return kv, tensors


def count(paths):
    seen, total, by_type, kv = {}, 0, collections.Counter(), {}
    for p in paths:
        k, ts = read_header(p)
        kv.update(k)
        for name, dims, ty, n in ts:
            if name in seen:  # tied / duplicated tensor name across shards: count once
                continue
            seen[name] = n
            total += n
            by_type[ty] += n
    return total, seen, by_type, kv


if __name__ == "__main__":
    total, seen, by_type, kv = count(sys.argv[1:])
    print(f"tensors={len(seen)} unique_params={total:,}")
    print("by ggml type id:", dict(by_type))
    ple = sum(n for k, n in seen.items() if "ple" in k.lower() or "ngram" in k.lower())
    print(f"ple/ngram tensors params={ple:,}")
    for k in ("general.architecture", "qwen4exp.block_count", "qwen4exp.expert_count", "qwen4exp.expert_used_count"):
        print(k, kv.get(k))
