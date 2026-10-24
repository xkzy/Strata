#!/usr/bin/env python3
"""Header-only GGUF reader (no weights loaded). Usage: gguf_dump.py out.json shard1 [shard2 ...]
Tolerates private ggml types (e.g. Strata type 42): only the type id is recorded."""
import struct, sys, json

T = {0: 'u8', 1: 'i8', 2: 'u16', 3: 'i16', 4: 'u32', 5: 'i32', 6: 'f32', 7: 'bool', 8: 'str', 9: 'arr', 10: 'u64', 11: 'i64', 12: 'f64'}
F = {0: '<B', 1: '<b', 2: '<H', 3: '<h', 4: '<I', 5: '<i', 6: '<f', 7: '<?', 10: '<Q', 11: '<q', 12: '<d'}


def rd(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def rstr(f):
    return f.read(rd(f, '<Q')).decode('utf8', 'replace')


def rval(f, t):
    if t == 8:
        return rstr(f)
    if t == 9:
        et = rd(f, '<I')
        n = rd(f, '<Q')
        head = []
        for i in range(n):
            v = rval(f, et)
            if i < 8 or n <= 64:  # keep small arrays whole (compress_ratios, ple.*)
                head.append(v)
        return {'elem': T[et], 'len': n, 'head': head}  # arrays summarised (vocab is huge)
    return rd(f, F[t])


def dump(path):
    with open(path, 'rb') as f:
        assert f.read(4) == b'GGUF'
        ver = rd(f, '<I')
        nt = rd(f, '<Q')
        nkv = rd(f, '<Q')
        kv = {}
        for _ in range(nkv):
            k = rstr(f)
            kv[k] = rval(f, rd(f, '<I'))
        tens = []
        for _ in range(nt):
            name = rstr(f)
            nd = rd(f, '<I')
            shp = [rd(f, '<Q') for _ in range(nd)]
            ty = rd(f, '<I')
            off = rd(f, '<Q')
            tens.append({'name': name, 'shape': shp, 'type': ty, 'off': off})
        data_start = (f.tell() + 31) // 32 * 32  # general.alignment absent -> default 32 (checked by the BF16 sanity test)
    return {'file': path, 'version': ver, 'kv': kv, 'tensors': tens, 'data_start': data_start}


if __name__ == '__main__':
    res = [dump(p) for p in sys.argv[2:]]
    json.dump(res, open(sys.argv[1], 'w'), indent=1)
    for r in res:
        print(r['file'].split('/')[-1], 'v', r['version'], 'kv', len(r['kv']), 'tensors', len(r['tensors']))
