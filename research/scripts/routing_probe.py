import struct, sys
b = open(sys.argv[1], 'rb').read()
print('bytes', len(b))
off, recs = 0, []
while off + 8 <= len(b) and len(recs) < 6:
    layer, k = struct.unpack_from('<ii', b, off)
    ids = struct.unpack_from(f'<{k}i', b, off + 8)
    w = struct.unpack_from(f'<{k}f', b, off + 8 + 4 * k)
    recs.append((layer, k, ids[:12], [round(x, 3) for x in w[:12]], round(sum(w), 3)))
    off += 8 + 8 * k
for r in recs: print(r)
# full parse check
off, n, layers, ks = 0, 0, [], []
while off + 8 <= len(b):
    layer, k = struct.unpack_from('<ii', b, off)
    if not (0 <= layer < 48 and 0 < k <= 80): print('BAD at', off, layer, k); break
    layers.append(layer); ks.append(k); off += 8 + 8 * k; n += 1
print('records', n, 'consumed', off, 'of', len(b), 'k set', sorted(set(ks)), 'first layers', layers[:50])
