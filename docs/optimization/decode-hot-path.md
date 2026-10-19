# Decode speed on the 5060 Ti + 4060 PC: the hot path, and the peer-cache default that cost 45%

Measured 2026-10-19 on the owner's PC: Ryzen 9 9950X3D (16 cores), 64 GB, RTX 5060 Ti 16 GB (x8 Gen4) + RTX 4060
8 GB behind the chipset (x4 Gen3), Linux driver 595.91.07. Model: Coder IQ1_M
(`strata-coder-iq1_m.json`), engine `build-fat/strata`, `--spec 4`, `--spec-min-p 0.5`, `--expert-cache auto`,
`--peer-device 1`, `--kv-resident 20480`, `--max-context 262144`, `--mmap-experts`.

Benchmark: a 415-token Go source prompt (tokenized with `pkg/tokenizer`), 128 new tokens, `--stats`. Decode
throughput only; the prompt read and the model load are excluded.

## Result

| Configuration | tok/s | cache hits | pool ms/round | ring wait ms/round |
| --- | ---: | ---: | ---: | ---: |
| `STRATA_PEER_HOT=0.45` (the old default) | 44.2 | 0.685 | 31.1 | 17.5 |
| `STRATA_PEER_HOT=0.25` | 57.3 | 0.854 | 15.5 | 18.3 |
| `STRATA_PEER_HOT=0.1` | 64.5 | 0.857 | 13.7 | 17.5 |
| `STRATA_PEER_HOT=0` | 63.6 / 64.8 | 0.892 | 11.5–12.7 | 18.2 |
| no peer at all (5060 Ti alone) | 49.3 | 0.737 | 27.0 | 17.9 |

The 45% is the whole story: the default gave the **peer card 45% of the hottest expert pairs**. On this PC the two
cards have no peer access (they meet at the CPU's host bridge), so every hot expert on the 4060 is a copy across
host memory in every layer, and the 5060 Ti keeps *colder* experts instead — 31 ms of CPU expert work per round
against 11.5.

## What changed

`src/program/generate.cpp`, the `STRATA_PEER_HOT` block: when the variable is not set, the share of hot pairs
moved to the peer is now **0 unless `cudaDeviceCanAccessPeer` reports a direct link**, in which case it stays 0.45
(the old behaviour, which the previous measurement on other hardware preferred). A one-line note is printed when
the default changes:

```
strata generate: peer GPU 1: no direct peer access, so the hot experts stay on the primary
(STRATA_PEER_HOT=0.45 puts a share on the peer)
```

`STRATA_PEER_HOT` still overrides either way. On the 415-token benchmark the two defaults gave the identical
128 tokens; a different placement can move the last bits, because the GPU and the CPU experts round differently
(the engine says so at start), and the peer-less run did end on a different continuation.

End to end through the server, the same model and config, a 300-token reply to a short code request: **66.1 tok/s**
on the second request (46.9 on the first, which also reads the prompt and the first window). Before the change the
same workload ran at ~43-44 tok/s.

## The window in the engine's own profiler (serve path)

`STRATA_DECODE_TIMING=1 STRATA_VERIFY_PROFILE=1` with `--serve`, a short code request answered with 200 tokens
(second request, warm prompt cache, 74-token prompt, engine log `strata serve: ... 200 generated in 2531 ms
(79.0 tok/s)`):

```
83 windows, avg T 2.76, 2.41 tokens/window, 30.50 ms/window =
  verify 27.05 (GPU-reach wait 16.90 + per-layer host 8.35 [plan 0.60 actq 0.18 CPU 7.55])
  + commit/emit 0.01 + draft 2.83
per layer-window: CPU experts 0.74, VRAM hits 18.45, PCIe 0.00
```

The GPU's own stages per window (`strata decode GPU stages`, GDN + QSA layers summed; 26.94 ms of GPU work per
window over 84 windows): **hc-read1 + router 4.99** (3.74 GDN + 1.25 QSA), **VRAM expert hits 3.81** (2.85 + 0.96),
**waitCPU 3.58** (2.81 + 0.77), q8/qkv/q-idx gemv 2.40, hc0 norm/down/up 2.34, out-proj 1.73, head 1.58, GDN z
1.28, shared+quant 1.17, QSA q 0.99, attention 0.50, copy+combine 0.62, `waitA`/`waitB` 0.45, the rest under 0.5.

The first request on the same server (cold cache) is 45.48 ms/window with `waitCPU` 7.52 and CPU experts 1.85/layer,
which is the same effect the 415-token benchmark shows: 30.5 ms and 79 tok/s once the cache matches the traffic,
42 ms and 64.5 tok/s while it does not. The profiler times every stage with events and slows the decode a little,
so these are for comparison, not speed.

## The same window from the outside (`--stats`, 415-token prompt)

The benchmark above, one window = 2.63 accepted tokens in ~42 ms:

| Part | ms/round | What it is |
| --- | ---: | --- |
| ring wait | 18.2 | the host waits for the GPU to reach each layer (attention, router, VRAM experts, combine) |
| CPU expert pool | 12.7 | every routed expert that is not on a GPU, read from host RAM |
| MTP draft | 3.6 | the draft layer's proposals |
| the rest | ~7 | head, sampling, window bookkeeping |

The pool is now 30%, not 70%, of the round. It is still DRAM-bound: 2.9 distinct experts per layer × 48 layers ×
1.95 MB ≈ 270 MB per round at 24 GB/s. A plain read loop on this PC measures the ceiling — and it is low:

| Threads | 1 | 2 | 4 | 8 | 16 | 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GB/s | 14.7 | 26.4 | 26.6 | 25.1 | 24.3 | 27.1 |

Two threads saturate it; 32 add nothing. Two DDR5 channels at 5,200 MT/s are 83 GB/s on paper, one channel is
~26 GB/s in practice, so this machine is delivering about one channel's worth. That is the next lever: with
~52 GB/s the pool would halve again (~6 ms), a round would be ~36 ms and decode would be ~74 tok/s. It is a
BIOS/DIMM item (`sudo dmidecode -t memory`: module count, `Locator`, `Configured Memory Speed`), not code.

Other knobs, measured on this PC (a fixed 128-token continuation so every arm has the same drafts, or the
415-token prompt where noted):

| Arm | tok/s | Why it is not a default |
| --- | ---: | --- |
| `--expert-cache-per-layer` | 46.6 | hits 0.664 against 0.893: the shared counter's arrival order is better here |
| `--adapt-swaps 0` | 46.3 | hits 0.522: the adaptive tier is what keeps the hit rate |
| `--adapt-every 16` | 47.4 | hits 0.605 (staler cache); `--adapt-every 2` is +1 tok/s (65.7) for twice the swaps |
| `--spec 3` / `6` / `8` | 29.6 / 48.3 / 38.5 | 2.17 / 2.56 / 2.22 accepted per round: the window outgrows the acceptance |
| `--lookup-chain 2` / `4` | 50.6 / 50.4 | same acceptance on this continuation |
| `--pool-workers 31` (SMT) | 39.9 | 31 threads read at 14.9 GB/s, 15 cores at 23-24 |
| `--resident-experts` (PCIe share on) | 46.3 | the GPU's 39% share (`pcie experts` 1.51/layer) cut the pool 23.3 → 20.8 but raised the ring wait 16.7 → 21.0 |
| `--mtp-q4 all` | 60.5 | the draft layer's saving is 22 slots here (4447 → 4425) and the 4-bit draft rows cost more than that buys |

## About a ring buffer for the copies

The swap copies are already asynchronous: they run on their own stream (`adapt_stream`) while the MTP drafts, and
`--adapt-async 1` lets them advance between windows. What a ring buffer would hide is *latency*, and the measured
hot path is *bandwidth*: the CPU pool reads at 24 GB/s against a 26 GB/s ceiling, so hiding the copy behind the
compute cannot add bandwidth. Before the peer fix the host's own swap bookkeeping (choose, pin, prefetch) was
5.25 ms/round, which is where a background thread or a two-deep slot set would pay; at the 0.89 hit rate it is off
the critical path, and `--adapt-every 2` is worth +1 tok/s (65.7).
