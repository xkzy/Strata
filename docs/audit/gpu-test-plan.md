# GPU test plan

## Rules

- No GPU test runs without the owner's approval for that run, with its abort conditions stated first.
- No GPU test runs while a device is missing from `nvidia-smi` or the driver reports a fault. Waiting for a clean
  driver state is not a test of any fix.
- A reboot, driver reload, GPU reset or retry is not evidence that a defect is fixed.
- Test the RTX 4060 alone first (`CUDA_VISIBLE_DEVICES=1`), then the pair.
- Abort at once on: any Xid in the kernel log, GPU temperature above 85 C, `nvidia-smi` not answering within 6 s,
  a link width below the one recorded at the start.

## Matrix

| Test | Needs GPU | Result |
|---|---|---|
| `ctest -R gpu_wait` (ready, failed, timeout, unbounded, clock check) | No | Pass |
| `go test ./pkg/...` (includes link check, GPU snapshot, monitor prefill history/sparkline source) | No | Pass, on 2026-10-19 with both cards answering `nvidia-smi`. One test needed its fixed 1.2 s wait replaced by a 10 s poll first (finding 18: dozens of per-test collectors queue `nvidia-smi`) |
| Full `ninja` build with the bounded waits | No | Pass |
| Full `ctest` (130 tests) | Mostly | 77 pass, 53 fail. The failures I checked stop at `cudaMalloc: unknown error` (GPU parity tests) or a missing `pack/full/experts.bin` fixture (`pool_test`). The driver is in a degraded state with GPU 0 absent, so this is not a result for the code. Not re-run under a healthy driver |
| 4060 alone: 300-token decode, 3 requests, record tok/s and `nvidia-smi` before and after | Yes | Attempted 2026-10-19 19:45 (`-binary-path build-fat/strata`, engine sha256 2a02f1ec): the engine never became ready - it exited 4 times on the prompt-path VRAM plan (finding 21), so there is no tok/s. Hardware clean: 0 aborts in 51 monitor samples, no Xid, links x8/x4 unchanged, 38-43 C on the 4060 |
| 5060 Ti + 4060 peer tier (both cards visible): the same 3 chats and 50%/85% prompts, native then PTX | Yes | 2026-10-19 20:41 native (`build-fat/strata`): 131,062 and 222,805 tokens at 1,362 / 1,179 tok/s, 122 samples, 0 Xid. 20:45 PTX (then `build/strata`, now kept as `build/strata-sm89-ptx`): `runs code compiled for sm_89`, 3 chats at 39.3-76.8 tok/s, then **Xid 79 27 s into the 131k prompt** (79 C, 150 W). Both runs had the peer tier on the 4060 (3606 experts, 6.89 GiB) because the recreated config's `env.CUDA_VISIBLE_DEVICES` was still `"0,1"` - see "Recreating the run configs" |
| 5060 Ti alone, `build/strata` (compute_89 PTX, now `build/strata-sm89-ptx`): the same 3 chats and 50%/85% prompts | Yes | Run 2026-10-19 20:26: **fall-off**. `runs code compiled for sm_89` printed; 3 chats at 40.1/50.0/54.1 tok/s decode (= native speed); the 131,062-token prompt raised `prefill mmq: quantize: unspecified launch failure` at the 32k checkpoint and `Xid 79, GPU has fallen off the bus` at 20:26:58, 80 C, 147 W, link x8. Card unavailable since; server stopped, no reboot |
| 5060 Ti alone, native sm_120: ready, 3 short requests, 50% and 85% of the cap | Yes | Done 2026-10-19 20:12 (after the finding 23 admission fix). Ready in ~20 s, native sm_120 (no arch line), 334 MiB free in the plan. Served 3 chats at 38.6-56.7 tok/s decode, then raw completions of 131,062 and 222,805 tokens (1,354 and 1,174 tok/s prefill). No fall-off: 245 monitor samples, 0 aborts, 0 Xid, x8 held, max 82 C / 153.9 W, both cards 2 MiB after the stop. The 20:06 attempt on the same card is the run that found 23 (ready, every request 429 memory_critical). Re-run on the installed engine 21:00-21:07 (`engine/strata`, native): 201,932- and 257,467-token prompts served, 0 Xid, x8 held, max 81 C ("Comparative stress" below) |
| 4060 alone: Compute Sanitizer memcheck on a short decode | Yes | Not run |
| Pair: zero-copy self-test message at start, then the same 3 requests | Yes | Not run |
| Fault injection: `STRATA_GPU_WAIT_S=1` with a stalled peer | Yes | Not run |
| Link check on a machine with a x8 link on a x16 slot | No (sysfs fixture) | Pass (unit test) |

## Numbers measured earlier (5060 Ti + 4060, zero-copy on)

3 requests, 300 tokens each with a 3,126-token prompt: 41.9-44.6 tok/s overall. With zero-copy off: 35.8 and
42.4 tok/s, then Xid 79 on the third request. That is one short sample and the fault was on GPU 0, not the peer, so
it says nothing about the zero-copy path being good or bad.

## Next run, ready to start (needs the owner's go-ahead)

**A: the 5060 Ti alone, with native sm_120 code** - the fall-off question, and the run that tells us whether the code
that has been crashing on that card was ever the built-for-it code (finding 0). Highest information, and the run that
can force a reboot. **Run 2026-10-19 20:12: served, no fall-off** (see "Runs made"). What it leaves open: one clean
run is not a fix. The next comparison is the same long workload on this card with `build/strata-sm89-ptx` (compute_89
PTX, the code that was crashing) as the control, then a second native run - if PTX falls off and native does not, finding 0 is
the cause rather than the hardware. A second native run alone (same 85% prompt, twice, back to back) would at least
say whether one clean pass is repeatable.

```sh
CUDA_VISIBLE_DEVICES=GPU-da4ac99c-634b-cb80-8ac6-499ec76fff39 ./bin/strata serve \
    -binary-path build-fat/strata -config strata-coder-iq1_m.json   # sha256 140d3f6c..., 58 sm_120 cubins
```

- Abort conditions, written before the start and checked every 2 s (`/tmp/strata-run-monitor.sh` does exactly this):
  any `NVRM: Xid` / fallen-off-the-bus / UVM fatal line; a card missing from `nvidia-smi -L`; `nvidia-smi` silent
  more than 6 s; a GPU over 85 C; a link narrower than the start value (5060 Ti x8, 4060 x4); the engine exiting by
  itself. On any of them: stop the server, no reboot, no retry, report.
- Record at start: no `runs code compiled for sm_<n>` line (with `build-fat/strata` there should be none - with
  `build/strata`/`engine/strata` the 5060 Ti prints one), the effective context from the engine's own plan lines, and
  `nvidia-smi -q` power/clocks before the first request.
- Workload: 3 requests of 300 tokens with a short prompt, then a long one at ~50% of the cap and, if that holds, one
  at ~85% (the range the owner says precedes the fall-off). Record tok/s, peak temperature and power, `memory.used`,
  every Xid, and whether the card is still in `nvidia-smi -L` afterwards.
- A fall-off is a result: stop and report, do not reboot into another run (a recovered driver is not evidence of a
  fix).

**A-control: the same run with `build/strata-sm89-ptx` (compute_89 PTX, the code that was on the card when it fell off;
it is the unchanged copy of `build/strata` before `build/` was reconfigured to 89;120).**
**Run 2026-10-19 20:26: the card fell off the bus (Xid 79) 27 s into the 131,062-token prompt**, after the marker
line `runs code compiled for sm_89` and three short chats at the same speed as native (40.1/50.0/54.1 tok/s). Same
card, config, server binary and monitor script, only `-binary-path` changed:

```sh
CUDA_VISIBLE_DEVICES=GPU-da4ac99c-634b-cb80-8ac6-499ec76fff39 ./bin/strata serve \
    -binary-path build/strata-sm89-ptx -config /tmp/strata-5060ti-alone.json
```

- `build/strata-sm89-ptx` sha256 877d86b8…, `cuobjdump --list-elf`: **58 `sm_89.cubin`, 0 `sm_120.cubin`** - the driver
  compiles compute_89 PTX for the 5060 Ti at load time. Expected marker: the engine prints `GPU 0 (…, sm_120) runs code compiled
  for sm_89: this engine has no native code for it…`. Record it and the startup time (JIT adds to it); do not abort on
  it, it is the point of the run.
- Same abort conditions as A (header of `/tmp/strata-5060ti-ptx-monitor.sh`, which writes
  `/tmp/strata-5060ti-ptx-monitor.log`), checked every 2 s. **Result: fall-off**, so the JIT'd path is the trigger in
  this comparison - kernel `Xid 79`, then UVM fatal 0x60 and `Xid 154 ... Node Reboot Required`, identical to the five
  historical incidents. Not thermal (80 C) or power (147 W) or the link (x8). The server refused to reload the model
  (`not restarting: the driver reports a lost or faulted GPU`). The card needs a full power-off before it can be used
  again; that is the owner's action, and it is a driver recovery, not a fix.
- Same workload, same order, so KV reuse is comparable: 3 chats of 300 max tokens (backends differ), then raw
  `/v1/completions` at 131,062 and 222,805 tokens.
- After the power-off, the only remaining question for this comparison is repeatability: a second native pass on the
  same 85% prompt (if it is clean again, the matched pair stands) and optionally a second PTX pass (if it falls off
  again, the trigger is confirmed rather than a one-off).
- **B (low risk, same setup, answers "can the 8 GB card serve at all"):** the RTX 4060 alone
  (`CUDA_VISIBLE_DEVICES=GPU-7dce6fe4-4534-3975-e733-e239764a859a`) with the plan given more room - the 19:45 run left
  it 162 of 7807 MiB free where the prompt path needed ~795: try `--vram-reserve-mib 1600` first, then
  `--kv-resident 8192` on top of it. Same monitor and abort conditions; the 5060 Ti stays idle.

## Runs made

One, on 2026-10-19 19:45: the RTX 4060 alone, abort conditions written first, stopped on "the engine exit by itself"
(after 4 identical exits). The 5060 Ti was not used and the host was left idle. Full record in
`gpu-reliability-audit.md` ("Run record: 2026-10-19 19:45, RTX 4060 alone").

Three more on 2026-10-19, option A (5060 Ti alone, same monitor and abort conditions), in order:

- 20:06 - engine ready, but the server refused every request (`429 memory_critical`, finding 23);
- 20:12 - after that fix, the full workload served with no fall-off: 3 short chats, then 50% and 85% of the 262144
  context in raw completions (82 C, 154 W, 0 Xid, x8 held);
- 20:26 - the PTX control (`build/strata`), which **reproduced the fall-off**: ready at ~20:26:20, three chats, then
  Xid 79 at 20:26:58 while reading the 131,062-token prompt.

Records: "Run record: 2026-10-19 20:06", "20:12" and "20:26" in `gpu-reliability-audit.md`.

After the 20:37 reboot (`/tmp` is tmpfs and was wiped), the repeatability pair was run at 20:41 (native) and 20:45
(PTX), both on the two-card configuration by mistake - see the records "20:41" and "20:45" in the audit. Result:
native clean again (1,362 / 1,179 tok/s), PTX fell off again at the same 32k checkpoint. Two native passes clean, two
PTX passes dead at the same place, in one-card and two-card configurations.

After the 20:52 power-off (the card was in the failed state since 20:46), 21:00-21:07 was run on the installed engine
with the 5060 Ti alone: native `89;120`, no arch line, 3 chats at ~43 tok/s, then 201,932- and 257,467-token
completions served, 0 aborts, 0 Xid, x8 held, max 81 C (record "21:00" in the audit). Three native passes clean.
Since 21:11 `engine/strata`, `build-fat/strata` and `build/strata` all hold native code for both cards; the PTX
control is preserved at `build/strata-sm89-ptx`.

## Recreating the run configs (after a reboot, `/tmp` is empty)

```sh
python3 - <<'EOF'
import json
cfg = json.load(open("strata-coder-iq1_m.json"))
for name, uuid in [("5060ti", "GPU-da4ac99c-634b-cb80-8ac6-499ec76fff39"),
                   ("4060",   "GPU-7dce6fe4-4534-3975-e733-e239764a859a")]:
    cfg["env"]["CUDA_VISIBLE_DEVICES"] = uuid      # the engine reads this (generate.cpp:1587)
    cfg["CUDA_VISIBLE_DEVICES"] = uuid             # the top-level key is documentation only
    json.dump(cfg, open(f"/tmp/strata-{name}-alone.json", "w"), indent=1)
EOF
```

A UUID only in the top-level key does **not** restrict the engine: the 20:41/20:45 runs silently used both cards
(peer tier, 6.89 GiB on the 4060) because `env.CUDA_VISIBLE_DEVICES` still said `"0,1"`. The chain, checked in the
source: the config's `env` object becomes `KEY=VALUE` (`pkg/server/engine.go:299`), the engine's command gets
`os.Environ()` plus those entries (`pkg/engineipc/orchestrator.go:378`; Go's `exec` keeps the last duplicate), and the
engine sets it before the first CUDA call (`src/program/generate.cpp:1587`). The top-level key has no reader under
`pkg/`. Check a fresh config by the
engine's start log: a one-card run prints `--peer-device 1: CUDA1 is not there` (or nothing about a peer), a two-card
run prints `peer GPU 1: ...` and `nvidia-smi` shows the second card's memory.

For long prompts use `/v1/completions`: a chat request is served from the virtual-context working set (a
200,000-character message became 15,074 prompt tokens), so the chat API cannot reach 50% of the context.

## Before the first run

Record `nvidia-smi -q`, `lspci -vvv -s 01:00.0` (LnkCap/LnkSta; needs root), and the BIOS slot mode, so a later
failure can be compared with a known start state.

Pick the engine by its device code (finding 0). `cuobjdump --list-elf` on 2026-10-19: `engine/strata`,
`build-fat/strata` (sha256 140d3f6c...) and `build/strata` (75f9670f..., after `build/` was reconfigured to `89;120`)
hold 58 `sm_89.cubin` + 58 `sm_120.cubin` (native for both cards); the sm_89-only control is kept unchanged as
`build/strata-sm89-ptx` (877d86b8..., 58 `sm_89.cubin`, 0 `sm_120`), which is the engine whose driver-compiled PTX fell
off. Record, per run, whether the engine prints a "runs code compiled for sm_<n>" line - it should not with
`engine/strata` or `build-fat/strata` - and the `cuobjdump --list-elf` count of the binary used.

The 5060 Ti's x8 Gen4 link is its full link (PCIe 5.0 x8 card in a
PCIe 4.0 x16 CPU slot), so do not abort on x8; abort if the width drops below the recorded value.

## Comparative stress: original engine vs current build (owner-directed, 2026-10-19)

Question: the 5060 Ti falls off (Xid 79) under Strata with a long context. Is the current build responsible, or does
the original engine do the same on this hardware?

Both runs use the same server binary and model config; only the engine binary changes:

```sh
CUDA_VISIBLE_DEVICES=<uuid> ./bin/strata serve -binary-path engine/strata         -config strata-coder-iq1_m.json
CUDA_VISIBLE_DEVICES=<uuid> ./bin/strata serve -binary-path build/strata-sm89-ptx -config strata-coder-iq1_m.json
```

Order: RTX 4060 alone first (control, short), then the 5060 Ti alone with the same short workload, then the long
context that has preceded the fall-off. Between runs, check `nvidia-smi -L` and the kernel log.

Abort a run at once on: any new Xid, a card missing from `nvidia-smi`, `nvidia-smi` silent for more than 6 s, GPU
temperature above 85 C, a link width below the value recorded before the run, or the engine/server exiting by itself.
Record per run: engine binary hash, card, effective context cap from the engine start line, prompt and output
tokens, tok/s, peak temperature and power, VRAM used, and any Xid. A fall-off is a result, not a reason to retry;
after one, stop and report instead of rebooting into another run.

**Run 2026-10-19 21:00-21:07, installed engine (`engine/strata` = the `89;120` fat build, 58 sm_120 cubins), 5060 Ti alone, no peer.** No `runs code compiled for sm_` line. 3 chats of 300 tokens at ~43 tok/s decode (6.9-10 s each). Raw completions of 201,932 tokens (2m41s, ~1,250 tok/s prefill) and 257,467 tokens (3m34s), both served; this is past the 131,062-token prompt that killed the PTX engine twice. Monitor (2 s, abort on Xid / missing card / link under x8 / over 85 C): 0 aborts, 0 Xid, x8 held, max 81 C. Third clean native pass on this card; n=3 per arm, so it is not proof of the root cause.
