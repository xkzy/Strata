# Handoff: GPU reliability audit (state after the second pass)

State on 2026-10-19. Nothing is committed. Read `docs/audit/gpu-reliability-audit.md`,
`gpu-resource-lifecycle.md` and `gpu-test-plan.md` first; they hold the evidence and the findings table.

The 5060 Ti **serves with native sm_120 code and falls off with the PTX one**. 20:12 (`build-fat/strata`, 58 sm_120
cubins): 3 chats at 38.6-56.7 tok/s and 131k/222k-token prompts at 1,354/1,174 tok/s prefill, 82 C, 154 W, no Xid.
20:26 (`build/strata`, sm_89 only, driver JITs compute_89 PTX): same card, config, server and workload, **`Xid 79, GPU
has fallen off the bus` 27 s into the first 131k prompt** (80 C, 147 W, x8), then UVM fatal 0x60 and `Xid 154 ... Node
Reboot Required`. That is the reproduction the audit was looking for. **GPU0 is unusable until a full power-off**
(owner's action; a power cycle is driver recovery, not a fix). The 4060 still answers.

After the 20:37 reboot the pair was repeated on a recreated config, so both runs used **both cards** by mistake (the
recreated config's `env.CUDA_VISIBLE_DEVICES` was still `"0,1"`; a UUID in the top-level key is documentation only:
the engine reads the environment, `generate.cpp:1587`). Same result: **native clean again** (20:41: 131,062 tokens at
1,362 tok/s, 222,805 at 1,179, 0 Xid) and **PTX fell off again** (20:45: Xid 79 27 s into the 131k prompt, same 32k
checkpoint, `pid=23429, name=nvtop`). Both arms had the peer tier on the 4060 (3606 experts, 6.89 GiB), so the trigger
travels with the binary, not the peer tier. Two native passes clean, two PTX passes dead at the same place. GPU0 went
down again and needed a power-off; the configs in `/tmp` are now correct (and `/tmp` is wiped by every reboot - the
recipe is in `gpu-test-plan.md`).

The power-off was done at 20:52. Both cards came back in the same boot (0 Xid lines in the kernel journal this boot),
and the third native pass ran at 21:00-21:07 on the **installed** engine (`engine/strata`, see below): 3 chats at
~43 tok/s, then 201,932- and 257,467-token prompts served, 0 Xid, x8 held, max 81 C. Three native passes clean,
two PTX passes dead in the same place.

**The local engines are native now.** `engine/strata` (the binary a user runs) and `build-fat/strata` are the same
89;120 build (sha256 140d3f6c..., 58 `sm_89` + 58 `sm_120` cubins). `build/strata` was the sm_89-only PTX control
(877d86b8...); that binary is kept unchanged as `build/strata-sm89-ptx` for any further control run, and `build/` was
reconfigured with `-DCMAKE_CUDA_ARCHITECTURES="89;120"` so a plain `ninja -C build strata` is native for both cards
here (now 75f9670f..., 58 + 58). `bin/strata` and `bin/strata-server` were rebuilt from this tree at 21:11.

## Ground rules (from the owner)

- Never recommend "reboot and try again", and never count a reboot, driver reload, GPU reset or retry as proof of a fix.
- No GPU test without the owner's approval for that run and its abort conditions written first. Do not run one while
  `nvidia-smi` is missing a card or the driver reports a fault. Test the RTX 4060 alone before the pair.
- Do not change the system driver. Do not weaken a check or a test to make a run pass.
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claim without a measurement.
- Run shell commands that need `sudo` through the owner (`! sudo ...`). Do not use `pkill -f` / `pgrep -f` in this
  tool (it kills its own shell); use `ps -eo pid,comm | awk '$2 ~ /^strata/'`.

## What is known

- The 5060 Ti (01:00.0) drops with Xid 79 first in all 5 incident boots; UVM 0x60 and Xid 154 follow. No AER, no
  app-fault Xids. It links at x8 Gen4 every boot, which is its full link: the RTX 5060 Ti is a PCIe 5.0 x8 card and
  the B840 board's CPU slot is PCIe 4.0 x16. The link is not a lead. Root cause not established.
- On 2026-10-19 `nvidia-smi` reported "Unable to determine the device handle for GPU0: 0000:01:00.0:
  Unknown Error" while GPU 1 (RTX 4060) answered: the card was in the failed state, so no GPU run was allowed.
  Later that day (02:56) `nvidia-smi -L` listed both cards again, with GPU 0 answering. A recovered driver state is
  not evidence of a fix, and no GPU run has been made since (the owner's approval is still needed).
- Done in the first pass (all build; no GPU run): bounded GPU waits (`gpu_wait.hpp`) in peer experts, verify commit,
  MTP draft, session ring; quarantine of a failed peer and drafter/verifier destructor; checked syncs in `verify.cpp`
  and `mtp.cpp kv_restore`; prefill non-compact route only with working P2P; KV page table re-register fix; startup
  link check; GPU snapshot in the failure log; no auto restart into a GPU the driver reports lost.
- Done in the second pass (builds; Go tests pass; no GPU run): see the tasks below and the audit's findings 7, 7b,
  11-17. The flaky `TestEngineDeathMidRequestReportsTheReason` was not a lost error: the assertion always passed —
  the failure was `t.TempDir` cleanup racing the background `writeGPUSnapshot` goroutine started by `afterFailure`.

## Tasks

1. **DONE — flaky test.** Cause and fix are in finding 12: `Stop` waits for background failure-log writes
   (`snapshotWG`/`stopping`), no new one starts after it; test `TestStopWaitsForBackgroundSnapshot`.
   `go test -count=20 ./pkg/engineipc/` passes.
2. **OPEN (GPU).** `expert_cache.cpp` syncs in `release_segmented()` and `shrink()`: unbounded device drains before a
   VMM unmap. Bounding them needs the streams that read the arena (a registry) and a GPU run to validate, so they
   are unchanged; comments say so. `sync_queued` already checks its result. A lost device returns an error rather
   than spinning.
3. **DONE.** The setup/capture syncs listed before are checked: `pinned.cu` `stream_bandwidth` (create and both
   passes; a failed pass reports `seconds < 0`), `layer.cpp` `qsa_state_init` (returns 0) and
   `stage_timing_report` (prints that it was skipped), `mtp.cpp` state zero / draft-head gather (init fails) and
   `finish_capture` (capture fails), `verify.cpp` window and batch capture (capture fails). Remaining bare syncs are
   in test/bench/parity and diagnostic paths; the one production one still open is `prefill.cpp`'s `KvDrain`
   destructor (an early-return DMA drain the destructor cannot turn into "do not refill").
4. **DONE (reviewed).** `Prefill::release` and the `MtpDrafter` destructor now drain bounded and free nothing on
   timeout; `RemoteExperts::close` likewise and its `finish()` is bounded with a `failed_` quarantine.
   `RouterLookahead`'s thread is host-only (CPU dot + `src_->warm`), so its join is safe. `NativeHead`/`NativeEmbed`/
   `NativeDense` and `RamExpertCache` hold no stream handle: they free after the run, and a bound is not possible
   from their destructors; `RamExpertCache` is unused in this tree. `ExpertCache` is task 2.
5. **ANSWERED (unproven).** CUDA 13.2's `cudaGraphExecDestroy` documentation only says use of the handle after the
   call is undefined; it does not say a launch in flight is safe. It is not relied on: destructors drain first or
   quarantine, and the failed-capture paths destroy only an exec that was never launched.
6. **DONE.** `reapProcess` reports whether the process is gone; `Start` refuses while the old process is still alive
   after SIGKILL (pid + reason in `eo.reason`). Test `TestStartRefusesWhileTheOldEngineIsStillAlive` (seam
   `reapEngine`).
7. **DONE.** `KVPageTable.Lookup`/`GetPage` return a copy (with `DependentPages`); `RegisterPage` documents that the
   table keeps the pointer. `TestLookupReturnsACopyUnderRace`; `go test -race ./pkg/virtualcontext/` passes.
8. **DONE (reviewed).** The KV streaming code checks every copy: `kv_ring_restore`/`kv_stage_from_host`/
   `kv_unstage_to_host` print and exit on a failed `cudaMemcpyAsync` instead of continuing; `kv_stream_resolve`
   checks both launches; a block it cannot place is page -1 and decode attention masks those cells, while the prompt
   path uses the staging pool's identity table. A device fault inside `copy_kernel` stops the stream and surfaces at
   the next checked sync.
9. **OPEN (needs a device).** Per-site fault injection cannot reach `cudaStreamQuery` without a GPU; only the
   `wait_bounded` template is tested without one. A seam would have to mock the query, not just the poll.
10. **UNCHANGED.** Port `set_route_resident` (#1578, speed only) and the #1237 fence if pinned stage buffers
    (`475912c6`) are ever ported. The `--ram-cache-target` code exists only in the old binary
    (`engine/strata.bak-pre-direct`); the flag is out of `strata-coder-iq1_m.json` (backup
    `strata-coder-iq1_m.json.bak-pre-ramtarget`).
11. **DONE.** Prefill speed history for the web UI sparkline. The web app reads `history.prefill_tok_s_mean`
    (it already read it, but telemetry published only `tok_s`). `TelemetryCollector` now has `SetPrefillSource`,
    `historyPrefill` and the `prefill_tok_s_mean` history key (bounded at `HistoryCapacity`, like the others);
    `NewStrataServer` wires it to `requestMonitor.currentPrefillTokS`, which samples the running request's
    `live.prefill_tok_s_mean` (0 when idle). Tests: `TestPrefillHistoryFollowsItsSource`,
    `TestPrefillHistoryWithoutASourceIsZero`, `TestBothSpeedHistoriesAreBounded`; the monitor test asserts the
    source follows reading → generating → idle, and `TestMetricsFeedsTheMonitorView` asserts the /metrics key the
    page reads exists. While doing this, a pre-existing timing flake in that last test surfaced: it slept 1.2 s for
    the first telemetry sample, but every server the test binary builds starts a collector and each sample runs
    `nvidia-smi` (0.03 s alone, 0.71 s for 50 at once on this PC), so the sample could land late. It now polls for
    the first sample (up to 10 s); the assertions are unchanged. The per-server collectors are still never stopped
    in tests (the server has no shutdown method), which is the source of that contention.

## Needs the owner and a healthy GPU (do not start without approval)

- `engine/strata` and `build-fat/strata` now carry native sm_120 code, so the 5060 Ti stops running driver-compiled
  compute_89 PTX by default; a PTX run has to name `-binary-path build/strata-sm89-ptx` (the preserved control). Record at start whether either card gets a "runs code
  compiled for sm_<n>" line (with this binary it should not; `build/strata` and `engine/strata` do on the 5060 Ti).
  If the card still falls off with native code, the JIT path is out as a suspect; if it does not, that is one clean
  run and not yet a fix.
- Inspect hardware first, read-only: `lspci -vvv -s 01:00.0` (LnkCap/LnkSta; needs `! sudo`), BIOS slot mode (no
  x8/x8 split), PSU cable for the 5060 Ti. The 2026-10-19 02:56 baseline is in the audit: idle 5060 Ti is P8, 39 C,
  3.84 W of 180 W, no throttle active, link Gen1 x8 (Gen1 is the idle downshift; x8 is the card's width).
- 4060-only run: 3 requests, 300 tokens, record tok/s and `nvidia-smi -q` before and after; then Compute Sanitizer
  memcheck on a short decode; then the pair with the zero-copy self-test; then `STRATA_GPU_WAIT_S=1` fault injection.
- With a GPU: run the new bounded paths (remote-experts `finish` timeout, prefill release, VMM shrink) and the
  capture-failure paths. Run finding 2's bounded VMM drain once it has a stream registry.
- Re-run the full `ctest` (130 tests). On this PC without GPU 0, 77 pass and 53 fail at `cudaMalloc: unknown error`
  or a missing `pack/full/experts.bin`.

## Verification done in this pass (no GPU)

- `ninja -C build strata` and `ninja -C build-fat strata` after the architecture-warning change (the peer device and
  the helper cache devices are now checked, not CUDA0 and the split devices alone); both link clean, and both
  binaries are newer than `src/program/generate.cpp`.
- `CUDA_VISIBLE_DEVICES= ./build/strata --peer-device 1` prints the unchanged "CUDA1 is not there ..." notice; adding
  `--expert-cache-device1 100 --expert-cache-device2 100 --expert-cache-device3 100 --tokens 1` prints it too. Both
  end at `--ple-gguf is required` (exit 2, no pack in this tree). The new lines print nothing with no device,
  because `device_code_arch_note` returns "" when the ordinal is past the driver's list; that a peer or helper card
  now GETS the note is only checkable on a GPU.
- `build-fat/strata` — the `89;120` engine of finding 0: 58 `sm_89.cubin` + 58 `sm_120.cubin`
  (`cuobjdump --list-elf`), against 58 `sm_89.cubin` and no `sm_120` in `build/strata` and `engine/strata`. The
  configure and build commands are in the audit ("The engine binary for the next run"); `build-fat/` was stale
  (`STRATA_ENABLE_CUDA=OFF`, no `strata` target).
- `./build-fat/strata --help` and `CUDA_VISIBLE_DEVICES= ./build-fat/strata --peer-device 1` — the second prints the
  same "CUDA1 is not there ..." notice as `build/strata`, so the fat build carries this pass's peer-device change.
- `ninja -C build strata` and `ninja -C build-fat strata` after finding 21's engine change - both build; `go test
  ./pkg/...` passes (28 packages ok, rc=0) and `go test ./pkg/engineipc/` passes with the two new tests. `bin/strata`
  and `bin/strata-server` were rebuilt from the tree (20:02) so the server carries the plan-failure rule.
- `ninja -C build strata` — builds.
- `./build/gpu_wait_test` — PASS (Ready, Failed, TimedOut, and the time bound).
- `go test ./pkg/...` — passes, with both cards answering `nvidia-smi` (2026-10-19). Earlier the same command
  failed `TestMetricsFeedsTheMonitorView` on its pre-existing `hardware.cpu` assertion because the test binary's
  telemetry collectors (one per test server) queue `nvidia-smi`; the fixed wait was replaced by a poll.
- `go test -count=20 ./pkg/engineipc/` — passes.
- `go test -race ./pkg/virtualcontext/` — passes.
- `go test ./pkg/telemetry/ ./pkg/server/` — passes (prefill history and the /metrics key included).

## Files changed by this work (uncommitted)

`include/strata/core/gpu_wait.hpp` (first pass), `device.hpp` (arch note, `gpu_missing_note`), `peer_experts.hpp`,
`mtp.hpp`, `expert_source.hpp`, `remote_experts.hpp`, `remote_expert_opt.hpp`; `src/core/peer_experts.cpp`,
`verify.cpp`, `mtp.cpp`, `session.cpp`, `layer.cpp`, `pinned.cu`, `remote_experts.cpp`, `remote_expert_opt.cu`,
`expert_cache.cpp` (comments); `src/program/generate.cpp` (peer/helper skip); `src/prefill/prefill.cpp`;
`tests/core/gpu_wait_test.cpp`, `tests/core/gpu_skip_test.cpp`, `CMakeLists.txt`; `pkg/engineipc/` (`linkcheck`, `gpusnapshot`, `restart`,
`orchestrator`, tests); `pkg/server/monitor.go`, `monitor_test.go`, `server.go`; `pkg/telemetry/telemetry.go` and
its test; `pkg/virtualcontext/kv_page_table.go` and its test; `cmd/strata/main.go`;
`docs/audit/`; `reset-gpu-soft.sh`. The engine for the GPU runs is the new `build-fat/strata` (`build-fat/` is
ignored by `.gitignore`; `build/strata`, `engine/strata` and its backups still hold `sm_89` code only).
The tree also holds earlier uncommitted work from the owner (see `git status`);
do not revert it. The existing `HANDOFF.md` and `HANDOFF_RT.md` belong to that earlier work.

---

## Addendum 2026-10-19 (08:00-09:00): the 07:29 fall-off, burn test, what is still open

Written after the third Xid 79 in this audit. No repository code was changed in this pass.

### Facts established (journal and logs)

- Xid 79 on 01:00.0 (the 5060 Ti) at 07:29:38, 3 min 36 s after the card's driver init line
  (`nvidia 0000:01:00.0: Using 52-bit DMA addresses`, 07:26:02). Same shape on the two earlier boots: init 00:16:54 ->
  Xid 00:20:15 (3 m 21 s), init 01:27:02 -> Xid 01:33:33 (6 m 31 s). Earlier text in this thread that quoted "4.5 h"
  was boot uptime, not time to failure; it was wrong.
- `prefill copy_i32: unspecified launch failure` in the engine log came after the card was already lost; it is a
  consequence. The kernel log shows no Xid 13/31/43/48 and no AER lines before Xid 79.
- Owner's observation: it always falls off after Strata with a long context, at roughly 80-90% of the maximum
  context. Not yet reproduced under monitoring. `strata-config.json` has `max_context` 262144 with `fit_max_tokens`
  on, so the real cap may be far lower; the effective cap printed at engine start was NOT recorded. Get it first.
- The failing request (07:29) was a 1981-token prompt right after a 25213-token one (24018 reused). Decode before it
  showed 34% of expert lookups served over PCIe and 404 MiB of KV read from RAM. It died in the prefill step that
  copies from mapped host memory.
- A 27-minute gpu-burn (tensor cores, 5060 Ti only, about 135 W avg, 148 W peak of 180 W) ran clean: 0 Xid, 0
  errors, Gen4 x8 held. Temperature 78-83 C at fan 100% with thermal slowdown reasons (0x48) in 84 of 798 samples,
  which is hot for 135 W. Burn has almost no PCIe traffic, so it does not clear "PCIe traffic plus load".
- The static scan of unchecked `cudaMalloc`/`cudaHostAlloc` found them only in bench/parity tools, not the serve
  path. No evidence of a null dereference or out-of-bounds access in Strata. Compute Sanitizer exists at
  `/usr/local/cuda-13.2/bin/compute-sanitizer`; not run.
- Voltage is not readable here: no voltage in `nvidia-smi` on GeForce, no board rails in hwmon (modules `nct6775`,
  `asus-ec-sensors`, `asus_wmi_sensors` exist but are not loaded; loading needs sudo). RAPL
  (`/sys/class/powercap/intel-rapl:0/energy_uj`) is root-only. No UPS or smart plug seen.

### Tools left in `/home/khing/gpu-burn-logs/` (outside the repo)

- `run.sh <secs>`: gpu-burn on the 5060 Ti only, 2 s CSV (power, clocks, temp, pstate, link, throttle mask), aborts
  on any new Xid, temp > 85 C, `nvidia-smi` silent > 6 s, or link width drop. gpu-burn is built in the session
  scratchpad (`.../scratchpad/burn/gpu-burn`, built with `-ccbin /usr/bin/g++-15`, `COMPUTE=120`); that path is
  temporary, rebuild if it is gone. It must be run from its own directory (it looks for `compare.fatbin` there).
- `power.sh <secs>`: 1 s CSV of CPU package W (needs the RAPL `chmod`), each GPU's power, and their sum. The sum is
  not wall power. The 4060 reports `[N/A]` power.
- Do not `pkill -f run.sh`-style patterns inside the same shell command: it matches itself. Kill by PID.

### One run was made (2026-10-19 19:45, RTX 4060 alone; stopped on its abort condition)

`./bin/strata serve -binary-path build-fat/strata -config <copy of strata-coder-iq1_m.json with CUDA_VISIBLE_DEVICES=1>`.
The engine never became ready: 4 identical exits (restarts 1/8 to 4/8) on the prompt-path VRAM plan - a 1024-token
chunk needs ~795 MiB, the plan leaves 162 of 7807 MiB free on the 8 GB card, and the step down to 512 is not smaller
(657 + 256 = ~913 MiB; the free VRAM before each attempt was the same, so nothing leaks). Deterministic: the byte
counts match the 09:52 run. New finding 21, open. Hardware clean (0 aborts in 51 samples, no Xid, x8/x4 held, 34/43 C).
Finding 20 is now **verified on a GPU**: `WARNING --peer-device 1: CUDA1 is not there ... continuing on the primary
GPU` printed on each start and the engine set up on the 4060, where 09:52 died. `bin/strata` was rebuilt (it was
built 01:41, before this audit's Go work, and printed the withdrawn x8/Xid-79 link line; finding 22; old copy in
`bin/strata.bak-pre-audit-gobuild`). Details in the audit's "Run record".

### Two more runs were made (2026-10-19, RTX 5060 Ti alone, native sm_120; option A of the test plan)

Same abort conditions and monitor as above. `build-fat/strata` sha256 140d3f6c… (the 20:00:09 build, 58 sm_89 + 58
sm_120 cubins; the `2a02f1ec` in the monitor headers and the old records is the 19:40 build), `CUDA_VISIBLE_DEVICES=GPU-da4ac99c-...`.

- **20:06** - the engine **became ready with native sm_120 code** (no `runs code compiled for sm_<n>` line,
  `compute capability 12.0`, `expert cache 4447 slots, 8.46 GiB`, `prompt chunk auto: 8192 tokens`, `334 MiB of VRAM
  free with everything loaded`) and then **refused every request** with `429 server_overloaded: memory_critical`: the
  collector read 15.5 of 16 GiB (95.5%) and the controller's VRAM-critical rule (95%) closed admission, on a card whose
  engine plan had just left 334 MiB by design. Finding 23.
- **20:12**, after the finding 23 fix - the same config **served everything**: `vram_percent` 95.53 with
  `engine.vram_free_mib` 334, tier `retain`, admission open; 3 chats at 38.6-56.7 tok/s decode, then raw completions
  of **131,062 tokens (50% of the cap) at 1,354 tok/s prefill** and **222,805 tokens (85%) at 1,174 tok/s**. **No
  fall-off**: 245 monitor samples, 0 aborts, 0 Xid, x8 held, max 82 C / 153.9 W, both cards 2 MiB after the stop, no
  processes left.
- **20:26, the PTX control** (`build/strata`, 58 sm_89 cubins, no sm_120): the engine printed `runs code compiled for
  sm_89`, served the same 3 chats at **40.1/50.0/54.1 tok/s** (JIT is not slower here), then **fell off 27 s into the
  131,062-token prompt**: `prefill mmq: quantize: unspecified launch failure`, `Xid 79, GPU has fallen off the bus`,
  UVM fatal 0x60, `Xid 154 ... Node Reboot Required`, `nvidia-smi`: `Unable to determine the device handle for GPU0`.
  80 C, 147 W, x8 - not thermal, power or link. The server's `gpuFaultsInRow` rule stopped the restart loop after
  one attempt (`not restarting: the driver reports a lost or faulted GPU`). Stopped there: no reboot, no driver
  reload, no retry.

So the fall-off is reproduced in a matched pair: the same card, config, server binary, monitor and workload, one
variable (`-binary-path`). The native arm served 50% and 85% of the context; the PTX arm died on the first 50%. That
is n=1 per arm and does not prove the JIT'd code is the root cause, but it is the first controlled evidence, and the
kernel sequence is identical to the five historical incidents. The suspect is the compute_89 PTX path (the same class
as issues #220/#224: sm_120 engines built with CUDA 12.8 crashed on long prompts); this machine's `build/strata` and
`engine/strata` are sm_89-only, so any 5060 Ti run with them JITs. Long prompts must go through `/v1/completions`: a
chat request is served from the virtual-context working set (a 200,000-character message became 15,074 tokens).

### Next steps, in order (each GPU run needs the owner's go-ahead; each can force a reboot)

0. **Finding 21**: the *reporting* and the *restart loop* are fixed (engine: real numbers and the real levers, no
   step down to a chunk that needs no less; server: `planFault` stops the restarts on the second consecutive plan
   failure and names the flags; tests `TestPlanFaultMarkers`, `TestRepeatedPlanFailureStopsRestarts`). **Still open:
   making the plan fit** - reserving the prompt path's own footprint when no cache slot can be lent, or lowering the KV
   residency from the plan's own numbers. That is what would let an 8 GB card serve, so a 4060 control run before it
   needs `--kv-resident` lowered or a smaller `--max-context` (the plan's own count: a 1024-token chunk needs ~640 MiB
   more free VRAM than the card had).
1. **The card is down again**: GPU0 needs a full power-off (owner's action; a power cycle is driver recovery, not a
   fix). The repeatability pair is done - native clean twice, PTX dead twice, one- and two-card. What is left is the
   fix, not more diagnosis: (a) stop running sm_89-only engines on this card - `build/strata` and `engine/strata` are
   `cuobjdump`: 58 sm_89, 0 sm_120, while `CMakeLists.txt` defaults to `CMAKE_CUDA_ARCHITECTURES=120`, so build the
   local engines native or fat (`89;120`); (b) decide whether the engine should refuse (or loudly warn before a long
   prompt) when the device runs non-native code - the warning exists but is only a stderr line; (c) then the 4060
   alone (`GPU-7dce6fe4-4534-3975-e733-e239764a859a`) once finding 21's plan-fit change is in. GPU order differs
   between tools; use the UUIDs.
2. If 5060 Ti drops and 4060 does not: run burn plus a pinned-memory H2D stress together (`stream_bandwidth` in
   `src/core/pinned.cu`, or `nvbandwidth`) to test "PCIe traffic under load" without Strata.
3. Power-cap test (owner's idea): `sudo nvidia-smi -pm 1`, then `-pl <Min Power Limit>` on the 5060 Ti (default
   180 W), same workload. A drop at minimum cap makes power unlikely; a clean run needs a long soak to mean anything.
4. Compute Sanitizer memcheck on the 4060 alone, short decode, per `docs/audit/gpu-test-plan.md`.
5. Hardware checks the owner can do: reseat the card and its power cable, BIOS PCIe Gen3 for the CPU slot, try
   `pcie_aspm=off`, check the cooler (83 C at 135 W).
6. **DONE for the peer and helper tiers, and now verified on a GPU** (2026-10-19 19:45, RTX 4060 alone, four
   starts: the warning printed each time and the engine set up on the card that was there). The 09:52 failure is the
   evidence: after a driver
   reload CUDA listed the 5060 Ti alone (the 4060 was gone from its list), and `--peer-device 1` ended the engine on
   every restart ("peer experts: CUDA device 1 is not visible (1 devices; ...)") instead of serving on the card that
   was there. The peer tier and the helper caches now skip a device the driver does not list, or one that fails to
   open, with a warning that names it, and continue on the primary GPU; the option is cleared so later checks agree.
   Test `build/gpu_skip_test` (no GPU) covers the sentence; `CUDA_VISIBLE_DEVICES= ./build/strata --peer-device 1`
   shows the warning and goes on. Split stages still refuse (the placement is the user's measured choice), and
   losing the primary is still fatal: at 07:29 the kernel log shows the UVM fault was global (0x60, "requiring os
   reboot to recover"). Naming the card and Xid in the reason was already done in `pkg/engineipc` (finding 12 and
   `gpuFaultSummary`).

### Rules that still apply

No GPU test while a device is missing or the driver reports a fault, and a reboot or driver reload is not evidence a
defect is fixed (`gpu-test-plan.md`). Nothing here is committed.
