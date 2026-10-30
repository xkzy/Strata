# Handoff: GPU reliability audit (state after the second pass)

State on 2026-10-19. The audit's work is committed: `3b66c9c4` (engine and server code, tests, `docs/audit`,
including this file) and `f781e9b1` (the rest of the working tree: start-up tuning, the telemetry fallback, model
cards, the pack CLI, and the `/pack/` ignore fix - `pkg/pack` had been invisible to git because of that rule). At
both commits `go build ./...` and `go test ./pkg/...` pass in a clean worktree; the C++ engine target was built in
this tree at 21:11 with the same file contents, and `gpu_wait_test`/`gpu_skip_test` pass. Read
`docs/audit/gpu-reliability-audit.md`, `gpu-resource-lifecycle.md` and `gpu-test-plan.md` first; they hold the
evidence and the findings table.

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

One caution about this checkout: while the audit work was committed, git wrote several 0-byte objects into
`.git/objects` (the `src/prefill` tree and the `prefill.cpp` blob among them), so the first commit had a missing
subtree. It was rebuilt from a fresh index after the blob and tree were rewritten from the working tree, `git fsck`
is clean, and a clean worktree at both commits builds. If another commit here ever reports "loose object ... is empty",
check `find .git/objects -type f -size 0` and `git fsck` before trusting the tree.

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


---

## Addendum 2026-10-26: Lean 4 kernel in the math runtime (separate work; unrelated to the GPU audit above)

State on 2026-10-26. Nothing is committed. Request that started it: "add lean to math verifier ... port it and merge into our math runtime".

### What exists

A Go port of the Lean 4 kernel (derived from leanprover/lean4 v4.35.0, Apache-2.0) inside `pkg/mathruntime`, used through a new `prove` operation. No external Lean binary is used.

| Where | What |
|---|---|
| `pkg/mathruntime/lean/` | the kernel port (levels, exprs, env, type checker, inductives, prelude, claim elaborator); `LICENSE` is Lean's |
| `pkg/mathruntime/lean_backend.go` | `LeanBackend`, `ProveWithLean` (replaces the old `VerifyWithLean`), CAS cross-check |
| `pkg/mathruntime/runtime.go`, `types.go` | `OpProve`, `kernel_certified`, stats, intercept of `example : <claim> := by decide` |
| `pkg/mathruntime/fast_numeric.go` | CAS bug fix: `^` is now right-associative (`2^3^2` = 512, was 64) |
| `pkg/server/server.go`, `generate.go` | `lean_*` fields on `/v1/strata/math/backends`; `factLine` renders a Lean verdict as `[Lean kernel: claim => refuted]` instead of `claim = refuted` |
| `docs/MATH_RUNTIME.md` | "Lean 4 kernel" section (grammar, limits, axioms, benchmarks) |
| `third_party/lean4/` | vendored Lean source, gitignored, never built. Only a reference for reading `src/kernel` and `src/Init` |

The 2026-10-26 pass added `Int`: the type with `ofNat`/`negSucc`, `Int.NonNeg` as an **indexed** family (so the
kernel's indexed-inductive path is now used, not just tested), `add sub mul pow neg natAbs ediv emod dvd gcd lcm
min max`, `le`/`lt` defined as Lean defines them, and `decEq`/`decLe`/`decLt`/`decNonneg`/`decDvd`. Claims with a
negative numeral are read over `Int` (see "Claim grammar"); Nat behaviour is unchanged.

Kernel features: levels, de Bruijn exprs, type checker (whnf, lazy-delta defeq, proof irrelevance, eta, structure eta, Nat literal acceleration, iota with K, quotients, projections), inductives that are single, **mutual** (`Env.AddMutual`), **nested** (aux-type elimination in `nested.go`, recursors `Main.rec_N`) or **indexed** (`Int.NonNeg : Int → Prop`, which `Int.le` is defined with).

Claim grammar (`claim.go`): closed Nat arithmetic `+ - * / % ^`, `= ≠ < > ≤ ≥ ∣`, `∧ ∨ ¬`, functions `gcd lcm min max factorial/fact`, and bounded quantifiers `∀ x < n, P` / `∃ x ≤ n, P`. A quantifier is **expanded** into a conjunction/disjunction (max 256 values each, 4096 cases total); the kernel checks the expansion, not a real `∀`.

A claim that contains a **negative numeral** (`-3`, `(-2)`) is read over `Int` instead: numerals are `Int.ofNat n` /
`Int.negSucc (n-1)`, `+ - *` are `Int.add/sub/mul`, `/ %` are `Int.ediv`/`Int.emod` (E-rounding: the remainder is
never negative, `-12 / 7 = -2`, `-12 % 7 = 2`, `x / 0 = 0`, `x % 0 = x`), `^` is `Int.pow` (its exponent stays a
`Nat`), `∣` is `Int.dvd` (`Exists (fun c => b = a * c)`), `gcd`/`lcm` are `Int.gcd`/`Int.lcm` and `min`/`max` are
`Int.min`/`Int.max`, the relations are `Int.le`/`Int.lt` and quantifier variables are `Int.ofNat k`. A claim without
one stays `Nat`, where `-` truncates: `3 - 5 = 0` is Nat (true) and `3 - 5 = -2` is Int (true). `factorial` has no
`Int` version and is rejected there with a reason.
`Int.le` is `Int.NonNeg (b - a)` and `Int.lt` is `(a + 1) ≤ b`, as in Lean, so a verdict certifies the same
proposition Lean states.

### How to run

```
go vet ./pkg/...
go test -race -count=1 ./pkg/mathruntime/... ./pkg/server/
go test -count=1 ./pkg/...        # full; densecpu takes ~260 s
go test -bench . ./pkg/mathruntime/lean
```

Last results: `go test -race -count=1 ./pkg/mathruntime/...` passes, as do `./pkg/server` and the full `go test ./pkg/...`
(all 30 packages ok on 2026-10-26, `densecpu` 266 s). One full run before that had `pkg/engineipc`
`TestUnreadableTokenLineIsAnError` fail once while every package ran in parallel; it passes 3x alone and in the repeat
full run, so it is a pre-existing load flake in the GPU-audit package, not this work.

Try it: `POST /v1/strata/math/evaluate` with `{"operation":"prove","expression":"∀ x < 10, x * x < 100"}`.

### Things you must know

- **No admitted axioms, nothing opaque.** `PreludeAxioms` is an empty list, `BuildPrelude` also fails on an `opaque` declaration, and `TestPreludeAxiomSetIsTheDeclaredList` / `TestPreludeAdmitsNothing` fail if `Env.Axioms()` is not empty; `GET /v1/strata/math/backends` reports `lean_admitted_axioms: []`. The last two - `Nat.le_of_ble_eq_true` and `Nat.not_le_of_not_ble_eq_true` - are now theorems: `Nat.ble` is a definition here (`Nat.rec` on both arguments), so the reflection is proved (`Nat.ble_refl`, `Nat.ble_step`, `Nat.ble_true_of_le` by `Nat.le.rec`, `Nat.le_of_ble_eq_true` by `Nat.rec` on both arguments, and the negative direction as the contrapositive). `Int.decDvd` decides `a ∣ b` through `Int.natAbs_dvd_natAbs`, and `Nat.div`/`Nat.mod` are definitions with `Nat.dvd_iff_mod_eq_zero` proved from them, so every verdict rests on the kernel's rules alone.
- **`Nat.gcd` is Lean's Euclid recursion with an explicit countdown.** Lean defines it (`Init/Data/Nat/Gcd.lean`: `gcd m n = if m = 0 then n else gcd (n % m) m`) by well-founded recursion, which this kernel cannot state; `Nat.gcdIter` maps `(a, b)` to `(b % a, a)` until the first argument is 0 and `Nat.gcd a b` runs it from `a`, since each step strictly lowers the first argument. The equations `gcd 0 b = b`, `gcd (succ a) b = gcdIter a (b % (succ a)) (succ a)`, `gcd 1 b = 1` and `gcd n 0 = n` hold definitionally (`Nat.gcd_succ`, `Nat.gcd_one_left`, `Nat.gcd_zero_left`, `Nat.gcd_zero_right`, with `Nat.gcdIter_zero` for the last), and `Nat.lcm` is `a * b / gcd a b` as in Lean.
- **`Nat.dvd_mul` and `Coprime.gcd_mul`/`mul_gcd` are proved** (`Init/Data/Nat/Gcd.lean`, `Coprime.lean`). Lean's `dvdProdDvdOfDvdProd` returns a pair; here it is the statement `Nat.dvdProd_exists` (`k ∣ m*n` gives `a ∣ m`, `b ∣ n`, `a*b = k`), split on `gcd k m = 0` by `Nat.rec`, with `Nat.dvd_mul` the `Iff` Lean states, then `Nat.gcd_mul_right_dvd_mul_gcd`, `Nat.Coprime.gcd_mul` and `Nat.Coprime.mul_gcd`. Covered by `TestNatGcdProductDivisibilityLemmas` (it was failing on these four names before).
- **`Nat.gcd_dvd_left`/`right` are proved**, where Lean derives them from its well-founded recursion. `Nat.gcdIter_dvd` carries the invariant `gcdIter f a b ∣ a ∧ ∣ b` under `a ≤ f` (the countdown keeps `a ≤ f`, so it can only stop on `a = 0`, where the value is `b` and divides `0` and `b`); going back up, a value that divides `b % a` and `a` divides `b`, because `b = a * (b / a) + b % a`. Supporting lemmas, all under Lean's names and statements from `Init/Data/Nat/Basic.lean`, `Dvd.lean`, `Mod.lean`: `Nat.le_trans`, `Nat.mul_zero`, `Nat.mul_one`, `Nat.add_assoc`, `Nat.mul_add`, `Nat.mul_assoc`, `Nat.dvd_mul_right_of_dvd`, `Nat.dvd_add`, `Nat.mod_lt` (`n % k < k` when `0 < k`, from `Nat.divStep_rem_lt`/`Nat.divIter_rem_lt`: a step keeps the machine's remainder below the divisor).
- **`Nat.dvd_gcd`, `Nat.gcd_le_left`/`right`, `Int.gcd_dvd_left`/`right`** are proved too, which is the divisibility half of the gcd API in Lean's `Init/Data/Nat/Gcd.lean` and `src/Init/Data/Int/Gcd.lean` (`gcd_dvd`, `gcd_le_left`/`right`, `dvd_gcd`, on `Nat` and on `Int`). The rest of the gcd, lcm and Coprime API was ported later (see the Lean 4 kernel addendum).
- **`Nat.gcd_comm` and `Nat.gcd_assoc` are proved**, and they never unfold the countdown: with `dvd_gcd` and `gcd_dvd` both sides of each equation are common divisors of the same arguments, so `Nat.dvd_antisymm` settles it. The order and multiplication lemmas that needs are now in the prelude too, under Lean's names: `Nat.mul_comm`, `Nat.add_le_add` (with `Nat.add_le_add_right`/`left`), `Nat.le_antisymm` (`Nat.le_trans` and `Nat.not_le_succ_self`), `Nat.dvd_antisymm` (via `Nat.le_of_dvd` and `Nat.pos_of_dvd_of_pos`), `Nat.dvd_trans` (via `Nat.mul_assoc`) and `Nat.eq_zero_of_zero_dvd`. `Nat.dvd_gcd` runs the countdown forwards (`Nat.gcdIter_dvd_of_dvd`: `k` divides both arguments at every step) and needs `k ∣ b % a` from `k ∣ a`, `k ∣ b`, i.e. `k ∣ b - a * (b / a)` (`Nat.dvd_mod_of_dvd`). That made the prelude need subtraction for the first time, so it now carries the layer from `Init/Data/Nat/Basic.lean`: `Nat.succ_add`, `Nat.add_comm`, `Nat.add_right_comm`, `Nat.succ_mul` (`*` recurses on its second argument here, so it is an induction), `Nat.add_sub_add_right`, `Nat.add_sub_add_left`, `Nat.add_sub_cancel`, `Nat.add_sub_cancel_left`, `Nat.succ_sub_succ`, `Nat.sub_self`, `Nat.zero_sub`, `Nat.sub_sub`, `Nat.mul_pred`, `Nat.mul_sub_left_distrib` (unconditional: Nat subtraction truncates), `Nat.dvd_sub`, plus `Nat.sub_congr` for the rewrites. `Nat.gcd_le_left`/`right` go through `Nat.le_of_dvd` (`Nat.le_add_left`, `Nat.zero_le`). `Int.gcd_dvd_left`/`right`/`Int.gcd_dvd` need no new multiplication lemma: `Int.natAbs_dvd_natAbs` carries `Nat.gcd_dvd_left` to `Int.dvd (ofNat (gcd a b)) a` because `natAbs (ofNat (gcd a b))` is `gcd a b` by definition. Not ported from `Init/Data/Nat`: files other than `Gcd`, `Lcm` and `Coprime` (those are complete, see the addendum), none of which a claim needs. `gcd_test.go` reduces `Nat.gcdIter` itself and requires it to equal `big.Int`'s GCD for every pair up to 30, checks that extra fuel never changes the answer, and checks the definition and the literal extension agree.
- **`Nat.div`/`Nat.mod` are a state machine, not Lean's well-founded recursion**, which this kernel cannot state. `Nat.divMod k n` runs `Nat.divStep` `n` times from `mk 0 0`; a step bumps the remainder, or wraps to `mk (q+1) 0` when the remainder would reach `k`, so `mk q r` always satisfies `q * k + r = steps taken`, `n / 0 = 0` and `n % 0 = n`. `Nat.div`/`Nat.mod` take the dividend first (as Lean's do) and project the state. Literals never unfold it: `reduceNat` answers `Nat.div`/`Nat.mod` with `big.Int` division first, so a big quotient is still one kernel step; `divmod_test.go` reduces `Nat.divMod` itself and requires quotients and remainders to equal `big.Int`'s for divisors 0-9 and dividends 0-40. What is proved from the model: `Nat.divMod_spec` (`n = k * (n / k) + n % k`), `Nat.divIter_lt`/`Nat.divIter_wrap` (where the machine stands after `j` steps), `Nat.divMod_mul` (`k ∣ k * c`), and from those `Nat.dvd_iff_mod_eq_zero` both ways. Remaining difference: Lean's `Nat.div`/`Nat.mod` are opaque builtins and it proves things about them through `wf`/`omega`; here only what these lemmas say is known, and every `Nat` arithmetic fact used (`Nat.le`, `Nat.succ.inj`, `Nat.not_le_succ_self`, `Nat.zero_add`, `Nat.zero_mul`, …) had to be proved in the prelude first.
- **Never compared with a real Lean build.** Recursor shapes in tests (`Nat.rec`, `Or.rec`, `Rose.rec_1`, `Int.NonNeg.rec`, mutual `Even/Odd`, ...) come from my reading of `inductive.cpp`, not from running Lean. Naming (`motive_N`, `rec_N`) and minor-premise order are the likeliest places to differ.
- **Inductive order comes from `Int.NonNeg`.** The one indexed family in the prelude is `Int.NonNeg` (Lean's own definition); its recursor eliminates only into `Prop`, which is how `Int.decNonneg` proves a negative integer is not non-negative.
- **`Decidable` deviates from 4.35**: the prelude uses the classic two-constructor inductive (`isFalse`/`isTrue`); 4.35 defines a structure with `decide` and `reflects_decide`.
- **Operators elaborate to what Lean's instances unfold to** (`a + b` becomes `Nat.add a b`, `Int.le` is `NonNeg (b - a)`). The type-class layer is not ported.
- **The `Int` operations are recursor/Bool.rec terms, not Lean's `match` bodies** (`Int.subNatNat`, `Int.negOfNat`, `Int.add`, `Int.mul`, `Int.ediv`, `Int.emod`), and `Int.pow` replaces Lean's `if n % 2 = 0` with a `Bool.rec` on `Nat.beq (Nat.mod n 2) 0`; the values are the same. `Int.decEq` uses five port helpers (`Int.ctorNat`, `Int.ofNat.inj`, `Int.negSucc.inj`, `Int.ofNat_ne_negSucc`, `Int.negSucc_ne_ofNat`) where Lean uses `Int.noConfusion`. `Int` declarations were read from `third_party/lean4/src/Init/Data/Int/Basic.lean`, `src/Init/Data/Int/DivMod/Basic.lean`, `src/Init/Data/Int/DivMod/Bootstrap.lean` (for `dvd_of_emod_eq_zero` / `emod_eq_zero_of_dvd`) and `src/Init/Data/Int/Gcd.lean` (`gcd`/`lcm`); `Int.decDvd` mirrors `decidableDvd` in `DivMod/Lemmas.lean` and `Int.min`/`Int.max` are `minOfLe`/`maxOfLe` from `Init/Prelude.lean`.
- **`prove` bypasses the result cache** because the cache key canonicalizer can mangle symbols.
- **CAS cross-check** (`certify`) only runs for expressions of `+ * ^ ( )` over unsigned integers; `- / %` differ between Lean's Nat and the CAS rationals.
- Budgets: `MaxSteps` (default 5,000,000) and the runtime timeout, checked on every kernel step. Nat literal ops refuse results over `natMaxBits = 1<<22`.

### Tests

`pkg/mathruntime/lean`: `kernel_test.go` (claims incl. quantifiers/functions, 1,500-claim big-integer oracle, budgets, refutation certificates), `divmod_test.go` (the div/mod model against `big.Int` for divisors 0-9 and dividends 0-40), `negative_test.go` (bad terms, defeq, universes, inductive accept/reject, recursor types), `mutual_test.go`, `mutual_nested_test.go` (a mutual block whose types nest a sibling: three motives, the auxiliary in the block, elimination from either type, the counts), `nested_test.go` (Rose, parameterised `T α`, nesting inside a mutual block), `gcd_test.go` (the gcd countdown against `big.Int`, fuel independence, the equations by reduction, large literals through the extension), `gcd_dvd_test.go` (`Nat.mod_lt` for divisors 1-9 and dividends 0-40 against `big.Int`, the refusals for a bound that is one too small and for `k = 0`, `Nat.gcd_dvd_left`/`right` for every pair up to 20 with the type compared against the computed gcd and `Nat.decDvd` agreeing, the divisibility laws on numerals, `mod_lt` through `Nat.le_of_succ_le`, and now `Nat.dvd_gcd` for common divisors up to 6 with arguments up to 12, `Nat.gcd_le_left`/`right` for every pair up to 20, and `Int.gcd_dvd_left`/`right`/`gcd_dvd` on positive, negative and zero pairs with `Int.decDvd` agreeing), `three_index_test.go` (`Cube α i j k`: the three-index recursor's type, reductions at fixed indices, the index mismatch refused, and a `Slice` nesting `Cube Slice 2 3 4` whose auxiliary carries all three indices and takes an induction hypothesis for the recursive parameter), `indexed_nested_test.go` (`Vector` as the nested family: a fixed index, an index bound in the constructor, an index computed from a constructor argument, two indices sharing one auxiliary, an indexed and a plain family in one declaration), `nat_order_test.go` (`Nat.mul_comm`, `Nat.add_le_add` and its halves, `Nat.le_antisymm`, `Nat.dvd_antisymm`, `Nat.dvd_trans`, `Nat.pos_of_dvd_of_pos`, `Nat.eq_zero_of_zero_dvd` applied to numerals with the refused reversed/off-by-one statements, `Nat.gcd_comm` for every pair up to 20 and `Nat.gcd_assoc` for every triple up to 8 against `big.Int`), `mat_nested_test.go` (`Mat α rows cols`, a two-index family: the family's own recursor, a fixed 2x2 occurrence whose auxiliary nests `Vector` and so adds a third motive, and both indices bound in the constructor), `int_test.go` (Int claims incl. negative literals, both readings of `-`, E-rounding `/` and `%`, `∣` as `Int.dvd`, the `gcd`/`lcm`/`min`/`max`, the `Int`-unsupported `factorial`, a refutation certificate, the proposition shape, a 400-claim Int oracle and a literal-cost check), `prelude_test.go`, `bench_test.go`.
`pkg/mathruntime/lean_runtime_test.go`: OpProve through `ProcessRequest`, budget, deadline (20 ms deadline returns within 250 ms), CAS cross-check, intercept, concurrency, `TestCASPowerIsRightAssociative`.
`pkg/server/lean_endpoint_test.go`: endpoint and prompt-injection rendering.

Measured (AMD Ryzen 9 9950X3D, `go test -bench . -benchtime=500x ./pkg/mathruntime/lean`, one run; the prelude and gcd readings are `-benchtime=300x`, after `Nat.gcd` became a definition): prelude build 10.7 ms (one `-benchtime=300x` run; 10.4 ms before the order/multiplication lemmas, 6.95 ms before the subtraction layer, 5.6 and 6.2 ms before the order and divisibility lemmas, 1.8 before the divisibility work; the div/mod model and its lemmas are the bulk, the `Nat.ble`, order and subtraction lemmas are the last declarations, the gcd countdown adds two definitions and five equations, and `Nat.mod_lt`, `Nat.gcdIter_dvd`/`Nat.gcdIter_dvd_of_dvd` with the lemmas under them add the rest); `2^10 = 1024` ~14 µs; big multiplication claim ~14 µs; `gcd(12, 18) = 6` ~11 µs and `gcd(123456789, 987654321) = 9` ~13 µs - a nine-digit pair costs what a two-digit one does, because the literal extension answers `Nat.gcd` before the definition unfolds; `3 ∣ 12 ∧ 5 < 7 ∧ ¬ (9 = 10)` ~46 µs; `-3 < 2` ~18 µs; `-123456789 * 987654321 = -121932631112635269` ~28 µs; `-12 / 7 = -2 ∧ -12 % 7 = 2` ~54 µs; `-3 ∣ 12 ∧ ¬ (-3 ∣ 13)` ~52 µs (`∣` on `Int` goes through `natAbs`); `gcd(-6, -9) = 3 ∧ min(-3, 5) = -3` ~50 µs. The same benchmarks move by up to a third between runs on this machine; an order claim now embeds a real proof term instead of an axiom's. The Nat oracle covers `+ - * / % ^ ∣ ∧ ∨ ¬` and comparisons; the Int oracle covers `+ - * / % ^ ∣ gcd lcm min max` and the six relations (with E-rounding); quantifiers only have hand-picked cases. `-2^64 = 18446744073709551616` takes 89 kernel steps, the big Int product 95, and `-123456789 / 7` / `% 7` 89 / 97, so the literals stay constant-cost.

### Not done

- Not ported: `String` literals, the elaborator, tactics, the compiler, the type-class layer.
- Not supported in claims: unbounded quantifiers, free variables, and over `Int`: `factorial` (Lean has no `Int` version). Nothing is admitted or opaque any more, so the next real step is coverage, not trust: `Nat.ble`/`Nat.le` are only proved for the constructors the prelude has, the Nat arithmetic library stops at what `Nat.gcd`/`Nat.dvd` need (`Gcd`, `Lcm` and `Coprime` are complete bar the items named in the addendum), and unbounded quantifiers would need `omega`-style machinery.
- Kernel cases now all exercised: single, mutual, nested, indexed, nested-through-indexed, two-index, three-index (`three_index_test.go`), and a mutual block whose types nest a sibling through `List` (`mutual_nested_test.go`). Both of the last two needed no change to the kernel: `nested.go` builds an auxiliary from the family's own type with the parameters instantiated, so three indices carry over for free, and `replaceNested` skips the block's own names, so a nested occurrence whose parameter is a sibling leaves the auxiliary in the same mutual block (every recursor then takes one motive per block type plus the auxiliary, and eliminates from any of them).
- `factorial(n)` unfolds one step per unit of `n`; large `n` hits the step budget rather than computing.
- The CAS itself was not audited beyond the `^` fix.

### Repo state

All changes are uncommitted. Mine: `.gitignore` (adds `/third_party/lean4`), `docs/MATH_RUNTIME.md`, `pkg/mathruntime/*` (including the new `lean/`, `lean_backend.go`, `lean_runtime_test.go`), `pkg/server/server.go`, `pkg/server/generate.go`, `pkg/server/lean_endpoint_test.go`. Not mine, already dirty before this work: `src/prefill/moe_mmq.cu`, `src/prefill/prefill.cpp`, `run_upstream.sh`, `strata-coder-iq1_m.json.bak*`. The status snapshot also shows modifications to `pkg/mathruntime` files I did not list (`cas_engine.go`, `dimensions.go`, `formulas.go`, `theorems.go`, `unified_cas_backend.go`, `math_test.go`); check the diff before committing.

### Lean port: Gcd / Lcm / Coprime now complete (2026-10-30)

`pkg/mathruntime/lean/prelude_nat_extra.go` … `prelude_nat_extra15.go` add the rest of `Init/Data/Nat/{Gcd,Lcm,Coprime}.lean`
under Lean's names (about 300 theorems), written with a small Go DSL (`eqN`, `tr`, `cg`, `tp`, `thm`, `dvIntro`) that
emits the same raw proof terms as `prelude.go`. `BuildPrelude` appends `preludeExtra` to `preludeDecls`. The extra files
register in their `init` functions, so their names must stay zero-padded (`extra02` … `extra15`): init order is filename order.
Binder order: explicit arguments first, then implicit ones, hypotheses last.

- Not ported: `Nat.gcd_def` (needs `ite`), `dvdProdDvdOfDvdProd` (data; its Prop form is `Nat.dvdProd_exists`), and
  `coprime_one_left_eq_true`/`right_eq_true` (need `propext`/`eq_true`).
- Every proof is checked only by this kernel; none was compared against real Lean's output. `nat_order_test.go` and
  `nat_lcm_coprime_test.go` pin a sample of statements; the rest are covered by the prelude build type-checking each one.
- Prelude build time went 10.7 → 45 ms. Nothing here is committed.
- Leftover untracked files in `lean/`: `prelude.go.active/.bak/.orig`, `probe0/1/2_test.go`.

### Lean port: Int Gcd / Lcm added (2026-10-30)

`pkg/mathruntime/lean/prelude_zint_gcd01.go` … `prelude_zint_gcd18.go` port every theorem of `Init/Data/Int/Gcd.lean`
(helpers `ithm`, `icg`, `icgN`, `tpI`, `iDvIntro`; most are the Nat lemma at absolute values). The `zint` names sort after
`prelude_nat_extra15.go` on purpose: init order is filename order and `preludeExtra` is declared in `prelude_nat_extra.go`.
Three lemmas the file needed were added: `Nat.mul_add_div_of_lt`, `Int.mul_ediv_cancel_left`, `Int.natAbs_ediv_of_dvd`.
`Int.dvd_mul` goes through `Nat.dvdProd_exists` (the Prop form), not `dvdProdDvdOfDvdProd`.

- Only `Nat.gcd_def` (needs `ite`) is still missing from Nat/Int Gcd, Lcm and Coprime; checked by comparing theorem names.
- `int_gcd_test.go` pins five statements (accept) and two drifted ones (reject); the rest are checked by the prelude build.
- Proofs are kernel-checked only, never compared with real Lean. Prelude build 45 → 56 ms (100 runs). Nothing is committed.
- Other `Init/Data/{Nat,Int}` files (Basic, Lemmas, Dvd, Mod, MinMax …) are not ported and are far larger.
