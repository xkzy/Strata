# GPU reliability audit

Scope: why the RTX 5060 Ti (GPU 0, 0000:01:00.0) falls off the bus on this PC, and which Strata code could make a
GPU hang or lose its device. Machine: ASUS PRIME B840-PLUS WIFI, Fedora 44, kernel 7.2.11, NVIDIA driver 595.91.07,
CUDA 13.2, GPU 1 = RTX 4060 (0000:05:00.0). Repository HEAD 20c1092d plus uncommitted work. Nothing was committed.

## What the incident evidence says

Read from the kernel journal of 22 boots; 5 of them contain a GPU fault.

| Fact | Evidence |
|---|---|
| The first NVIDIA error in all 5 incident boots is Xid 79 ("GPU has fallen off the bus") on 01:00.0, the 5060 Ti | journal, 5 of 5 boots |
| Xid 154 "Node Reboot Required" on both GPUs and "uvm encountered global fatal error 0x60" come after it, in the same second (UVM about 100-200 log lines later) | journal order |
| No AER, DPC or pcieport error lines, and no application-fault Xids (13, 31, 43, 48) | journal grep |
| The 5060 Ti links at x8 Gen4 on every boot. That is its full link, not a narrowing: the RTX 5060 Ti is a PCIe 5.0 x8 card and the PRIME B840-PLUS WIFI's CPU slot is PCIe 4.0 x16, so x8 is the card's width and Gen4 the slot's top speed | sysfs, dmesg ("limited by 16.0 GT/s PCIe x8 link"), card and board specifications |
| The 4060 sits behind the chipset at x4 of x8, which is normal for that slot | sysfs |

The task text names the RTX 4060 as the failing card. The kernel log names 01:00.0, which is the 5060 Ti. This audit
follows the log.

## Root cause

Not established. The pattern (a card that vanishes with no earlier warning, nothing from AER, normal idle power and
temperature) is consistent with a hardware or platform cause: slot contact or power delivery. It is also consistent
with a driver fault. The x8 link is not part of the pattern: it is the card's full width on this board (see below). It does not point at an application error: the driver reports no illegal
access or invalid launch before Xid 79. An earlier theory in this work, that `cudaDeviceEnablePeerAccess` raises
Xid 96, is not supported by the logs. The UVM 0x60 error is a result of the device loss, not its cause.

On 2026-10-19 02:56 both cards answered `nvidia-smi -L` again after the period in which GPU 0 reported
"Unable to determine the device handle". No run was made on that state, and a driver that recovers is not evidence
that the cause is gone.

### Baseline recorded then (read-only, idle, no GPU test)

| Card | State | Link (current) | Link (max) | Temp | Power | Throttle reason |
|---|---|---|---|---|---|---|
| 5060 Ti, 01:00.0 | P8, both cards listed | Gen1 x8 | `nvidia-smi`: Gen4 x16; sysfs `max_link_speed`: 32 GT/s | 38-39 C | 3.84 W of 180 W | none active |
| 4060, 05:00.0 | P8 | Gen1 x4 | Gen3 x8 | 42 C | `[N/A]` of 115 W | none active |

- Gen1 / 2.5 GT/s is the normal idle downshift; the width does not usually drop, so x8 at idle is the same narrowing
  the journal shows under load.
- `nvidia-smi` reports max Gen4 for the 5060 Ti while sysfs reports 32 GT/s. The Gen4 side is the board (its CPU
  slot is PCIe 4.0 x16), so Gen4 is the link's top speed and 32 GT/s is the card's own capability. `lspci -vvv`
  needs root for LnkCap/LnkSta and was not run.
- `Clocks Event Reasons Counters` on the 5060 Ti: SW Power Capping 227892 us (0.23 s of its uptime), all others 0.
- The current kernel ring has no Xid lines. `lspci -vvv` link capability/status fields need root
  (`! sudo lspci -vvv -s 01:00.0`), which was not run.

Evidence still needed (all read-only): `nvidia-smi -q` power and clock-throttle fields before a failure,
`lspci -vvv` for 01:00.0 (link capability and status, LnkSta2), BIOS slot settings (x8/x8 split, Gen mode), the
PSU rail and cable for the 5060 Ti, and a run of the 4060 alone.

## Findings (ranked)

| # | Severity | Confidence | Where | Finding | Status |
|---|---|---|---|---|---|
| 0 | High | High (measured) | build configuration (`build/CMakeCache.txt`: `CMAKE_CUDA_ARCHITECTURES=89`) | The engine in use (`engine/strata`, both backups, and `build/strata`) holds device code for sm_89 only (`cuobjdump --list-elf`: 58 sm_89 images, no sm_120), plus PTX for compute_89. The RTX 5060 Ti is sm_120, so every kernel on it is compiled by the driver from compute_89 PTX at run time (`~/.nv/ComputeCache` is 452 MB). The 4060 (sm_89) runs native code. `CMakeLists.txt` says the engine was developed and measured on sm_120. The code that runs on the 5060 Ti is therefore not the code that was built and tested for it. This is not shown to cause Xid 79: an invalid instruction would raise Xid 13 or CUDA error 715 and normally leave the card on the bus, and the journal has no Xid 13/31/43/48. It is a verified difference between the two cards and a reason to rebuild before any more 5060 Ti testing | Fixed (built; sm_89 half verified on a GPU, sm_120 half not): the engine warns at start when a GPU runs another architecture's code (`device_code_arch_note`), now for every GPU the run uses - the peer tier and the helper caches included, not CUDA0 alone (before this, the warning missed exactly the card in this finding, which ran as the peer on 2026-10-19). `build-fat/strata` is  the `89;120` build - 58 `sm_89.cubin` and 58 `sm_120.cubin` each by `cuobjdump --list-elf` (see "The engine binary for the next run"). Checked on a GPU 2026-10-19 19:45: the RTX 4060 printed no `runs code compiled for sm_<n>` line with `build-fat/strata`, as it should (the binary has native sm_89). **The 20:12 and 20:26 runs made the difference a measurement**: with `build-fat/strata` (58 sm_120 cubins) the 5060 Ti printed no arch line and served 131,062- and 222,805-token prompts; with `build/strata` (PTX only) the same card printed `runs code compiled for sm_89` and **fell off the bus (Xid 79) 27 s into the same 131,062-token prompt** - same config, server, workload and monitor, one variable. All five earlier incidents were seen while this machine had no sm_120-native engine at all (`build-fat/strata` was first produced at 19:40 that day; `engine/strata` and `build/strata` are sm_89-only), so **no earlier fall-off was ever made with native code**, and the native build is the only configuration that has served a long prompt on this card. **Repeated once per arm** ("20:41" and "20:46" records, two-card config): native clean again at 1,362/1,179 tok/s, PTX fell off again at the same checkpoint (32k tokens into the same 131,062-token prompt). Two native passes clean, two PTX passes dead at the same place. That does not prove the JIT'd code is the root cause (n=1 per arm, and the JIT path could expose a hardware fault the native path happens not to), but it is no longer only a difference: the only configuration in this audit that has served a long prompt on this card without a fall-off is the native one, and the only one that fell off in a controlled comparison is the PTX one. **Third native pass** 2026-10-19 21:00-21:07 on the installed engine (`engine/strata` = the 89;120 build): 201,932- and 257,467-token prompts served, 0 Xid. **Build state after the 20:52 power-off**: `engine/strata`, `build-fat/strata` and `build/strata` all hold 58 `sm_89` + 58 `sm_120` cubins (`build/` was reconfigured to `89;120`); the sm_89-only control was kept unchanged as `build/strata-sm89-ptx` (sha256 877d86b8...), so the PTX arm is still reproducible without rebuilding it |
| 1 | High | High (code read) | `src/core/peer_experts.cpp` `finish()`/`close()` | A wait on the peer GPU had no time limit; a stuck peer left the thread spinning, and `close()` freed memory the stuck work might still use | Fixed: bounded wait, peer marked failed, memory of a failed peer is left alone |
| 2 | High | High | `src/core/verify.cpp` `pl_commit_async` | Spin on `cudaEventQuery` with no limit | Fixed: bounded |
| 3 | High | High | `src/core/mtp.cpp` draft waits (2 loops) | Spin on a mapped value, the stream query only checked its return for "not ready", so a hang never ended and an error was dropped in the stepped path | Fixed: bounded, stream error returned, drafter quarantined (`failed()`; destructor leaves its memory alone) |
| 4 | High | High | `src/core/session.cpp` layer ring poll | Loop could spin forever if the graph never finished and never rang | Fixed: bounded |
| 5 | Medium | Medium | `peer_experts.cpp` `launch()` | Zero-copy result path assumed pinned memory without checking | Fixed: self-test at open, pointer attribute check per call, staged fallback |
| 6 | Medium | Medium | `src/prefill/prefill.cpp` ~2887, ~2897 | Two `cudaMemcpyPeerAsync` calls without a P2P check (non-compact path only; the default compact path does not reach them) | Fixed: the non-compact route (`STRATA_PF_PEER_COMPACT=0`) is ignored when P2P is not working; no GPU run yet |
| 6b | Medium | Medium | `src/core/expert_cache.cpp` `release_segmented`, `shrink` | `cudaDeviceSynchronize` with no limit before VMM segments are unmapped (`sync_queued` already checks its result). Unmapping while work runs is unsafe, so these must wait; a bounded version needs the streams that read the arena and can only be validated with a GPU. A lost device makes the call return an error; it does not spin. Comment in the code | Open; needs a GPU run |
| 7 | Low | Medium | whole tree | The first count (about 521 launches against 44 checks) was a rough grep and overstated the risk. Rechecked by API: every `cudaGraphLaunch`, `cudaGraphInstantiate`, `cudaStreamBeginCapture` and `cudaHostRegister` result in `src/` is checked. Ignored results remain on async copies/records/memsets (61, 64, 25 bare calls); their errors are sticky and appear at the next checked sync. Of 68 bare sync calls, the ones on production paths were fixed: `verify.cpp` expert-copy syncs (3 sites) now return the error, `mtp.cpp` `kv_restore` stops the drafter on error. The setup and capture syncs left open in the first pass are now checked too: `pinned.cu` `stream_bandwidth` (create + both passes; a failure reports `seconds < 0`, not a number), `layer.cpp` `qsa_state_init` (returns 0) and `stage_timing_report` (says it was skipped), `mtp.cpp` state zero / draft-head gather (init fails) and `finish_capture` (capture fails), `verify.cpp` window and batch capture (capture fails). Bare syncs remain in test/bench/parity files and in diagnostic, timing and `STRATA_*_CHECK` paths; the one production one still open is the `KvDrain` destructor in `prefill.cpp` (an early-return DMA drain before the borrowed expert slots are refilled, which a destructor cannot turn into "do not refill") | Fixed for the listed production and setup paths |
| 7b | Low | High | `pkg/virtualcontext/kv_page_table.go` `RegisterPage`, `Lookup`, `GetPage` | Registering an existing page ID again counted its bytes in the tier stats twice and left the old key resolving. Bookkeeping only (no GPU calls in this package). `Lookup` and `GetPage` handed out the stored `*KVPage` that `UpdateResidency`/`EvictTier`/`Pin`/`RecordAccess` write under the lock; they now return a copy (including `DependentPages`), and `RegisterPage` documents that the table keeps the pointer | Fixed; tests `TestRegisterPageTwiceCountsOnce`, `TestLookupReturnsACopyUnderRace` (`go test -race` passes) |
| 8 | Medium | Medium | `expert_source.hpp` `note_async_read` | Upstream #1237 fence not ported. Not a live bug here: stage buffers are pageable | Documented in code; must be ported with pinned stage buffers |
| 9 | Low | High | `mtp.hpp` `set_route_resident` | No-op stub; affects speed only | Open |
| 10 | Info | High | link x8 on a x16 slot | Withdrawn as a lead: the RTX 5060 Ti is a PCIe 5.0 x8 card (vendor specifications) and the PRIME B840-PLUS WIFI's CPU slot is PCIe 4.0 x16 (ASUS specifications), so x8 Gen4 is its full link. The startup link check used to call this a fault, because sysfs reports the slot's maximum width, not the card's | Closed; link-check message corrected |
| 11 | Medium | High (code read) | `pkg/engineipc/orchestrator.go` `reapProcess`/`Start` | After SIGKILL `reapProcess` waited 15 s and then went on; a process stuck in the driver (D state) still held the GPU, and a new engine could start on the same card. `reapProcess` now reports whether the process is gone and `Start` refuses, with the pid and the reason, while it is alive | Fixed; test `TestStartRefusesWhileTheOldEngineIsStillAlive` (seam `reapEngine`) |
| 12 | Low | High | `pkg/engineipc` background GPU snapshot | `afterFailure` started `writeGPUSnapshot` in a goroutine (nvidia-smi can hang); the write could land after `Stop`, which made `TestEngineDeathMidRequestReportsTheReason` fail about 1 run in 4 with "TempDir RemoveAll cleanup: directory not empty". The assertion itself always passed, so the error event did carry the reason | Fixed: `snapshotWG`/`stopping`; `Stop` waits for background writes and no new one starts after it; test `TestStopWaitsForBackgroundSnapshot` |
| 13 | Medium | Medium | `src/core/remote_experts.cpp` `finish`/`close` | The helper GPU's `finish()` synced its stream with no limit on the request path, and `close()` synced and then freed everything even if the stream had not drained. `finish()` is now bounded and quarantines the helper on timeout/error (`failed_`); `begin()`/`finish()` refuse afterwards and `close()` frees nothing | Fixed (code); needs a GPU run to exercise |
| 14 | Medium | Medium | `src/prefill/prefill.cpp` `Prefill::release` | Three unbounded `cudaStreamSynchronize` calls, then events, pinned buffers and owned device memory were freed regardless | Fixed: bounded drain of `cs`/`copy`/`kv_copy`; on timeout nothing is freed and the message says so |
| 15 | Low | Medium | `src/core/mtp.cpp` drafter destructor | Unbounded `cudaStreamSynchronize(cs_)` before destroying execs and freeing buffers (the `failed_` quarantine covered only waits that had already timed out) | Fixed: bounded drain; on timeout nothing is freed |
| 16 | Info | High | `cudaGraphExecDestroy` with a launch in flight | CUDA 13.2's `cudaGraphExecDestroy` docs only note that use of the handle after the call is undefined; they do not state that a launch in flight is safe. Treated as unproven and not relied on: the destructors drain the stream first (or quarantine), and the failed-capture paths destroy only an exec that was never launched | Unproven; no code relies on it |
| 17 | Info | High | `src/kernels/cuda/kv_stream.cu` | KV paths reviewed for a half-failed copy: `kv_ring_restore`/`kv_stage_from_host`/`kv_unstage_to_host` check every `cudaMemcpyAsync` and exit with the message rather than continue; `kv_stream_resolve` checks both launches, and a device fault stops the stream and surfaces at the next checked sync. A block resolve cannot place (overflow) is left at page -1, and the decode attention masks those cells; the prompt path uses the staging pool's identity table, never this one | Reviewed; no change |
| 18 | Low | High (measured) | `pkg/server/monitor_test.go` `TestMetricsFeedsTheMonitorView` | The test slept 1.2 s for the first telemetry sample. Every server the test binary builds starts a `TelemetryCollector`, and none is stopped (the server has no shutdown method), so dozens of collectors each run `nvidia-smi` every second: 0.03-0.08 s for one call here, 0.71 s for 50 launched at once. With both cards answering `nvidia-smi`, the first sample could land after the fixed wait and the pre-existing `hardware.cpu` assertion failed (it passed when the test ran alone). Not a product defect: the poll for the sample is now bounded at 10 s and the assertions are unchanged | Fixed (test timing); the leaked per-test collectors remain |
| 19 | Info | High | web Monitor prefill sparkline | `pkg/frontend/web/app.js` read `history.prefill_tok_s_mean`, but `pkg/telemetry` published only `tok_s`, so the sparkline had no data. Telemetry now samples the running request's prompt rate (`SetPrefillSource`, wired to the monitor's `currentPrefillTokS`) into a bounded `prefill_tok_s_mean` history | Fixed; no GPU needed |
| 20 | Medium | High (journal + failure log) | `src/program/generate.cpp` peer / helper tiers | The engine refused to start when a configured second GPU was no longer in the driver's list. On 2026-10-19 09:52, after the driver was reloaded and CUDA listed the 5060 Ti alone (the 4060 was gone from its list, while nvidia-smi still showed both), `--peer-device 1` ended the engine on every restart ("peer experts: CUDA device 1 is not visible (1 devices; set CUDA_VISIBLE_DEVICES)") instead of serving on the card that was there. The peer tier and the helper caches are speed tiers: a device the driver does not list, or one that fails to open, is now skipped with a warning that names it, and the run continues on the primary GPU (the option is cleared, so every later stage agrees). `--layer-split`/`--split-device` still refuse: the placement is the user's measured choice, and running every layer on one remaining card is not the same run. Losing the primary device is still fatal: at 07:29 the kernel log says the UVM fault was global ("uvm encountered global fatal error 0x60, requiring os reboot to recover"), so another card would not have run either | Fixed; **verified on a GPU** 2026-10-19 19:45 (RTX 4060 alone, `CUDA_VISIBLE_DEVICES=1`, `build-fat/strata` 2a02f1ec): the warning printed on each of the 4 engine starts and the run went on to set up on the card that was there. Before this change the same configuration died on every start (09:52) |
| 21 | High | High (measured) | the VRAM plan (`src/program/generate.cpp` prompt-path sizing, `src/prefill/prefill.cpp` `carve`) | With the shipped config on the 8 GB RTX 4060 alone the prompt path cannot fit at any chunk size, so the engine exits before it serves and the server restarts it: `--max-context 262144` + `--kv-resident 20480` (20480 of 262144 cells in VRAM, 6.00 GiB of pinned K/V) + an 866 MiB draft layer + a 0.65 GiB cache leave **162 of 7807 MiB free**, while the prompt path's own buffers (no cache slots were lendable) need 539 + 256 = ~795 MiB for a 1024-token chunk. The step-down to 512 does not help: its layout took 657 + a 256 MiB buffer = ~913 MiB against the same ~700 MiB, so "a smaller chunk only reads slower" does not hold for own buffers and the loop runs out of sizes. Deterministic, not a race: the byte counts are identical to the 09:52/09:53 run 10 h earlier and the free VRAM before each attempt was the same (162+539 = 701, 42+657 = 699), so nothing leaks between attempts. The pre-check that knows this (`own_fits`) leaves `prefill_chunk` as it was when no size fits and lets `init` fail | **Partly fixed** 2026-10-19 (code; no GPU run): the engine no longer advises `--prefill 512` after it has tried 512 - it prints the plan's own numbers (device, MiB free, MiB needed for the last chunk priced, "every size down to 512 tokens was priced") and the real levers (`--kv-resident`, `--max-context`, `--vram-reserve-mib`, `--vision no`), and it no longer steps down to a smaller chunk when the plan's count says that chunk needs no less (the 19:45 run's 512-token layout took more than the 1024-token one). The server no longer reloads the model for it: `planFault`/`planFailsInRow` in `pkg/engineipc/restart.go` stop the restarts on the second consecutive plan failure with the flags to change, the same shape as the GPU-fault rule; tests `TestPlanFaultMarkers`, `TestRepeatedPlanFailureStopsRestarts`. **Still open: making the plan fit** (reserve the prompt path's own footprint when no cache slot can be lent, or lower the KV residency from the plan's own numbers) - that is what would let the 4060 serve, and it needs a GPU run to validate. No card fault, no Xid in the 19:45 run |
| 23 | High | High (measured) | `pkg/resourcemanager/controller.go` `tier`, `pkg/server/runtime.go` `runtimeSample` | The controller read VRAM as a percentage and nothing else. On a card the model only just fits, that percentage is the engine's plan: the 2026-10-19 20:06 run (RTX 5060 Ti alone, native sm_120) was ready with `334 MiB of VRAM free with everything loaded` and the collector read 15.5 of 16 GiB (97%), so the VRAM-critical tier (95%) closed admission and every request got `429 server_overloaded: memory_critical` - a card with 334 MiB to give refused work. The engine already says what the plan left: its `INFO` line's `vram_free_mib`. The controller now lets that plan override the percentage while it is at or above the engine's own 256 MiB stall floor (below the floor the engine itself warns the card only just fits, and the percentage keeps its old meaning; no plan reported = old meaning). `tight`/`aggressive` from RAM, temperature and device state are unchanged | Fixed (Go only); tests `TestPlannedVRAMFillIsNotPressure` (334 MiB + 97% keeps admission open, 200 MiB or no plan still closes it), `TestApplyEnginePlan`. **Verified on the GPU** in the 20:12 re-run: the 20:06 config (same card, engine and plan) reported `vram_percent` 95.53, `engine.vram_free_mib` 334, tier `retain`, `admission_open` true, and served the requests (see the run record) |
| 22 | Medium | High (measured) | `bin/strata` (the server the owner runs) | The binary the 19:45 run used was built 2026-10-19 01:41, before this audit's Go changes (03:07 onward), so it printed the withdrawn link line - "the PCIe link trained at x8, the slot and card allow x16. Slower, and a marginal link can drop the card off the bus under load (Xid 79)" - for the 5060 Ti, whose x8 is its full width (finding 10), and it lacked the rest of the Go-side work (GPU snapshot naming, `Stop` waiting for background writes, the `gpuFaultsInRow` rule, prefill telemetry). The source and `TestLinkWarnings` say the x8 message must not claim the card allows x16. Both Go binaries were rebuilt from this tree at 19:49 (`go build -o bin/strata ./cmd/strata`, `-o bin/strata-server ./cmd/strata-server`); the old one is kept as `bin/strata.bak-pre-audit-gobuild` | Fixed (rebuilt 19:49, again 20:02 with finding 21's restart rule, and again 20:12 with finding 23's admission fix); the corrected wording was seen at the 20:12 and 20:26 server starts |

## Changes made

- `include/strata/core/gpu_wait.hpp`: `wait_bounded` and `gpu_wait_limit()` (`STRATA_GPU_WAIT_S`, default 60 s, 0
  disables the limit for debugging only). No CUDA in it, so it is tested without a GPU.
- `PeerExperts`: `failed_` quarantine; `launch()` refuses after a failure; `close()` does not sync or free a failed
  device. Startup self-test of the zero-copy path; P2P tried only for same-architecture pairs after a test copy.
- `RemoteExperts`: the same shape: bounded `finish()`, `failed_` quarantine, `begin()`/`finish()` refuse afterwards,
  `close()` frees nothing when the stream did not drain or the helper failed.
- `Prefill::release` and the `MtpDrafter` destructor: bounded drains; on timeout no event, buffer or allocation is
  freed.
- Setup and capture syncs check their result (see finding 7): `pinned.cu`, `layer.cpp`, `mtp.cpp`, `verify.cpp`.
- `pkg/engineipc/linkcheck.go`: reads sysfs at startup and warns about a dead card, a link below x8, or two cards
  that meet only at the host bridge. An x8 link in a x16 slot is reported with its caveat (an 8-lane card links that
  way by design) because sysfs gives the slot's maximum, not the card's; the dead-card message no longer says a
  reboot is the remedy.
- `pkg/engineipc/gpusnapshot.go`: writes `nvidia-smi` state and the last Xid lines to the failure log when the
  engine dies, so the cause is visible from outside the process. `Stop` waits for the background write.
- `pkg/engineipc/orchestrator.go`: `Start` refuses while the previous engine process is still alive after SIGKILL.
- `pkg/virtualcontext/kv_page_table.go`: `Lookup`/`GetPage` return copies.
- `pkg/telemetry/telemetry.go`: `SetPrefillSource` and a bounded `prefill_tok_s_mean` history, so the Monitor's
  prefill sparkline shows the running request's prompt speed (the page already read that key).
- `pkg/resourcemanager/controller.go`: `Sample.KnownVRAMPlan`/`VRAMPlanFreeByte` and `vramPlanFloorByte`; `tier()`
  reads the used percentage only when the engine's plan did not leave at least 256 MiB free (finding 23).
- `pkg/server/runtime.go` `applyEnginePlan`: copies the ready engine's `INFO` `vram_free_mib` into the sample (only
  while the engine is ready, so a previous plan's permission does not survive a restart).
- `pkg/server/monitor.go`: `currentPrefillTokS` for that source; `server.go` wires it at construction.
- `pkg/server/monitor_test.go`: the 1.2 s wait for the first telemetry sample became a 10 s poll (finding 18).
- `include/strata/core/device.hpp` `gpu_missing_note()`: the sentence for a device the driver does not list, pure and
  tested without a GPU (`tests/core/gpu_skip_test.cpp`).
- `src/program/generate.cpp`: `--peer-device` is dropped with that warning before it shapes the profile order, and
  again if `PeerExperts::open` fails; `--expert-cache-remote` slots are dropped the same way when no GPU is left for
  them (layer split) or their `RemoteExperts::open` fails. `RemoteExpertOpt::detach()` keeps the decode optimization
  off a helper that was skipped. The primary device, split stages and a missing model are unchanged.
- `src/program/generate.cpp`, the architecture warning (finding 0): the check ran on CUDA0 and the split devices
  only, so a peer tier or helper cache on a card the binary has no code for went unreported - on 2026-10-19 the
  5060 Ti ran a peer tier built for sm_89 with nothing said about it. The peer device and the helper cache devices
  are now checked too (sorted and de-duplicated). Advisory only: no device behavior changes, and the message text is
  `device_code_arch_note`'s, unchanged.

## The engine binary for the next run (finding 0)

`build-fat/strata` (98,243,704 bytes, built 2026-10-19 19:40 from this working tree including the changes above) holds
native device code for both cards: `cuobjdump --list-elf` counts 58 `sm_89.cubin` and 58 `sm_120.cubin`, with PTX for
both as a fallback. `build/strata` (re-linked at 19:39, same tree) and `engine/strata` count 58 `sm_89.cubin` and no
`sm_120`, so every kernel on the 5060 Ti is compiled by the driver from compute_89 PTX at run time - the difference
in finding 0.

```sh
cmake -S . -B build-fat -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="89;120" -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.2/bin/nvcc \
      -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15 \
      -DCMAKE_CUDA_FLAGS="-allow-unsupported-compiler -ccbin /usr/bin/g++-15" \
      -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
ninja -C build-fat strata
```

(`build-fat/` had been left configured with `STRATA_ENABLE_CUDA=OFF`, so it had no `strata` target at all - `ninja`
said "unknown target 'strata'". Its stale cache was removed and it was configured again; the directory is ignored by
`.gitignore` (`build*/`).) A run picks it up with `-binary-path build-fat/strata`.

Checked without a GPU: the cubin counts above, `./build-fat/strata --help`, and
`CUDA_VISIBLE_DEVICES= ./build-fat/strata --peer-device 1`, which prints the same "CUDA1 is not there" notice as
`build/strata`. This is not evidence about the fall-off: it removes "the 5060 Ti was running code never built for it"
from the next run's list of differences.

## Run record: 2026-10-19 19:45, RTX 4060 alone (one run, stopped on its abort condition)

- Command: `./bin/strata serve -binary-path build-fat/strata -config /tmp/strata-4060-alone.json` - a copy of
  `strata-coder-iq1_m.json` with `CUDA_VISIBLE_DEVICES=1`, `port` 8099 and its own log, so the owner's config was not
  touched. Engine sha256 2a02f1ec..., server sha256 cd11edad... (the 01:41 binary, rebuilt after this run: finding
  22).
- Card: RTX 4060 (05:00.0) alone. The 5060 Ti was not visible to the run and stayed idle at 34 C, 4 W, 2 MiB.
- Abort conditions were written and recorded before the start (any NVRM Xid / fallen-off-the-bus / UVM fatal line; a
  card missing from `nvidia-smi -L`; `nvidia-smi` silent > 6 s; over 85 C; a link narrower than the start value; the
  engine exiting by itself) and checked every 2 s into `/tmp/strata-run-monitor.log`.
- Result: the engine never became ready. It exited with status 1 four times (19:45:32 through 19:46:08, restarts
  1/8 to 4/8) for one reason, finding 21, and the run was stopped there - no request was served, so there is no
  tok/s for it.
- Plan numbers from the engine's own start: `KV streaming: 20480 of 262144 cells per QSA layer in VRAM, the K/V in
  6.00 GiB of pinned RAM`; `expert cache auto: 1.55 GiB free, 700 MiB reserved (+218 MiB for the draft head) -> 261
  slots`, ending at 335 slots and 0.65 GiB; `draft layer loaded, 866 MiB of VRAM`. The server printed Physical
  Context 262144, so the fit cap was not lowered.
- Hardware: 51 monitor samples, 0 aborts; no `NVRM: Xid`, no fallen-off-the-bus and no UVM fatal line in
  `journalctl -k -b` at the time (this boot has since recorded the 20:26 Xid 79, see that record); both cards listed in every sample; 4060 38 -> 43 C, 5060 Ti 34 C, power 3.8-4.1 W; links x8 and
  x4 unchanged; 2 MiB used on each card after the stop. No engine process left behind.
- Finding 20 (peer device skipped) printed on each of the four starts and the engine went on to set up on the 4060;
  finding 0's warning did not print (correct here: the binary has native sm_89).
- After the run, finding 21's reporting and restart rule were changed (see its row): the same run repeated would now stop
  after the second start instead of eight, and say which flags free the VRAM. Not re-run (no approval for a second
  GPU run in this pass).

## Run record: 2026-10-19 20:06, RTX 5060 Ti alone (the fall-off question; stopped at the "every request refused" blocker)

- Command: `CUDA_VISIBLE_DEVICES=GPU-da4ac99c-634b-cb80-8ac6-499ec76fff39 ./bin/strata serve -binary-path build-fat/strata
  -config /tmp/strata-5060ti-alone.json`; monitor `/tmp/strata-5060ti-monitor.sh` every 2 s, abort conditions written
  in its header and in `gpu-test-plan.md`. This is option A of the test plan: the 5060 Ti with native sm_120 code.
  Engine `build-fat/strata` sha256 140d3f6c…, built 20:00:09 from this tree; `cuobjdump --list-elf` counts **58
  `sm_89.cubin` and 58 `sm_120.cubin`** (rechecked 2026-10-19 20:23 with `/usr/local/cuda-13.2/bin/cuobjdump`, which
  is not in `PATH` — a bare `cuobjdump` call reports zero ELF files). The monitor header's `2a02f1ec` is the earlier
  19:40 build's hash, carried over from the test plan: `build-fat/strata` was re-linked at 20:00:09 when finding 21's
  reporting change went in.
- The engine **became ready with native sm_120 code**: `GPU 0: NVIDIA GeForce RTX 5060 Ti, compute capability 12.0`, no
  `runs code compiled for sm_<n>` line, `expert cache 4447 slots, 8.46 GiB of VRAM`, `prompt chunk auto: 8192 tokens`,
  `the prompt path borrows 2069 CUDA0 cache slots (3.90 GiB)`, `334 MiB of VRAM free with everything loaded` (the plain
  line, not the LOW warning). So finding 0's "was the 5060 Ti ever running code built for it" question answered itself:
  with `build-fat/strata` it does, and it reaches a serving state at the full 262144 context.
- **Every request was refused**: `429 server_overloaded: memory_critical`. The collector read 15,516 of 16,303 MiB
  (95.5-97%), the controller's 95% VRAM-critical rule closed admission, and the engine's own plan (334 MiB free) was
  invisible to it. Finding 23. Zero tokens were generated on the card in this run.
- Hardware: 170 monitor samples, 0 aborts, 0 `NVRM: Xid`; both cards listed throughout; the 5060 Ti 35-37 C (no work
  was admitted), link x8 Gen1-4 power states, 15,516 MiB used as planned; 4060 idle 47 C, 2 MiB.
- The server was left running after this turn ended (15.5 GiB held, no request served); it was stopped before the
  re-run, and both cards returned to 2 MiB.

## Run record: 2026-10-19 20:12, RTX 5060 Ti alone, after the finding 23 fix (the same option A, completed)

- Binary: `bin/strata` rebuilt 20:12 from this tree (finding 23's Go change), engine `build-fat/strata` sha256
  140d3f6c… (the 20:00:09 build, 58 sm_89 + 58 sm_120 cubins). Same config and card as 20:06. Monitor
  `/tmp/strata-5060ti-monitor2.sh`, same abort conditions.
- Engine ready in ~20 s; the same plan as 20:06 (4447 slots, 8.46 GiB, prompt chunk 8192, 2069 borrowed slots, 334 MiB
  free). Server state with the fix: `resource.adapt.sample.vram_percent` **95.53**, `engine.vram_free_mib` **334**,
  `resource.adapt.knobs.tier` **retain**, `admission_open` **true**.
- Served: the startup measurement (84-token prompt); 3 chat requests of 300 max tokens - 172, 203 and 130 tokens
  generated at **38.6, 49.2 and 56.7 tok/s** decode (65-token prompt, 39-57 tok/s prefill, draft acceptance
  92/112, 106/132, 80/84); a 12,492-token prompt (1,271 tok/s prefill, 58.4 tok/s decode); and raw completions -
  `prompt 38,373 tokens` (1,413 tok/s), `prompt 131,062 tokens` = 50% of the cap in 96.8 s (1,354.5 tok/s prefill,
  6 checkpoints) and `prompt 222,805 tokens` = **85% of the cap** in 92.1 s (114,688 reused + 108,117 read,
  1,174.2 tok/s).
- **No fall-off**: the 5060 Ti stayed on the bus for all of it. 245 monitor samples, 0 aborts, 0 `NVRM: Xid` this
  boot; both cards in `nvidia-smi -L` at the end; link x8 throughout (Gen1/2/4 power states, width never below 8);
  max 82 C and 153.9 W (limit 180 W, no HW/SW thermal slowdown), 15,608 MiB used as planned; the 4060 idle at 40-47 C,
  2 MiB. After the stop: both cards 2 MiB, 0 processes left.
- What this does and does not say: the earlier fall-offs happened on this card; this run at 85% of the context with
  native sm_120 code, 82 C and 154 W did not reproduce one. One run is not a fix: it removes "the 5060 Ti cannot
  serve at long context at all with this build" and leaves "under what load it drops" open. The control that would
  separate the code path from the hardware was run 14 minutes later (20:26, below) and **fell off on the same prompt**,
  so this clean run is now the native half of a matched pair rather than an isolated pass.
- Note for future long-context tests: a *chat* request never sends more than the virtual-context working set (the
  200,000-character message above became 15,074 prompt tokens and the trace said `working set: 49 pages, 19777
  tokens`). The raw `/v1/completions` path sends the whole prompt, so the 50%/85% steps use it.

## Run record: 2026-10-19 20:26, RTX 5060 Ti alone, `build/strata` (compute_89 PTX) - **fall-off reproduced**

- Command: `CUDA_VISIBLE_DEVICES=GPU-da4ac99c-634b-cb80-8ac6-499ec76fff39 ./bin/strata serve -binary-path build/strata
  -config /tmp/strata-5060ti-alone.json`; monitor `/tmp/strata-5060ti-ptx-monitor.sh` (same checks as the 20:12 run),
  abort conditions written before the start. Only `-binary-path` differs from the 20:12 run. Engine `build/strata`
  sha256 877d86b8…, `cuobjdump --list-elf`: 58 `sm_89.cubin`, **0 `sm_120.cubin`**.
- The expected marker printed at start: `WARNING GPU 0 (NVIDIA GeForce RTX 5060 Ti, sm_120) runs code compiled for
  sm_89: this engine has no native code for it, so the driver compiles PTX at run time`. Same plan as 20:12 (4447
  slots, 8.46 GiB, prompt chunk 8192, 2069 borrowed slots, 334 MiB free), ready in well under a minute.
- Served the same 3 short chats as 20:12: 176, 250 and 300 tokens at **40.1, 50.0 and 54.1 tok/s** decode (native run:
  38.6, 49.2, 56.7 - the JIT'd path is not slower here, it is the one that dies).
- **Then the 131,062-token prompt killed the card at 20:26:58**, 27 s after the request was sent: the engine logged
  `KVEV v1 checkpoint tier=checkpoint tokens=0-32768` and then `prefill mmq: quantize: unspecified launch failure`,
  and the card was gone. Kernel log: `NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus`, followed by many
  `nvAssertFailedNoLog: ... NV_ERR_GPU_IN_FULLCHIP_RESET`, `kgmmuInvalidateTlb_GM107: TLB invalidation failed`, and
  `nvGpuOpsReportFatalError: uvm encountered global fatal error 0x60, requiring os reboot to recover`, plus
  `Xid 154, GPU recovery action changed from 0x0 (None) to 0x2 (Node Reboot Required)` (also reported for 05:00.0).
  The same sequence as the five historical incidents recorded in `handoff.md`. `nvidia-smi -L` has reported
  `Unable to determine the device handle for GPU0: 0000:01:00.0: Unknown Error` since; the 4060 still answers.
- Conditions at the fall-off: 80 C, 147 W (limit 180 W, no slow-down flag), link x8 Gen4, 15,608 MiB used as
  planned, 0 aborts in the monitor before the card vanished (the device-handle check fired first). Not thermal, not
  power, not the link.
- The server did not reload the model: `not restarting: the driver reports a lost or faulted GPU`, the
  `gpuFaultsInRow` rule from this audit, after one restart attempt. The failure log holds the engine's last lines,
  the `nvidia-smi` state and the kernel lines (`strata-engine-failures.log`).
- Stopped there, per the abort rule: server stopped, no reboot, no driver reload, no retry. The card needs a full
  power-off before it can be used again (the UVM error says a reboot; the link check says the same), and that is the
  owner's action - a power cycle is a recovery for the driver, not a fix for the fall-off.

## Run record: 2026-10-19 20:41, RTX 5060 Ti primary + 4060 peer tier (native sm_120) - repeat, clean

- Command: `CUDA_VISIBLE_DEVICES=GPU-da4ac99c-... ./bin/strata serve -binary-path build-fat/strata -config
  /tmp/strata-5060ti-alone.json`, monitor `/tmp/strata-monitor.sh native2`. This was meant to be the single-card repeat
  of 20:12 and **was not**: `/tmp` was wiped by the 20:37 reboot, the config was recreated from `strata-coder-iq1_m.json`
  with the UUID only in the top-level `CUDA_VISIBLE_DEVICES` key, and the engine reads the variable from its
  *environment* (`src/program/generate.cpp:1587`), which the config's `env` block still set to the repo's `"0,1"`. The
  Go server does not read the top-level key (no reference under `pkg/`). So both cards were visible and **the peer tier
  ran on the 4060**: `peer GPU 1: 3606 experts, 6.89 GiB; with the primary's 4442 that is 8048 of 12288`, and
  `nvidia-smi` showed 7,392 MiB on the 4060 for the length of the run. The configs have since been fixed (the `env`
  block carries the UUID; a one-card run needs that, not the top-level key).
- Native engine: no `runs code compiled` line; 4442 slots, 8.46 GiB, 334 MiB free, prompt chunk 8192; the peer's
  prefill buffers did not fit, so `the prompt path stays on the primary GPU`.
- Served: `prompt 131,062 tokens` in 96.2 s (**1,362.2 tok/s**), then `prompt 222,805 tokens` in 91.7 s (114,688
  reused + 108,117 read, **1,178.6 tok/s**). 122 monitor samples, 0 aborts, 0 Xid, x8 throughout, max 80 C / 150.6 W.
- A user `nvtop` (pid 23429) started at 20:42:26 and was running for the 222,805-token prompt, which stayed clean.

## Run record: 2026-10-19 20:45, same two-card config, `build/strata` (PTX) - **fall-off again**

- Same server binary, config and monitor (`/tmp/strata-monitor.sh ptx2`), only `-binary-path build/strata` changed.
  `runs code compiled for sm_89` printed; the same peer tier came up (6.89 GiB on the 4060).
- Three chats first: 122, 145 and 146 tokens at **39.3, 69.9 and 76.8 tok/s** decode. Then the same 131,062-token
  prompt: dead **27 s in** at 20:46:09, `KVEV v1 checkpoint tokens=0-32768` then `prefill mmq: quantize: unspecified
  launch failure`; kernel `NVRM: Xid (PCI:0000:01:00): 79, pid=23429, name=nvtop, GPU has fallen off the bus`, then
  `Xid 154 ... Node Reboot Required`, `uvm encountered global fatal error 0x60`, and the monitor aborted on
  `Unable to determine the device handle for GPU0` at 20:46:11. 79 C, 150.3 W, x8. The server did not reload the model
  (`not restarting: the driver reports a lost or faulted GPU`).
- Same abort rule: stopped, no reboot, no driver reload. The card needs a full power-off again.
- Second PTX fall, at the same checkpoint of the same prompt as 20:26, in a different (two-card vs one-card)
  configuration - the trigger travels with the binary, not with the peer tier.

## Run record: 2026-10-19 21:00, RTX 5060 Ti alone, installed engine (native `89;120`) - clean

- Command: `./bin/strata serve -config strata-coder-iq1_m.json` with the installed `engine/strata` (sha256
  140d3f6c..., byte-identical to `build-fat/strata`, 58 `sm_89` + 58 `sm_120` cubins), 5060 Ti alone, no peer tier.
  No `runs code compiled for sm_<n>` line.
- Served: 3 chats of 300 max tokens at ~43 tok/s decode (6.9-10 s each); raw completions of 201,932 tokens in 2m41s
  (~1,250 tok/s) and 257,467 tokens in 3m34s (~1,200 tok/s) - both past the 131,062-token prompt that killed the PTX
  engine at the 32k checkpoint in the 20:26 and 20:46 runs.
- Monitor (2 s, the abort conditions of `gpu-test-plan.md`): 0 aborts, 0 Xid, x8 held, max 81 C. The kernel journal
  for this boot (20:52 onward) has no Xid line at all.
- Third native pass on this card. It is a run of the fix's precondition - the card executing code built for it - not
  of any hardware change: the card, slot, cable and PSU are the same as in the fall-offs.

## What this does not claim

The bounded waits stop Strata from hanging on a dead GPU. They do not stop the card from dropping. No change here
was shown to prevent Xid 79, and a clean run after a driver reload or reboot would not show it fixed.

Finding 20 was checked without touching a card: `ninja -C build strata gpu_skip_test` builds, `./build/gpu_skip_test`
passes, and `CUDA_VISIBLE_DEVICES= ./build/strata --peer-device 1` prints "CUDA1 is not there: the driver lists no
CUDA device; continuing on the primary GPU..." and goes on to its token check. A device that is listed but fails to
open (out of memory, its context does not start) takes the `PeerExperts::open`/`RemoteExperts::open` path, which no
run has exercised.

The `89;120` build of finding 0 was made in this pass (`build-fat/strata`, verified with `cuobjdump`) and has now run:
three times on the 5060 Ti (20:12, 20:41, 21:00), each clean, against two fall-offs on the sm_89-only control (20:26,
20:46). The installed `engine/strata` and both build trees are native since 21:11 (finding 0).

No GPU run was made for this pass: the C++ changes build (`ninja -C build strata`) and `gpu_wait_test` passes, the
Go changes pass `go test ./pkg/...` (including `-race` for `pkg/virtualcontext` and `-count=20` for
`pkg/engineipc`), but a bounded wait that times out on a real stuck stream is not measured here. Finding 6b (the
VMM unmap drains) still needs a stream registry and a GPU run; finding 9 (per-site fault injection) cannot reach
`cudaStreamQuery` without a device, so only the `wait_bounded` template has a no-GPU test.
