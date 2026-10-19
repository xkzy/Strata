# GPU resource lifecycle

How Strata's GPU resources are created, used and released, and what must hold for a device that has failed.
Based on reading the code; not yet confirmed by a Compute Sanitizer run (no GPU run was possible during the audit).

## Rules

1. A GPU wait is bounded (`STRATA_GPU_WAIT_S`, default 60 s). On timeout or a CUDA error the owner of that device
   is marked failed.
2. A failed device is quarantined: nothing more is submitted to it, and its memory, streams and events are not
   freed or synchronized, because work that is stuck may still read or write them. The process frees them at exit.
3. A CUDA error is returned to the caller with the CUDA message. It is not dropped and work does not continue on
   the resource that raised it.
4. A mapped (zero-copy) buffer is used only if the pointer is pinned host memory; otherwise the staged copy is used.

## Per component

| Component | Created | Used by | Released | State |
|---|---|---|---|---|
| Peer experts (`peer_experts.cpp`) | `open()`: device buffers, stream, events, pinned/mapped result buffer | `launch()`, `finish()` | `close()`; skipped when `failed_` | Rules 1, 2, 4 in place |
| Verify commit (`verify.cpp`) | per session | `pl_commit_async` | session end | Commit wait bounded. The destructor drains its streams with a bounded wait; on a timeout it frees nothing |
| MTP drafter (`mtp.cpp`) | graphs, mapped outputs `h_out_`, `h_prob_` | draft rounds | drafter destructor | Both waits bounded; error returned. A timeout or stream error marks the drafter failed: `draft()` refuses and the destructor does not sync or free |
| Session loop (`session.cpp`) | graphs, doorbell in mapped memory | `session_loop`, `session_run_token` | session end | Ring poll bounded (the token path already had a 20 s limit) |
| Prefill (`prefill.cpp`) | host staging and device buffers | prefill pass | freed at the end of the pass | Peer copies on the non-compact path are used only with working P2P |
| Expert arenas | `cudaHostRegister` pinned arenas, `--mmap-experts` | GPU reads over PCIe | unregister at shutdown | Not audited in depth |

## Not yet audited

- Free-before-synchronize order in the other destructors (295 `cudaFree` sites). Done: `PeerExperts`, `MtpDrafter`, `Verifier`. Still unbounded: `expert_cache.cpp` lines 351, 395 and 741 (`cudaDeviceSynchronize`, `cudaStreamSynchronize`), and the rest.
- CUDA graph lifetime: read `verify.cpp` batch-window eviction (the evicted exec is from a window already synced by `cudaStreamSynchronize(cs_)`) and `session_graphs_free`/`token_graph_free` (called at the end of the benchmark and diagnostic programs in `generate.cpp`). No defect found by reading; whether `cudaGraphExecDestroy` is safe with a launch still in flight was not checked against the CUDA documentation and is not relied on.
- KV-cache transfers on the device (`kv_ring_restore`, KV streaming) when a copy fails halfway. The Go page table (`pkg/virtualcontext`) is bookkeeping only and was read.
- Idempotence of shutdown and cancellation.
- Partial failure during multi-GPU start-up.
