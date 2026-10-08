# Measured KV ledger: real numbers behind the virtual KV page table

Date: 2026-10-08. Status: draft for review.

## Why

The virtual KV page table (`include/strata/rt/kv_page_table.hpp`, runtime layer) is bookkeeping with made-up values:
`byte_size = tokens * 4096`, `prefill_cost_ms = tokens * 0.05`, `model_hash = 0` (`src/rt/virtual_context.cpp:328-341`).
The real KV lives in the engine process (`generate --serve`): the conversation cache (live session, up to 6
checkpoints of ~118 MB, parked conversations in a host-RAM budget, session files on disk). Today nothing says how
much a chat switch costs in retrieval, restore (page-in) and prefill (recompute), so we cannot tell which of
cost-aware eviction, automatic NVMe spill, mid-prompt shared checkpoints or prefetch is worth building.

## Goal and success criteria

1. For every request, the Monitor and `/metrics` show three separate costs: retrieval ms, KV restore ms, prefill ms
   (plus tokens reused vs read, and an estimate of prefill time saved).
2. The page table is fed real bytes, token ranges and times, or its metrics are labeled "model, not measured".
3. A measured run on this machine (hardware named) is recorded in `docs/DETAILS.md`, with what was and was not measured.
4. No change to what the engine caches, evicts or restores.

## Out of scope

Eviction policy, tiers, compression, prefetch, mid-prompt checkpoints, any change to cache behavior.

## Design

### 1. Engine: structured cache-event lines (small, additive)
At each park, restore, checkpoint and prefix reuse, the engine prints one line next to today's free-text log line
(`src/program/generate.cpp`, around the "conversation cache:" print):

    KVEV v1 <event> tier=<live|checkpoint|parked> tokens=<first>-<last> bytes=<n> ms=<float>

`event` is one of `park`, `restore`, `checkpoint`, `reuse`. Values come from data the code already has (snapshot
bytes, token counts, elapsed time). The `v1` prefix lets the server skip unknown versions.

### 2. Go server: ledger and cost split
- `pkg/server` parses `KVEV v1` lines from the engine's log stream (the path that already reads `RESUME`, `REUSED`,
  `PP`) into a ledger: recent events, per-tier bytes, per-request restore ms.
- Per request: retrieval ms (existing runtime timing), restore ms (sum of that request's `restore` events), prefill ms
  (prompt tokens read / measured prefill rate), tokens reused, estimated prefill ms saved (reused tokens / same rate).
- Exposed in the Monitor's Conversation cache card and under `"conversation_cache"` in `/metrics`, extending
  `conversationCacheView` in `pkg/server/enginemetrics.go`.

### 3. Page table
Feed `VirtualKVPageTable` from the ledger (real bytes, token ranges, measured restore and prefill times) where the
runtime layer can see them; where it cannot, label the `virtual_kv` metrics as modeled. The 138 ns lookup benchmark
stays, described as a table-only microbenchmark.

### 4. Measurement
Run a two-agent alternating scenario and a long single-chat scenario, record restore, prefill and reuse numbers with
the machine, model and settings, and state the unmeasured parts. The numbers decide the next project.

## Error handling
Unknown or malformed `KVEV` lines are ignored. A missing event leaves a metric empty (`null`), never zero. Parsing
never blocks the request path.

## Testing
- Go: table tests for the parser (valid, unknown version, malformed) and the cost split, from recorded engine lines.
- `tools/parking_test.py`: assert the new lines appear, bytes and restore ms are plausible, and the follow-up still
  decodes the same tokens.
- Re-run `test_kv_page_table` and `go test ./pkg/...`.

## Risks
The line format becomes an engine-server interface (mitigated by `KVEV v1`). Prefill ms is derived from a rate, so it
is an estimate and is labeled as one. The engine change needs a CUDA build to verify.
