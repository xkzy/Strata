# Hardware discovery

Strata discovers the machine before it chooses a configuration. Discovery never fails startup and never requires
a flag: every source is optional, and a question that cannot be answered stays `unknown` instead of becoming an
assumption. The code is in [`pkg/autotune/capability.go`](../../pkg/autotune/capability.go) (schema) and
[`pkg/autotune/discover.go`](../../pkg/autotune/discover.go) (sources).

## The capability schema

`Capabilities` is a plain value with schema version 1. It is persisted in the tuning record, so a field's meaning
may only change with `CapabilitySchema`.

| Group | Fields | Why the solver needs it |
|---|---|---|
| Host | `os`, `arch`, `driver`, `backends` | Backend support is a hard constraint; `backends` always contains `cpu`. |
| `cpu` | `model`, `logical`, `physical`, `available`, `numa[]`, `simd[]`, `mem_bandwidth_gbs`, `mem_bandwidth_source` | Kernel support (AVX-512 / AVX2 / NEON), the worker-count search space, and CPU roofline rates. `available` is this process's affinity mask, not the box's core count. |
| `memory` | `total_bytes`, `available_bytes`, `swap_total_bytes`, `swap_free_bytes`, `cgroup_limit_bytes`, `cgroup_used_bytes`, `pinned_limit_bytes`, `spread_bytes`, `pressure` | Host feasibility, the reserve, and pinned-staging limits. Swap is capacity only: the plan never relies on it for inference. |
| `accelerators[]` | `id`, `vendor`, `backend`, `name`, `memory_total_bytes`, `memory_free_bytes`, `compute_class`, `data_types[]`, `link_gen`, `link_width`, `link_gbs`, `memory_gbs`, `streams`, `health`, `health_reason`, `temperature_c`, `power_w`, `power_limit_w`, `throttled` | Placement, memory feasibility, format support and reliability. |
| `peers[]` | `from`, `to`, `kind`, `p2p`, `bandwidth_gbs` | Whether a device pair has a real peer path. Absence is not a claim of P2P. |
| `unknown[]` | free text | Everything discovery could not establish, with the reason. |

`Health` is `ok`, `unknown` or `failed`. Only `ok` devices may receive benchmark work or be placed; `unknown` is
not treated as healthy. `MemoryInfo.UsableBytes()` is the smallest of available host memory, the cgroup's remaining
limit and the total (halved when no availability figure exists).

## Sources

Discovery is Linux-first because the engine's measured targets are Linux and Windows-on-Linux-tooling. On other
platforms the CPU count comes from the Go runtime and the remaining fields stay unknown.

- **CPU**: `/proc/cpuinfo` (model, `flags`/`Features`, physical cores from `physical id` + `core id`),
  the process affinity mask, and `/sys/devices/system/node/node*` for NUMA. A missing SIMD list is recorded as
  unknown; it is not assumed to include AVX2.
- **Memory**: `/proc/meminfo` twice, 50 ms apart; the observed change in `MemAvailable` becomes `spread_bytes`
  and is part of the reserve. Cgroup v2 and v1 memory files are read for container limits, and `RLIMIT_MEMLOCK`
  becomes `pinned_limit_bytes`.
- **NVIDIA**: one `nvidia-smi --query-gpu=...` call (identity, total/free memory, compute capability, link
  generation/width, temperature, power, throttle reasons, driver) and `nvidia-smi topo -m` for peer links. The
  link estimate uses the *maximum* generation and width because an idle card reports a downshifted current link.
  `PHB`, `NODE` and `SYS` are recorded as host-staged and never as P2P.
- **AMD**: sysfs (`vendor`, `device`, `mem_info_vram_total`, `mem_info_vis_vram_total`,
  `current_link_speed`/`current_link_width`). No ROCm tool is required. VRAM free memory is often unavailable from
  sysfs; then the device is kept with `memory_free_bytes = 0` and the solver rejects it as memory-uncertain.
- **Intel / unknown**: sysfs PCI identity and memory files. A device that reports no backend is not added to
  `backends`, so no plan can claim to use it.

Vendor-specific code is confined to the `discoverNVIDIA`, `discoverAMD*` and `discoverIntel*` functions. The
schema, solver and record never contain a vendor concept such as "CUDA stream" or "ROCm queue"; backend names are
opaque strings checked against `backends`.

## Partial failure policy

- Missing optional telemetry (no `nvidia-smi`, no cgroup, unreadable sysfs) does not abort discovery; it adds a
  line to `unknown` and leaves the related field zero.
- A device whose memory did not report is `unknown`, gets no benchmark work and no plan. The rest of the machine
  remains usable.
- A device reporting a fatal throttle/brake (`clocks_throttle_reasons.active` bits `0x8`, `0x40`, `0x80`) or a
  temperature of at least 90 °C is `failed`. Software power caps (`0x4`, `0x20`) are normal and are not failures.
- If no accelerator is usable, the CPU placement remains in the domain. CPU-only execution is a supported result,
  not an error.
- Discovery itself never returns an error and never blocks on a missing command for longer than its context.

## Fingerprint

`Capabilities.Fingerprint()` is a 96-bit SHA-256 prefix over the schema version, OS/arch, driver, CPU model and
counts, sorted SIMD flags, NUMA nodes, memory totals/cgroup/pinned limits, every accelerator's identity, vendor,
backend, name, total memory, compute class, sorted data types and link generation/width, and sorted peer links.

Deliberately excluded: free memory, temperatures, utilisation, process count and discovery time. They change
between runs and would invalidate every record for no reason. A record whose fingerprint differs is not used;
the fingerprints are checked again on every start.

## Inspecting what was found

```sh
go run ./cmd/strata tune --save=false | jq .capabilities
go test ./pkg/autotune/ -run 'TestDiscover|TestRecord' -v
```

The discovery tests inject a fake `Host` (file tree) and `CommandRunner`, so they run without a GPU, without
`nvidia-smi`, and without root.

## Tested and untested

| Combination | Status |
|---|---|
| Linux, x86-64, AVX-512, one NVIDIA GPU, 64 GiB RAM | exercised on the development machine (see `startup-benchmark.md`) |
| Linux, two heterogeneous NVIDIA GPUs | discovery and planning exercised; no split is ever selected without a measured pair rate |
| Linux, AMD via sysfs | discovery exercised (a 2 GiB sysfs device was found; its free memory was reported) |
| Windows | CPU and memory fields come from the same logic where the paths exist; no Windows machine was available for this change |
| macOS / ARM (NEON) | code path exists (`neon`, `sve`); untested |
| AMD ROCm engine path | discovery is vendor-neutral but no ROCm device was benchmarked |
| Intel Arc / oneAPI | discovery only; untested for planning |

Unknown capability is always visible in `capabilities.unknown`; it is never silently replaced by a plausible
default.
