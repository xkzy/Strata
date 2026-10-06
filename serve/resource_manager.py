"""serve/resource_manager.py - #18 Auto-Adaptive Resource Utilization (the server layer).

A feedback controller over what Strata's server already measures: serve/telemetry.py's 1 Hz hardware sampler, the
engine's INFO/DONE counters and the request history.  The loop mirrors the issue's diagram:

    ResourceMonitor -> WorkloadAnalyzer -> ResourceEstimator -> Scheduler -> Execution -> Telemetry
                                                       ^______________________________| feedback

The knobs it turns - and the ONLY things it turns (sampling semantics and numerics are never touched):
    concurrency    how many requests may be active at once (capped by the engine's own batch slots)
    queue_limit    how many requests may wait before new ones get a structured 429 with retry_after
    cache_tier     the memory-pressure tier (balanced | retain | tight | aggressive)
    power_policy   the targets: MAX_THROUGHPUT | LOW_LATENCY | BALANCED | ENERGY_SAVING (config)

Rules it follows (the issue's "final design principles"):
- No hardcoded device, vendor, core-count or GPU-count assumptions: the monitor reads whatever the OS and the
  engine report, and the estimator compares measurements against their own recent baseline - never FLOPS.
- Hysteresis: a direction must hold for STABLE_TICKS evaluations, changes sit behind a cooldown (growth waits
  twice as long after a shrink), and a concurrency level that did not add throughput is held back for
  SATURATION_WINDOW seconds - so the loop never oscillates increase/decrease every interval.
- Workload classes are INFERRED (prompt size, requested answer length, measured prefill/decode split, expert hit
  rate, PCIe share) - users never name them.
- A memory-pressure tier tightens admission instead of overcommitting: exhaustion ends as a structured status
  (Overloaded -> HTTP 429 with retry_after), never as system-wide OOM.
- Every knob change lands in a bounded decision log with its reason, which /metrics' "resource" section shows -
  enough to answer "why is this request using this resource?" and "why is this resource idle?".

The numbers below are policy (targets, cooldowns), not hardware facts: they say how hard the controller leans,
never which device is faster.
"""
from __future__ import annotations

import collections
import threading
import time
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- workload classes (inferred)
SHORT_INTERACTIVE = "SHORT_INTERACTIVE"
LONG_CONTEXT = "LONG_CONTEXT"
HIGH_THROUGHPUT = "HIGH_THROUGHPUT"
LOW_LATENCY = "LOW_LATENCY"
MEMORY_BOUND = "MEMORY_BOUND"
COMPUTE_BOUND = "COMPUTE_BOUND"
TRANSFER_BOUND = "TRANSFER_BOUND"
MOE_SPARSE = "MOE_SPARSE"
MOE_DENSE = "MOE_DENSE"
PREFILL_HEAVY = "PREFILL_HEAVY"
DECODE_HEAVY = "DECODE_HEAVY"

WORKLOAD_CLASSES = (SHORT_INTERACTIVE, LONG_CONTEXT, HIGH_THROUGHPUT, LOW_LATENCY, MEMORY_BOUND, COMPUTE_BOUND,
                    TRANSFER_BOUND, MOE_SPARSE, MOE_DENSE, PREFILL_HEAVY, DECODE_HEAVY)

MAX_THROUGHPUT = "MAX_THROUGHPUT"
LOW_LATENCY_POLICY = "LOW_LATENCY"
BALANCED = "BALANCED"
ENERGY_SAVING = "ENERGY_SAVING"
POWER_POLICIES = (MAX_THROUGHPUT, LOW_LATENCY_POLICY, BALANCED, ENERGY_SAVING)

CACHE_TIERS = ("retain", "balanced", "tight", "aggressive")

# --------------------------------------------------------------------------- policy numbers (targets, not hardware)
QUEUE_BOUNDS = {                       # (min, max) queue_limit per power policy
    MAX_THROUGHPUT: (16, 512),
    LOW_LATENCY_POLICY: (4, 32),
    BALANCED: (8, 128),
    ENERGY_SAVING: (4, 64),
}
DEFAULT_QUEUE = {MAX_THROUGHPUT: 64, LOW_LATENCY_POLICY: 8, BALANCED: 32, ENERGY_SAVING: 16}
DEGRADE_RATIO = {                      # p95 must rise this much over its own baseline before a shrink (per policy)
    MAX_THROUGHPUT: 1.6, LOW_LATENCY_POLICY: 1.2, BALANCED: 1.35, ENERGY_SAVING: 1.3,
}
GROW_UTIL = 35.0                       # util below this % with latency flat means room to grow
PRESERVE_UTIL = 60.0                   # MAX_THROUGHPUT grows while below this util
PRESSURE_TIGHT = 0.85                  # host RAM: the cache tier tightens here
PRESSURE_CRITICAL = 0.93               # ... and refuses new work here (floor of MIN_FLOOR in flight kept)
MIN_FLOOR = 2                          # never backpressure down to zero from memory alone
STABLE_TICKS = 2                       # consecutive evaluations in one direction before a change
SHRINK_COOLDOWN_S = 2.0                # a shrink separates itself from the next change by this
GROW_COOLDOWN_MULT = 2                 # after a shrink, growth waits this multiple of the cooldown
SATURATION_RATIO = 1.03                # a grown concurrency must add at least 3% throughput to be kept
SATURATION_WINDOW_S = 45.0             # ... otherwise growth is held back this long
LOOP_TTL_S = 90.0                      # how long a suspected/confirmed loop limits a session's admission
PENDING_TTL_S = 120.0                  # an admitted request that never reached run() stops reserving after this
HISTORY = 256                          # observations kept for percentiles and classification
DECISIONS = 64                         # decisions kept for /metrics


class Overloaded(RuntimeError):
    """#18 backpressure: the server refuses NEW work with a structured status instead of queueing until memory
    runs out.  The handler maps it to HTTP 429 with `retry_after` seconds (a measured estimate: how long the
    current queue takes to drain at the observed completion rate)."""

    status = 429

    def __init__(self, message: str, retry_after: float = 1.0, reason: str = "capacity"):
        super().__init__(message)
        self.retry_after = max(1, int(round(retry_after)))
        self.reason = reason


# --------------------------------------------------------------------------- observations & classification
@dataclass
class Observation:
    """One finished request as the controller measures it: characteristics, the engine's own clock, and the
    classes inferred from both (post-run classes replace the pre-run guesses for this window)."""
    prompt_tokens: int = 0
    output_tokens: int = 0
    max_new: int = 0
    queue_s: float = 0.0
    ttft_s: float | None = None        # start -> first token (prompt read + prefill)
    prompt_ms: float | None = None      # the engine's clock (None on a mock engine)
    decode_ms: float | None = None
    pcie_share: float | None = None    # the share of routed experts computed elsewhere (#588)
    hit_rate: float | None = None      # expert-cache hit rate of the DONE line
    finish: str = ""
    at: float = 0.0
    classes: frozenset = field(default_factory=frozenset)


def classify(prompt_tokens: int, max_new: int, context: int = 0) -> frozenset:
    """The classes of a request from its OWN characteristics, before it runs - no user input required.
    Thresholds are relative to the request (and its context window), never to a device."""
    out = set()
    if prompt_tokens <= 1024 and 0 < max_new <= 256:
        out.add(SHORT_INTERACTIVE)
    if max_new and max_new <= 64:
        out.add(LOW_LATENCY)
    if prompt_tokens >= 8192 or (context and prompt_tokens >= context // 4):
        out.add(LONG_CONTEXT)
    if max_new >= 4096 or (prompt_tokens and max_new >= 4 * prompt_tokens):
        out.add(DECODE_HEAVY)
    if max_new >= 4096:
        out.add(HIGH_THROUGHPUT)
    if prompt_tokens >= 2048 and prompt_tokens >= 4 * max_new:
        out.add(PREFILL_HEAVY)
    return frozenset(out or {SHORT_INTERACTIVE if prompt_tokens <= 1024 else PREFILL_HEAVY if
                             prompt_tokens >= 2048 else LOW_LATENCY})


def measured_classes(o: Observation) -> frozenset:
    """The classes the RUN itself proves (or adds to) the pre-run guess: which half of the work dominated, where
    the experts were computed, how much memory the decode needed.  Keeps the guess and the measurement together."""
    out = set(o.classes)
    if o.prompt_ms is not None and o.decode_ms is not None:
        if o.prompt_ms >= o.decode_ms and o.prompt_ms >= 100:
            out.add(PREFILL_HEAVY)
        if o.decode_ms > o.prompt_ms and o.decode_ms >= 100:
            out.add(DECODE_HEAVY)
    if o.pcie_share is not None and o.pcie_share >= 0.25:
        out.add(TRANSFER_BOUND)
    if o.hit_rate is not None:
        (out.add(MOE_DENSE) if o.hit_rate >= 0.9 else out.add(MOE_SPARSE) if o.hit_rate < 0.6 else None)
    return frozenset(out)


# --------------------------------------------------------------------------- the five stages
class ResourceMonitor:
    """Stage 1: reads - without new polling - what the server already samples.  `source()` returns one dict from
    the live system: telemetry's 1 Hz hardware readings (cpu, ram, gpu), the queue counters and engine counters.
    Anything missing is None; nothing here can stop the server."""

    def __init__(self, source):
        self._source = source

    def snapshot(self) -> dict:
        try:
            snap = dict(self._source() or {})
        except Exception:                                   # a broken source must never block admission
            snap = {}
        return snap

    @staticmethod
    def pressure(snap: dict):
        """0..1 host-RAM pressure - the memory the parked conversations, queues and buffers share.  VRAM is
        deliberately not part of it: a loaded model fills its card by design (the expert cache takes the rest),
        so the raw used/total would read as pressure forever.  A VRAM floor still counts when free runs out."""
        used, total = snap.get("ram_used"), snap.get("ram_total")
        if used and total and total > 0:
            p = used / total
        else:
            p = None
        free = snap.get("vram_free_mib")
        if isinstance(free, (int, float)) and free < 400:   # below any usable working set: count it as critical
            p = PRESSURE_CRITICAL if p is None else max(p, PRESSURE_CRITICAL)
        return None if p is None else max(0.0, min(1.0, p))

    @staticmethod
    def util(snap: dict):
        """The best available busy-% reading (GPU under a GPU run, CPU under a CPU run): None when neither is
        reported.  The controller asks "is the machine busy?", never "which vendor is it?"."""
        vals = [snap.get(k) for k in ("gpu_util", "cpu") if isinstance(snap.get(k), (int, float))]
        return max(vals) if vals else None


class WorkloadAnalyzer:
    """Stage 2: the window of recent requests - what the workload IS (classes), how fast requests arrive, and the
    p95 of queue wait / time-to-first-token / decode rate, which the estimator measures against their own baseline."""

    def __init__(self, maxlen=HISTORY):
        self._lock = threading.Lock()
        self._obs: collections.deque = collections.deque(maxlen=maxlen)
        self._starts: collections.deque = collections.deque(maxlen=1024)   # (at, classes)
        self.classes: dict = collections.Counter()

    def observe_start(self, prompt_tokens: int, max_new: int, context: int = 0, now=None):
        cls = classify(prompt_tokens, max_new, context)
        with self._lock:
            self._starts.append((now or time.time(), cls))
        return cls

    def observe_finish(self, o: Observation):
        # the request's own characteristics + what the run measured: no user input, no phase to remember
        if not o.classes:
            o.classes = classify(o.prompt_tokens, o.max_new)
        o.classes = measured_classes(o)
        with self._lock:
            self._obs.append(o)
            for c in o.classes:
                self.classes[c] += 1

    @staticmethod
    def _pct(vals, p):
        if not vals:
            return None
        s = sorted(vals)
        return s[min(len(s) - 1, int(round((len(s) - 1) * p)))]

    def stats(self) -> dict:
        """Percentiles over the recent window plus arrival/completion rates of the last 30 s (per second)."""
        with self._lock:
            obs = list(self._obs)
            starts = list(self._starts)
        now = time.time()
        recent = [o for o in obs if now - o.at <= 30.0]
        span = max(1.0, min(30.0, now - min((o.at for o in recent), default=now - 30.0)))
        ttft = [o.ttft_s for o in recent if o.ttft_s is not None] or \
               [o.prompt_ms / 1000 for o in recent if o.prompt_ms]
        queue = [o.queue_s for o in recent if o.queue_s > 0]
        hits = [o.hit_rate for o in recent if o.hit_rate is not None]
        return {
            "samples": len(recent),
            "p95_ttft_s": self._pct(ttft, 0.95),
            "p95_queue_s": self._pct(queue, 0.95),
            # aggregate useful work: the tokens actually completed per second over the window
            "tok_s": sum(o.output_tokens for o in recent) / span if recent else None,
            "hit_rate": sum(hits) / len(hits) if hits else None,
            "arrival_per_s": sum(1 for t, _ in starts if now - t <= span) / span,
            "completion_per_s": len(recent) / 30.0,
            "output_tokens": sum(o.output_tokens for o in recent),
        }

    def dominant(self, last=100) -> list:
        """[[class, share], ...] over the most recent window - what the workload currently IS."""
        with self._lock:
            obs = list(self._obs)[-last:]
        cnt: dict = collections.Counter()
        for o in obs:
            for c in o.classes:
                cnt[c] += 1
        total = sum(cnt.values()) or 1
        return [[c, round(n / total, 3)] for c, n in cnt.most_common()]


class ResourceEstimator:
    """Stage 3: turns measurements into a direction per knob - grow / hold / shrink, each with a reason.  It
    compares against the workload's OWN recent baseline (EWMA), never an absolute latency target: the loop says
    "p95 TTFT rose from 1.2 s to 1.9 s", not "TTFT is bad for this GPU"."""

    def __init__(self, policy: str = BALANCED):
        self.policy = policy
        self.baseline_ttft = None
        self.baseline_queue = None

    def set_policy(self, policy: str):
        if policy not in POWER_POLICIES:
            raise ValueError(f"power_policy must be one of {', '.join(POWER_POLICIES)}, not {policy!r}")
        self.policy = policy

    def _ewma(self, old, new, alpha=0.3):
        return new if old is None else old + alpha * (new - old)

    def propose(self, stats: dict, snap: dict, knobs: dict, idle_evals: int) -> list:
        """-> [(knob, direction, reason), ...]; direction: +1 grow, -1 shrink, 0 hold."""
        out = []
        pressure = ResourceMonitor.pressure(snap)
        util = ResourceMonitor.util(snap)
        p_ttft, p_queue = stats.get("p95_ttft_s"), stats.get("p95_queue_s")
        have = stats.get("samples", 0) >= 4
        degrade = DEGRADE_RATIO[self.policy]

        # --- shrink signals (any one is enough; the reason names the measurement)
        if have and p_ttft is not None and self.baseline_ttft is not None and \
                p_ttft > self.baseline_ttft * degrade:
            out.append(("queue_limit", -1,
                        f"p95 TTFT rose from {self.baseline_ttft:.2f}s to {p_ttft:.2f}s"))
            out.append(("concurrency", -1, "latency degraded; fewer requests at once"))
        elif have and p_queue is not None and self.baseline_queue is not None and \
                p_queue > self.baseline_queue * degrade and p_queue > 1.0:
            out.append(("queue_limit", -1,
                        f"p95 queue wait rose from {self.baseline_queue:.2f}s to {p_queue:.2f}s"))
        if pressure is not None and pressure >= PRESSURE_CRITICAL:
            out.append(("queue_limit", -1, f"host memory {pressure:.0%} used"))
            out.append(("cache_tier", -1, f"memory pressure {pressure:.0%}: evict aggressively"))
        elif pressure is not None and pressure >= PRESSURE_TIGHT:
            out.append(("cache_tier", -1, f"memory pressure {pressure:.0%}: tighten admission"))

        # --- grow signals (only when nothing above fired: hysteresis at the signal level too)
        if not any(d == -1 for _, d, _ in out):
            q = snap.get("queued", 0)
            lim = knobs.get("queue_limit", 0)
            latency_flat = not (have and self.baseline_ttft is not None and p_ttft is not None and
                                p_ttft > self.baseline_ttft * 1.1)
            if lim and q >= 0.8 * lim and latency_flat and \
                    (pressure is None or pressure < PRESSURE_TIGHT):
                out.append(("queue_limit", +1, f"queue at {q} of {lim} with latency flat"))
                out.append(("concurrency", +1, "queue pressure with headroom"))
            elif util is not None and util < GROW_UTIL and latency_flat and \
                    (pressure is None or pressure < PRESSURE_TIGHT) and (q or snap.get("active", 0)):
                out.append(("concurrency", +1, f"machine {util:.0f}% busy with work waiting"))
                if util < PRESERVE_UTIL:
                    out.append(("queue_limit", +1, f"machine {util:.0f}% busy: room for the next burst"))
            elif idle_evals >= 3 and (q == 0 and not snap.get("active", 0)):
                out.append(("queue_limit", +1, "idle: restore room for the next burst"))
                if self.policy == ENERGY_SAVING:
                    out.append(("concurrency", -1, "idle: give resources back (ENERGY_SAVING)"))

        # --- the cache tier also opens back up when memory is comfortable and reuse is good
        if pressure is not None and pressure < PRESSURE_TIGHT:
            hit = stats.get("hit_rate")
            if not any(k == "cache_tier" and d == -1 for k, d, _ in out):
                out.append(("cache_tier", +1,
                            f"memory {pressure:.0%} used"
                            + (f", expert reuse {hit:.0%}" if hit is not None else "")))

        # --- keep the baselines moving only while nothing degraded (they must not drift up with an outage)
        if have and p_ttft is not None:
            degraded = any(d == -1 and k == "queue_limit" and "TTFT" in r for k, d, r in out)
            self.baseline_ttft = self._ewma(self.baseline_ttft, p_ttft) if not degraded \
                else self.baseline_ttft
            self.baseline_queue = self._ewma(self.baseline_queue, p_queue) \
                if p_queue is not None and not degraded else self.baseline_queue
        return out


class Scheduler:
    """Stage 4: applies the estimator's proposals to the actual knobs, with the hysteresis the issue asks for -
    direction stability, cooldowns (growth waits longer after a shrink), saturation detection - and records every
    change with its reason for /metrics."""

    def __init__(self, policy: str = BALANCED, min_interval: float = SHRINK_COOLDOWN_S):
        self.policy = policy
        self.min_interval = min_interval
        lo, hi = QUEUE_BOUNDS[policy]
        self.queue_limit = DEFAULT_QUEUE[policy]
        self.concurrency = 1
        self.cache_tier = "balanced"
        self.max_concurrency = 1          # the engine's batch slots raise it (set_engine); 1 = serial
        self.decisions: collections.deque = collections.deque(maxlen=DECISIONS)
        self._streak: dict = {}
        self._last_change: dict = {}
        self._last_dir: dict = {}
        self._idle_evals = 0
        self._last_grow = None            # (concurrency, throughput at that grow, when)
        self._hold_until = 0.0
        self._hold_reason = ""

    def set_policy(self, policy: str):
        if policy not in POWER_POLICIES:
            raise ValueError(f"power_policy must be one of {', '.join(POWER_POLICIES)}, not {policy!r}")
        self.policy = policy
        lo, hi = QUEUE_BOUNDS[policy]
        self.queue_limit = max(lo, min(hi, self.queue_limit))

    def set_engine(self, batch: int):
        """The engine's own batch slots are the hard ceiling of `concurrency` - the controller never assumes more
        parallelism exists than the engine reported."""
        self.max_concurrency = max(1, int(batch or 0))

    def note(self, knob, old, new, reason, now):
        self.decisions.append({"at": round(now, 1), "knob": knob, "from": old, "to": new, "reason": reason})

    def _bounds(self, knob):
        if knob == "queue_limit":
            return QUEUE_BOUNDS[self.policy]
        if knob == "concurrency":
            return 1, self.max_concurrency
        return 0, 0                                   # cache_tier: ordered by CACHE_TIERS, applied elsewhere

    def apply(self, proposals: list, stats: dict, now: float):
        """Proposals -> knob changes, or nothing.  A direction must repeat STABLE_TICKS evaluations and sit out
        the cooldown; a grow right after a shrink waits GROW_COOLDOWN_MULT x longer; a grow whose throughput did
        not improve over SATURATION_RATIO is held back for SATURATION_WINDOW_S (learn the operating point)."""
        by_knob: dict = {}
        for knob, direction, reason in proposals:
            by_knob.setdefault(knob, (direction, reason))     # first reason per knob wins (most specific)
        for knob, (direction, reason) in by_knob.items():
            key = knob
            if direction == 0:
                self._streak[key] = 0
                continue
            if self._last_dir.get(key) != direction:
                self._streak[key], self._last_dir[key] = 1, direction
            else:
                self._streak[key] = self._streak.get(key, 0) + 1
            if self._streak[key] < STABLE_TICKS:
                continue
            last = self._last_change.get(key, 0.0)
            cooldown = self.min_interval * (GROW_COOLDOWN_MULT if direction == 1 and
                                            self._last_change.get("_shrink_at", 0.0) > last else 1)
            if now - last < cooldown:
                continue
            if knob == "cache_tier":
                self._apply_tier(direction, reason, now)
                continue
            lo, hi = self._bounds(knob)
            cur = getattr(self, knob)
            new = cur + direction
            if new < lo or new > hi or new == cur:
                continue
            if knob == "concurrency" and direction == 1 and self._hold_until and now < self._hold_until:
                if not self._hold_reason:                        # say it once, not every evaluation
                    self.note("concurrency", cur, cur, f"growth held: {self._hold_reason}", now)
                continue
            if knob == "concurrency" and direction == 1:
                tp = stats.get("tok_s")
                if self._last_grow and tp is not None and self._last_grow[1] is not None:
                    lvl, base_tp, _ = self._last_grow
                    if cur >= lvl and tp < base_tp * SATURATION_RATIO:
                        self._hold_until = now + SATURATION_WINDOW_S
                        self._hold_reason = (f"throughput flat at {tp:.1f} tok/s after growing "
                                             f"(need +{int((SATURATION_RATIO - 1) * 100)}%)")
                        self.note("concurrency", cur, cur, f"stop growing: {self._hold_reason}", now)
                        self._last_change[key] = now          # hold the cooldown too
                        continue
                self._last_grow = (new, tp, now)
            setattr(self, knob, new)
            self._last_change[key] = now
            self._streak[key] = 0
            if direction == -1:
                self._last_change["_shrink_at"] = now
            self.note(knob, cur, new, reason, now)
            if knob == "concurrency" and direction == -1:      # a shrink cancels any growth hold
                self._hold_until, self._hold_reason = 0.0, ""

    def _apply_tier(self, direction: int, reason: str, now: float):
        # CACHE_TIERS is ordered gentle -> strict, so direction -1 (pressure) moves UP the index.
        i = CACHE_TIERS.index(self.cache_tier)
        j = max(0, min(len(CACHE_TIERS) - 1, i + (1 if direction == -1 else -1)))
        if j != i:
            self.note("cache_tier", self.cache_tier, CACHE_TIERS[j], reason, now)
            self.cache_tier = CACHE_TIERS[j]
            self._last_change["cache_tier"] = now
            self._streak["cache_tier"] = 0


class ResourceManager:
    """The facade the server talks to: admission control (structured backpressure), observations feeding the
    loop, loop-protection hooks, and the /metrics view.  Thread-safe; safe to leave enabled by default because
    the gates only bind once measurements show the server is full."""

    def __init__(self, source=None, policy: str = BALANCED, min_interval: float = SHRINK_COOLDOWN_S):
        self._source = source or (lambda: {})
        self._lock = threading.Lock()
        self.enabled = True
        self.monitor = ResourceMonitor(self._source)
        self.analyzer = WorkloadAnalyzer()
        self.estimator = ResourceEstimator(policy)
        self.scheduler = Scheduler(policy, min_interval)
        self.admitted = 0
        self.rejected: dict = collections.Counter()
        self.loop_events = 0
        self.loop_limited = 0
        self._suspected: dict = {}                         # session -> until (monotonic-less: time.time())
        self._pending: dict = {}                           # token -> when it was admitted, not yet in run()
        self._tokens = iter(range(1, 1 << 62))
        self._last_eval = 0.0

    # ---------------------------------------------------------------- knobs & config
    def set_power_policy(self, policy: str):
        self.estimator.set_policy(policy)
        self.scheduler.set_policy(policy)

    def set_enabled(self, on: bool):
        self.enabled = bool(on)

    def set_engine(self, batch: int):
        self.scheduler.set_engine(batch)

    # ---------------------------------------------------------------- the loop
    def _tick(self, now: float, snap: dict | None, force: bool = False):
        """One evaluation of the loop (rate-limited).  `snap` may be None when the caller must not touch the
        source (it holds a lock the source needs): the estimator then skips hardware signals for that pass."""
        if not self.enabled or (not force and now - self._last_eval < self.scheduler.min_interval / 2):
            return
        self._last_eval = now
        if snap is None:
            snap = {}
        if snap.get("batch") is not None:
            # the engine's own batch slots are the hard ceiling of concurrency - learned, never assumed
            self.scheduler.set_engine(int(snap.get("batch") or 0))
        stats = self.analyzer.stats()
        knobs = self.scheduler.__dict__
        if "active" in snap and "queued" in snap:              # an empty snapshot must not count as idle
            idle = not (snap.get("active") or snap.get("queued"))
            self.scheduler._idle_evals = self.scheduler._idle_evals + 1 if idle else 0
        proposals = self.estimator.propose(stats, snap, knobs, self.scheduler._idle_evals)
        self.scheduler.apply(proposals, stats, now)

    # ---------------------------------------------------------------- admission (execution side)
    def admit(self, session: str = "", now: float | None = None) -> int:
        """-> a token when the request may proceed (paired with begin_run / cancel_pending, so a request counted
        here can never leak capacity); Overloaded (HTTP 429 + retry_after) when it may not.  Runs the loop first,
        so a burst of new work is measured against the same numbers as a completion."""
        if not self.enabled:
            return 0
        now = now if now is not None else time.time()
        snap = self.monitor.snapshot()                     # OUTSIDE the lock: the source may need status_lock
        with self._lock:
            self._expire_pending(now)
            self._tick(now, snap)
            active, queued = int(snap.get("active") or 0), int(snap.get("queued") or 0)
            limit = self.scheduler.concurrency + self.scheduler.queue_limit
            if self.scheduler.cache_tier in ("tight", "aggressive"):
                limit = max(MIN_FLOOR, limit // 2)         # memory is tight: fewer in flight before refusing
            load = active + queued + len(self._pending)
            pressure = ResourceMonitor.pressure(snap)
            if load >= limit:
                raise Overloaded(self._reject("capacity", load, limit, queued, now),
                                 retry_after=self._drain_s(queued), reason="capacity")
            if pressure is not None and pressure >= PRESSURE_CRITICAL and load >= MIN_FLOOR:
                raise Overloaded(self._reject("memory", load, limit, queued, now),
                                 retry_after=max(2, self._drain_s(queued)), reason="memory")
            until = self._suspected.get(session or "")
            if until and now < until and load >= max(1, limit // 2):
                self.loop_limited += 1
                raise Overloaded(self._reject("loop_protection", load, limit, queued, now),
                                 retry_after=max(2, self._drain_s(queued)), reason="loop_protection")
            token = next(self._tokens)
            self._pending[token] = now
            self.admitted += 1
            return token

    def begin_run(self, token=None, now: float | None = None):
        """The request reached run() and is now counted as queued/active by the source: its reservation here is
        handed over (run() is often another thread than prepare(), so a missing token is normal)."""
        if not self.enabled or token is None:
            return
        now = now if now is not None else time.time()
        with self._lock:
            self._pending.pop(token, None)
            self._expire_pending(now)

    def cancel_pending(self, token=None):
        """The request ended before run() (rejected, disconnected): give its reservation back at once."""
        if token is None:
            return
        with self._lock:
            self._pending.pop(token, None)

    def _expire_pending(self, now: float):
        self._pending = {t: at for t, at in self._pending.items() if now - at <= PENDING_TTL_S}

    def _reject(self, reason, load, limit, queued, now):
        self.rejected[reason] += 1
        self._last_rejection = {"at": round(now, 1), "reason": reason, "in_flight": load, "limit": limit}
        text = {"capacity": f"the server is running {load} of {limit} requests it can hold right now",
                "memory": "host memory is nearly exhausted",
                "loop_protection": "this session's recent generation loop limits its resource allocation"}[reason]
        return f"{text}; retry in a few seconds (backpressure, not an error in your request)"

    def _drain_s(self, queued: int) -> float:
        """The measured estimate: how long `queued` requests take at the window's completion rate."""
        if queued <= 0:
            return 1.0
        rate = self.analyzer.stats().get("completion_per_s") or 0
        return float(min(60, max(1, round(queued / max(rate, 0.05)))))

    # ---------------------------------------------------------------- observations (feedback)
    def observe_start(self, prompt_tokens: int, max_new: int, context: int = 0):
        if not self.enabled:
            return frozenset()
        return self.analyzer.observe_start(prompt_tokens, max_new, context)

    def observe_finish(self, prompt_tokens=0, output_tokens=0, max_new=0, queue_s=0.0, ttft_s=None,
                       prompt_ms=None, decode_ms=None, pcie_share=None, hit_rate=None, finish="", now=None):
        if not self.enabled:
            return
        now = now if now is not None else time.time()
        o = Observation(prompt_tokens=prompt_tokens, output_tokens=output_tokens, max_new=max_new,
                        queue_s=max(0.0, queue_s or 0.0), ttft_s=ttft_s, prompt_ms=prompt_ms,
                        decode_ms=decode_ms, pcie_share=pcie_share, hit_rate=hit_rate, finish=finish, at=now)
        with self._lock:
            self.analyzer.observe_finish(o)
            self._tick(now, None)                          # no hardware snapshot here: the caller may hold
            #                              the status lock the source itself takes (no lock inversion)

    # ---------------------------------------------------------------- anti-loop integration
    def note_loop(self, session: str = "", confirmed: bool = False, now: float | None = None):
        """A session whose generation loop was SUSPECTED (or ended with finish_reason=generation_loop) stops
        getting more than half of the capacity for LOOP_TTL_S: a runaway generation may not consume the whole
        server, and the next request from that session queues behind healthy ones.  Recovery = the TTL lapsing
        with no further suspicion; the decoding layer (dynamic temperature) keeps its own separation."""
        if not self.enabled:
            return
        now = now if now is not None else time.time()
        with self._lock:
            if confirmed:
                self.loop_events += 1
            key = session or ""
            if not key:
                return
            fresh = self._suspected.get(key, 0) <= now     # a suspicious verdict every token: say it once
            self._suspected[key] = now + LOOP_TTL_S
            if fresh or confirmed:
                self.scheduler.note("loop_protection", "normal", "limited" if confirmed else "suspected",
                                    ("generation loop confirmed" if confirmed else "generation loop suspected")
                                    + f" for {int(LOOP_TTL_S)}s", now)

    def active_suspects(self, now=None):
        now = now if now is not None else time.time()
        with self._lock:
            self._suspected = {k: u for k, u in self._suspected.items() if u > now}
            return dict(self._suspected)

    # ---------------------------------------------------------------- observability
    def metrics(self) -> dict:
        snap = self.monitor.snapshot()
        with self._lock:
            stats = self.analyzer.stats()
            s = self.scheduler
            if snap.get("batch") is not None:
                s.set_engine(int(snap.get("batch") or 0))
            active, queued = int(snap.get("active") or 0), int(snap.get("queued") or 0)
            limit = s.concurrency + s.queue_limit
            now = time.time()
            self._expire_pending(now)
            self._suspected = {k: u for k, u in self._suspected.items() if u > now}
            return {
                "enabled": self.enabled,
                "power_policy": s.policy,
                "workload": {"dominant": self.analyzer.dominant(), "arrival_per_s": round(stats["arrival_per_s"], 3),
                             "completion_per_s": round(stats["completion_per_s"], 3),
                             "p95_ttft_s": round(stats["p95_ttft_s"], 3) if stats["p95_ttft_s"] else None,
                             "p95_queue_s": round(stats["p95_queue_s"], 3) if stats["p95_queue_s"] else None,
                             "sampled": stats["samples"]},
                "knobs": {"concurrency": s.concurrency, "max_concurrency": s.max_concurrency,
                          "queue_limit": s.queue_limit, "capacity": limit,
                          "in_flight": active + queued, "reserving": len(self._pending),
                          "cache_tier": s.cache_tier,
                          "saturation_hold_s": round(max(0.0, s._hold_until - time.time()), 1)},
                "pressure": {"memory": round(ResourceMonitor.pressure(snap), 3)
                             if ResourceMonitor.pressure(snap) is not None else None,
                             "cpu": snap.get("cpu"), "gpu_util": snap.get("gpu_util"),
                             "ram_used": snap.get("ram_used"), "ram_total": snap.get("ram_total"),
                             "vram_free_mib": snap.get("vram_free_mib")},
                "backpressure": {"admitted": self.admitted, "rejected": sum(self.rejected.values()),
                                 "by_reason": dict(self.rejected), "last": getattr(self, "_last_rejection", None)},
                "loop_protection": {"suspected_sessions": len(self._suspected),
                                    "loop_events": self.loop_events, "limited_admissions": self.loop_limited},
                "decisions": list(s.decisions)[-12:],
            }
