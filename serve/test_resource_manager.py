"""serve/test_resource_manager.py - #18 Auto-Adaptive Resource Utilization: the workload classes are inferred
(never asked for), the feedback controller grows and shrinks the knobs with hysteresis and saturation, admission
backpressures with a structured 429 instead of queueing forever, the cache tier follows measured memory pressure,
a suspected generation loop limits its session, and /metrics carries the "resource" section - all against the
mock engine, so no GPU is needed.

    python -m unittest serve.test_resource_manager
"""
from __future__ import annotations

import json
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.resource_manager import (
    DECODE_HEAVY, ENERGY_SAVING, HIGH_THROUGHPUT, LONG_CONTEXT, LOW_LATENCY, MOE_DENSE, MOE_SPARSE,
    Observation, Overloaded, PREFILL_HEAVY, ResourceManager, SHORT_INTERACTIVE, TRANSFER_BOUND,
    classify, measured_classes,
)
from serve.server import ByteTokenizer, MockEngine, Service, serve

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ChatTemplate(ROOT / "serve/chat_template.jinja")


def mgr(source=None, **kw):
    """A controller under test: always evaluates (min_interval 0), a source the test controls."""
    src = source if source is not None else (lambda: {"queued": 0, "active": 0, "batch": 0})
    kw.setdefault("min_interval", 0)
    return ResourceManager(source=src, **kw)


def tick(m, now, snap, n=1):
    """n evaluations at `now` (force: the tests must not wait out the rate limit)."""
    for i in range(n):
        m._tick(now + i * 0.01, snap, force=True)


class Classify(unittest.TestCase):
    """The classes come from the request itself and from what the run measured - never from the user."""

    def test_short_interactive_and_low_latency(self):
        c = classify(300, 50, 4096)
        self.assertIn(SHORT_INTERACTIVE, c)
        self.assertIn(LOW_LATENCY, c)

    def test_long_context_and_prefill_heavy(self):
        c = classify(16000, 100, 32768)
        self.assertIn(LONG_CONTEXT, c)
        self.assertIn(PREFILL_HEAVY, c)

    def test_decode_heavy_and_high_throughput_from_the_asked_length(self):
        c = classify(100, 5000, 32768)
        self.assertIn(DECODE_HEAVY, c)
        self.assertIn(HIGH_THROUGHPUT, c)

    def test_the_run_proves_prefill_or_decode(self):
        pre = measured_classes(Observation(prompt_tokens=300, max_new=50, prompt_ms=900, decode_ms=100,
                                           classes=classify(300, 50)))
        self.assertIn(PREFILL_HEAVY, pre)
        self.assertIn(SHORT_INTERACTIVE, pre)               # the pre-run guess stays
        dec = measured_classes(Observation(prompt_tokens=300, max_new=50, prompt_ms=100, decode_ms=900,
                                           classes=classify(300, 50)))
        self.assertIn(DECODE_HEAVY, dec)

    def test_transfer_and_expert_classes_from_engine_counters(self):
        self.assertIn(TRANSFER_BOUND, measured_classes(Observation(pcie_share=0.4)))
        self.assertIn(MOE_DENSE, measured_classes(Observation(hit_rate=0.95)))
        self.assertIn(MOE_SPARSE, measured_classes(Observation(hit_rate=0.3)))
        self.assertEqual(measured_classes(Observation(hit_rate=0.75)) , frozenset())   # in between: no claim

    def test_p95_of_the_window(self):
        m = mgr()
        base = time.time()
        for i in range(10):
            m.observe_finish(prompt_tokens=10, output_tokens=10, ttft_s=float(i), now=base - 5 + i * 0.1)
        stats = m.analyzer.stats()
        self.assertEqual(stats["samples"], 10)
        self.assertGreaterEqual(stats["p95_ttft_s"], 8.5)     # the tail, not the mean
        self.assertGreater(stats["tok_s"], 0)


class FeedbackLoop(unittest.TestCase):
    """Grow when there is room and latency is flat, shrink when the workload's own baseline degrades, and never
    oscillate: a direction must repeat, changes sit behind a cooldown, and a grow without throughput is held."""

    def test_grows_the_queue_when_latency_is_flat(self):
        m = mgr()
        snap = {"queued": 29, "active": 1, "batch": 2, "cpu": 10}
        t0 = time.time()
        tick(m, t0, snap, n=2)
        self.assertEqual(m.scheduler.queue_limit, 33)          # 32 -> 33, room for the burst
        self.assertEqual(m.scheduler.concurrency, 2)           # learned the engine's 2 slots, capped by them
        d = [x for x in m.scheduler.decisions if x["knob"] == "queue_limit"]
        self.assertTrue(d and "queue at 29" in d[-1]["reason"], d)

    def test_shrinks_after_the_ttft_baseline_doubles(self):
        m = mgr()
        base = time.time()
        for i in range(4):                                     # a quiet minute: TTFT ~1.0 s
            m.observe_finish(prompt_tokens=100, output_tokens=10, ttft_s=1.0, now=base - 10 + i)
        snap = {"queued": 0, "active": 1, "batch": 2}
        m._tick(base, snap, force=True)
        self.assertAlmostEqual(m.estimator.baseline_ttft, 1.0, places=3)
        for i in range(4):                                     # now it degrades to 3.0 s
            m.observe_finish(prompt_tokens=100, output_tokens=10, ttft_s=3.0, now=base + 1 + i)
        tick(m, base + 2, snap, n=2)                           # the direction must repeat, then it acts
        self.assertLess(m.scheduler.queue_limit, 32)
        self.assertAlmostEqual(m.estimator.baseline_ttft, 1.0, places=3)   # the baseline must not drift up
        reasons = [x["reason"] for x in m.scheduler.decisions if x["knob"] == "queue_limit"]
        self.assertTrue(any("TTFT" in r for r in reasons), reasons)

    def test_no_reversal_inside_the_cooldown(self):
        m = mgr(min_interval=5.0)                              # shrink cooldown 5 s
        base = time.time()
        for i in range(4):                                     # a quiet minute: TTFT ~1.0 s
            m.observe_finish(prompt_tokens=10, output_tokens=10, ttft_s=1.0, now=base - 6 + i)
        bad = {"queued": 0, "active": 1, "batch": 0}
        m._tick(base, bad, force=True)                         # baseline 1.0
        for i in range(4):                                     # it degrades to 5.0 s
            m.observe_finish(prompt_tokens=10, output_tokens=10, ttft_s=5.0, now=base - 1 + i)
        m._tick(base + 4, bad, force=True)
        m._tick(base + 5, bad, force=True)                     # the direction held: shrink happens
        self.assertLess(m.scheduler.queue_limit, 32)
        after = m.scheduler.queue_limit
        m.analyzer._obs.clear()                                # the burst passed: latency is flat again
        good = {"queued": 29, "active": 0, "batch": 0}         # queue pressure says grow again...
        m._tick(base + 6, good, force=True)
        m._tick(base + 7, good, force=True)
        self.assertEqual(m.scheduler.queue_limit, after)       # ...but not inside the cooldown
        m._tick(base + 11, good, force=True)                   # 6 s after the shrink: growth again
        self.assertGreater(m.scheduler.queue_limit, after)

    def test_a_grow_without_throughput_is_held(self):
        m = mgr()
        base = time.time()
        for _ in range(2):                                     # ~120 completed tokens in the window
            m.observe_finish(prompt_tokens=10, output_tokens=60, now=base - 0.4)
        snap = {"queued": 29, "active": 1, "batch": 4, "cpu": 5}
        tick(m, base, snap, n=2)                               # grows 1 -> 2 with the throughput recorded
        self.assertEqual(m.scheduler.concurrency, 2)
        tick(m, base + 2, snap, n=2)                           # would grow again, but the work did not speed up
        self.assertEqual(m.scheduler.concurrency, 2)
        self.assertGreater(m.scheduler._hold_until, base + 4)
        flat = [x for x in m.scheduler.decisions if "stop growing" in x["reason"]]
        self.assertTrue(flat, list(m.scheduler.decisions))


class Admission(unittest.TestCase):
    """A structured 429 with a measured retry_after - never unbounded queueing, never OOM."""

    def test_capacity_is_a_structured_429(self):
        m = mgr(source=lambda: {"queued": 40, "active": 0, "batch": 0})
        with self.assertRaises(Overloaded) as cm:
            m.admit("s")
        e = cm.exception
        self.assertEqual((e.status, e.reason), (429, "capacity"))
        self.assertGreaterEqual(e.retry_after, 1)
        self.assertIn("retry", str(e))
        b = m.metrics()["backpressure"]
        self.assertEqual(b["by_reason"]["capacity"], 1)
        self.assertEqual(b["last"]["reason"], "capacity")

    def test_memory_pressure_keeps_a_floor(self):
        m = mgr(source=lambda: {"queued": 1, "active": 0, "batch": 0, "ram_used": 95, "ram_total": 100})
        m.admit("s", now=time.time())                          # alone: admitted (the floor never blocks the last)
        m2 = mgr(source=lambda: {"queued": 2, "active": 0, "batch": 0, "ram_used": 95, "ram_total": 100})
        with self.assertRaises(Overloaded) as cm:
            m2.admit("s", now=time.time())
        self.assertEqual(cm.exception.reason, "memory")

    def test_pending_requests_hold_capacity_until_run_or_cancel(self):
        m = mgr(source=lambda: {"batch": 0})                   # no queue counters: only the reservations count
        tokens = []
        with self.assertRaises(Overloaded):
            while True:
                tokens.append(m.admit("s"))
        self.assertEqual(len(tokens), 33)                       # concurrency 1 + queue_limit 32
        m.begin_run(tokens[0])                                  # one reached run(): its room is the queue's now
        m.admit("s")
        m.cancel_pending(tokens[1])                             # one ended before run(): room given back
        m.admit("s")

    def test_a_suspected_loop_limits_only_that_session(self):
        m = mgr(source=lambda: {"queued": 20, "active": 0, "batch": 0})
        t0 = time.time()
        m.note_loop("agent-a", confirmed=True, now=t0)
        with self.assertRaises(Overloaded) as cm:
            m.admit("agent-a", now=t0 + 1)
        self.assertEqual(cm.exception.reason, "loop_protection")
        m.admit("healthy", now=t0 + 1)                          # the rest of the server keeps its capacity
        m.admit("agent-a", now=t0 + 120)                        # suspicion lapsed: normal again
        lp = m.metrics()["loop_protection"]
        self.assertEqual(lp["loop_events"], 1)
        self.assertEqual(lp["limited_admissions"], 1)
        self.assertEqual(lp["suspected_sessions"], 1)

    def test_disabled_is_really_off(self):
        m = mgr(source=lambda: {"queued": 99, "active": 9})
        m.set_enabled(False)
        self.assertEqual(m.admit("s"), 0)
        self.assertFalse(m.metrics()["enabled"])


class CacheTier(unittest.TestCase):
    """The cache-pressure tier follows measured memory: tight when it is tight, back to retain when it is not."""

    def test_tier_follows_pressure_both_ways(self):
        state = {"queued": 0, "active": 1, "batch": 0, "ram_used": 90, "ram_total": 100}
        m = mgr(source=lambda: dict(state))
        tick(m, time.time(), state, n=2)
        self.assertEqual(m.scheduler.cache_tier, "tight")
        reasons = [x["reason"] for x in m.scheduler.decisions if x["knob"] == "cache_tier"]
        self.assertTrue(any("90%" in r for r in reasons), reasons)
        state["ram_used"] = 50
        tick(m, time.time() + 1, state, n=4)                    # pressure gone: gentle tier again
        self.assertEqual(m.scheduler.cache_tier, "retain")

    def test_a_tight_tier_halves_admission(self):
        state = {"queued": 20, "active": 0, "batch": 0, "ram_used": 90, "ram_total": 100}
        m = mgr(source=lambda: dict(state))
        tick(m, time.time(), state, n=4)                        # tight (and maybe aggressive) under pressure
        with self.assertRaises(Overloaded) as cm:
            m.admit("s")
        self.assertEqual(cm.exception.reason, "capacity")
        self.assertEqual(cm.exception.retry_after >= 1, True)
        self.assertEqual(m.metrics()["backpressure"]["last"]["limit"], 16)   # (1 + 32) // 2


class PowerPolicy(unittest.TestCase):
    def test_bounds_and_validation(self):
        m = mgr()
        m.scheduler.queue_limit = 100
        m.set_power_policy("LOW_LATENCY")
        self.assertEqual(m.scheduler.queue_limit, 32)           # that policy's own ceiling
        self.assertEqual(m.scheduler.policy, "LOW_LATENCY")
        with self.assertRaises(ValueError):
            m.set_power_policy("TURBO")

    def test_energy_saving_restores_room_when_idle(self):
        m = mgr(policy=ENERGY_SAVING)
        snap = {"queued": 0, "active": 0, "batch": 0}
        tick(m, time.time(), snap, n=6)
        reasons = [x["reason"] for x in m.scheduler.decisions if x["knob"] == "queue_limit"]
        self.assertTrue(any("idle" in r for r in reasons), list(m.scheduler.decisions))
        self.assertGreater(m.scheduler.queue_limit, 4)


class Observability(unittest.TestCase):
    def test_metrics_answers_why(self):
        m = mgr(source=lambda: {"queued": 0, "active": 0, "batch": 2})
        m.observe_start(300, 50, 4096)
        m.observe_finish(prompt_tokens=300, output_tokens=50, prompt_ms=200, decode_ms=400,
                         hit_rate=0.95, finish="stop")
        tick(m, time.time(), {"queued": 29, "active": 1, "batch": 2, "cpu": 70}, n=2)
        d = m.metrics()
        json.dumps(d)                                           # /metrics must serialize as it is
        for key in ("enabled", "power_policy", "workload", "knobs", "pressure", "backpressure",
                    "loop_protection", "decisions"):
            self.assertIn(key, d)
        self.assertTrue(d["workload"]["dominant"], "the window must say what the workload is")
        self.assertEqual(d["knobs"]["max_concurrency"], 2)      # from the engine's reported slots
        self.assertTrue(d["decisions"], "a knob changed: it must be on record")
        for dec in d["decisions"]:
            self.assertTrue(dec.get("reason"), dec)             # every decision explains itself


class ServiceLevel(unittest.TestCase):
    """The wiring: prepare() gates, run() feeds back, /metrics carries the section."""

    def setUp(self):
        self.tok = ByteTokenizer()
        self.svc = Service(MockEngine(self.tok, "ok", max_context=4096), self.tok, TEMPLATE)

    def test_a_request_feeds_the_controller(self):
        ids, thinking, max_new = self.svc.prepare(
            [{"role": "user", "content": "hello there"}], None, {})
        list(self.svc.run(ids, thinking, None, max_new, {}, threading.Event()))
        m = self.svc.metrics()["resource"]
        self.assertGreaterEqual(m["workload"]["sampled"], 1)
        self.assertGreaterEqual(m["backpressure"]["admitted"], 1)
        self.assertTrue(m["workload"]["dominant"])

    def test_prepare_backpressures_when_the_queue_is_full(self):
        self.svc.resources.scheduler.queue_limit = 0           # capacity: 1 + 0
        self.svc.resources._pending[1] = time.time()           # one request is already spoken for
        try:
            with self.assertRaises(Overloaded) as cm:
                self.svc.prepare([{"role": "user", "content": "hi"}], None, {})
            self.assertEqual(cm.exception.reason, "capacity")
        finally:
            self.svc.resources._pending.clear()
            self.svc.resources.scheduler.queue_limit = 32

    def test_a_suspected_session_is_gated_at_prepare(self):
        self.svc.resources.scheduler.queue_limit = 4           # capacity 5, half = 2
        self.svc.resources._pending[1] = time.time()
        self.svc.resources._pending[2] = time.time()
        scope_session = "default_session"                      # SecurityScope()'s own session id
        self.svc.resources.note_loop(scope_session, confirmed=True)
        try:
            with self.assertRaises(Overloaded) as cm:
                self.svc.prepare([{"role": "user", "content": "hi"}], None, {})
            self.assertEqual(cm.exception.reason, "loop_protection")
            self.svc.resources._suspected[scope_session] = time.time() - 1   # suspicion lapsed
            self.svc.request_trace.admitted = False
            self.svc.prepare([{"role": "user", "content": "hi"}], None, {})  # normal again
        finally:
            self.svc.resources._pending.clear()
            self.svc.resources.scheduler.queue_limit = 32


class HttpBackpressure(unittest.TestCase):
    """Over the wire: 429 with a structured body and a retry_after, and the resource section on /metrics."""

    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "ok", max_context=4096), tok, TEMPLATE)
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def chat(self):
        body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 16}
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())

    def test_a_full_queue_is_a_structured_429(self):
        self.assertIn("choices", self.chat())                   # room: it answers
        self.svc.resources.scheduler.queue_limit = 0           # capacity 1, and one is spoken for
        self.svc.resources._pending[7] = time.time()
        try:
            with self.assertRaises(urllib.error.HTTPError) as cm:
                self.chat()
            self.assertEqual(cm.exception.code, 429)
            err = json.loads(cm.exception.read().decode())["error"]
            self.assertEqual(err["code"], "server_overloaded")
            self.assertGreaterEqual(err["retry_after"], 1)
            self.assertTrue(err["message"])
            with urllib.request.urlopen(self.base + "/metrics", timeout=10) as r:
                res = json.loads(r.read().decode())["resource"]
            self.assertEqual(res["backpressure"]["by_reason"]["capacity"], 1)
            self.assertEqual(res["backpressure"]["admitted"], 1)   # the first request went through
            self.assertEqual(res["knobs"]["queue_limit"], 0)
            self.assertIsInstance(res["decisions"], list)
        finally:
            self.svc.resources._pending.clear()
            self.svc.resources.scheduler.queue_limit = 32


if __name__ == "__main__":
    unittest.main()
