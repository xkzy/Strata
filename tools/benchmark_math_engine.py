#!/usr/bin/env python3
"""
benchmark_math_engine.py: Comprehensive Comparative LLM Math, Physics & CAS Benchmark
Evaluates Strata LLM inference with Go-Native Unified CAS Engine ON vs OFF.
"""

import json
import re
import sys
import time
import urllib.request
import urllib.error

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"
MODEL_NAME = "qwen3.8-flash-next-coder-iq1_m"

# Comprehensive Benchmark Suite across 9 STEM domains (32 problems)
BENCHMARK_SUITE = [
    # Tier 1: Multi-Digit Large Integer Arithmetic
    {
        "category": "Multi-Digit Arithmetic",
        "prompt": "Calculate 123456789 * 987654321. Give only the exact final numerical answer.",
        "expected": "121932631112635269",
        "verify_fn": lambda text: "121932631112635269" in text.replace(",", "").replace(" ", ""),
    },
    {
        "category": "Multi-Digit Arithmetic",
        "prompt": "What is 987654321987654321 - 123456789123456789? Give only the exact final numerical answer.",
        "expected": "864197532864197532",
        "verify_fn": lambda text: "864197532864197532" in text.replace(",", "").replace(" ", ""),
    },
    {
        "category": "Multi-Digit Arithmetic",
        "prompt": "Compute 847291 * 392817. Give only the exact final numerical answer.",
        "expected": "332830308747",
        "verify_fn": lambda text: "332830308747" in text.replace(",", "").replace(" ", ""),
    },

    # Tier 2: Exact Rational & Fraction Arithmetic
    {
        "category": "Rational Fractions",
        "prompt": "Evaluate 1/3 + 1/6 + 1/12. Give only the final simplified fraction directly.",
        "expected": "7/12",
        "verify_fn": lambda text: "7/12" in text,
    },
    {
        "category": "Rational Fractions",
        "prompt": "Calculate (7/13) * (26/21). Give only the final simplified fraction directly.",
        "expected": "2/3",
        "verify_fn": lambda text: "2/3" in text,
    },
    {
        "category": "Rational Fractions",
        "prompt": "Simplify (5/8 - 1/4) / (3/16). Give only the final answer directly.",
        "expected": "2",
        "verify_fn": lambda text: "2" in text and ("= 2" in text or "is 2" in text or "**2**" in text or " 2\n" in text or " 2." in text or text.strip() == "2"),
    },

    # Tier 3: Linear Algebra & Matrices
    {
        "category": "Linear Algebra",
        "prompt": "Compute the determinant of [[17, 23], [41, 59]]. Give only the final numerical answer directly.",
        "expected": "60",
        "verify_fn": lambda text: "60" in text and ("= 60" in text or "is 60" in text or "**60**" in text or " 60\n" in text or " 60." in text or text.strip() == "60"),
    },
    {
        "category": "Linear Algebra",
        "prompt": "Find the determinant of [[2, 0, 1], [3, 0, 0], [5, 1, 1]]. Give only the final numerical answer directly.",
        "expected": "3",
        "verify_fn": lambda text: "3" in text and ("= 3" in text or "is 3" in text or "**3**" in text or " 3\n" in text or " 3." in text or text.strip() == "3"),
    },
    {
        "category": "Linear Algebra",
        "prompt": "Find the inverse of [[4, 7], [2, 6]]. Give only the final matrix directly.",
        "expected": "[[0.6, -0.7], [-0.2, 0.4]]",
        "verify_fn": lambda text: "0.6" in text and "-0.7" in text and "-0.2" in text and "0.4" in text,
    },
    {
        "category": "Linear Algebra",
        "prompt": "Compute the trace of [[15, 8], [4, 25]]. Give only the final numerical answer directly.",
        "expected": "40",
        "verify_fn": lambda text: "40" in text and ("= 40" in text or "is 40" in text or "**40**" in text or text.strip() == "40"),
    },

    # Tier 4: Symbolic Calculus
    {
        "category": "Symbolic Calculus",
        "prompt": "Differentiate 3*x^4 - 5*x^2 + 7 with respect to x. Give only the derivative.",
        "expected": "12*x^3 - 10*x",
        "verify_fn": lambda text: "12" in text and "x^3" in text and "10" in text,
    },
    {
        "category": "Symbolic Calculus",
        "prompt": "Differentiate sin(4*x) with respect to x. Give only the derivative.",
        "expected": "4*cos(4*x)",
        "verify_fn": lambda text: "4*cos(4*x)" in text or "4cos(4x)" in text or "4 * cos(4 * x)" in text,
    },
    {
        "category": "Symbolic Calculus",
        "prompt": "What is the limit of sin(x)/x as x -> 0? Give only the final numerical answer directly.",
        "expected": "1",
        "verify_fn": lambda text: "1" in text and ("= 1" in text or "is 1" in text or "**1**" in text or text.strip() == "1"),
    },

    # Tier 5: Statistics & Discrete Math
    {
        "category": "Statistics & Discrete",
        "prompt": "What is the mean of [12, 18, 24, 30, 36]? Give only the numerical answer.",
        "expected": "24",
        "verify_fn": lambda text: "24" in text and ("= 24" in text or "is 24" in text or "**24**" in text or text.strip() == "24"),
    },
    {
        "category": "Statistics & Discrete",
        "prompt": "What is the modular inverse of 7 mod 26? Give only the integer answer.",
        "expected": "15",
        "verify_fn": lambda text: "15" in text and ("= 15" in text or "is 15" in text or "**15**" in text or text.strip() == "15"),
    },
    {
        "category": "Statistics & Discrete",
        "prompt": "What is the combination C(12, 5)? Give only the final integer answer directly.",
        "expected": "792",
        "verify_fn": lambda text: "792" in text,
    },
    {
        "category": "Statistics & Discrete",
        "prompt": "What is the Greatest Common Divisor GCD(4620, 3696)? Give only the final integer answer directly.",
        "expected": "924",
        "verify_fn": lambda text: "924" in text,
    },

    # Tier 6: Classical Mechanics
    {
        "category": "Mechanics",
        "prompt": "What is the net force on 45 kg accelerating at 6.2 m/s^2? Give only the final answer directly.",
        "expected": "279 N",
        "verify_fn": lambda text: "279" in text,
    },
    {
        "category": "Mechanics",
        "prompt": "Compute the kinetic energy of an 8 kg mass moving at 15 m/s. Give only the final answer directly.",
        "expected": "900 J",
        "verify_fn": lambda text: "900" in text,
    },

    # Tier 7: Circuits & Electromagnetism
    {
        "category": "Circuits & EM",
        "prompt": "What is the voltage across 220 ohm with 4.5 A current? Give only the final answer directly.",
        "expected": "990 V",
        "verify_fn": lambda text: "990" in text,
    },
    {
        "category": "Circuits & EM",
        "prompt": "What is the time constant of 47 kohm and 10 uF? Give only the final answer directly.",
        "expected": "0.47 s",
        "verify_fn": lambda text: "0.47" in text or "470 ms" in text,
    },
    {
        "category": "Circuits & EM",
        "prompt": "What is the resonant frequency of 10 mH inductor and 100 nF capacitor? Give only the final answer directly.",
        "expected": "5032.9 Hz",
        "verify_fn": lambda text: "5032" in text or "5033" in text or "5.03" in text or "5032.9" in text,
    },
    {
        "category": "Circuits & EM",
        "prompt": "What is the Coulomb force between 2e-6 C and 3e-6 C at distance 0.5 m? Give only the final answer directly.",
        "expected": "0.2157 N",
        "verify_fn": lambda text: "0.215" in text or "0.216" in text or "0.22" in text,
    },

    # Tier 8: Thermodynamics
    {
        "category": "Thermodynamics",
        "prompt": "What is the Carnot efficiency between 300 K and 600 K? Give only the final answer directly.",
        "expected": "0.5 (50%)",
        "verify_fn": lambda text: "0.5" in text or "50%" in text or "50 %" in text,
    },
    {
        "category": "Thermodynamics",
        "prompt": "For an ideal gas with 2.0 mol at 300 K in 0.05 m^3, what is the pressure in Pa? Give only the numerical answer.",
        "expected": "99773.55 Pa",
        "verify_fn": lambda text: "99773" in text or "99774" in text or "9.98e4" in text or "99.77" in text or "99770" in text,
    },

    # Tier 9: Modern, Quantum & Optics
    {
        "category": "Modern Physics",
        "prompt": "Using E = m*c^2 with mass = 2.5 kg, what is the rest energy in Joules? Give only the final answer directly.",
        "expected": "2.2468879e17 J",
        "verify_fn": lambda text: "2.2468" in text or "2.247" in text or "224688794" in text or "2.25" in text,
    },
    {
        "category": "Quantum Physics",
        "prompt": "What is the photon energy E = h * f for light with frequency f = 6.0e14 Hz? Give only the final answer directly.",
        "expected": "3.975642e-19 J",
        "verify_fn": lambda text: "3.9756" in text or "3.976" in text or "3.98" in text or "3.97" in text,
    },
    {
        "category": "Optics",
        "prompt": "Using Snell's law with n1 = 1.0, theta1 = 30 deg, n2 = 1.5, what is the refraction angle? Give only the final answer in degrees.",
        "expected": "19.47 deg",
        "verify_fn": lambda text: "19.47" in text or "19.5" in text or "19.48" in text,
    },
]


def query_llm(prompt: str, math_engine: bool) -> tuple[str, float, int]:
    """Sends a completion request to Strata LLM with math engine ON or OFF."""
    headers = {
        "Content-Type": "application/json",
        "X-Strata-Math-Engine": "on" if math_engine else "off",
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a precise scientific calculator. Answer questions with the final value directly and concisely."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 128,
        "enable_thinking": False,
        "math_engine": math_engine,
    }

    req = urllib.request.Request(
        SERVER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST"
    )

    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - t0
            content = data["choices"][0]["message"].get("content") or ""
            reasoning = data["choices"][0]["message"].get("reasoning_content") or ""
            full_text = f"{reasoning}\n\n{content}" if reasoning else content
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            return full_text, elapsed, tokens
    except Exception as e:
        print(f"Error querying LLM (math_engine={math_engine}): {e}", flush=True)
        return f"ERROR: {e}", 0.0, 0


def main():
    print("=" * 80, flush=True)
    print(" STRATA LLM BENCHMARK: UNIFIED CAS MATH & PHYSICS ENGINE (ON vs OFF)", flush=True)
    print(f" Target Endpoint: {SERVER_URL}", flush=True)
    print(f" Model: {MODEL_NAME}", flush=True)
    print(f" Total Benchmark Problems: {len(BENCHMARK_SUITE)}", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)

    results = []
    correct_on = 0
    correct_off = 0
    total_time_on = 0.0
    total_time_off = 0.0

    for idx, item in enumerate(BENCHMARK_SUITE, start=1):
        cat = item["category"]
        prompt = item["prompt"]
        expected = item["expected"]
        verify = item["verify_fn"]

        print(f"[{idx:02d}/{len(BENCHMARK_SUITE):02d}] Category: {cat}", flush=True)
        print(f"     Prompt:   {prompt}", flush=True)
        print(f"     Expected: {expected}", flush=True)

        # 1. Query with Math Engine ON
        out_on, time_on, tok_on = query_llm(prompt, math_engine=True)
        is_pass_on = verify(out_on)
        if is_pass_on:
            correct_on += 1
        total_time_on += time_on

        # 2. Query with Math Engine OFF
        out_off, time_off, tok_off = query_llm(prompt, math_engine=False)
        is_pass_off = verify(out_off)
        if is_pass_off:
            correct_off += 1
        total_time_off += time_off

        # Extract last lines of answers for display
        ans_on = [line.strip() for line in out_on.strip().split("\n") if line.strip()][-1][:70] if out_on.strip() else ""
        ans_off = [line.strip() for line in out_off.strip().split("\n") if line.strip()][-1][:70] if out_off.strip() else ""

        status_on = " PASS" if is_pass_on else " FAIL"
        status_off = " PASS" if is_pass_off else " FAIL"

        print(f"     -> [Math Engine  ON]: {status_on} ({time_on:.2f}s, {tok_on} tok) | Answer: {ans_on}", flush=True)
        print(f"     -> [Math Engine OFF]: {status_off} ({time_off:.2f}s, {tok_off} tok) | Answer: {ans_off}", flush=True)
        print("-" * 80, flush=True)

        results.append({
            "id": idx,
            "category": cat,
            "prompt": prompt,
            "expected": expected,
            "pass_on": is_pass_on,
            "pass_off": is_pass_off,
            "time_on": time_on,
            "time_off": time_off,
            "ans_on": ans_on,
            "ans_off": ans_off,
        })

    # Summary Report
    acc_on = (correct_on / len(BENCHMARK_SUITE)) * 100.0
    acc_off = (correct_off / len(BENCHMARK_SUITE)) * 100.0

    print(flush=True)
    print("=" * 80, flush=True)
    print(" BENCHMARK SUMMARY REPORT", flush=True)
    print("=" * 80, flush=True)
    print(f" Total Problems Tested:    {len(BENCHMARK_SUITE)}", flush=True)
    print(f" Math Engine  ON Accuracy: {correct_on}/{len(BENCHMARK_SUITE)} ({acc_on:.1f}%) | Total Time: {total_time_on:.2f}s", flush=True)
    print(f" Math Engine OFF Accuracy: {correct_off}/{len(BENCHMARK_SUITE)} ({acc_off:.1f}%) | Total Time: {total_time_off:.2f}s", flush=True)
    print(f" Accuracy Delta:           +{acc_on - acc_off:.1f}% improvement with Go Unified CAS Engine", flush=True)
    print("=" * 80, flush=True)

    # Category breakdown table
    cat_stats = {}
    for r in results:
        c = r["category"]
        if c not in cat_stats:
            cat_stats[c] = {"total": 0, "pass_on": 0, "pass_off": 0}
        cat_stats[c]["total"] += 1
        if r["pass_on"]:
            cat_stats[c]["pass_on"] += 1
        if r["pass_off"]:
            cat_stats[c]["pass_off"] += 1

    print("\nCategory Breakdown:", flush=True)
    print(f"{'Category':<25} | {'Engine ON':<12} | {'Engine OFF':<12} | {'Delta'}", flush=True)
    print("-" * 65, flush=True)
    for c, s in cat_stats.items():
        pct_on = (s["pass_on"] / s["total"]) * 100
        pct_off = (s["pass_off"] / s["total"]) * 100
        print(f"{c:<25} | {s['pass_on']}/{s['total']} ({pct_on:4.0f}%)   | {s['pass_off']}/{s['total']} ({pct_off:4.0f}%)   | +{pct_on - pct_off:4.0f}%", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
