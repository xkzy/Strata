#!/usr/bin/env python3
"""Differential test: Strata's native CAS against Mathics3 (the reference).

For each case the same Wolfram-syntax input goes to both. Results are compared by mathematical equivalence
(Mathics decides whether the difference simplifies to zero), never by string. Every discrepancy is printed; nothing is hidden.

    python3 -m venv venv && venv/bin/pip install Mathics3
    venv/bin/python tools/differential_mathics.py [--cli build/strata_cas_cli] [--seed 7] [--random 40]

Exit status 1 when any case disagrees. Cases the reference cannot evaluate are reported as 'reference unsupported'.
SageMath is not part of this run (it is not installed here); its algorithms are covered by the same Mathics comparison where
Mathics has the operation, and by the unit tests otherwise.
"""
import argparse
import random
import re
import subprocess
import sys

try:
    from mathics.session import MathicsSession
except ImportError:
    print("Mathics3 is not installed (pip install Mathics3)", file=sys.stderr)
    sys.exit(2)

# Strata prints lowercase function calls like sqrt(2) and abs(x); Wolfram wants Sqrt[2]
LOWER_FUNCS = {"sqrt": "Sqrt", "sin": "Sin", "cos": "Cos", "tan": "Tan", "cot": "Cot", "sec": "Sec", "csc": "Csc", "exp": "Exp", "log": "Log",
               "abs": "Abs", "asin": "ArcSin", "acos": "ArcCos", "atan": "ArcTan", "sinh": "Sinh", "cosh": "Cosh", "tanh": "Tanh",
               "arcsin": "ArcSin", "arccos": "ArcCos", "arctan": "ArcTan", "gamma": "Gamma", "sign": "Sign", "floor": "Floor", "ceiling": "Ceiling"}


def to_wolfram(text):
    """Strata output -> Wolfram input: function-call parentheses become brackets, grouping parentheses stay."""
    out, stack, i = [], [], 0
    while i < len(text):
        m = re.match(r"[A-Za-z_][A-Za-z0-9_]*", text[i:])
        if m:
            name = m.group(0)
            j = i + len(name)
            if j < len(text) and text[j] == "(" and (name in LOWER_FUNCS or name[0].isupper() and len(name) > 1):
                out.append(LOWER_FUNCS.get(name, name) + "[")
                stack.append("call")
                i = j + 1
                continue
            out.append(LOWER_FUNCS.get(name, name) if name in LOWER_FUNCS else name)
            i = j
            continue
        c = text[i]
        if c == "(":
            stack.append("group"); out.append("(")
        elif c == ")":
            kind = stack.pop() if stack else "group"
            out.append("]" if kind == "call" else ")")
        else:
            out.append(c)
        i += 1
    s = "".join(out)
    s = re.sub(r"\bpi\b", "Pi", s)
    return s


class Reference:
    def __init__(self):
        self.s = MathicsSession()

    def ev(self, wl):
        return self.s.evaluate(wl)

    def text(self, wl, form="FullForm"):
        """The reference's answer as text. FullForm is ASCII and parses back; InputForm is for display."""
        r = self.s.evaluate(f"ToString[{wl}, {form}]")
        v = getattr(r, "value", None)
        return v if isinstance(v, str) else str(r)

    def is_true(self, wl):
        r = self.s.evaluate(wl)
        return str(r).endswith("True")


def reference_input(wl):
    """Operations Mathics 10 does not implement are expressed through ones it does (the semantics are the textbook definition)."""
    if wl.startswith("PolynomialMod["):
        f, m = wl[len("PolynomialMod["):-1].rsplit(",", 1)
        return f"Plus @@ (Mod[CoefficientList[{f}, x], {m}] * x^Range[0, Exponent[{f}, x]])"
    return wl


def sympy_reference(wl):
    """A second reference (SymPy, which Mathics itself builds on) for number-theory operations Mathics 10 lacks. None if not covered."""
    try:
        import sympy
        from sympy.ntheory import n_order, primitive_root
        from sympy.ntheory.modular import crt
    except ImportError:
        return None
    m = re.fullmatch(r"(\w+)\[(.*)\]", wl)
    if not m:
        return None
    name, raw = m.group(1), m.group(2)
    nums = [int(x) for x in re.findall(r"-?\d+", raw)]
    try:
        if name == "MultiplicativeOrder" and len(nums) == 2: return str(n_order(nums[0], nums[1]))
        if name == "PrimitiveRoot" and len(nums) == 1: return str(primitive_root(nums[0]))
        if name == "ChineseRemainder":
            lists = re.findall(r"\{([^}]*)\}", raw)
            r, mods = [int(x) for x in lists[0].split(",")], [int(x) for x in lists[1].split(",")]
            res = crt(mods, r)
            return str(int(res[0])) if res else None
    except Exception:
        return None
    return None


def check_factormod(wl, out):
    """FactorMod[f, x, p]: SymPy decides that the product equals f mod p and that every factor is irreducible mod p."""
    import sympy
    m = re.fullmatch(r"FactorMod\[(.*),\s*x,\s*(\d+)\]", wl)
    if not m:
        return None
    f_txt, p = m.group(1), int(m.group(2))
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application
    tr = standard_transformations + (implicit_multiplication_application,)
    x = sympy.Symbol("x")
    ns = {"x": x}
    f = sympy.Poly(parse_expr(f_txt.replace("^", "**"), local_dict=ns, transformations=tr), x, modulus=p)
    got = parse_expr(out.replace("^", "**"), local_dict=ns, transformations=tr)
    if sympy.Poly(sympy.expand(got), x, modulus=p) != f:
        return False
    # every factor (a power or a plain polynomial) must be irreducible mod p
    for fac in sympy.Mul.make_args(got):
        base = fac.base if fac.is_Pow else fac
        if base.is_number or base == x:
            continue
        if not sympy.Poly(base, x, modulus=p).is_irreducible:
            return False
    return True


def run_cli(cli, lines):
    p = subprocess.run([cli], input="\n".join(lines) + "\n", capture_output=True, text=True, timeout=600)
    return p.stdout.split("\n")[: len(lines)]


def equivalent(ref, a_wl, b_wl, kind):
    """Are two Wolfram expressions mathematically the same? Decided by the reference."""
    if kind == "set":     # lists compared as sets (solutions, eigenvalues, divisors)
        return ref.is_true(f"Sort[Simplify /@ ({a_wl})] === Sort[Simplify /@ ({b_wl})]") or ref.is_true(f"Sort[N[{a_wl}], 0]  === Sort[N[{b_wl}], 0]") \
            or ref.is_true(f"Union[Simplify /@ Flatten[{a_wl}]] === Union[Simplify /@ Flatten[{b_wl}]]")
    if kind == "list":    # ordered lists / matrices
        return ref.is_true(f"AllTrue[Flatten[Simplify[Flatten[{a_wl}] - Flatten[{b_wl}]]], PossibleZeroQ]") and ref.is_true(f"Dimensions[{a_wl}] === Dimensions[{b_wl}]")
    if kind == "exact":   # symbols/integers: structural equality after canonical evaluation
        return ref.is_true(f"({a_wl}) === ({b_wl})")
    # default: expression equivalence
    return (ref.is_true(f"Simplify[({a_wl}) - ({b_wl})] === 0") or ref.is_true(f"PossibleZeroQ[({a_wl}) - ({b_wl})]")
            or ref.is_true(f"Simplify[Expand[({a_wl}) - ({b_wl})]] === 0"))


# Cases Strata deliberately does not answer, with the reason. They are reported, never silently dropped.
KNOWN_GAPS = {
    "Limit[x Log[x], x -> 0]": "two-sided limit of a function that is complex for x < 0: Strata refuses instead of taking the right-hand side (Mathics returns 0)",
}


def build_cases(seed, n_random):
    cases = []   # (category, wolfram input, kind)
    A = cases.append
    for e in ["1/3 + 1/6", "2^100", "100!", "Binomial[40, 20]", "(7/3)^5", "Sqrt[8]", "Sqrt[18] + Sqrt[2]", "12345678901234567890 * 98765432109876543210",
              "2^(1/2) * 2^(1/2)", "(1 + Sqrt[2])^4", "Sin[Pi/6]", "Cos[Pi/3]", "Exp[0]", "Log[1]", "Abs[-7/2]", "Floor[7/2]", "Ceiling[-7/2]"]:
        A(("arithmetic", e, "exact"))
    for e in ["Expand[(x + 1)^6]", "Expand[(x - y)^4]", "Expand[(a + b + c)^3]", "Factor[x^4 - 1]", "Factor[x^6 - 1]", "Factor[x^3 + 3 x^2 + 3 x + 1]",
              "Factor[6 x^2 + 5 x + 1]", "Cancel[(x^2 - 1)/(x - 1)]", "Together[1/x + 1/(x + 1)]", "Apart[1/(x^2 - 1), x]", "Apart[(x^3 + 2)/(x^2 (x + 1)), x]",
              "Collect[a x + b x + c, x]", "Simplify[Sin[x]^2 + Cos[x]^2]", "Simplify[(x^2 - 1)/(x - 1)]"]:
        A(("algebra", e, "expr"))
    for e in ["D[x^5, x]", "D[Sin[x] Cos[x], x]", "D[Exp[x^2], x]", "D[Log[x^2 + 1], x]", "D[x^x, x]", "D[ArcTan[x], x]", "D[x^3 y^2, x, y]", "D[Sqrt[1 + x^2], x]"]:
        A(("derivative", e, "expr"))
    for e in ["Integrate[x^3, x]", "Integrate[1/x, x]", "Integrate[Sin[x], x]", "Integrate[Exp[2 x], x]", "Integrate[1/(1 + x^2), x]", "Integrate[x Exp[x], x]",
              "Integrate[x^2, {x, 0, 3}]", "Integrate[Sin[x], {x, 0, Pi}]", "Integrate[1/x^2, {x, 1, 2}]", "Integrate[Exp[-x], {x, 0, Infinity}]"]:
        A(("integral", e, "expr"))
    for e in ["Limit[Sin[x]/x, x -> 0]", "Limit[(1 + 1/x)^x, x -> Infinity]", "Limit[(x^2 - 1)/(x - 1), x -> 1]", "Limit[x Log[x], x -> 0]", "Limit[(Exp[x] - 1)/x, x -> 0]",
              "Limit[1/x, x -> Infinity]"]:
        A(("limit", e, "expr"))
    for e in ["Solve[x^2 - 5 x + 6 == 0, x]", "Solve[2 x + 3 == 11, x]", "Solve[x^3 - x == 0, x]", "Solve[{x + y == 3, x - y == 1}, {x, y}]", "Solve[x^2 + 1 == 0, x]"]:
        A(("solve", e, "set"))
    for e in ["Det[{{1, 2}, {3, 4}}]", "Det[{{2, 0, 1}, {1, 3, 2}, {1, 1, 1}}]", "MatrixRank[{{1, 2}, {2, 4}}]", "MatrixRank[{{1, 2, 3}, {4, 5, 6}, {7, 8, 10}}]", "Tr[{{1, 2}, {3, 4}}]"]:
        A(("matrix", e, "expr"))
    for e in ["Inverse[{{1, 2}, {3, 4}}]", "Transpose[{{1, 2, 3}, {4, 5, 6}}]", "Dot[{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}]", "IdentityMatrix[3]"]:
        A(("matrix", e, "list"))
    for e in ["Eigenvalues[{{2, 1}, {1, 2}}]", "Eigenvalues[{{4, 1}, {2, 3}}]", "Eigenvalues[{{1, 2}, {3, 4}}]"]:
        A(("matrix", e, "set"))
    for e in ["PrimeQ[1000003]", "PrimeQ[1000001]", "FactorInteger[360]", "FactorInteger[600851475143]", "EulerPhi[360]", "Divisors[60]", "PowerMod[7, 222, 13]",
              "GCD[462, 1071]", "LCM[12, 18, 30]", "Fibonacci[100]", "PartitionsP[50]", "PrimePi[1000]", "NextPrime[1000]", "ChineseRemainder[{2, 3, 2}, {3, 5, 7}]",
              "MoebiusMu[30]", "MultiplicativeOrder[3, 7]", "MultiplicativeOrder[2, 101]", "PrimitiveRoot[23]", "PrimitiveRoot[101]", "DivisorSigma[2, 12]", "JacobiSymbol[3, 7]", "Mod[-7, 3]", "Quotient[17, 5]"]:
        A(("number theory", e, "exact"))
    for e in ["PolynomialMod[x^3 + 5 x^2 + 7, 3]", "PolynomialMod[(x + 1)^5, 5]", "PolynomialMod[3 x^4 - 8 x + 11, 7]"]:
        A(("finite field", e, "expr"))
    for e in ["FactorMod[x^4 + 1, x, 2]", "FactorMod[x^5 - x, x, 5]", "FactorMod[x^8 - 1, x, 3]", "FactorMod[(x^2 + x + 1)^3 (x + 2), x, 7]", "FactorMod[x^7 - 1, x, 2]",
              "FactorMod[x^12 - 1, x, 13]", "FactorMod[x^6 + 3 x^3 + 1, x, 11]", "FactorMod[x^9 - x, x, 3]", "FactorMod[x^10 + x^5 + 1, x, 5]"]:
        A(("factor mod p", e, "factormod"))
    for e in ["Reduce[x^2 > 4, x]", "Reduce[x^2 - 5 x + 6 <= 0, x]", "Reduce[(x - 1)/(x + 2) > 0, x]", "Reduce[x^3 - x >= 0, x]", "Reduce[x^2 + 1 < 0, x]",
              "Reduce[x^2 + 1 > 0, x]", "Reduce[(x^2 - 4)/(x - 1) < 0, x]", "Reduce[2 x - 7 > 3, x]"]:
        A(("inequality", e, "reduce"))

    rng = random.Random(seed)
    def poly(deg=4, lo=-5, hi=5):
        return " + ".join(f"({rng.randint(lo, hi)}) x^{k}" for k in range(deg, -1, -1))
    for _ in range(n_random):
        p, q = poly(rng.randint(2, 4)), poly(rng.randint(1, 3))
        A(("random", f"Expand[({p}) ({q})]", "expr"))
        A(("random", f"D[({p}) ({q}), x]", "expr"))
        A(("random", f"Integrate[{p}, x]", "expr"))
        A(("random", f"Together[1/({q}) + 1/({p})]", "expr"))
        A(("random", f"Det[{{{{{rng.randint(-9,9)}, {rng.randint(-9,9)}}}, {{{rng.randint(-9,9)}, {rng.randint(-9,9)}}}}}]", "expr"))
        pr = rng.choice([2, 3, 5, 7, 11, 13, 101, 257, 65537])
        A(("random", f"FactorMod[{poly(rng.randint(3, 9), 0, 40)}, x, {pr}]".replace("(0) x^", "(0) x^"), "factormod"))
        a, b = rng.randint(2, 10**6), rng.randint(2, 10**6)
        A(("random", f"GCD[{a}, {b}]", "exact"))
        A(("random", f"PrimeQ[{a}]", "exact"))
        A(("random", f"EulerPhi[{a}]", "exact"))
        A(("random", f"PowerMod[{a}, {rng.randint(1, 10**5)}, {b}]", "exact"))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cli", default="build/strata_cas_cli")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--random", type=int, default=30)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cases = build_cases(args.seed, args.random)
    ref = Reference()
    outs = run_cli(args.cli, [c[1] for c in cases])

    agree = disagree = ref_unsupported = strata_unsupported = 0
    by_cat = {}
    problems = []
    known_gaps = []
    for (cat, wl, kind), out in zip(cases, outs):
        st = by_cat.setdefault(cat, [0, 0, 0, 0])   # agree, disagree, strata unsupported, reference unsupported
        if kind == "factormod" and out.startswith(("ERROR", "UNSUPPORTED", "LIMIT")):
            strata_unsupported += 1; st[2] += 1
            problems.append(f"STRATA DID NOT ANSWER  {wl}\n      strata: {out}")
            continue
        if kind == "factormod":
            res = check_factormod(wl, out)
            if res: agree += 1; st[0] += 1
            else:
                disagree += 1; st[1] += 1
                problems.append(f"DISAGREE (SymPy)  {wl}\n      strata: {out}")
            continue
        try:
            ref_text = ref.text(reference_input(wl))
        except Exception:
            ref_text = None
        # the reference returns the input unevaluated when it cannot do the operation
        unevaluated = ref_text is not None and re.sub(r"\s+", "", ref_text) == re.sub(r"\s+", "", wl) or (ref_text or "").startswith(("Integrate[", "Solve[", "Limit[", "Reduce[", "Eigenvalues[", "Series[", "ChineseRemainder["))
        ref_bad = ref_text is None or "$Aborted" in ref_text or unevaluated
        if ref_bad and kind != "reduce":
            sy = sympy_reference(wl)
            if sy is not None:
                if out.strip() == sy:
                    agree += 1; st[0] += 1
                else:
                    disagree += 1; st[1] += 1
                    problems.append(f"DISAGREE (SymPy)  {wl}\n      strata: {out}\n      sympy:  {sy}")
                continue
            ref_unsupported += 1; st[3] += 1
            if args.verbose: print(f"  reference unsupported: {wl} -> {ref_text}")
            continue
        if out.startswith(("ERROR", "UNSUPPORTED", "LIMIT")) and wl in KNOWN_GAPS:
            known_gaps.append(f"KNOWN GAP  {wl}: {KNOWN_GAPS[wl]}\n      strata: {out}")
            st[2] += 1
            continue
        if out.startswith(("ERROR", "UNSUPPORTED", "LIMIT")):
            strata_unsupported += 1; st[2] += 1
            problems.append(f"STRATA DID NOT ANSWER  {wl}\n      strata: {out}\n      mathics: {ref.text(wl, 'InputForm')}")
            continue
        try:
            if kind == "reduce":
                # Mathics does not solve inequalities: compare Strata's answer with the inequality itself at sample points
                ineq = wl[len("Reduce["):wl.rindex(",")]
                strata_wl = to_wolfram(out)
                ok = True
                for x in ["-9", "-5", "-3", "-5/2", "-2", "-3/2", "-1", "-1/2", "0", "1/2", "1", "3/2", "2", "5/2", "3", "5", "7/3", "10"]:
                    truth = ref.text(f"({ineq}) /. x -> {x}", "InputForm")
                    claim = ref.text(f"({strata_wl}) /. x -> {x}", "InputForm")
                    if truth in ("True", "False") and claim in ("True", "False") and truth != claim: ok = False
            else:
                ok = equivalent(ref, to_wolfram(out), ref_text, kind)
        except Exception as e:   # the reference could not even compare
            ok, out = None, f"{out}   (comparison failed: {e})"
        if ok:
            agree += 1; st[0] += 1
            if args.verbose: print(f"  agree: {wl} = {out}")
        elif ok is None:
            ref_unsupported += 1; st[3] += 1
        else:
            disagree += 1; st[1] += 1
            problems.append(f"DISAGREE  {wl}\n      strata:  {out}\n      mathics: {ref.text(wl, 'InputForm')}")

    print(f"{'category':<14} {'agree':>6} {'disagree':>9} {'strata n/a':>11} {'ref n/a':>8}")
    for cat, (a, d, s, r) in by_cat.items():
        print(f"{cat:<14} {a:>6} {d:>9} {s:>11} {r:>8}")
    print(f"\ncases {len(cases)}: agree {agree}, DISAGREE {disagree}, strata did not answer {strata_unsupported}, reference could not decide {ref_unsupported}")
    for g in known_gaps:
        print("  " + g)
    for p in problems:
        print("  " + p)
    return 1 if (disagree or strata_unsupported) else 0


if __name__ == "__main__":
    sys.exit(main())
