# Mathematical Backing Engine & Mathics Runtime

Strata integrates **Mathics3 `mathics-core`** (and native `FastNumeric` rational arithmetic) as an authoritative, deterministic mathematical calculation backend.

## Core Principle

Do NOT rely on the LLM itself to perform calculations when the calculation can be evaluated deterministically by a mathematical CAS.

```
User
  ↓
LLM understands problem
  ↓
LLM constructs mathematical expression
  ↓
Math Runtime
  ↓
Mathics-Core / FastNumeric
  ↓
verified result
  ↓
LLM explains result
```

The LLM remains responsible for:
* Understanding natural language
* Identifying variables and formulating problems
* Deciding what calculation is required
* Explaining the verified result

The Math Runtime provides:
* Exact arithmetic by default (e.g. `1/3 + 1/6` -> `1/2` rather than `0.5`)
* High-speed native fast path for rational arithmetic, large integer calculations, factorials, and 2x2 determinants
* Symbolic algebra, differentiation, integration, equation solving, limits, and series via Mathics3 / SymPy
* Non-bloating virtual context integration (storing full calculations externally while injecting compact bounded observations)
* Verification loop detecting LLM arithmetic hallucinations
* Calculation-aware generation loop recovery

## Architecture

```
                    Strata Server
                          │
          ┌───────────────┼────────────────┐
          │               │                │
      LLM Runtime    Context Runtime   Tool Runtime
          │               │                │
          └───────────────┼────────────────┘
                          │
                    Math Runtime
                          │
          ┌───────────────┼────────────────┬───────────────┐
          │               │                │               │
     Fast Numeric    Mathics-Core       SageMath     Lean 4 kernel
       Backend          Backend          Backend      (Go port)
          │               │                │               │
          └───────────────┼────────────────┴───────────────┘
                          │
                    Math Result Store
```

## API Endpoints

* `POST /v1/strata/math/evaluate`: Evaluates structured math requests with exact or numeric mode.
* `POST /v1/strata/math/verify`: Verifies LLM output arithmetic against ground truth computation.
* `POST /v1/strata/math/intercept`: Detects calculation intents in prompt or generation and augments with deterministic results.
* `GET /v1/strata/math/metrics`: Real-time calculation statistics, cache hit rates, and execution timings.
* `GET /v1/strata/math/backends`: Status and versions of available mathematical execution engines.

## Lean 4 kernel (`pkg/mathruntime/lean`)

The runtime can prove or refute a claim with a Go port of the Lean 4 kernel. It runs inside the server process: no
`lean` binary, no subprocess. The port is derived from `leanprover/lean4` v4.35.0 (`src/kernel`, Apache-2.0; a copy is
vendored, git-ignored, in `third_party/lean4`).

**What is ported:** universe levels (normalisation, `max`/`imax`, `≥`), expressions with de Bruijn indices, the
environment, `add_inductive` (parameters, indices, strict positivity, universe checks, the elimination-level rule,
generated recursors and their computation rules, K-like recursors), quotients, and the type checker: `infer`, `whnf`
(beta, delta, zeta, iota, projections, structure eta), `is_def_eq` (lazy delta with reducibility hints, proof
irrelevance, eta, structure eta, unit-like types, `Nat.succ` against literals) and the kernel's `Nat` literal extension
(`add sub mul div mod gcd pow beq ble land lor xor shiftLeft shiftRight` on big integers, with the same
refusal of oversized results). `whnf` and `is_def_eq` stop at a step budget and at the runtime's timeout.

Inductive types may be mutual (`Env.AddMutual`, one recursor per type taking every motive and minor premise), nested
(`inductive Rose | node : List Rose → Rose`: the nested occurrence becomes an auxiliary type while checking, and its
recursor is kept as `Rose.rec_1`) or indexed (`Int.NonNeg : Int → Prop`, whose single constructor returns
`NonNeg (ofNat n)`). A nested occurrence of an *indexed* family (`inductive Tree | node : Vector Tree 2 → Tree`) makes
the auxiliary indexed too: its motive takes the index before the value, one auxiliary serves every index of the same
family, and an index bound in the constructor (`Vector Forest n`) or computed from one (`Vector Bar (Nat.succ n)`, whose
minor then receives `n` and the vector at `n+1`) is carried into the auxiliary's own constructor. A family with two
indices (`Mat α rows cols`) gives an auxiliary with two, whose motive takes both before the value; and because an
auxiliary is built by the same path, one that contains a nested occurrence of its own gets a further auxiliary (a
matrix whose rows are `Vector α n` adds a `Vector` motive to the tree's recursor). A mutual block whose types nest each
other (`A.mk : List B → A` and `B.mk : A → B`) puts the auxiliary *inside* the block, so every recursor of the block
takes one motive per type plus one for the nested family and eliminates from either type; a family with three indices
(`Cube α i j k`) takes all three in its motive, before the value, and an auxiliary built from it carries all three. Tests
compare these recursor types with the shapes Lean generates and reduce them on small values, including a type that
nests an indexed and a plain family at once, the mutual case above and a three-index family nested with its parameter
recursive; no real Lean build was available to compare against.

**What is not ported:** `String` literals, the elaborator, tactics, the compiler, and the
type-class layer. A claim's operators elaborate to what Lean's instances unfold to (`a + b` becomes `Nat.add a b`).

**The prelude** (`prelude.go`) is built at start-up by the same kernel from text: `Nat`, `Int` (both constructors,
`add sub mul pow neg ediv emod dvd gcd lcm min max`, `le` as `NonNeg (b - a)` and `lt` as `(a + 1) ≤ b`, and
their decidable instances), `Bool`,
`Eq`, `And`, `Or`, `Iff`, `Exists`, `Decidable`, `Nat.le`, the `Nat` operations written with `Nat.rec`, hand-written
proof terms for `Nat.eq_of_beq_eq_true` and `Nat.ne_of_beq_eq_false`, and the `Decidable` instances. **Nothing in it is
admitted or opaque:** there is no axiom, `BuildPrelude` refuses an `opaque` declaration, and a test fails if
`Env.Axioms()` is not empty (the list is also reported by `GET /v1/strata/math/backends` as `lean_admitted_axioms`, now
always `[]`).

`Nat.ble` is a definition - `Nat.rec` on both arguments: `ble 0 m = true`, `ble (k+1) 0 = false`,
`ble (k+1) (m+1) = ble k m` - so the two directions of its reflection are proved. `Nat.ble_refl` and `Nat.ble_step`
(`ble n m = true → ble n (m+1) = true`, by `Nat.rec` on `n` and a case split on `m`, `False.elim` where `ble` is
false) give `Nat.ble_true_of_le : n ≤ m → ble n m = true` by `Nat.le.rec`; `Nat.le_of_ble_eq_true : ble n m = true →
n ≤ m` is `Nat.rec` on both arguments, ending in `Nat.succ_le_succ`; and `Nat.not_le_of_not_ble_eq_true` is the
contrapositive of `Nat.ble_true_of_le`. `Nat.decLe` decides `n ≤ m` with the last two, so an order claim is checked
from `Nat.rec` and `Nat.le.rec` like every other claim, and none of them requires the equation compiler.

`Nat.div` and `Nat.mod` are definitions, not axioms: Lean builds them by well-founded recursion, which this kernel
cannot state, so they are a state machine run once per unit of the dividend (`Nat.divMod`). `Nat.DivMod.mk q r` keeps
`q * k + r` equal to the number of steps taken, and each step either bumps `r` or, when `r` would reach `k`, wraps to
`mk (q+1) 0`; for `k = 0` the remainder never wraps, so `n / 0 = 0` and `n % 0 = n`. From that model the prelude
proves `n = k * (n / k) + n % k` (`Nat.divMod_spec`) and `k ∣ n ↔ n % k = 0` (`Nat.dvd_iff_mod_eq_zero`), which is
what `Nat.decDvd` decides with. A big dividend still costs one kernel step, because the kernel's `Nat` literal
extension answers `Nat.div`/`Nat.mod` on numerals with `big.Int` division before unfolding the model; a test
(`divmod_test.go`) reduces the model itself and requires quotients and remainders to agree with `big.Int` for every
divisor up to 9 and dividend up to 40.

`Nat.gcd` is a definition too: Lean gives it one in `Init/Data/Nat/Gcd.lean` (`gcd m n = if m = 0 then n else gcd
(n % m) m`, by well-founded recursion, with an `@[extern]` fast path only for evaluation), so here it is the same
recursion with the countdown made explicit. `Nat.gcdIter` maps `(a, b)` to `(b % a, a)` until the first argument is 0,
and `Nat.gcd a b` runs it from `a`; each step strictly lowers the first argument, so `a` steps always suffice and extra
fuel never changes the answer (`gcd_test.go` checks both). The equations `gcd 0 b = b`, `gcd (succ a) b = gcdIter a
(b % succ a) (succ a)`, `gcd 1 b = 1` and `gcd n 0 = n` hold definitionally, and `Nat.lcm` is `a * b / gcd a b` as in
Lean. **The prelude has no `opaque` declaration left** - `BuildPrelude` fails if one is added - and on two numerals
`Nat.gcd`/`Nat.lcm` are answered by the literal extension before the definition unfolds, so `gcd(12, 18) = 6` costs
12.2 µs and `gcd(123456789, 987654321) = 9` costs 13.8 µs (one reading each, `go test -bench . -benchtime=300x` on a
Ryzen 9 9950X3D).

`Nat.gcd_dvd_left`/`Nat.gcd_dvd_right` (`gcd m n ∣ m`, `gcd m n ∣ n`) are proved from that countdown, where Lean
derives them from its well-founded recursion: if the value divides `b % a` and `a` it divides `b`, because
`b = a * (b / a) + b % a`, and the induction runs over the fuel backwards - the countdown keeps `a ≤ f`, so it can only
stop on `a = 0`, where the value is `b` and divides `0` and `b`. That needed the supporting lemmas first, all under
Lean's names and statements: `Nat.le_trans`, `Nat.mul_zero`, `Nat.mul_one`, `Nat.add_assoc`, `Nat.mul_add`,
`Nat.mul_assoc`, `Nat.dvd_mul_right_of_dvd`, `Nat.dvd_add`, the remainder bound `Nat.mod_lt` (`n % k < k` when
`0 < k`, from `Nat.divStep_rem_lt`/`Nat.divIter_rem_lt`, which say a step keeps the remainder below the divisor) and
`Nat.gcdIter_dvd`, the invariant `gcdIter f a b ∣ a ∧ gcdIter f a b ∣ b` under `a ≤ f`.

`Nat.gcd_mul_left` (`gcd (m * n) (m * k) = m * gcd n k`) rests on `Nat.gcdIter_mul_left`: with `a ≤ f ≤ F`,
`gcdIter F (m * a) (m * b) = m * gcdIter f a b`. The two fuels differ because `gcd (m * n) (m * k)` starts with fuel
`m * n` and `gcd n k` with `n`. The induction is on `f` and needs no separate "fuel does not matter" lemma: each step
uses `Nat.mul_mod_mul_left` and only has to keep the remainder `≤ f'`. `Nat.gcd_mul_right`, `Nat.lcm_dvd` and
`Nat.lcm_assoc` follow, and so do `Nat.Coprime.symm` and `Nat.Coprime.dvd_of_dvd_mul_right`/`_left` (`k` coprime to
`n` and `k ∣ m * n` give `k ∣ m`: `k` divides `gcd (m * k) (m * n) = m * gcd k n = m`, Lean's proof) and
`Nat.Coprime.mul_dvd_of_dvd_of_dvd` (`k ∣ m`, `n ∣ m`, `k` coprime to `n` give `k * n ∣ m`), `Nat.div_one` and
`Nat.Coprime.lcm_eq_mul` (`lcm m n = m * n`), `Nat.Coprime.coprime_dvd_left` (a divisor of a number coprime to `k` is
coprime to `k`), `Nat.Coprime.of_mul_left`/`of_mul_right` (`Coprime (m * n) k` gives `Coprime m k`, `Coprime n k`),
the divisibility-to-coprime reductions `Nat.Coprime.coprime_dvd_right` and `coprime_mul_left`/`right`/`left_right`/`right_right`,
`Nat.Coprime.gcd_left`/`right`/`both`, `Nat.Coprime.pow_left`/`pow_right`/`pow`, `Nat.Coprime.eq_one_of_dvd` and
`Nat.coprime_mul_iff_right`. The product converse is `Nat.Coprime.mul_left`/`mul_right` with
`Nat.coprime_mul_iff_left`/`right`. It needs no Bezout: with
`g = gcd k (m * n)`, `g` is coprime to `m`, so `g ∣ n`, so `g` divides `gcd k n = 1`.

The order layer has `lt_irrefl`, `le_succ`, `lt_succ_self`, `le_of_lt`, `succ_le_of_lt`, `lt_of_succ_le`, `lt_of_le_of_lt`,
`lt_trans`, `pos_of_ne_zero`, `le_total` and `lt_or_ge`, and the negations built from them: `not_lt_of_le`,
`not_le_of_lt`, `lt_asymm`, `le_of_not_lt`, `lt_of_not_le` and `lt_or_eq_of_le`. Subtraction has `sub_zero`, `pred_le`, `sub_le`, `succ_pred`,
`sub_add_cancel` (`m ≤ n` gives `n - m + m = n`), `add_sub_of_le`, `sub_eq_zero_of_le` and `sub_lt` (`0 < n` and
`0 < m` give `n - m < n`). Cancellation and order: `add_right_cancel`, `add_left_cancel` (both from `add_sub_cancel`),
`lt_succ_of_le` and `le_of_lt_succ`. Subtraction also has `sub_sub_self` (`m ≤ n` gives `n - (n - m) = m`) and
`add_sub_assoc` (`k ≤ m` gives `n + m - k = n + (m - k)`). Monotonicity has `pred_le_pred`, `sub_le_sub_right`
(`n ≤ m` gives `n - k ≤ m - k`), `le_of_add_le_add_left`, `le_of_add_le_add_right` and `add_lt_add_right`.
`mul_eq_zero` (`m * n = 0` gives `m = 0 ∨ n = 0`) is by cases: `succ i * succ j` is a successor. `eq_zero_of_mul_eq_zero_pos` (`0 < n`, `n * k = 0` give `k = 0`) and
`mul_left_cancel` (`0 < n`, `n * m = n * k` give `m = k`) follow, the latter by induction on `m` with `add_right_cancel`. `mul_right_cancel` is the same with the factor on the right, and `le_of_mul_le_mul_left` (`0 < n`,
`n * m ≤ n * k` give `m ≤ k`) is the same induction for `≤` (`le_of_mul_le_mul_right` commutes the factors), and `lt_of_mul_lt_mul_left` (`n * m < n * k` gives
`m < k`) follows by contradiction from `mul_le_mul_left`. Divisibility also has `one_dvd` and `dvd_zero`, and remainders of
multiples are 0: `mul_mod_right` (`a * b % a`), `mul_mod_left` (`a * b % b`), `mod_self`, `mod_one` and `zero_mod`. Division has `zero_div`, `mul_div_le` (`n * (m / n) ≤ m`), `div_mul_le_self` (`m / n * n ≤ m`) and `lt_mul_div_succ`
(`0 < n` gives `m < n * (m / n + 1)`), `le_div_iff_mul_le` (`0 < k` gives `x ≤ y / k ↔ x * k ≤ y`) and its negation `div_lt_iff_lt_mul`
(`x / k < y ↔ x < y * k`), and `div_le_of_le_mul` (`0 < k` and `m ≤ k * n` give `m / k ≤ n`). Division has the identity `div_add_mod`
(`k * (n / k) + n % k = n`, from `divMod_spec`), `mod_add_div`, and the bounds `mod_le` and `div_le_self`.
Modular addition has `add_mul_mod_self_left` (`(a + b * n) % b = a % b`, with no `0 < b` side condition: `b = 0` is
`mod_zero`), `mod_add_mod`, `add_mod_mod` and `add_mod` (`(a + b) % n = (a % n + b % n) % n`). Products have
`add_mul` (right distributivity), `mod_mod`, `mod_mul_mod` (`(a % n) * b % n = a * b % n`) and `mul_mod`
(`a * b % n = (a % n) * (b % n) % n`).

Powers are monotone in the exponent: `pow_le_pow_right` (`0 < b` and `n ≤ m` give `b^n ≤ b^m`) and `pow_dvd_pow`
(`n ≤ m` gives `b^n ∣ b^m`), both by induction on `n ≤ m`.

`Nat.pow` has Lean's basic lemmas: `pow_zero`, `pow_succ`, `pow_one`, `pow_two`, `one_pow`, `zero_pow`, `pow_add`,
`mul_pow` (with `mul_left_comm` and `mul_mul_mul_comm`), `pos_pow_of_pos` (with `mul_pos`), `pow_succ'`, `pow_mul`,
`pow_le_pow_left` (with `mul_le_mul_right` and `mul_le_mul`) and `lt_two_pow` (`n < 2 ^ n`). `Nat.factorial` has
`factorial_zero`, `factorial_succ`, `factorial_pos`, `dvd_factorial` (`0 < m ≤ n` gives `m ∣ n!`) and
`self_le_factorial`. Lean core defines `factorial` only (Mathlib has the lemmas); the statements follow Mathlib's.
Checked by the
kernel with `go test ./pkg/mathruntime/lean`.

`Nat.dvd_gcd` (a common divisor divides the gcd) and `Nat.gcd_le_left`/`right` (`gcd m n ≤ m` when `0 < m`) then close
the divisibility side of the Nat gcd. `Nat.dvd_gcd` runs the countdown forwards: the invariant is "`k` divides both arguments at every step", and
a step needs `k ∣ b % a` from `k ∣ a` and `k ∣ b`, which is `k ∣ b - a * (b / a)`. That is the first place the prelude
needs subtraction, so it carries the layer Lean's `Init/Data/Nat/Basic.lean` has: `Nat.succ_add`, `Nat.add_comm`,
`Nat.add_right_comm`, `Nat.succ_mul` (`*` recurses on its second argument here, so this is an induction too),
`Nat.add_sub_add_right`, `Nat.add_sub_add_left`, `Nat.add_sub_cancel`, `Nat.add_sub_cancel_left`, `Nat.succ_sub_succ`,
`Nat.sub_self`, `Nat.zero_sub`, `Nat.sub_sub`, `Nat.mul_pred`, `Nat.mul_sub_left_distrib` (unconditional, because Nat
subtraction truncates), `Nat.dvd_sub`, and `Nat.sub_congr` for the rewrites; `Nat.gcd_le_left`/`right` go through
`Nat.le_of_dvd`, which needs `Nat.le_add_left` and the `0 ≤ n` base of the order. `Int.gcd_dvd_left`, `Int.gcd_dvd_right`
and the pair `Int.gcd_dvd` need no new multiplication lemma: `Int.natAbs_dvd_natAbs` turns `Nat.gcd_dvd_left` into
`Int.dvd (ofNat (gcd a b)) a` directly, since `natAbs (ofNat (gcd a b))` is `gcd a b` by definition. `Int.natAbs_ofNat`
and `Int.natAbs_negSucc` state those two definitional values, and `Int.natAbs_neg` (`natAbs (-a) = natAbs a`) follows by
cases, using `natAbs_negOfNat` for `ofNat k`. `Int.neg_neg` (`- -a = a`) is cases on `a` and on `k`, each case `rfl`. `Int.natAbs_eq_zero` (`natAbs a = 0 ↔ a = 0`) is cases forwards and `congrArg natAbs` backwards. `Int.dvd_refl`, `Int.dvd_trans` and `Int.dvd_neg`
(`a ∣ -b ↔ a ∣ b`) all go through `natAbs_dvd_natAbs`, so they reuse the Nat lemmas and `natAbs_neg`. The same route gives `dvd_mul_right`, `dvd_mul_left` (with `natAbs_mul`),
`dvd_zero` and `neg_dvd` (`-a ∣ b ↔ a ∣ b`).

`Int.gcd` and `Int.lcm` are the Nat functions on `natAbs`, so the Nat lemmas apply unchanged: `Int.gcd_comm`,
`Int.lcm_comm`, `Int.gcd_self` and `Int.gcd_zero_right` are the Nat statements at `natAbs`, and `Int.gcd_neg_left` and
`Int.gcd_neg_right` are `congrArg` of `natAbs_neg`.

`Int.mul_comm` is the first arithmetic law proved on Int itself rather than through `natAbs`: `Int.mul` is four constructor
cases, and each case of `m * n = n * m` is `congrArg ofNat` or `congrArg negOfNat` of `Nat.mul_comm`.
`Int.mul_ofNat_negOfNat` (`ofNat a * negOfNat k = negOfNat (a * k)`, both cases `rfl`) and `Int.negOfNat_mul_ofNat` (its
commuted form) are helpers for the mixed-sign cases of `Int.mul_assoc`, with two more: `Int.negOfNat_mul_negSucc`
(`negOfNat k * negSucc c = ofNat (k * succ c)`; `k = 0` needs `zero_mul`) and `Int.negSucc_mul_negOfNat`. `Int.mul_assoc`
is then eight constructor cases, each `congrArg ofNat` or `congrArg negOfNat` of `Nat.mul_assoc`, with a helper moving
`negOfNat` across one factor where the signs are mixed. The units follow: `Int.mul_zero` (`rfl` on both constructors),
`Int.mul_one` (`congrArg` of `Nat.mul_one`), and `Int.zero_mul` and `Int.one_mul` by `mul_comm`.

`Int.add_comm` is four constructor cases: the same-sign cases are `congrArg` of `Nat.add_comm`, and the mixed cases are
`rfl`, because `Int.add` gives `subNatNat` of the same pair in either order. (`add_assoc` is harder: it needs facts about
`subNatNat`, and is proved further down.) `Int.add_zero` is `rfl` on both constructors, since `subNatNat 0 (succ k)` reduces to
`negSucc k`; `Int.zero_add` is `add_comm` then that, and `Int.neg_zero` is `rfl`.

`Int.subNatNat_self` (`subNatNat n n = 0`) is the first lemma about `subNatNat`: its `Bool.rec` is on `Nat.ble n n`, which
`Nat.ble_refl` makes `true`, leaving `ofNat (n - n)` and `Nat.sub_self`. With it, `Int.add_neg_cancel` (`a + -a = 0`) is
cases on `a`, where the nonzero cases are `subNatNat (succ _) (succ _)`, and `Int.neg_add_cancel` is `add_comm` then that.

`Int.neg_negOfNat` (`-(negOfNat n) = ofNat n`, both cases `rfl`) feeds `Int.neg_mul` (`-a * b = -(a * b)`): four constructor
cases using the `negOfNat` helpers from `mul_assoc`. `Int.mul_neg` (`a * -b = -(a * b)`) is `mul_comm` and `neg_mul`.

`Nat.div_lt_self` (`0 < n`, `1 < k` give `n / k < n`) goes through `div_lt_iff_lt_mul`, which turns it into `n < n * k`; that is
`mul_lt_mul_of_pos_left` on `1 < k`, with `n * 1` rewritten to `n` by `mul_one`.

`Nat.lt_add_of_pos_right` (`0 < k` gives `n < n + k`) is `add_lt_add_left` as is, since `n + 0` reduces to `n`. `Nat.add_pos_left`
(`0 < m` gives `0 < m + n`) is `lt_of_lt_of_le` with `le_add_right`.

`Nat.sub_pos_of_lt` (`a < b` gives `0 < b - a`) argues by `lt_of_not_le`: if `b - a <= 0` then `b - a = 0` (`le_zero`), so
`add_sub_of_le` gives `a + 0 = b`, which is `a = b`, and `ne_of_lt` refutes it.

`Nat.pow_lt_pow_succ` (`1 < b` gives `b ^ n < b ^ (n + 1)`): `b ^ (n + 1)` is `b ^ n * b` by `rfl`, and `mul_lt_mul_of_pos_left`
on `1 < b` gives `b ^ n * 1 < b ^ n * b`, as `b ^ n > 0` (`pos_pow_of_pos`); `mul_one` rewrites the left side.

`Nat.pow_lt_pow_right` (`1 < b`, `m < n` give `b ^ m < b ^ n`) is induction on `succ m <= n` with `Nat.le.rec`: the base case is
`pow_lt_pow_succ`, and each step chains one more `pow_lt_pow_succ` by `lt_trans`.

`Nat.lt_of_add_lt_add_left` (`n + m < n + k` gives `m < k`) is `le_of_add_le_add_left` at `succ m`, since `succ (n + m)` is
`n + succ m` by `rfl`.

`Nat.add_lt_add` (`a < b`, `c < d` give `a + c < b + d`) chains `add_lt_add_right` and `add_lt_add_left` by `lt_trans`, through
`b + c`.

`Nat.mul_lt_mul_of_pos_right` (`n < m`, `0 < k` give `n * k < m * k`) is `mul_lt_mul_of_pos_left` with both products turned round
by `mul_comm`.

`Nat.lt_of_mul_lt_mul_right` (`n * k < m * k` gives `n < m`) is `lt_of_mul_lt_mul_left` after turning both products round by
`mul_comm`.

`Nat.lt_of_add_lt_add_right` (`n + k < m + k` gives `n < m`) is `lt_of_add_lt_add_left` after turning both sums round by
`add_comm`.

`Nat.lt_succ_iff` (`n < succ m <-> n <= m`) pairs `le_of_lt_succ` with `lt_succ_of_le`, and `Nat.succ_le_iff` (`succ n <= m <-> n < m`)
pairs `lt_of_succ_le` with `succ_le_of_lt`.

`Nat.lt_of_sub_pos` (`0 < b - a` gives `a < b`) is the converse of `sub_pos_of_lt`: by `lt_of_not_le`, if `b <= a` then
`sub_eq_zero_of_le` makes `b - a = 0`, so `0 < 0`, which `lt_irrefl` refutes.

`Nat.sub_pos_iff_lt` (`0 < b - a <-> a < b`) pairs `lt_of_sub_pos` with `sub_pos_of_lt`. `Nat.sub_eq_zero_iff_le`
(`n - m = 0 <-> n <= m`) has `sub_eq_zero_of_le` one way; the other is `le_of_not_lt`, since `m < n` would give `0 < n - m`.

Four Int facts are `rfl`: `Int.ofNat_add` and `Int.ofNat_mul` (`ofNat` commutes with `+` and `*`), `Int.neg_ofNat_succ`
(`-(ofNat (succ n)) = negSucc n`) and `Int.neg_negSucc` (`-(negSucc n) = ofNat (succ n)`). They are named so later proofs can cite them.

Integer subtraction is `a + -b`: `Int.sub_eq_add_neg` is `rfl`, and `Int.sub_self`, `Int.sub_zero` and `Int.zero_sub` are
`add_neg_cancel`, `add_zero` and `zero_add` at that form.

First `subNatNat` building blocks for `Int.add_assoc`: `Int.subNatNat_add_left` (`subNatNat (m + n) m = ofNat n`) and
`Int.subNatNat_add_right` (`subNatNat m (succ (m + n)) = negSucc n`). The first has `Nat.ble m (m + n)` true, so the `Bool.rec`
takes the `ofNat (m + n - m)` branch and `add_sub_cancel_left` finishes. In the second `Nat.ble (succ (m + n)) m` is false: true would
give `succ m <= m`, against `lt_irrefl`; the `negSucc` branch then needs `succ_sub_succ` and `add_sub_cancel_left`.

`Int.subNatNat_succ_succ` (`subNatNat (succ m) (succ n) = subNatNat m n`): `ble (succ n) (succ m)` reduces to `ble n m`, and the two
`Bool.rec` branches each differ by one `succ_sub_succ`. `Int.subNatNat_add_add` (`subNatNat (m + k) (n + k) = subNatNat m n`) is
induction on `k` with that as the step.

`Nat.add_left_comm` (`n + (m + k) = m + (n + k)`) is new. With the `subNatNat` blocks it gives `ofNat a + subNatNat b c = subNatNat (a + b) c`
in two shapes, one per way `b` and `c` can sit: `Int.ofNat_add_subNatNat_le` for `b = c + d` (both sides are `ofNat (a + d)`, by
`subNatNat_add_left`) and `Int.ofNat_add_subNatNat_gt` for `c = succ (b + d)` (both sides are `subNatNat a (succ d)`, by
`subNatNat_add_right` and `subNatNat_add_add`). `Int.ofNat_add_subNatNat` is the general statement (`ofNat a + subNatNat b c = subNatNat (a + b) c`): split by `Nat.lt_or_ge b c`, write
`c = succ (b + (c - succ b))` or `b = c + (b - c)` with `add_sub_of_le`, and rewrite into the matching shape.

`Int.negSucc_add_subNatNat` (`negSucc a + subNatNat b c = subNatNat b (succ (a + c))`) uses the same split. For `b = c + d` both sides are
`subNatNat d (succ a)`, by `subNatNat_add_add` once `c + d` and `succ (a + c)` are turned round; for `c = succ (b + d)` both are
`negSucc (succ (a + d))`, by `subNatNat_add_right` and `Nat.add_left_comm`.

`Int.add_assoc` (`(a + b) + c = a + (b + c)`) is eight constructor cases. `ofNat`, `ofNat`, `ofNat` is `Nat.add_assoc`; `negSucc` three times is
`Nat.add_assoc` under two `succ`s. In the six mixed cases, `Int.add` reduces to `subNatNat` sums, which `ofNat_add_subNatNat` and
`negSucc_add_subNatNat` move across; where the `subNatNat` is on the left, `Int.add_comm` swaps it first, then `Nat.add_comm` fixes the order of the
two numbers.

Three consequences of `add_assoc`: `Int.add_left_cancel` (`a + b = a + c` gives `b = c`) adds `-a` on the left and uses `neg_add_cancel` and
`zero_add`; `Int.add_right_cancel` goes through `add_comm` to that; and `Int.add_left_comm` (`a + (b + c) = b + (a + c)`) is `add_assoc` both ways
around an `add_comm`.

`Int.neg_add` (`-(a + b) = -a + -b`) is `add_left_cancel` at `a + b`: `(a + b) + -(a + b)` is `0` by `add_neg_cancel`, and
`(a + b) + (-a + -b)` reassociates, through `add_comm`, `add_neg_cancel` and `add_zero`, to `b + -b`, which is also `0`.

First steps toward `Int.mul_add`: `Int.negOfNat_add_negSucc` (`negOfNat (u + succ n) = negOfNat u + negSucc n`) is induction on `u`, where
the step is `succ_add` under `negSucc`; and `Int.ofNat_succ_mul` (`ofNat (succ k) * x = ofNat k * x + x`) is `Nat.succ_mul` for `x = ofNat n`, and
`succ_mul` under `negOfNat` followed by the first lemma for `x = negSucc n`.

Distributivity follows. `Int.add_add_add_comm` (`(p + q) + (b + c) = (p + b) + (q + c)`) is `add_assoc` around `add_left_comm`. `Int.ofNat_mul_add`
(`ofNat k * (b + c) = ofNat k * b + ofNat k * c`) is induction on `k`: `0` is `zero_mul` three times, and the step is `ofNat_succ_mul`, the
induction hypothesis and `add_add_add_comm`. `Int.mul_add` takes `a = ofNat k` from that; for `a = negSucc k`, which is `-(ofNat (succ k))`, it goes
through `neg_mul`, `ofNat_mul_add` and `neg_add`. `Int.add_mul` is `mul_comm`, `mul_add`, `mul_comm` again.

Subtraction versions, all using `a - b = a + -b`: `Int.mul_sub` (`a * (b - c) = a * b - a * c`) is `mul_add` then `mul_neg`; `Int.sub_mul`
is `add_mul` then `neg_mul`; `Int.neg_sub` (`-(a - b) = b - a`) is `neg_add`, `neg_neg`, `add_comm`; and `Int.sub_add_cancel` (`a - b + b = a`) is
`add_assoc`, `neg_add_cancel`, `add_zero`.

`Int.eq_of_sub_eq_zero` (`a - b = 0` gives `a = b`) rewrites `a` as `(a - b) + b` by `sub_add_cancel`, then `0 + b`; `Int.sub_eq_zero_of_eq` is
`sub_self` after substituting; `Int.sub_eq_zero` is the `Iff` of the two. `Int.neg_inj` (`-a = -b` gives `a = b`) goes through `neg_neg` on both sides.

`Int.mul_eq_zero` (`a * b = 0` gives `a = 0` or `b = 0`) needs no case split on signs: `natAbs` of both sides gives `natAbs a * natAbs b = 0` by
`natAbs_mul`, `Nat.mul_eq_zero` splits it, and `natAbs_eq_zero` brings each side back to Int. `Int.mul_left_cancel` (`a != 0`, `a * b = a * c` gives
`b = c`) shows `a * (b - c) = 0` by `mul_sub` and `sub_eq_zero_of_eq`; `mul_eq_zero` then leaves `b - c = 0`, and `eq_of_sub_eq_zero` finishes.

`Int.mul_right_cancel` (`c != 0`, `a * c = b * c` gives `a = b`) is `mul_left_cancel` after `mul_comm` on both sides. `Int.neg_eq_of_add_eq_zero`
(`a + b = 0` gives `-a = b`) rewrites `-a` as `-a + (a + b)`, reassociates to `(-a + a) + b`, and uses `neg_add_cancel` and `zero_add`.

First order lemmas on Int, where `a <= b` is `NonNeg (b - a)`. `Int.NonNeg_add` (`NonNeg x`, `NonNeg y` give `NonNeg (x + y)`) is `NonNeg.rec` twice, since
the one constructor is on `ofNat` and `ofNat n + ofNat m` reduces to `ofNat (n + m)`. `Int.le_refl` is `NonNeg 0`, moved onto `a - a` by `sub_self`.
`Int.le_trans` shows `(c - b) + (b - a) = c - a` with `add_assoc`, `neg_add_cancel` and `zero_add`, then applies `NonNeg_add`.

`Int.add_le_add_right` (`a <= b` gives `a + c <= b + c`) rewrites `NonNeg (b - a)` onto `(b + c) - (a + c)`, which equals `b - a` by `neg_add`,
`add_add_add_comm`, `sub_self` and `add_zero`. `Int.add_le_add_left` is the same with the sums turned round by `add_comm`.

The link between Int order and Nat order. `Int.not_nonneg_negSucc` (`NonNeg (negSucc k)` is false) is `NonNeg.rec` with the index equation, the same
argument `Int.decNonneg` uses. `Int.subNatNat_nonneg` (`NonNeg (subNatNat m n) <-> n <= m`) splits on `Nat.ble n m`: true leaves `ofNat (m - n)`, which is
`NonNeg.mk`; false leaves a `negSucc`, which is refuted. `Int.ofNat_sub_ofNat` (`ofNat n - ofNat m = subNatNat n m`) is `rfl` for `m = 0` and
`m = succ k`, and `Int.ofNat_le` (`ofNat m <= ofNat n <-> m <= n`) combines the two.

`Int.le_total` (`a <= b` or `b <= a`) cases on `b - a`: if it is `ofNat n`, that is `NonNeg.mk n`; if it is `negSucc k`, then
`a - b = -(b - a) = ofNat (succ k)` by `neg_sub`. `Int.le_of_eq` (`a = b` gives `a <= b`) is `le_refl` rewritten along the equation.

`Int.eq_zero_of_nonneg_neg` (`NonNeg x` and `NonNeg (-x)` give `x = 0`) cases on `x`: `ofNat 0` is `rfl`; for `ofNat (succ j)`, `-x` is `negSucc j`, which is
not `NonNeg`; and `negSucc k` is not `NonNeg` itself. `Int.le_antisymm` (`a <= b` and `b <= a` give `a = b`) applies it to `b - a`, whose negation is
`a - b` by `neg_sub`, and finishes with `eq_of_sub_eq_zero`.

Strict order on Int, where `a < b` is `a + 1 <= b`. `Int.add_one_sub` (`(a + 1) - a = 1`) and `Int.sub_add_one_self` (`a - (a + 1) = negSucc 0`) are
`add_comm`, `add_assoc`, `add_neg_cancel` and `neg_add` chains. They give `Int.le_add_one` (`a <= a + 1`) and `Int.lt_irrefl`, since `a < a` would make
`NonNeg (negSucc 0)`. `Int.le_of_lt`, `Int.lt_trans`, `Int.lt_of_le_of_lt` and `Int.lt_of_lt_of_le` are `le_trans` with `le_of_lt` or `add_le_add_right`.

`Int.NonNeg_mul` (`NonNeg x`, `NonNeg y` give `NonNeg (x * y)`) is the same two `NonNeg.rec` as `NonNeg_add`, ending in `NonNeg.mk (n * m)`. With it,
`Int.mul_le_mul_of_nonneg_left` (`NonNeg c`, `a <= b` give `c * a <= c * b`) takes `NonNeg (c * (b - a))` and rewrites it by `mul_sub` to `c * b - c * a`.

`Int.mul_le_mul_of_nonneg_right` is the left form with both products turned round by `mul_comm`. `Int.not_le_of_lt` (`a < b` rules out `b <= a`) is
`lt_of_lt_of_le` giving `a < a`, against `lt_irrefl`. Division by and of zero is proved: `ediv_zero` (`a / 0 = 0`), `emod_zero` (`a % 0 = a`), `zero_ediv` and `zero_emod`, by cases on the integer using `Nat.div_zero`, `Nat.mod_zero`, `Nat.zero_div` and `Nat.zero_mod`, with the `negSucc` cases closing by `rfl`. Division by one is proved too: `ediv_one` (`a / 1 = a`) and `emod_one` (`a % 1 = 0`), using `Nat.div_one` and `Nat.mod_one`. `emod_self` (`a % a = 0`) is proved for every `a`: `ofNat k` by `Nat.mod_self`, `negSucc k` by rewriting `k % succ k` to `k` (`Nat.mod_eq_of_lt`) and then `subNatNat_self`. The first try used `rfl` for that last step and the kernel rejected it, because `subNatNat (succ k) (succ k)` does not reduce for a variable `k`. `ediv_self` (`a / a = 1` for `a != 0`) is proved: `ofNat 0` is absurd, `ofNat (succ m)` uses `Nat.div_self`, and `negSucc k` uses `k / succ k = 0` (`Nat.div_eq_of_lt`). `lt_iff_add_one_le` (`a < b` iff `a + 1 <= b`, the identity both ways since that is the definition) and `not_lt_of_le` (`b <= a` rules out `a < b`) are proved too. `NonNeg_cases` says a `NonNeg` integer `j` is `0` or has `NonNeg (j - 1)`: `NonNeg.rec`, then `Nat.rec` on the payload, where `ofNat (succ m) - 1` reduces to `ofNat m` by `rfl`. With `sub_add_eq_sub_sub` (`b - (a + c) = (b - a) - c`, from `neg_add` and `add_assoc`) it gives `eq_or_lt_of_le`: `a <= b` means `a = b` or `a < b`. The `0` case is `eq_of_sub_eq_zero`; the other moves `NonNeg ((b - a) - 1)` to `NonNeg (b - (a + 1))`. `lt_of_le_of_not_le` follows: `eq_or_lt_of_le` leaves `a = b` (then `b <= a`, absurd) or `a < b`. With `le_of_lt` and `not_le_of_lt` that gives `lt_iff_le_not_le`: `a < b` iff `a <= b` and `not (b <= a)`. `emod_nonneg` proves `0 <= a % b` for `b != 0`: `ofNat k` is `NonNeg.mk`; `negSucc k` is `subNatNat |b| (succ (k % |b|))`, which is `NonNeg` by `subNatNat_nonneg` and `Nat.mod_lt` (`|b| > 0` from `natAbs_eq_zero`), moved to `x - 0` by `sub_zero`. The first try mixed up an `Int` and a `Nat` equality and the kernel rejected it. `nonneg_of_le_zero` and `le_zero_of_nonneg` say `0 <= a` and `NonNeg a` are the same, moving between `a - 0` and `a` with `sub_zero`. They give `ediv_nonneg`: `0 <= a / b` for `0 <= a` and `0 <= b`; only `ofNat / ofNat` is left (`NonNeg.mk`), the other three cases contradict `not_nonneg_negSucc`. `ediv_add_emod_ofNat` is the division identity on `ofNat`: `k * (m / k) + m % k = m`. The `Int` operations on `ofNat` reduce to `ofNat` of the `Nat` ones, so it is `congrArg ofNat` of `Nat.mod_add_div` after `add_comm`. The general `ediv_add_emod` (negative `a` or `b`) is not proved yet. `emod_lt_ofNat` bounds the remainder for `ofNat a`: `a % b < |b|` for `b != 0`. It unfolds to `NonNeg (subNatNat |b| (succ (a % |b|)))`, which is `subNatNat_nonneg` and `Nat.mod_lt`. The `negSucc a` case needs `sub_sub_cancel` (`a - (a - b) = b`, from `neg_add`, `neg_neg`, `add_assoc`, `add_neg_cancel`) and `subNatNat_succ_lt` (`subNatNat n (succ r) < n` for every `n` and `r`: it equals `n - s`, so `n - ((n - s) + 1) = s - 1 = r`, `NonNeg` by `mk`). Together they give `emod_lt`: `a % b < |b|` for `b != 0`. `ediv_add_emod_ofNat_negSucc` is the division identity for a natural `m` and divisor `negSucc n`: `negSucc n * (m / negSucc n) + m % negSucc n = m`. It uses the existing `negSucc_mul_negOfNat` and `Nat.mod_add_div`, as for the `ofNat` divisor. `ediv_add_emod_zero` covers divisor `0` for every `a`: `0 * (a / 0) + a % 0 = a`, from `emod_zero`, `zero_mul` and `zero_add`. `negOfNat_add_subNatNat` (`negOfNat (t + s) + subNatNat s (succ r) = negSucc (t + r)`) writes both sides as `-(T + S) + (S - (R + 1))`, regroups, cancels `-S + S`, and uses `-T + -(R + 1) = -(T + (R + 1))`. With `t = succ j * (a / succ j)` and `r = a % succ j` it gives `ediv_add_emod_negSucc_ofNat_succ`, the division identity for `negSucc a` and divisor `succ j`. `ediv_add_emod_negSucc_negSucc` is the same computation for a `negSucc` divisor (`negSucc n * ofNat (succ q)` is `negOfNat (succ n * succ q)`). The six cases (two signs of `a`, and for `b` the `ofNat` with `0` and `succ`, and `negSucc`) are put together in `ediv_add_emod`: `b * (a / b) + a % b = a` for every `Int` `a` and `b`, with no side condition. `Int.min` and `Int.max` are a `Bool.rec` on `decide (a <= b)`, so they are now pinned down by `decide_eq_true'` and `decide_eq_false'` (`decide p` is `true` if `p`, `false` if `not p`; primed to leave Lean's own names free): `min_eq_left` (`a <= b` gives `min a b = a`), `min_eq_right_of_not_le`, `max_eq_right` and `max_eq_left_of_not_le`. With them `min_le_left`, `min_le_right`, `le_max_left` and `le_max_right` are proved by splitting on `decide (a <= b)`: the equation moves the goal to `a` or `b`, which `le_refl`, the hypothesis, or (when `not (a <= b)`) `b <= a` from `le_total` closes. `le_min` (`c <= a` and `c <= b` give `c <= min a b`) and `max_le` (`a <= c` and `b <= c` give `max a b <= c`) are proved the same way. They make `min_comm` and `max_comm` a `le_antisymm` of two such bounds. The same method gives `min_self`, `max_self` (`min a a = a`, `max a a = a`) and `min_assoc`, `max_assoc`, using `le_trans` to chain the bounds. `add_le_add` (`a <= b` and `c <= d` give `a + c <= b + d`) is `add_le_add_right` then `add_le_add_left` joined by `le_trans`. `neg_le_neg` (`a <= b` gives `-b <= -a`) uses `-a - -b = -a + b = b + -a = b - a`, so the `NonNeg (b - a)` carries over. The strict forms follow: `add_lt_add_right` (`a < b` gives `a + c < b + c`) is `add_le_add_right` on `a + 1 <= b` after `(a + c) + 1 = (a + 1) + c` (`add_assoc`, `add_comm`); `add_lt_add_left` is that moved by `add_comm`; `add_lt_add` joins the two with `lt_trans`. Cancelling `+` in `<=` is proved too: `add_neg_cancel_right` (`(x + c) + -c = x`), then `le_of_add_le_add_right` (`a + c <= b + c` gives `a <= b`, by adding `-c` to both sides) and `le_of_add_le_add_left` (the same after `add_comm`). The strict forms `lt_of_add_lt_add_right` and `lt_of_add_lt_add_left` follow: `(a + c) + 1 = (a + 1) + c` turns `a + c < b + c` into `(a + 1) + c <= b + c`, then `le_of_add_le_add_right` finishes. `sub_nonneg` (`0 <= a - b` iff `b <= a`) is `nonneg_of_le_zero` and `le_zero_of_nonneg`. `le_of_neg_le_neg` applies `neg_le_neg` to `-b <= -a` and rewrites `--x` to `x`, which with `neg_le_neg` gives the iff `neg_le_neg_iff`. Subtraction is monotone: `a - c` is `a + -c`, so `sub_le_sub_right` is `add_le_add_right`, and `sub_le_sub_left` and `sub_le_sub` (`a <= b` and `c <= d` give `a - d <= b - c`) add `neg_le_neg`. `mul_nonneg` (`0 <= a` and `0 <= b` give `0 <= a * b`) is `NonNeg_mul` seen through `0 <= x` iff `NonNeg x`. `mul_pos` (`0 < a` and `0 < b` give `0 < a * b`): `0 < a` is `1 <= a`, so `1 * b <= a * b` by `mul_le_mul_of_nonneg_right`, `1 * b = b`, and `1 <= b <= a * b` by `le_trans`. `neg_lt_neg` (`a < b` gives `-b < -a`): `-b < -a` is `NonNeg (-a - (-b + 1))`, and `-a - (-b + 1) = (-a - -b) - 1 = (b - a) - 1 = b - (a + 1)` by `sub_add_eq_sub_sub` and `-a - -b = b - a`, which is the `NonNeg` that `a < b` already is. `min_add_add_right` (`min (a + c) (b + c) = min a b + c`) is `le_antisymm`. `min a b + c <= min (a + c) (b + c)` is `le_min` of two `add_le_add_right`. For the other way, `m + -c <= a` and `<= b` (by `add_neg_cancel_right`) give `m + -c <= min a b`, and `le_of_add_le_add_right` moves `-c` across. `max_add_add_right` (`max (a + c) (b + c) = max a b + c`) mirrors it: `max_le` of two `add_le_add_right` one way; the other way, `a, b <= m + -c` gives `max a b <= m + -c`, and `le_of_add_le_add_right` moves `-c` across. The left forms `min_add_add_left` and `max_add_add_left` (`min (c + a) (c + b) = c + min a b`) rewrite with `add_comm` inside both arguments, use the right form, and swap back with `add_comm`. `neg_min` (`-(min a b) = max (-a) (-b)`) is `le_antisymm`: `max_le` with `neg_le_neg` of `min_le_left` and `min_le_right` gives one side; for the other, split on `decide (a <= b)`, where `min` is `a` or `b` and `le_max_left` or `le_max_right` finishes. `neg_max` (`-(max a b) = min (-a) (-b)`) follows: `neg_min` at `-a` and `-b` reads `-(min (-a) (-b)) = max (--a) (--b)`, so rewrite `--x` to `x` and negate both sides. `lt_add_one_iff` (`a < b + 1` iff `a <= b`) is `a + 1 <= b + 1` read through `add_le_add_right` and `le_of_add_le_add_right`. It gives `lt_add_one` (`a < a + 1`), and `sub_one_lt` (`a - 1 < a`) follows from `(a - 1) + 1 = a`. `mul_lt_mul_of_pos_left` (`a < b` and `0 < c` give `c * a < c * b`): `c * a + 1 <= c * a + c` because `1 <= c`, and `c * a + c = c * (a + 1) <= c * b` by `mul_add`, `mul_one` and `mul_le_mul_of_nonneg_left`. `mul_lt_mul_of_pos_right` (`a * c < b * c`) is that with `mul_comm` on both products. `lt_of_mul_lt_mul_left` (`c * a < c * b` and `0 < c` give `a < b`) cancels the factor: `b <= a` would give `c * b <= c * a` (`mul_le_mul_of_nonneg_left`), against `not_le_of_lt`, so `le_total` leaves `a <= b` and `lt_of_le_of_not_le` finishes. `lt_of_mul_lt_mul_right` is the same after `mul_comm`. `mul_le_mul` is two-sided monotonicity: `a <= b`, `c <= d`, `0 <= c` and `0 <= b` give `a * c <= b * c <= b * d` by the right and left one-sided forms and `le_trans`. `mul_neg` and `neg_mul` already existed; `neg_mul_neg` (`(-a) * (-b) = a * b`) follows from them and `neg_neg`. `mul_self_nonneg` (`0 <= a * a`): if `0 <= a` it is `mul_nonneg`; if `a <= 0` then `0 <= -a` by `neg_le_neg`, and `(-a) * (-a) = a * a`. `natAbs_neg` and `natAbs_mul` already existed. `le_natAbs` (`a <= natAbs a`): `le_refl` for `ofNat k`; for `negSucc k` the difference `succ k + succ k` is a `NonNeg.mk`. `neg_natAbs_le` (`-natAbs a <= a`): `le_natAbs (-a)`, then `neg_le_neg`, with `natAbs_neg` and `neg_neg` rewriting the two ends. `mul_min_of_nonneg` and `mul_max_of_nonneg` (`0 <= c` gives `c * min a b = min (c * a) (c * b)`, same for `max`) are `le_antisymm`: one side is `le_min` / `max_le` of `mul_le_mul_of_nonneg_left`; the other splits on `decide (a <= b)`, rewrites `min a b` / `max a b` to `a` or `b`, and closes with `min_le_*` / `le_max_*`. Sign rules: `add_nonneg`, `add_pos`, `neg_nonneg_of_nonpos`, `neg_nonpos_of_nonneg`, `neg_pos_of_neg`, `neg_neg_of_pos` are `add_le_add`, `add_lt_add`, `neg_le_neg` and `neg_lt_neg` at `0`, since `0 + 0` and `-0` reduce to `0`. Product signs: `mul_nonneg_of_nonpos_of_nonpos` is `mul_nonneg` on `-a`, `-b` with `neg_mul_neg`. `mul_nonpos_of_nonneg_of_nonpos`, `mul_nonpos_of_nonpos_of_nonneg` and `mul_neg_of_pos_of_neg` take `0 <= a * -b = -(a * b)` (or `(-a) * b`; `mul_pos` for the strict one), then `neg_le_neg` / `neg_lt_neg` and `neg_neg` flip it. Strict order: `lt_asymm` (from `not_lt_of_le`), `ne_of_lt` (a proof of `a = b` would turn `a < b` into `a < a`, against `lt_irrefl`), `le_of_not_lt` (`le_total`, then `eq_or_lt_of_le` for the equal case) and `lt_trichotomy` (`le_total`, then `eq_or_lt_of_le` on each side). `mul_pos_of_neg_of_neg` is `mul_pos` on the negations with `neg_mul_neg`; `mul_neg_of_neg_of_pos` is `mul_pos` on `-a`, `b` flipped like the non-strict ones. From `mul_eq_zero`: `mul_ne_zero` (nonzero factors give a nonzero product) and `mul_self_eq_zero` (`a * a = 0` gives `a = 0`). `mul_self_pos` (`a != 0` gives `0 < a * a`): `lt_trichotomy a 0` leaves `a < 0` (`mul_pos_of_neg_of_neg`), `a = 0` (contradiction) or `0 < a` (`mul_pos`). `sub_pos` (`0 < a - b <-> b < a`), `sub_neg` (`a - b < 0 <-> a < b`) and `sub_nonpos` (`a - b <= 0 <-> a <= b`): add `b` with `add_lt_add_right` / `add_le_add_right` and rewrite with `sub_add_cancel` and `zero_add`; for the way back add `-b` and use `add_neg_cancel`. Adding a nonneg or positive term: `le_add_of_nonneg_right` / `_left` (`a <= a + b`, `a <= b + a`) and `lt_add_of_pos_right` / `_left` are `add_le_add_*` / `add_lt_add_*` at `0` with `add_zero` / `zero_add`; `sub_le_self` and `sub_lt_self` go the same way through `-b <= 0` / `-b < 0`. `add_lt_add_of_le_of_lt` and `add_lt_add_of_lt_of_le` chain one strict and one non-strict step with `lt_of_lt_of_le`. `ofNat_natAbs_of_nonneg` (`0 <= a` gives `ofNat (natAbs a) = a`): `ofNat k` is `refl`, `negSucc k` contradicts `not_nonneg_negSucc`. `ofNat_natAbs_of_nonpos` (`a <= 0` gives `ofNat (natAbs a) = -a`) applies it to `-a` and rewrites with `natAbs_neg`. `natAbs_eq` (`a = natAbs a` or `a = -natAbs a`): `le_total 0 a`, then the two `ofNat_natAbs_of_*` lemmas, the second after `neg_neg`. `neg_le_natAbs` (`-a <= natAbs a`) is `le_natAbs (-a)` with `natAbs_neg`. The triangle inequality `natAbs_add_le` (`natAbs (a + b) <= natAbs a + natAbs b`) is proved in `Int` and moved to `Nat` by `ofNat_le`: if `0 <= a + b`, then `a + b <= |a| + |b|` from `le_natAbs`; if `a + b <= 0`, then `-(a + b) = -a + -b <= |a| + |b|` from `neg_le_natAbs` and `neg_add`. `natAbs_sub_le` (`natAbs (a - b) <= natAbs a + natAbs b`): `a - b` is `a + -b`, so it is `natAbs_add_le` with `natAbs_neg`. `natAbs_sub_comm` (`natAbs (a - b) = natAbs (b - a)`) is `natAbs_neg` and `neg_sub`.

`Nat.gcd_comm` and `Nat.gcd_assoc` finish that: neither unfolds the countdown, because with `dvd_gcd` and `gcd_dvd` both
sides of each equation are common divisors of the same arguments, and `Nat.dvd_antisymm` then makes them equal. That is
where the order and multiplication lemmas come in: `Nat.le_antisymm` (`Nat.le_trans` plus
`Nat.not_le_succ_self` in the strict case), `Nat.dvd_antisymm` (from `Nat.le_of_dvd` and
`Nat.pos_of_dvd_of_pos`), `Nat.dvd_trans` (from `Nat.mul_assoc`), `Nat.mul_comm`, `Nat.add_le_add` with its
right/left halves, and `Nat.eq_zero_of_zero_dvd`. The core `Nat.Coprime` API is also present: its gcd equivalence,
symmetry and commutativity, zero/one/self cases, cancellation of a coprime factor from divisibility, products of
coprime factors, powers of coprime numbers, `eq_one_of_dvd`, and `lcm_eq_mul`. The broader `Init/Data/Nat` arithmetic library is still not ported; this is not
a complete import of Lean's standard library.
The statements, `Dvd Nat` and the `Int` declarations were compared by hand with `Init` in the vendored Lean 4.35
source (`src/Init/Data/Int/Basic.lean`). Deliberate differences, none of which changes a statement:

* `Decidable` here is the classic two-constructor inductive (`isFalse`/`isTrue`); 4.35 defines it as a structure with
  `decide` and `reflects_decide` fields.
* Lean writes the `Int` operations (`ediv` and `emod` included) as `match` expressions and `if n % 2 = 0`; here they
  are `Int.rec` and `Bool.rec` applications with the same values, so the kernel can check them without the equation
  compiler.
* `Int.decEq` uses `Int.ctorNat`, `Int.ofNat.inj`, `Int.negSucc.inj` and two cross-constructor lemmas where Lean uses
  `Int.noConfusion`; those five are port helpers, not Lean declarations.
* `Int.min`/`Int.max` are Lean's `Min Int`/`Max Int` (`minOfLe`/`maxOfLe`: `ite (a ≤ b) a b` and `ite (a ≤ b) b a`),
  with the `ite` written as a `Bool.rec` on `Decidable.decide (Int.le a b) (Int.decLe a b)`. Lean calls `Int.gcd`
  and `Int.lcm` `gcd`/`lcm` through the instance layer; they are `Int.gcd`/`Int.lcm` here, same `Nat` results.
* A claim has one type and the coercion layer is not ported, so in an `Int` claim `gcd`/`lcm` (whose Lean results are
  `Nat`) are read as `Int.ofNat (Int.gcd a b)`. `ofNat` is injective and order-preserving, so the verdict is Lean's;
  the certified proposition is `Eq Int (Int.ofNat (Int.gcd …)) (Int.ofNat 3)` rather than Lean's `Nat` equality.
* `a ∣ b` on `Int` is `Int.dvd` (`Exists (fun c => b = a * c)`, Lean's `Dvd` instance). `Int.decDvd` decides it
  through `Int.natAbs_dvd_natAbs : (natAbs a ∣ natAbs b) ↔ (a ∣ b)`, proved here: the forward direction is the four
  constructor cases and the backward direction is `natAbs_mul` under `congrArg`, so nothing about `Int` has to be
  trusted beyond `Nat.decDvd` (`DivMod/Bootstrap.lean:66` proves `natAbs_dvd_natAbs`; Lean obtains
  `dvd_iff_emod_eq_zero` from `mul_ediv_cancel_of_emod_eq_zero` instead, which is also available now that
  `Nat.div`/`Nat.mod` are definitions, but the `natAbs` route needs no div/mod lemmas at all).

**Using it.** `POST /v1/strata/math/evaluate` with `{"operation":"prove","expression":"2^10 = 1024"}` returns
`exact_result` `proved` or `refuted`, plus `metadata.proof_term` and `metadata.kernel_steps`. Both verdicts are backed
by a term the kernel checked (`of_decide_eq_true p inst (Eq.refl true)` for `p`, or for `¬ p`). The claim grammar is closed
arithmetic: numbers, `+ - * / % ^`, `= ≠ < > ≤ ≥ ∣` (ASCII `<= >= != |` also work), `∧ ∨ ¬` with
parentheses, and bounded quantifiers `∀ x < 10, x * x < 100` / `∃ x ≤ 7, x * x = 49` (`forall`/`exists` also work). A
quantifier is expanded into a conjunction or disjunction (at most 256 values each, 4096 cases in all) and the kernel checks
the expanded proposition, which is what `metadata.proposition` shows. The functions `gcd(a, b)`, `lcm(a, b)`, `min(a, b)`, `max(a, b)` and `factorial(n)` (`fact`) are
kernel definitions (`gcd` is computed by the kernel on literals; `lcm` is `a * b / gcd a b` as in Lean). Subtraction truncates at 0 and `/` rounds down, as on Lean's `Nat`.
`coprime(a, b)` is a proposition (not a function) on `Nat`: Lean's `Nat.Coprime a b`, that is `gcd a b = 1`, decided by `Nat.decEq`. In a claim with a negative numeral it is `unsupported_operation`, as `gcd` is not ported for `Int` there.

**A claim is read over `Nat` unless it contains a negative numeral** (`-3`, `(-2)`): then it is read over `Int`, where
`+ - *` are `Int.add/sub/mul`, `/` and `%` are `Int.ediv`/`Int.emod`, `^` is `Int.pow` (the exponent stays a `Nat`),
the relations are `Int.le`/`Int.lt` and the quantifier variables are `Int.ofNat k`. So `3 - 5 = 0` is a `Nat` claim
(true, truncated) and `3 - 5 = -2` is an `Int` claim (true, and its `metadata.proposition` is
`(Eq Int (Int.sub (Int.ofNat 3) (Int.ofNat 5)) (Int.negSucc 1))`).

`/` and `%` on `Int` follow Lean's E-rounding, so the remainder is never negative and `x = y * (x / y) + x % y` holds:
`-12 / 7 = -2` and `-12 % 7 = 2` are proved, `12 % -7 = 5`, and `x / 0 = 0` while `x % 0 = x`. A claim's `∣` is
`Int.dvd` (`-3 ∣ 12`, `3 ∣ -12`, `-5 ∣ 0`, but not `0 ∣ -5`). `gcd`/`lcm` are `Int.gcd`/`Int.lcm` (`gcd(-6, -9) = 3`,
`lcm(-4, 6) = 12`; their Lean results are `Nat`, read through `Int.ofNat`) and `min`/`max` are `Int.min`/`Int.max`
(`min(-3, 5) = -3`, `max(-3, 5) = 5`). (`Nat` claims keep the truncated `Nat.sub` and floor `Nat.div`.) `factorial`
has no `Int` version and is rejected there with a reason rather than read as `Nat`. Anything else (variables,
unbounded quantifiers, unknown functions) returns `unsupported_operation`. In Go: `MathRuntime.ProveWithLean(claim)`
(it replaces the old `VerifyWithLean`; no external Lean binary is used).

Two places use it without being asked:

* `VerifyCalculation` re-derives a CAS result with the kernel when the expression uses only `+ * ^` and parentheses
  over integers (where Lean's `Nat` and the CAS agree). Agreement sets `kernel_certified` and `confidence` 1.0;
  that as `2^(3^2)` = 512, this CAS reads it as 64.
* `InterceptAndVerifyIntent` recognises `example : <claim> := by decide` (also `theorem`/`lemma`, and `rfl`, `norm_num`,
  `simp`, `omega`) in model output and checks the claim. The tactic named is not run; the kernel always checks the claim.

**Measured** (AMD Ryzen 9 9950X3D, `go test -bench . ./pkg/mathruntime/lean`, 500 runs each, one run of the machine):
building the prelude 1.8 ms → 10.7 ms over the port's history (the Int divisibility proofs, the div/mod model, the
`Nat.ble` order lemmas, the order and divisibility lemmas under `Nat.gcd_dvd_left`/`right`, the subtraction layer under
`Nat.dvd_gcd`/`Nat.gcd_le_left`, and the order/multiplication lemmas under `Nat.gcd_comm`/`Nat.gcd_assoc`) up to 10.7 ms, then the rest of the `Gcd`, `Lcm` and `Coprime` files (`prelude_nat_extra*.go`, about 300 more theorems) up to 45 ms, then `Init/Data/Int/Gcd.lean` (`prelude_zint_gcd01–18.go`, about 190 more theorems (counted from `ithm` calls) plus `Int.mul_ediv_cancel_left`, `Int.natAbs_ediv_of_dvd`, `Nat.mul_add_div_of_lt` they needed) up to 56 ms (100 runs of `BenchmarkBuildPrelude`, same machine); checking
`2^10 = 1024` 14 µs; `123456789 * 987654321 = 121932631112635269` 14 µs;
`3 ∣ 12 ∧ 5 < 7 ∧ ¬ (9 = 10)` 43 µs; the `Int` claims `-3 < 2` 17 µs,
`-123456789 * 987654321 = -121932631112635269` 26 µs, `-12 / 7 = -2 ∧ -12 % 7 = 2` 51 µs,
`-3 ∣ 12 ∧ ¬ (-3 ∣ 13)` 50 µs and `gcd(-6, -9) = 3 ∧ min(-3, 5) = -3` 50 µs. An order claim now carries a real
proof term (`Nat.decLe` applies `Nat.le_of_ble_eq_true`, a `Nat.rec` term) instead of an axiom's, and the same
benchmarks on this machine move by up to a third between runs, so treat these as one reading, not a trend; literals
never unfold the div/mod model either way. The literal extension is why: `2^64 = 18446744073709551616` checks inside a 1000-step
budget, which unary `Nat.rec` could not, and the same holds for `Int` (89 kernel steps for
`-2^64 = 18446744073709551616`, 95 for the big product, 89 and 97 for `-123456789 / 7` and `-123456789 % 7`).

**Tests.** The kernel accepts a claim exactly when plain big-integer evaluation says it is true, on 1,500 random
`Nat` claims with `+ - * / % ^`, `∣`, `∧`, `∨`, `¬` and on 400 random `Int` claims with `+ - * / % ^`, `∣`,
`gcd`/`lcm`/`min`/`max` and the six relations (both directions; the oracle divides with E-rounding, so a mismatch in
the rounding would fail it). The generated recursors of `Nat`, `Bool`, `False`, `Eq`, `And`, `Or`, `Exists`,
`Decidable` and `Nat.le` equal Lean's types. Ill-formed terms, a non-positive inductive, a field in too high a
universe, an ill-typed definition and a theorem about data are rejected. Nothing has been compared against a real
Lean build: the vendored source was configured but not built.
