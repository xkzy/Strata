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
     Fast Numeric    Mathics-Core       SageMath      Future CAS
       Backend          Backend          Backend
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
