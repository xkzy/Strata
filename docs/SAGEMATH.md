# SageMath Execution Backend

Strata accepts **SageMath-style syntax** in the Math Runtime and runs it on its own native CAS (`src/math/cas`). It does not run SageMath or Python, and it is not a port of Sage: it is a Sage-syntax front end over the operations the CAS implements.

## Overview

The SageMath backend (`SageBackend`) provides Python- and SageMath-compatible execution:
- **Syntax Compatibility**:
  - Variable declarations: `var('x')`, `var('x y z')`, `x, y = var('x y')`
  - Polynomial rings: `R.<x> = PolynomialRing(QQ)`
  - Dot-method chaining: `(x^3 - 8).factor()`, `f.diff(x)`, `f.integrate(x)`, `f.roots(x)`, `f.expand()`, `f.simplify()`, `M.det()`, `M.inverse()`, `M.transpose()`
  - Matrix constructors and brackets: `matrix([[1, 2], [3, 4]])`, `[[1, 2], [3, 4]]`
  - Python power operators: `**` and `^`

- **Calculus Operations**:
  - `diff(f, x, [order])`, `derivative(f, x)`
  - `integrate(f, x)`, `integral(f, x, a, b)`
  - `limit(f, x=a, [dir])`
  - `taylor(f, x, a, n)`

- **Algebra & Equation Solving**:
  - `factor(expr)`: Exact polynomial factorization over $\mathbb{Q}$ and integers
  - `expand(expr)`: Full algebraic expansion
  - `simplify(expr)`: Rational function canonicalization
  - `solve(eq, x)`: Complete algebraic equation solution sets
  - `roots(expr, x)`: Solution sets formatted as Python equations `[x == a, x == b]`

- **Linear Algebra**:
  - `det(M)`, `M.det()`: Exact determinant calculation
  - `inverse(M)`, `M.inverse()`: Exact matrix inversion
  - `transpose(M)`, `M.transpose()`: Exact matrix transposition

- **Computational Number Theory**:
  - `is_prime(n)`: Miller-Rabin deterministic / Baillie-PSW primality test (`True` / `False`)
  - `euler_phi(n)`: Euler's totient function $\phi(n)$
  - `fibonacci(n)`: Exact arbitrary-precision Fibonacci sequence calculation
  - `lucas_number(n)`: Lucas numbers
  - `divisors(n)`, `sigma(n, [k])`: Exact divisor analysis
  - `power_mod(a, b, m)`: Fast modular exponentiation $a^b \pmod m$
  - `xgcd(a, b)`: Extended Euclidean algorithm (Bézout coefficients)
  - `kronecker(a, b)`, `jacobi_symbol(a, b)`: Jacobi/Kronecker symbol
  - `moebius(n)`: Möbius inversion function $\mu(n)$
  - `binomial(n, k)`, `factorial(n)`, `gcd(a, b)`, `lcm(a, b)`

## Limits

- Input is limited to 4096 characters; longer input is refused (`invalid_expression`), not truncated.
- Only the call shapes listed above are translated. A one-sided limit (`limit(f, x=0, dir='plus')`) is refused, not answered as a two-sided one.
- Sage's `sigma(n, k)` is translated to the CAS's `DivisorSigma(k, n)` (argument order differs).
- No Python: statements, assignments other than `var(...)` and ring declarations, loops and imports are not supported and fail to parse.
- Everything else Sage offers (groups, rings other than `QQ[x]`, plotting, ...) is not implemented.

## Architecture

```
                    Strata Math Runtime
                             │
            ┌────────────────┼────────────────┐
            │                │                │
       Fast Numeric      Mathics3          SageMath
      Native Backend    CAS Backend     Backend Engine
            │                │                │
            └────────────────┼────────────────┘
                             │
                    Deterministic Cache
```

## Verification

Run the test suite:
```bash
./build/test_sage_backend
go test -v ./pkg/mathruntime
```
