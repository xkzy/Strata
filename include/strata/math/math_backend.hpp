// include/strata/math/math_backend.hpp - Mathematical Execution Backends
//
// Defines abstract backend interface and native fast numeric and CAS backends.
#pragma once

#include "strata/math/math_types.hpp"

#include <memory>
#include <string>
#include <vector>

namespace strata::math {

class IMathBackend {
public:
    virtual ~IMathBackend() = default;

    virtual MathBackendType backend_type() const = 0;
    virtual std::string name() const = 0;
    virtual std::string version() const = 0;
    virtual bool is_available() const = 0;

    virtual bool supports_operation(MathOperation op, MathMode mode) const = 0;
    virtual MathResult execute(const MathRequest& request) = 0;
};

// High-speed native arithmetic backend for exact rational calculations, powers, roots,
// factorials, determinants, and standard numerical operations.
class FastNumericBackend : public IMathBackend {
public:
    FastNumericBackend();
    ~FastNumericBackend() override = default;

    MathBackendType backend_type() const override { return MathBackendType::kFastNumeric; }
    std::string name() const override { return "FastNumericNative"; }
    std::string version() const override { return "1.0.0"; }
    bool is_available() const override { return true; }

    bool supports_operation(MathOperation op, MathMode mode) const override;
    MathResult execute(const MathRequest& request) override;

private:
    MathResult execute_checked(const MathRequest& request);   // throws on int64 overflow
    struct Rational {
        int64_t num = 0;
        int64_t den = 1;

        Rational() = default;
        Rational(int64_t n, int64_t d = 1);
        void reduce();
        Rational operator+(const Rational& o) const;
        Rational operator-(const Rational& o) const;
        Rational operator*(const Rational& o) const;
        Rational operator/(const Rational& o) const;
        std::string to_string() const;
        double to_double() const;
    };

    bool evaluate_arithmetic_exact(const std::string& expr, Rational& out_rat, double& out_dbl, std::string& err);
    bool evaluate_determinant(const std::string& expr, double& out_det, std::string& err);
    bool evaluate_combinatorics(const std::string& expr, int64_t& out_val, std::string& err);
};

// Symbolic backend: the native C++ port of the Mathics3 evaluation model (src/math/cas, GPL-3.0-or-later).
// The class keeps its historical name because the router and tests refer to it. Every operation either returns a
// computed result or an error status (kUnsupportedOperation, kResourceLimitExceeded, ...): it never echoes the
// input back as a "result".
class MathicsBackend : public IMathBackend {
public:
    explicit MathicsBackend(bool mock_mode = false);
    ~MathicsBackend() override = default;

    MathBackendType backend_type() const override { return MathBackendType::kMathics; }
    std::string name() const override { return "StrataCAS"; }
    std::string version() const override { return "0.1.0"; }
    bool is_available() const override { return available_; }

    void set_available(bool avail) { available_ = avail; }

    bool supports_operation(MathOperation op, MathMode mode) const override;
    MathResult execute(const MathRequest& request) override;

private:
    bool available_ = true;
    bool mock_mode_ = false;
};

// SageMath backend: provides native execution and translation of SageMath syntax,
// functions, and conventions (diff, integrate/integral, taylor, factor, expand,
// simplify, solve, roots, matrix, det, is_prime, euler_phi, fibonacci, etc.).
class SageBackend : public IMathBackend {
public:
    explicit SageBackend(bool mock_mode = false);
    ~SageBackend() override = default;

    MathBackendType backend_type() const override { return MathBackendType::kSageMath; }
    std::string name() const override { return "SageMath"; }
    std::string version() const override { return "10.4"; }
    bool is_available() const override { return available_; }

    void set_available(bool avail) { available_ = avail; }

    bool supports_operation(MathOperation op, MathMode mode) const override;
    MathResult execute(const MathRequest& request) override;

    // Translates SageMath Python expressions, dot methods, and declarations into standard CAS statements
    static std::string translate_sage_syntax(const std::string& sage_expr);

private:
    bool available_ = true;
    bool mock_mode_ = false;
};

} // namespace strata::math
