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

// Unified CAS backend: merges Mathics3 (Wolfram Language compatible) and SageMath (Python-style)
// execution and syntax translation into a single unified CAS subsystem on top of the native Strata CAS kernel.
class UnifiedCasBackend : public IMathBackend {
public:
    explicit UnifiedCasBackend(bool mock_mode = false);
    ~UnifiedCasBackend() override = default;

    MathBackendType backend_type() const override { return MathBackendType::kUnifiedCAS; }
    std::string name() const override { return "StrataCAS"; }
    std::string version() const override { return "1.0.0"; }
    bool is_available() const override { return available_; }

    void set_available(bool avail) { available_ = avail; }

    bool supports_operation(MathOperation op, MathMode mode) const override;
    MathResult execute(const MathRequest& request) override;

    // Translates / normalizes either SageMath Python expressions (dot methods, var/ring declarations,
    // matrix/vector syntax) or Wolfram/Mathics expressions into canonical CAS statements.
    static std::string translate_syntax(const std::string& expr);

    // Compatibility alias for Sage syntax translation
    static std::string translate_sage_syntax(const std::string& sage_expr) {
        return translate_syntax(sage_expr);
    }

private:
    bool available_ = true;
    bool mock_mode_ = false;
};

// Seamless aliases for backwards compatibility
using MathicsBackend = UnifiedCasBackend;
using SageBackend = UnifiedCasBackend;

} // namespace strata::math
