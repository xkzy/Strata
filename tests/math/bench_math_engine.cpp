// tests/math/bench_math_engine.cpp - latency of the math runtime: cold vs cache hit, big numbers, sequential vs parallel batches
#include "strata/math/math_runtime.hpp"

#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

using namespace strata::math;
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

int main() {
    std::printf("hardware threads: %u\n", std::thread::hardware_concurrency());
    struct Case { const char* name; MathOperation op; const char* expr; };
    const Case cases[] = {
        {"exact arithmetic", MathOperation::kEvaluate, "(12345 * 6789 + 1) / 3 - 7^20"},
        {"2^10000", MathOperation::kEvaluate, "2^10000"},
        {"1000!", MathOperation::kEvaluate, "1000!"},
        {"expand (x+1)^40", MathOperation::kExpand, "(x+1)^40"},
        {"factor x^24-1", MathOperation::kFactor, "x^24 - 1"},
        {"diff", MathOperation::kDifferentiate, "x^3*Sin(x)*Exp(x^2)"},
        {"integrate", MathOperation::kIntegrate, "x^3*Exp(x)"},
        {"solve cubic", MathOperation::kSolve, "x^3 - 6*x^2 + 11*x - 6 == 0"},
        {"det 6x6", MathOperation::kEvaluate, "Det({{2,1,0,0,0,1},{1,3,1,0,0,0},{0,1,4,1,0,0},{0,0,1,5,1,0},{0,0,0,1,6,1},{1,0,0,0,1,7}})"},
        {"FactorInteger", MathOperation::kEvaluate, "FactorInteger(600851475143)"},
    };
    std::printf("%-20s %12s %12s %10s\n", "operation", "cold (ms)", "cache hit", "speedup");
    for (const auto& c : cases) {
        MathRuntime rt;
        MathRequest r; r.expression = c.expr; r.operation = c.op; r.variable = "x";
        auto t = Clock::now();
        auto first = rt.process_request(r);
        const double cold = ms_since(t);
        double warm = 0;
        const int reps = 2000;
        t = Clock::now();
        for (int i = 0; i < reps; ++i) rt.process_request(r);
        warm = ms_since(t) / reps;
        std::printf("%-20s %12.4f %12.5f %9.0fx %s\n", c.name, cold, warm, warm > 0 ? cold / warm : 0.0, first.status == MathStatus::kSuccess ? "" : "(not successful)");
    }

    // a batch of 400 distinct requests
    std::vector<MathRequest> reqs;
    for (int i = 0; i < 100; ++i)
        for (auto op : {MathOperation::kExpand, MathOperation::kFactor, MathOperation::kDifferentiate, MathOperation::kIntegrate}) {
            MathRequest r; r.operation = op; r.variable = "x";
            r.expression = "(x + " + std::to_string(i + 1) + ")^" + std::to_string(8 + i % 12) + " + " + std::to_string(i * 7 + 3) + "*x^3 - 5";
            reqs.push_back(r);
        }
    for (size_t threads : {1u, 2u, 4u, 8u}) {
        MathRuntime rt;
        auto t = Clock::now();
        auto out = rt.process_batch(reqs, threads);
        const double ms = ms_since(t);
        size_t ok = 0;
        for (auto& o : out) ok += o.status == MathStatus::kSuccess;
        std::printf("batch of %zu distinct requests, %zu thread(s): %9.1f ms  (%zu ok)\n", reqs.size(), threads, ms, ok);
    }
    return 0;
}
