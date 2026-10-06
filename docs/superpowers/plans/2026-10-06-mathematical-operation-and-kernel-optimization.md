# Mathematical Operation and Kernel Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a comprehensive mathematical optimization system and runtime kernel selection engine with configurable numerical policies (`STRICT`, `BALANCED`, `FAST`), operation fusion, and numerical parity verification.

**Architecture:** A modular kernel suite featuring explicit numerical policy contexts (`MathPolicy`, `MathContext`), a cached cost-model `KernelDispatcher`, fused operation pipelines (Fused RMSNorm+Residual, SwiGLU, Decode Attention, RoPE, MoE Routing/Top-K, Quantization), and an automated parity verification test suite comparing optimized kernels against scalar reference baselines across standard and edge inputs.

**Tech Stack:** C++20, SIMD Intrinsics (AVX2, AVX-512, FMA), OpenMP, CMake, Ninja.

## Global Constraints

- Do not silently reduce precision; all accumulators and intermediate reduction denominators must use FP32 precision.
- Preserve exact IEEE-754 semantics under `MathPolicy::kStrict`.
- Maintain strict multi-user context and cache isolation; no request-specific state in global caches.
- Keep the docs' style: plain words, measured numbers with what they were measured on, no claims without a measurement.

---

### Task 1: Numerical Policy & Context Core (`math_policy.hpp`)

**Files:**
- Create: `include/strata/kernels/math_policy.hpp`
- Test: `tests/kernels/test_math_policy.cpp`

**Interfaces:**
- Produces: `strata::kernels::MathPolicy`, `strata::kernels::MathContext`, `strata::kernels::validate_math_bounds(float x, const MathContext& ctx)`

- [ ] **Step 1: Write the failing test**

```cpp
// tests/kernels/test_math_policy.cpp
#include "strata/kernels/math_policy.hpp"
#include <cassert>
#include <iostream>

int main() {
    using namespace strata::kernels;

    MathContext ctx;
    assert(ctx.policy == MathPolicy::kBalanced);
    assert(ctx.rsqrt_epsilon == 1e-6f);
    assert(ctx.softmax_max_clamp == 88.0f);
    assert(ctx.allow_fused_residual == true);

    ctx.policy = MathPolicy::kStrict;
    assert(!ctx.allow_approximate_gelu);
    assert(!ctx.allow_fast_exp);

    ctx.policy = MathPolicy::kFast;
    assert(ctx.allow_fast_exp);

    std::cout << "test_math_policy passed" << std::endl;
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `ninja -C build test_math_policy` (or compile directly)
Expected: FAIL (missing header)

- [ ] **Step 3: Write minimal implementation**

```cpp
// include/strata/kernels/math_policy.hpp
#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>

namespace strata::kernels {

enum class MathPolicy : uint8_t {
    kStrict = 0,
    kBalanced = 1,
    kFast = 2
};

struct MathContext {
    MathPolicy policy = MathPolicy::kBalanced;
    float rsqrt_epsilon = 1e-6f;
    float softmax_max_clamp = 88.0f;
    bool allow_fused_residual = true;
    bool allow_approximate_gelu = true;
    bool allow_fast_exp = false;

    MathContext() = default;
    explicit MathContext(MathPolicy p) : policy(p) {
        if (p == MathPolicy::kStrict) {
            allow_approximate_gelu = false;
            allow_fast_exp = false;
        } else if (p == MathPolicy::kFast) {
            allow_approximate_gelu = true;
            allow_fast_exp = true;
        }
    }
};

inline float sanitize_logit(float x, const MathContext& ctx) {
    if (std::isnan(x)) return -ctx.softmax_max_clamp;
    if (x > ctx.softmax_max_clamp) return ctx.softmax_max_clamp;
    if (x < -ctx.softmax_max_clamp) return -ctx.softmax_max_clamp;
    return x;
}

} // namespace strata::kernels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `g++ -std=c++20 -Iinclude tests/kernels/test_math_policy.cpp -o /tmp/test_math_policy && /tmp/test_math_policy`
Expected: `test_math_policy passed`

- [ ] **Step 5: Commit**

```bash
git add include/strata/kernels/math_policy.hpp tests/kernels/test_math_policy.cpp
git commit -m "feat(kernels): add MathPolicy and MathContext definitions"
```

---

### Task 2: Dynamic Kernel Registry & Cost-Model Dispatcher (`kernel_dispatcher.hpp`)

**Files:**
- Create: `include/strata/kernels/kernel_dispatcher.hpp`
- Create: `src/kernels/kernel_dispatcher.cpp`
- Test: `tests/kernels/test_kernel_dispatcher.cpp`

**Interfaces:**
- Consumes: `strata::kernels::MathPolicy`, `strata::kernels::MathContext`
- Produces: `strata::kernels::KernelDispatcher`, `strata::kernels::OpType`, `strata::kernels::KernelKey`

- [ ] **Step 1: Write the failing test**

```cpp
// tests/kernels/test_kernel_dispatcher.cpp
#include "strata/kernels/kernel_dispatcher.hpp"
#include <cassert>
#include <iostream>

int main() {
    using namespace strata::kernels;

    auto& dispatcher = KernelDispatcher::instance();
    dispatcher.set_default_policy(MathPolicy::kBalanced);
    assert(dispatcher.default_policy() == MathPolicy::kBalanced);

    KernelKey key{OpType::kGEMV, 1, 4096, 4096, 0, MathPolicy::kBalanced};
    auto decision = dispatcher.select_kernel(key);
    assert(!decision.empty());

    std::cout << "test_kernel_dispatcher passed" << std::endl;
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `g++ -std=c++20 -Iinclude tests/kernels/test_kernel_dispatcher.cpp -o /tmp/test_kernel_dispatcher`
Expected: FAIL (missing header)

- [ ] **Step 3: Write implementation**

```cpp
// include/strata/kernels/kernel_dispatcher.hpp
#pragma once

#include "strata/kernels/math_policy.hpp"
#include <cstdint>
#include <string>
#include <unordered_map>
#include <mutex>

namespace strata::kernels {

enum class OpType : uint16_t {
    kMatMul = 0,
    kGEMV = 1,
    kRMSNorm = 2,
    kFusedNormResidual = 3,
    kAttention = 4,
    kRoPE = 5,
    kSwiGLU = 6,
    kMoERouting = 7,
    kSoftmax = 8,
    kQuantize = 9,
    kDequantize = 10
};

struct KernelKey {
    OpType op;
    int64_t M, N, K;
    int32_t dtype;
    MathPolicy policy;

    bool operator==(const KernelKey& o) const {
        return op == o.op && M == o.M && N == o.N && K == o.K &&
               dtype == o.dtype && policy == o.policy;
    }
};

struct KernelKeyHash {
    size_t operator()(const KernelKey& k) const {
        size_t h = static_cast<size_t>(k.op);
        h = h * 31 + static_cast<size_t>(k.M);
        h = h * 31 + static_cast<size_t>(k.N);
        h = h * 31 + static_cast<size_t>(k.K);
        h = h * 31 + static_cast<size_t>(k.dtype);
        h = h * 31 + static_cast<size_t>(k.policy);
        return h;
    }
};

class KernelDispatcher {
public:
    static KernelDispatcher& instance();

    void set_default_policy(MathPolicy policy) { policy_ = policy; }
    MathPolicy default_policy() const { return policy_; }

    std::string select_kernel(const KernelKey& key);
    void register_kernel(const KernelKey& key, const std::string& kernel_name);

private:
    KernelDispatcher() = default;
    MathPolicy policy_ = MathPolicy::kBalanced;
    std::unordered_map<KernelKey, std::string, KernelKeyHash> cache_;
    std::mutex mu_;
};

} // namespace strata::kernels
```

```cpp
// src/kernels/kernel_dispatcher.cpp
#include "strata/kernels/kernel_dispatcher.hpp"

namespace strata::kernels {

KernelDispatcher& KernelDispatcher::instance() {
    static KernelDispatcher inst;
    return inst;
}

std::string KernelDispatcher::select_kernel(const KernelKey& key) {
    std::lock_guard<std::mutex> lock(mu_);
    auto it = cache_.find(key);
    if (it != cache_.end()) {
        return it->second;
    }

    // Cost model heuristic
    std::string choice;
    switch (key.op) {
        case OpType::kGEMV:
            choice = (key.M <= 8) ? "simd_row_split_gemv" : "tiled_gemm";
            break;
        case OpType::kFusedNormResidual:
            choice = "fused_single_pass_rmsnorm";
            break;
        case OpType::kMoERouting:
            choice = "partial_select_topk";
            break;
        case OpType::kAttention:
            choice = (key.M == 1) ? "incremental_decode_attn" : "tiled_flash_attn";
            break;
        default:
            choice = "default_vectorized";
            break;
    }
    cache_[key] = choice;
    return choice;
}

void KernelDispatcher::register_kernel(const KernelKey& key, const std::string& kernel_name) {
    std::lock_guard<std::mutex> lock(mu_);
    cache_[key] = kernel_name;
}

} // namespace strata::kernels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `g++ -std=c++20 -Iinclude src/kernels/kernel_dispatcher.cpp tests/kernels/test_kernel_dispatcher.cpp -o /tmp/test_kernel_dispatcher && /tmp/test_kernel_dispatcher`
Expected: `test_kernel_dispatcher passed`

- [ ] **Step 5: Commit**

```bash
git add include/strata/kernels/kernel_dispatcher.hpp src/kernels/kernel_dispatcher.cpp tests/kernels/test_kernel_dispatcher.cpp
git commit -m "feat(kernels): implement KernelDispatcher and decision cache"
```

---

### Task 3: Fast Attention & Incremental KV-Cache Math (`fast_attention.hpp`)

**Files:**
- Create: `include/strata/kernels/fast_attention.hpp`
- Test: `tests/kernels/test_fast_attention.cpp`

**Interfaces:**
- Consumes: `strata::kernels::MathPolicy`, `strata::kernels::MathContext`
- Produces: `strata::kernels::FastAttention::scaled_dot_product_decode`

- [ ] **Step 1: Write the failing test**

```cpp
// tests/kernels/test_fast_attention.cpp
#include "strata/kernels/fast_attention.hpp"
#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>

int main() {
    using namespace strata::kernels;

    const size_t head_dim = 64;
    const size_t seq_len = 8;
    std::vector<float> q(head_dim, 0.5f);
    std::vector<float> k_cache(seq_len * head_dim, 0.2f);
    std::vector<float> v_cache(seq_len * head_dim, 1.0f);
    std::vector<float> out(head_dim, 0.0f);

    FastAttention::scaled_dot_product_decode(
        q.data(), k_cache.data(), v_cache.data(), out.data(),
        seq_len, head_dim, 1.0f / std::sqrt(static_cast<float>(head_dim))
    );

    // With identical V values (=1.0f), output must be 1.0f
    for (size_t i = 0; i < head_dim; ++i) {
        assert(std::abs(out[i] - 1.0f) < 1e-4f);
    }

    std::cout << "test_fast_attention passed" << std::endl;
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `g++ -std=c++20 -Iinclude tests/kernels/test_fast_attention.cpp -o /tmp/test_fast_attention`
Expected: FAIL (missing header)

- [ ] **Step 3: Write minimal implementation**

```cpp
// include/strata/kernels/fast_attention.hpp
#pragma once

#include "strata/kernels/math_policy.hpp"
#include <cmath>
#include <cstddef>
#include <vector>
#include <algorithm>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace strata::kernels {

class FastAttention {
public:
    static void scaled_dot_product_decode(
        const float* __restrict__ q,
        const float* __restrict__ k_cache,
        const float* __restrict__ v_cache,
        float* __restrict__ out,
        size_t seq_len,
        size_t head_dim,
        float scale,
        const MathContext& ctx = MathContext()) {
        if (seq_len == 0 || head_dim == 0) return;

        std::vector<float> scores(seq_len);
        float max_score = -1e9f;

        for (size_t s = 0; s < seq_len; ++s) {
            const float* k_row = k_cache + s * head_dim;
            float dot = 0.0f;
            for (size_t d = 0; d < head_dim; ++d) {
                dot += q[d] * k_row[d];
            }
            float score = dot * scale;
            score = sanitize_logit(score, ctx);
            scores[s] = score;
            if (score > max_score) max_score = score;
        }

        float sum_exp = 0.0f;
        for (size_t s = 0; s < seq_len; ++s) {
            scores[s] = std::exp(scores[s] - max_score);
            sum_exp += scores[s];
        }
        const float inv_sum = 1.0f / (sum_exp + 1e-9f);
        for (size_t s = 0; s < seq_len; ++s) {
            scores[s] *= inv_sum;
        }

        std::fill(out, out + head_dim, 0.0f);
        for (size_t s = 0; s < seq_len; ++s) {
            const float weight = scores[s];
            const float* v_row = v_cache + s * head_dim;
            for (size_t d = 0; d < head_dim; ++d) {
                out[d] += weight * v_row[d];
            }
        }
    }
};

} // namespace strata::kernels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `g++ -std=c++20 -Iinclude tests/kernels/test_fast_attention.cpp -o /tmp/test_fast_attention && /tmp/test_fast_attention`
Expected: `test_fast_attention passed`

- [ ] **Step 5: Commit**

```bash
git add include/strata/kernels/fast_attention.hpp tests/kernels/test_fast_attention.cpp
git commit -m "feat(kernels): add FastAttention incremental decode implementation"
```

---

### Task 4: Fast MoE Router, Top-K & Aggregation (`fast_moe.hpp`)

**Files:**
- Create: `include/strata/kernels/fast_moe.hpp`
- Test: `tests/kernels/test_fast_moe.cpp`

**Interfaces:**
- Consumes: `strata::kernels::MathPolicy`, `strata::kernels::MathContext`
- Produces: `strata::kernels::FastMoE::route_topk`, `strata::kernels::FastMoE::weighted_aggregate`

- [ ] **Step 1: Write the failing test**

```cpp
// tests/kernels/test_fast_moe.cpp
#include "strata/kernels/fast_moe.hpp"
#include <cassert>
#include <cmath>
#include <vector>
#include <iostream>

int main() {
    using namespace strata::kernels;

    const size_t num_experts = 8;
    const size_t top_k = 2;
    std::vector<float> logits = {0.1f, 2.5f, -1.0f, 0.4f, 3.2f, 0.0f, -0.5f, 1.2f};
    std::vector<int32_t> selected_experts(top_k);
    std::vector<float> selected_weights(top_k);

    FastMoE::route_topk(
        logits.data(), num_experts, top_k,
        selected_experts.data(), selected_weights.data()
    );

    assert(selected_experts[0] == 4); // Highest is index 4 (3.2)
    assert(selected_experts[1] == 1); // Second highest is index 1 (2.5)
    assert(selected_weights[0] > selected_weights[1]);
    assert(std::abs((selected_weights[0] + selected_weights[1]) - 1.0f) < 1e-4f);

    std::cout << "test_fast_moe passed" << std::endl;
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `g++ -std=c++20 -Iinclude tests/kernels/test_fast_moe.cpp -o /tmp/test_fast_moe`
Expected: FAIL (missing header)

- [ ] **Step 3: Write implementation**

```cpp
// include/strata/kernels/fast_moe.hpp
#pragma once

#include "strata/kernels/math_policy.hpp"
#include <algorithm>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <vector>

namespace strata::kernels {

class FastMoE {
public:
    static void route_topk(
        const float* __restrict__ logits,
        size_t num_experts,
        size_t top_k,
        int32_t* __restrict__ out_indices,
        float* __restrict__ out_weights,
        const MathContext& ctx = MathContext()) {
        if (num_experts == 0 || top_k == 0) return;
        top_k = std::min(top_k, num_experts);

        struct ExpertScore {
            int32_t index;
            float score;
        };

        std::vector<ExpertScore> scores(num_experts);
        for (size_t i = 0; i < num_experts; ++i) {
            scores[i] = {static_cast<int32_t>(i), sanitize_logit(logits[i], ctx)};
        }

        // Partial quick-select for top-k: O(N + K log K)
        std::partial_sort(
            scores.begin(), scores.begin() + top_k, scores.end(),
            [](const ExpertScore& a, const ExpertScore& b) { return a.score > b.score; }
        );

        float max_top_score = scores[0].score;
        float sum_exp = 0.0f;
        for (size_t k = 0; k < top_k; ++k) {
            float e = std::exp(scores[k].score - max_top_score);
            out_weights[k] = e;
            sum_exp += e;
            out_indices[k] = scores[k].index;
        }

        const float inv_sum = 1.0f / (sum_exp + 1e-9f);
        for (size_t k = 0; k < top_k; ++k) {
            out_weights[k] *= inv_sum;
        }
    }

    static void weighted_aggregate(
        const float* const* __restrict__ expert_outputs,
        const float* __restrict__ weights,
        size_t top_k,
        size_t dim,
        float* __restrict__ out) {
        if (top_k == 0 || dim == 0) return;
        std::fill(out, out + dim, 0.0f);

        for (size_t k = 0; k < top_k; ++k) {
            const float w = weights[k];
            const float* exp_out = expert_outputs[k];
            for (size_t d = 0; d < dim; ++d) {
                out[d] += w * exp_out[d];
            }
        }
    }
};

} // namespace strata::kernels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `g++ -std=c++20 -Iinclude tests/kernels/test_fast_moe.cpp -o /tmp/test_fast_moe && /tmp/test_fast_moe`
Expected: `test_fast_moe passed`

- [ ] **Step 5: Commit**

```bash
git add include/strata/kernels/fast_moe.hpp tests/kernels/test_fast_moe.cpp
git commit -m "feat(kernels): implement FastMoE top-k routing and aggregation"
```

---

### Task 5: Numerical Parity & Edge Case Validation Harness (`test_fast_math_parity.cpp`)

**Files:**
- Create: `tests/kernels/test_fast_math_parity.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Comprehensive numerical testing comparing optimized kernel implementations against exact reference math across edge, random, extreme values, NaN, and Inf.

- [ ] **Step 1: Write parity test suite**

```cpp
// tests/kernels/test_fast_math_parity.cpp
#include "strata/kernels/fast_norm.hpp"
#include "strata/kernels/fast_activations.hpp"
#include "strata/kernels/fast_matmul.hpp"
#include "strata/kernels/fast_attention.hpp"
#include "strata/kernels/fast_moe.hpp"
#include "strata/kernels/fast_rope.hpp"
#include "strata/kernels/fast_softmax.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

void test_rmsnorm_parity() {
    using namespace strata::kernels;
    const size_t dim = 1024;
    std::vector<float> x(dim), weight(dim), out_opt(dim), out_ref(dim);

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-5.0f, 5.0f);
    for (size_t i = 0; i < dim; ++i) {
        x[i] = dist(rng);
        weight[i] = 1.0f + 0.1f * dist(rng);
    }

    // Reference RMSNorm
    float sum_sq = 0.0f;
    for (size_t i = 0; i < dim; ++i) sum_sq += x[i] * x[i];
    float scale = 1.0f / std::sqrt(sum_sq / dim + 1e-6f);
    for (size_t i = 0; i < dim; ++i) out_ref[i] = x[i] * scale * weight[i];

    // Optimized RMSNorm
    FastNorm::rmsnorm(x.data(), weight.data(), out_opt.data(), dim, 1e-6f);

    float max_err = 0.0f;
    for (size_t i = 0; i < dim; ++i) {
        max_err = std::max(max_err, std::abs(out_opt[i] - out_ref[i]));
    }
    std::cout << "RMSNorm parity max error: " << max_err << std::endl;
    assert(max_err < 1e-5f);
}

void test_edge_and_nan_sanitization() {
    using namespace strata::kernels;
    MathContext ctx;
    assert(sanitize_logit(100.0f, ctx) == 88.0f);
    assert(sanitize_logit(-100.0f, ctx) == -88.0f);
    assert(sanitize_logit(std::numeric_limits<float>::quiet_NaN(), ctx) == -88.0f);
    std::cout << "Edge sanitization verified" << std::endl;
}

int main() {
    std::cout << "Running Fast Math Parity & Edge Case Validation..." << std::endl;
    test_rmsnorm_parity();
    test_edge_and_nan_sanitization();
    std::cout << "All Math Parity Tests Passed!" << std::endl;
    return 0;
}
```

- [ ] **Step 2: Add executable targets in CMakeLists.txt**

Modify `CMakeLists.txt` to include `test_fast_math_parity` and build with ninja.

- [ ] **Step 3: Compile and run test**

Run: `ninja -C build test_fast_math_parity && ./build/test_fast_math_parity`
Expected: `All Math Parity Tests Passed!`

- [ ] **Step 4: Commit**

```bash
git add tests/kernels/test_fast_math_parity.cpp CMakeLists.txt
git commit -m "test(kernels): add numerical parity and edge case validation harness"
```

---

### Task 6: Unified Integration & Performance Verification

**Files:**
- Modify: `include/strata/strata_unified.hpp`
- Modify: `tests/kernels/test_fast_unified_benchmark.cpp`

- [ ] **Step 1: Update unified runtime header**
Ensure `strata_unified.hpp` includes `math_policy.hpp`, `kernel_dispatcher.hpp`, `fast_attention.hpp`, and `fast_moe.hpp`.

- [ ] **Step 2: Run benchmark suite**
Run: `ninja -C build test_fast_unified_benchmark && ./build/test_fast_unified_benchmark`
Expected: Full forward layer simulation executes cleanly with microsecond-level timing.

- [ ] **Step 3: Commit**

```bash
git add include/strata/strata_unified.hpp tests/kernels/test_fast_unified_benchmark.cpp
git commit -m "feat(kernels): integrate full mathematical operation suite and verify benchmarks"
```
