// include/strata/core/gpu_wait.hpp - a bounded wait for GPU work, with no CUDA in it (so it is tested without a GPU).
// A GPU that stopped answering must not leave a pool thread spinning forever: the wait ends with TimedOut, and the
// caller treats that device as unhealthy (stops submitting to it, never frees what the stuck work may still touch).
#pragma once

#include <chrono>
#include <cstdint>
#include <cstdlib>

namespace strata::core {

enum class PollState { Ready, Pending, Failed };
enum class WaitResult { Ready, Failed, TimedOut };

/// The longest a single GPU wait may take: STRATA_GPU_WAIT_S seconds (default 60; 0 = no limit, for debugging only).
inline std::chrono::milliseconds gpu_wait_limit() {
    static const std::chrono::milliseconds v = [] {
        const char* e = std::getenv("STRATA_GPU_WAIT_S");
        const long s = e != nullptr ? std::atol(e) : 60;
        return std::chrono::milliseconds(s < 0 ? 60000 : s * 1000);
    }();
    return v;
}

/// Spins on `poll` (low latency for a decode step), reading the clock only every 256 polls.  limit 0 = unbounded.
template <class Poll>
WaitResult wait_bounded(Poll&& poll, std::chrono::milliseconds limit) {
    using Clock = std::chrono::steady_clock;
    const Clock::time_point t0 = Clock::now();
    for (uint32_t n = 1;; ++n) {
        switch (poll()) {
        case PollState::Ready: return WaitResult::Ready;
        case PollState::Failed: return WaitResult::Failed;
        case PollState::Pending: break;
        }
        if (limit.count() > 0 && (n & 255u) == 0 && Clock::now() - t0 > limit) return WaitResult::TimedOut;
    }
}

}  // namespace strata::core
