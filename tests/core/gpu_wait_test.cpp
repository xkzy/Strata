// tests/core/gpu_wait_test.cpp - the bounded GPU wait: ready, failed, and never-finishing work, with no GPU.
#include "strata/core/gpu_wait.hpp"

#include <chrono>
#include <cstdio>
#include <string>

using namespace strata::core;
using namespace std::chrono_literals;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-70s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    int polls = 0;
    check(wait_bounded([&] { return ++polls < 1000 ? PollState::Pending : PollState::Ready; }, 5s) == WaitResult::Ready,
          "work that finishes after many polls is Ready");
    check(wait_bounded([] { return PollState::Ready; }, 5s) == WaitResult::Ready, "work that is already done is Ready");
    check(wait_bounded([&] { return ++polls > 2000 ? PollState::Failed : PollState::Pending; }, 5s) == WaitResult::Failed,
          "a device error ends the wait as Failed, not as a hang");
    const auto t0 = std::chrono::steady_clock::now();
    const WaitResult r = wait_bounded([] { return PollState::Pending; }, 100ms);
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - t0).count();
    check(r == WaitResult::TimedOut, "work that never finishes ends as TimedOut");
    check(ms >= 100 && ms < 2000, "...after about the limit (" + std::to_string(ms) + " ms)");
    std::printf("%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
