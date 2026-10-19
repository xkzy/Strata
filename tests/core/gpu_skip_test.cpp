// tests/core/gpu_skip_test.cpp - the sentence a run prints when it goes on without a peer or helper GPU the driver no
// longer lists (a card that fell off the bus, or a driver reloaded with fewer cards).  No GPU: the decision is pure.
#include "strata/core/device.hpp"

#include <cstdio>
#include <string>

using namespace strata::core;

namespace {
int g_fail = 0;
void check(bool ok, const std::string& what) {
    std::printf("  %-78s %s\n", what.c_str(), ok ? "ok" : "FAIL");
    if (!ok) ++g_fail;
}
}  // namespace

int main() {
    check(gpu_missing_note(1, 2).empty(), "a peer device the driver lists is usable (no note)");
    check(gpu_missing_note(0, 1).empty(), "the primary is usable when one card is listed");

    const std::string gone = gpu_missing_note(1, 1);
    check(!gone.empty(), "a peer device past the driver's list gets a note");
    check(gone.find("CUDA1") != std::string::npos, "...that names the device (" + gone + ")");
    check(gone.find("1 CUDA device") != std::string::npos, "...and how many the driver lists");
    check(gone.find("not there") != std::string::npos, "...and says the device is not there");

    check(gpu_missing_note(2, 1).find("CUDA2") != std::string::npos, "the note names whichever device is missing");
    check(gpu_missing_note(1, 0).find("no CUDA device") != std::string::npos,
          "a driver that lists nothing says so");

    std::printf("%s\n", g_fail == 0 ? "PASS" : "FAIL");
    return g_fail == 0 ? 0 : 1;
}
