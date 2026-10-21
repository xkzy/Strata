// include/strata/kernels/dp4a.hpp - the two instructions the kernels use that Pascal (sm_60) does not have.
//
// The engine is written for compute capability 7.5 and newer and refuses older cards at CMake configure time
// (a Pascal build passes -DSTRATA_EXPERIMENTAL_SM60=ON).  A GP100-class card is compute capability 6.0, which is
// missing exactly two things the i-quant kernels rely on:
//
//   * `__dp4a`, a byte-wise dot product, available from 6.1 (Pascal GP10x/GV11x).  Its fallback below is
//     llama.cpp's own (`ggml/src/ggml-cuda/common.cuh`), which is the reference for the kernels in this
//     directory: they are transcribed from llama.cpp's vecdotq.cuh, and that wrapper's operands are the same
//     sites.  It reinterprets the operands as SIGNED bytes and accumulates in int32, which is what `__dp4a`
//     does for these calls, so the fallback is bit-exact rather than merely close.
//   * `__nanosleep`, available from 7.0 (Volta).  It only paces single-thread doorbell waits, so a loop that
//     spins without it is correct, merely busier.
//
// Both are compile-time selections on `__CUDA_ARCH__`, so one source tree builds for Pascal and for RTX
// 20/30/40/50 without a #define at the call sites.  A HIP build never takes either fallback: `__CUDA_ARCH__` is
// undefined there, so `STRATA_DP4A` is `__dp4a` (which `hip_compat/intrinsics.hpp` provides) and the pause is
// HIP's own `__nanosleep`.
#pragma once

#include "strata/core/gpu_wait.hpp"

#include <cstdint>
#include <cstdlib>
#if defined(STRATA_HIP_GFX906)
#include <cuda_runtime.h>   // gfx906: the compat layer (__forceinline__, __nanosleep, __dp4a)
#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 610
__device__ __forceinline__ int strata_dp4a(const int a, const int b, const int c) {
    const int8_t* a8 = (const int8_t*) &a;
    const int8_t* b8 = (const int8_t*) &b;
    return c + a8[0] * b8[0] + a8[1] * b8[1] + a8[2] * b8[2] + a8[3] * b8[3];
}
#define STRATA_DP4A(a, b, c) strata_dp4a((a), (b), (c))
#else
#define STRATA_DP4A(a, b, c) __dp4a((a), (b), (c))
#endif

/// Yield the thread while a doorbell flag is being polled.  The length of the pause is a backoff hint, not a
/// contract, and only pre-Volta CUDA has no `__nanosleep` at all.
__device__ __forceinline__ void strata_spin_pause_ns(unsigned ns) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700
    // sm_6x: the loop spins.  Every call site is a single-thread doorbell wait, so nothing else is delayed.
    (void) ns;
#else
    __nanosleep(ns);
#endif
}
__device__ __forceinline__ void strata_spin_pause() { strata_spin_pause_ns(100); }

/// The GPU's own clock in ns (%globaltimer; HIP: the constant-rate wall clock).
__device__ __forceinline__ unsigned long long strata_gpu_ns() {
    unsigned long long t;
#if defined(STRATA_HIP_GFX906)
    t = wall_clock64() * 40ull;   // gfx906: the wall clock runs at 25 MHz (hipDeviceAttributeWallClockRate) -> ns
#elif defined(__HIPCC__)
    t = wall_clock64() * 10ull;   // gfx10.3 / gfx11 / gfx12: a constant 100 MHz counter, in ns
#else
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
#endif
    return t;
}

/// Doorbell waits poll a flag word in mapped host memory over PCIe.  The pause doubles from 100 ns up to
/// `max_ns`, so a long wait (the peer GPU's rows) polls host memory far less often; past `timeout_ns` (0 = never)
/// the wait gives up and counts itself in word kSpinGaveUpWord of the flag's 64-byte block, which the host checks
/// after the window: a stuck wait becomes an error instead of a GPU that never finishes its queue.
constexpr int kSpinGaveUpWord = 8;
struct SpinParams { unsigned max_ns; unsigned long long timeout_ns; };
/// STRATA_SPIN_MAX_NS (default 2000; 100 = the old fixed 100 ns poll).  STRATA_SPIN_TIMEOUT_S (default the host's
/// GPU wait limit + 10 s, so the host's own limits fire first; 0 = never give up).
inline SpinParams strata_spin_params() {
    static const SpinParams p = [] {
        SpinParams s{2000u, 0ull};
        if (const char* e = std::getenv("STRATA_SPIN_MAX_NS")) {
            const long v = std::atol(e);
            s.max_ns = (unsigned) (v < 1 ? 1 : v > 1000000 ? 1000000 : v);
        }
        const long host_s = (long) (strata::core::gpu_wait_limit().count() / 1000);
        long t = host_s > 0 ? host_s + 10 : 0;   // STRATA_GPU_WAIT_S=0 (no host limit): none here either
        if (const char* e = std::getenv("STRATA_SPIN_TIMEOUT_S")) t = std::atol(e);
        s.timeout_ns = t > 0 ? (unsigned long long) t * 1000000000ull : 0ull;
        return s;
    }();
    return p;
}
template <class Done>
__device__ __forceinline__ void strata_spin_wait(const volatile uint32_t* flag, Done done, unsigned max_ns,
                                                 unsigned long long timeout_ns) {
    if (done()) return;
    unsigned ns = max_ns < 100u ? max_ns : 100u;
    const unsigned long long t0 = strata_gpu_ns();
    while (!done()) {
        if (timeout_ns != 0 && strata_gpu_ns() - t0 > timeout_ns) {
            volatile uint32_t* w = const_cast<volatile uint32_t*>(flag) + kSpinGaveUpWord;
            *w = *w + 1u;
            __threadfence_system();
            return;
        }
        strata_spin_pause_ns(ns);
        ns = ns < max_ns / 2 ? ns * 2 : max_ns;
    }
}
