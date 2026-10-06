#pragma once
// include/strata/kernels/fast_parallel.hpp - minimum work before a fast kernel opens an OpenMP region
//
// A parallel region costs about 3.4 us on the 32-thread machine below (more for gemm_fp32, which has up to three
// regions), so a kernel whose serial run is only a few us is slower in parallel. These limits count the
// multiply-accumulates (or elements) of one call and are used in the `if (...)` clause of each region.
//
// Measured with a 1-thread vs 32-thread sweep on a 32-thread AVX2 CPU (no AVX-512), -O3 -mavx2 -mfma, K up to 2048,
// head_dim 128. A CPU with fewer threads has a cheaper region and wants lower limits; these are on the safe side.

#include <cstddef>
#include <cstdint>

namespace strata::kernels::fast_parallel {

// gemv_fp32, work = M * K. 65k: 0.87-0.95x (loses); 131k: 1.9x.
inline constexpr int64_t kGemvMinMacs = int64_t{1} << 17;
// batch_gemv_fp32, work = T * N * K. 35k: 1.2x; 131k: 3.8x.
inline constexpr int64_t kBatchGemvMinMacs = int64_t{1} << 16;
// gemm_fp32, work = M * N * K. 262k: 0.74x; 524k: 0.93x (still loses); 1M: 1.27x.
inline constexpr int64_t kGemmMinMacs = int64_t{1} << 20;
// fused_swiglu_gemm, work = M * N * K. 8k: 0.48x; 32k: 1.7x.
inline constexpr int64_t kSwigluMinMacs = int64_t{1} << 15;
// apply_multi_head_neox, work = n_heads * head_dim. 64 heads x 128: 0.6 us serial vs 3.5 us parallel.
inline constexpr int64_t kRopeMinElems = int64_t{1} << 17;
// Zero / scale pass over C before gemm_fp32, work = M * N. Not measured on its own; chosen so the pass takes
// several region costs serially.
inline constexpr int64_t kElementwiseMinElems = int64_t{1} << 16;
// flash_attention_prefill: parallel over 64-row query blocks, so it cannot beat the block count.
// 3 blocks (L=129): 0.90x; 4 blocks (L=256): 1.6x.
inline constexpr size_t kFlashPrefillMinQBlocks = 4;
// flash_attention_prefill_mha, work = seq_len^2 * num_q_heads * head_dim. L=16, 16 heads (2^19): 42 us serial;
// L=24 (2^20.5): 5.5x parallel. Chosen near the 7 us break-even (about L=8 with 16 heads).
inline constexpr size_t kPrefillMhaMinWork = size_t{1} << 17;

} // namespace strata::kernels::fast_parallel
