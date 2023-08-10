/**
 * This kernel is optimized for GPT model, where batch_size is always less
 * than 4. Typically the reduce_dim is 1280, and out_dim is 5120. Each warp
 * reduce one reduce_dim We Launch less than kGridSize blocks, each block has
 * kBlockSize threads.
 */
#define kBlockSize 256
#define kGridSize 84  // The number of SM on RTX3090 is 84
#include <cuda_fp16.h>
#include <stdio.h>

#include "../../../cuda_kernel_utils.h"

// template<int64_t batch_size, int64_t reduce_dim, int64_t out_dim>


template <int64_t batch_size, int64_t reduce_dim, int64_t out_dim>
__global__ void __launch_bounds__(kBlockSize)
    vector_matrix_mul_kernel_half2(const half *__restrict__ input,
                                   const half *__restrict__ weight,
                                   half *__restrict__ output) {
  const int warpIdx = threadIdx.x / 32;
  const int laneIdx = threadIdx.x % 32;
  const int numWarp = kBlockSize / 32;
  const int vectorLength = sizeof(half2) / sizeof(half);

  // Iterate over batch_size
  for (int64_t b = 0; b < batch_size; ++b) {
    // Iterate over out_dim
    for (int64_t idx = 0; UPDIV(out_dim, kGridSize * numWarp); ++idx) {
      // Each warp reduce one reduce_dim
      half2 local_sum(0, 0);
      half2 local_input, local_weight;
      int64_t weight_row_idx =
          (idx * kGridSize * numWarp + blockIdx.x * numWarp + warpIdx);
      // Guard against over indexing
      if (weight_row_idx >= out_dim) break;
#pragma unroll
      for (int64_t k = 0; k < reduce_dim; k += (warpSize * vectorLength)) {
        int64_t col_idx = k + laneIdx * vectorLength;
        // Guard against over indexing
        if (col_idx >= reduce_dim) break;
        local_input = ((half2 *)input)[(b * reduce_dim + col_idx) >> 1];
        local_weight =
            ((half2 *)weight)[(weight_row_idx * reduce_dim + col_idx) >> 1];
        local_sum += __hmul2(local_input, local_weight);
      }
      // Reduce within warp
      local_sum = warpReduceSum(local_sum);
      local_sum.x += local_sum.y;
      // Write to output
      if (laneIdx == 0) {
        output[b * out_dim + weight_row_idx] = local_sum.x;
      }
    }
  }
}
