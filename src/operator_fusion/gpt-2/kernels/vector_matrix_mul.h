#pragma once
/**
 * This kernel is optimized for GPT model, where batch_size is always less than 4.
 * Typically the reduce_dim is 1280, and out_dim is 5120.
 * Each warp reduce one reduce_dim
 * We Launch less than kGridSize blocks, each block has kBlockSize threads.
*/
#define kBlockSize 256
#define kGridSize 128
#include <cuda_fp16.h>
#include "../../../cuda_kernel_utils.h"
template<int64_t batch_size, int64_t reduce_dim, int64_t out_dim>
__global__ void __launch_bounds__(kBlockSize) vector_matrix_mul_kernel(
    const half* __restrict__ input, const half* __restrict__ weight, half* __restrict__ output){
    const int warpIdx = threadIdx.x / 32;
    const int laneIdx = threadIdx.x % 32;
    const int numWarp = kBlockSize / 32;
    const int vectorLength = sizeof(float4) / sizeof(half);

    // Iterate over batch_size
    for(int64_t b = 0; b < batch_size; ++b){
        // Iterate over out_dim
        for(int64_t idx = 0; UPDIV(out_dim, kGridSize); ++idx){
            // Each warp reduce one reduce_dim
            half2 sum[] = {half2(0, 0), half2(0, 0), half2(0, 0), half2(0, 0)};
            half2 local_input[4]; 
            half2 local_weight[4];
            int64_t weight_row_idx = (idx * kGridSize * numWarp + warpIdx);
            // Guard against over indexing
            if (weight_row_idx >= out_dim) break;
            # pragma unroll
            for(int64_t k = 0; k < reduce_dim; k += (warpSize*vectorLength)){
                int64_t col_idx = (k + laneIdx) * vectorLength;
                // Guard against over indexing
                if(col_idx >= reduce_dim) break;
                // auto tmp_input_ptr = 
                ((float4*)local_input)[0] = ((float4*)input)[(b * reduce_dim + col_idx) >> 3];
                ((float4*)local_weight)[0] = ((float4*)weight)[(weight_row_idx * reduce_dim + col_idx) >> 3];
                // (float4*)(&(local_input[0])) = __ldg((const float4*)(input + b * reduce_dim + col_idx));
                // (float4*)(&(local_weight[0])) = __ldg((const float4*)(weight + (weight_row_idx * reduce_dim + col_idx)));
                sum[0] += __hmul2(local_input[0], local_weight[0]);
                sum[1] += __hmul2(local_input[1], local_weight[1]);
                sum[2] += __hmul2(local_input[2], local_weight[2]);
                sum[3] += __hmul2(local_input[3], local_weight[3]);
            }
            // Reduce within thread
            sum[0] += sum[1];
            sum[0] += sum[2];
            sum[0] += sum[3];
            // Reduce within warp
            sum[0] = warpReduceSum(sum[0]);
            sum[0].x += sum[0].y;
            // Write to output
            if(laneIdx == 0){
                output[b * out_dim + weight_row_idx] = sum[0].x;
            }
        }
    }
}