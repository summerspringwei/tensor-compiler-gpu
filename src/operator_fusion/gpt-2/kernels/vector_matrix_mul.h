#pragma once
/**
 * This kernel is optimized for GPT model, where batch_size is always less than 4.
 * Typically the reduce_dim is 1280, and out_dim is 5120.
 * Each warp reduce one reduce_dim
 * We Launch less than kGridSize blocks, each block has kBlockSize threads.
*/
#define kBlockSize 256
#define kGridSize 168
#include <stdio.h>
#include <cuda_fp16.h>
#include "../../../cuda_kernel_utils.h"

typedef struct __align__(16) {
   half2 x;
   half2 y;
   half2 z;
   half2 w;
} half8;

// template<int64_t batch_size, int64_t reduce_dim, int64_t out_dim>
// __global__ void __launch_bounds__(kBlockSize) vector_matrix_mul_kernel(
//      half* __restrict__ input,  half* __restrict__ weight, half* __restrict__ output){
//     const int warpIdx = threadIdx.x / 32;
//     const int laneIdx = threadIdx.x % 32;
//     const int numWarp = kBlockSize / 32;
//     const int vectorLength = sizeof(float4) / sizeof(half);

//     // Iterate over batch_size
//     for(int64_t b = 0; b < batch_size; ++b){
//         // Iterate over out_dim
//         for(int64_t idx = 0; UPDIV(out_dim, kGridSize); ++idx){
//             // Each warp reduce one reduce_dim
//             half2 local_sum(0, 0);
//             // half2 local_input[4]; 
//             // half2 local_weight[4];
//             half8 local_input;
//             half8 local_weight;
//             int64_t weight_row_idx = (idx * kGridSize * numWarp + warpIdx);
//             // Guard against over indexing
//             if (weight_row_idx >= out_dim) break;
//             # pragma unroll
//             for(int64_t k = 0; k < reduce_dim; k += (warpSize*vectorLength)){
//                 int64_t col_idx = (k + laneIdx) * vectorLength;
//                 // Guard against over indexing
//                 if(col_idx >= reduce_dim) break;
//                 ((uint4*)(&local_input))[0] = ((uint4*)input)[(b * reduce_dim + col_idx) >> 3];
//                 ((uint4*)local_weight)[0] = ((uint4*)weight)[(weight_row_idx * reduce_dim + col_idx) >> 3];
//                 // reinterpret_cast<float4*>(local_input)[0] = reinterpret_cast<float4*>(input)[(b * reduce_dim + col_idx) >> 3];
//                 // reinterpret_cast<float4*>(local_weight)[0] = reinterpret_cast<float4*>(weight)[(weight_row_idx * reduce_dim + col_idx) >> 3];
//                 // ((float4*)local_input)[0] = ((float4*)input)[(b * reduce_dim + col_idx) >> 3];
//                 // ((float4*)local_weight)[0] = ((float4*)weight)[(weight_row_idx * reduce_dim + col_idx) >> 3];
//                 // (float4*)(&(local_input[0])) = __ldg((const float4*)(input + b * reduce_dim + col_idx));
//                 // (float4*)(&(local_weight[0])) = __ldg((const float4*)(weight + (weight_row_idx * reduce_dim + col_idx)));
//                 // local_sum += __hmul2(local_input[0], local_weight[0]);
//                 // local_sum += __hmul2(local_input[1], local_weight[1]);
//                 // local_sum += __hmul2(local_input[2], local_weight[2]);
//                 // local_sum += __hmul2(local_input[3], local_weight[3]);
//                 local_sum += __hmul2(local_input.x, local_weight.x);
//                 local_sum += __hmul2(local_input.y, local_weight.y);
//                 local_sum += __hmul2(local_input.z, local_weight.z);
//                 local_sum += __hmul2(local_input.w, local_weight.w);
//             }
//             // Reduce within warp
//             local_sum = warpReduceSum(local_sum);
//             local_sum.x += local_sum.y;
//             // Write to output
//             if(laneIdx == 0){
//                 output[b * out_dim + weight_row_idx] = local_sum.x;
//             }
//         }
//     }
// }


template<int64_t batch_size, int64_t reduce_dim, int64_t out_dim>
__global__ void __launch_bounds__(kBlockSize) vector_matrix_mul_kernel_half2(
    const half* __restrict__ input, const half* __restrict__ weight, half* __restrict__ output) {
    const int warpIdx = threadIdx.x / 32;
    const int laneIdx = threadIdx.x % 32;
    const int numWarp = kBlockSize / 32;
    const int vectorLength = sizeof(half2) / sizeof(half);

    // Iterate over batch_size
    for(int64_t b = 0; b < batch_size; ++b){
        // Iterate over out_dim
        for(int64_t idx = 0; UPDIV(out_dim, kGridSize * numWarp); ++idx){
            // Each warp reduce one reduce_dim
            half2 local_sum(0, 0);
            half2 local_input, local_weight;
            int64_t weight_row_idx = (idx * kGridSize * numWarp + blockIdx.x * numWarp + warpIdx);
            // Guard against over indexing
            if (weight_row_idx >= out_dim) break;
            # pragma unroll
            for(int64_t k = 0; k < reduce_dim; k += (warpSize*vectorLength)) {
                int64_t col_idx = k + laneIdx * vectorLength;
                // Guard against over indexing
                if(col_idx >= reduce_dim) break;
                local_input = ((half2*)input)[(b * reduce_dim + col_idx) >> 1];
                local_weight = ((half2*)weight)[(weight_row_idx * reduce_dim + col_idx) >> 1];
                local_sum += __hmul2(local_input, local_weight);
                // if(threadIdx.x == 0){
                //     // printf("blockIdx.x: %d, warpIdx: %d, laneIdx: %d, weight_row_idx: %d, col_idx: %d k: %d, local_sum: %f, %f\n", 
                //          weight_row_idx, col_idx, k, __half2float(local_sum.x), __half2float(local_sum.y));
                // }
            }
            // Reduce within warp
            local_sum = warpReduceSum(local_sum);
            local_sum.x += local_sum.y;
            // Write to output
            if(laneIdx == 0){
                // printf("blockIdx.x: %d, warpIdx: %d, laneIdx: %d, local_sum: %f\n", 
                //     blockIdx.x, warpIdx, laneIdx, __half2float(local_sum.x));
                output[b * out_dim + weight_row_idx] = local_sum.x;
            }
        }
    }
}
