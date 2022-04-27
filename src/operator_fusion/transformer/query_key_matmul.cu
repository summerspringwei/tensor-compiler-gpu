#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>

#include "../../utils.h"
#include "../../cuda_utils.h"

// dim3(6, 12, 12), dim3(32, 2, 1)
extern "C" __global__ void __launch_bounds__(64) default_function_kernel0(half* __restrict__ placeholder, half* __restrict__ placeholder1, half* __restrict__ compute) {
  __shared__ half compute_wmma_accumulator_shared[2560];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> compute_wmma_accumulator[4];
  __shared__ half placeholder_shared[2304];
  __shared__ half placeholder_d_shared[2304];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> placeholder_shared_wmma_matrix_a[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> placeholder_d_shared_wmma_matrix_b[2];
  for (int ax1_outer_outer = 0; ax1_outer_outer < 2; ++ax1_outer_outer) {
    for (int i_c_outer_init = 0; i_c_outer_init < 2; ++i_c_outer_init) {
      for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init) {
        (void)nvcuda::wmma::fill_fragment(compute_wmma_accumulator[((i_c_outer_init * 2) + j_c_outer_init)], 0.000000e+00f);
      }
    }
    __syncthreads();
    for (int ax1_ax2_fused_outer_outer_outer_outer = 0; ax1_ax2_fused_outer_outer_outer_outer < 8; ++ax1_ax2_fused_outer_outer_outer_outer) {
      ((uint2*)(placeholder_shared + (((((ax1_ax2_fused_outer_outer_outer_outer * 288) + (((int)threadIdx.y) * 144)) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)))))[0] = ((uint2*)(placeholder + (((((((((int)blockIdx.z) * 24576) + (((int)blockIdx.x) * 4096)) + (ax1_outer_outer * 2048)) + (ax1_ax2_fused_outer_outer_outer_outer * 256)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)))))[0];
    }
    for (int ax1_ax2_fused_outer_outer_outer_outer1 = 0; ax1_ax2_fused_outer_outer_outer_outer1 < 8; ++ax1_ax2_fused_outer_outer_outer_outer1) {
      ((uint2*)(placeholder_d_shared + (((((ax1_ax2_fused_outer_outer_outer_outer1 * 288) + (((int)threadIdx.y) * 144)) + ((((int)threadIdx.x) >> 4) * 72)) + ((((int)threadIdx.x) & 15) * 4)))))[0] = ((uint2*)(placeholder1 + ((((((((int)blockIdx.z) * 24576) + (((int)blockIdx.y) * 2048)) + (ax1_ax2_fused_outer_outer_outer_outer1 * 256)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 4)))))[0];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
      for (int ax1_outer = 0; ax1_outer < 2; ++ax1_outer) {
        (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_a[ax1_outer], ((half *)placeholder_shared + (((ax1_outer * 1152) + (k_outer_inner * 16)))), 72);
      }
      for (int ax1_outer1 = 0; ax1_outer1 < 2; ++ax1_outer1) {
        (void)nvcuda::wmma::load_matrix_sync(placeholder_d_shared_wmma_matrix_b[ax1_outer1], ((half *)placeholder_d_shared + (((ax1_outer1 * 1152) + (k_outer_inner * 16)))), 72);
      }
      for (int i_c_outer = 0; i_c_outer < 2; ++i_c_outer) {
        for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
          (void)nvcuda::wmma::mma_sync(compute_wmma_accumulator[((i_c_outer * 2) + j_c_outer)], placeholder_shared_wmma_matrix_a[i_c_outer], placeholder_d_shared_wmma_matrix_b[j_c_outer], compute_wmma_accumulator[((i_c_outer * 2) + j_c_outer)]);
        }
      }
    }
    for (int ax1_outer_inner = 0; ax1_outer_inner < 2; ++ax1_outer_inner) {
      for (int ax2_outer_inner = 0; ax2_outer_inner < 2; ++ax2_outer_inner) {
        (void)nvcuda::wmma::store_matrix_sync(((half *)compute_wmma_accumulator_shared + ((((ax1_outer_outer * 1280) + (ax1_outer_inner * 640)) + (ax2_outer_inner * 16)))), compute_wmma_accumulator[((ax1_outer_inner * 2) + ax2_outer_inner)], 40, nvcuda::wmma::mem_row_major);
      }
    }
  }
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 8; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    ((uint2*)(compute + ((((((((((int)blockIdx.z) * 147456) + (((int)blockIdx.x) * 24576)) + (i_inner_j_inner_fused_outer_outer_outer_outer * 3072)) + (((int)threadIdx.y) * 1536)) + ((((int)threadIdx.x) >> 3) * 384)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 7) * 4)))))[0] = ((uint2*)(compute_wmma_accumulator_shared + (((((i_inner_j_inner_fused_outer_outer_outer_outer * 320) + (((int)threadIdx.y) * 160)) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)))))[0];
  }
}

// Only valid for query-key-matmul, dim3(576, 1, 1), dim3(256, 1, 1)
extern "C" __global__ void __launch_bounds__(256) softmax(half* __restrict__ input, half* __restrict__ output){
  // Each warp reduce a row, each block reduce 8 rows
  const int kNumWarps = 8;
  __shared__ half shared_input[kNumWarps*384];
  const int warp_id = threadIdx.x / warpSize;
  const int row = blockIdx.x * kNumWarps + warp_id;
  // Load input to shared memory 
  const int kNumIters = 384 / 32;
  const int in_warp_index = threadIdx.x / warpSize;
  const int vec_size = 2;
  for(int i=0; i<kNumIters / vec_size; ++i){
    int col = i * warpSize + in_warp_index;
    h2exp(reinterpret_cast<half2*>(shared_input)[warp_id * 384 + col]) = h2exp(reinterpret_cast<half2*>(input)[row*384+col]);
  }

  // Reduce sum
  __shared__ half sum_in_warp[kNumWarps * 32];
  sum_in_warp[in_warp_index] = __float2half(0);
  for(int i=0; i<kNumIters; ++i){
    int col = i * warpSize + in_warp_index;
    sum_in_warp[warp_id*32+in_warp_index] += shared_input[warp_id * 384 + col];
  }
  // Reduce in a warp
  half sum = warpReduceSum(sum_in_warp[in_warp_index]);
  __syncthreads();
  if(threadIdx.x==0){
    sum_in_warp[0]=sum;
  }
  __syncthreads();

  // Normalize
  half2 sum2;
  sum2.x = sum; sum2.y=sum;
  for(int i=0; i<kNumIters/vec_size; ++i){
    int col = i * warpSize + in_warp_index;
    reinterpret_cast<half2*>(output)[row*384+col] = __h2div(reinterpret_cast<half2*>(shared_input)[warp_id*384+col], sum2);
  }
}
