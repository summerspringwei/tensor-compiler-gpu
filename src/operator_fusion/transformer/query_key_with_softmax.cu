// attr [IterVar(blockIdx.x: int32, (nullptr), "ThreadIndex", "blockIdx.x")] "thread_extent" = 24;
// attr [IterVar(blockIdx.y: int32, (nullptr), "ThreadIndex", "blockIdx.y")] "thread_extent" = 1 
// attr [IterVar(blockIdx.z: int32, (nullptr), "ThreadIndex", "blockIdx.z")] "thread_extent" = 12;
// attr [IterVar(threadIdx.z: int32, (nullptr), "ThreadIndex", "threadIdx.z")] "thread_extent" = 6;
// attr [IterVar(threadIdx.y: int32, (nullptr), "ThreadIndex", "threadIdx.y")] "thread_extent" = 1;
// attr [IterVar(threadIdx.x: int32, (nullptr), "ThreadIndex", "threadIdx.x")] "thread_extent" = 32;
// dim3(24, 1, 12), dim3(32, 1, 6)

__inline__ __device__
half warpReduceSum(half val) {
  for (int offset = warpSize/2; offset > 0; offset /= 2) 
    val += __shfl_down_sync(0xffffffff, val, offset);
  return val;
}


#include <cuda_fp16.h>
#include <mma.h>
extern "C" __global__ void __launch_bounds__(192) default_function_kernel0(half* __restrict__ placeholder, half* __restrict__ placeholder1, half* __restrict__ compute) {
  __shared__ half compute_wmma_accumulator_shared[6272];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> compute_wmma_accumulator[4];
  __shared__ half placeholder_shared[640];
  __shared__ half placeholder_d_shared[2560];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> placeholder_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> placeholder_d_shared_wmma_matrix_b[4];
  for (int ax2_outer_outer = 0; ax2_outer_outer < 6; ++ax2_outer_outer) {
    for (int j_c_outer_init = 0; j_c_outer_init < 4; ++j_c_outer_init) {
      (void)nvcuda::wmma::fill_fragment(compute_wmma_accumulator[j_c_outer_init], 0.000000e+00f);
    }
    for (int k_outer_outer = 0; k_outer_outer < 2; ++k_outer_outer) {
      __syncthreads();
      for (int ax1_ax2_fused_outer_outer_outer_outer = 0; ax1_ax2_fused_outer_outer_outer_outer < 3; ++ax1_ax2_fused_outer_outer_outer_outer) {
        if (((ax1_ax2_fused_outer_outer_outer_outer * 6) + ((int)threadIdx.z)) < 16) {
          if ((((ax1_ax2_fused_outer_outer_outer_outer * 192) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)) < 512) {
            if ((((((int)blockIdx.x) * 16) + (ax1_ax2_fused_outer_outer_outer_outer * 6)) + ((int)threadIdx.z)) < 384) {
              placeholder_shared[((((ax1_ax2_fused_outer_outer_outer_outer * 240) + (((int)threadIdx.z) * 40)) + ((int)threadIdx.x)))] = placeholder[(((((((((int)blockIdx.z) * 24576) + (((int)blockIdx.x) * 1024)) + (ax1_ax2_fused_outer_outer_outer_outer * 384)) + (((int)threadIdx.z) * 64)) + (k_outer_outer * 32)) + ((int)threadIdx.x)))];
            }
          }
        }
      }
      for (int ax1_ax2_fused_outer_outer_outer_outer1 = 0; ax1_ax2_fused_outer_outer_outer_outer1 < 11; ++ax1_ax2_fused_outer_outer_outer_outer1) {
        if (((ax1_ax2_fused_outer_outer_outer_outer1 * 6) + ((int)threadIdx.z)) < 64) {
          if ((((ax1_ax2_fused_outer_outer_outer_outer1 * 192) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)) < 2048) {
            if ((((ax2_outer_outer * 64) + (ax1_ax2_fused_outer_outer_outer_outer1 * 6)) + ((int)threadIdx.z)) < 384) {
              placeholder_d_shared[((((ax1_ax2_fused_outer_outer_outer_outer1 * 240) + (((int)threadIdx.z) * 40)) + ((int)threadIdx.x)))] = placeholder1[(((((((((int)blockIdx.z) * 24576) + (ax2_outer_outer * 4096)) + (ax1_ax2_fused_outer_outer_outer_outer1 * 384)) + (((int)threadIdx.z) * 64)) + (k_outer_outer * 32)) + ((int)threadIdx.x)))];
            }
          }
        }
      }
      __syncthreads();
      for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
        (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_a[0], ((half *)placeholder_shared + ((k_outer_inner * 16))), 40);
        for (int ax1_outer = 0; ax1_outer < 4; ++ax1_outer) {
          (void)nvcuda::wmma::load_matrix_sync(placeholder_d_shared_wmma_matrix_b[ax1_outer], ((half *)placeholder_d_shared + (((ax1_outer * 640) + (k_outer_inner * 16)))), 40);
        }
        for (int j_c_outer = 0; j_c_outer < 4; ++j_c_outer) {
          (void)nvcuda::wmma::mma_sync(compute_wmma_accumulator[j_c_outer], placeholder_shared_wmma_matrix_a[0], placeholder_d_shared_wmma_matrix_b[j_c_outer], compute_wmma_accumulator[j_c_outer]);
        }
      }
    }
    for (int ax2_outer_inner = 0; ax2_outer_inner < 4; ++ax2_outer_inner) {
      (void)nvcuda::wmma::store_matrix_sync(((half *)compute_wmma_accumulator_shared + (((ax2_outer_outer * 64) + (ax2_outer_inner * 16)))), compute_wmma_accumulator[ax2_outer_inner], 392, nvcuda::wmma::mem_row_major);
    } 
  }
  __syncthreads();
  // Keep all scores in registers
  half arr_scores[32];
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 32; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    // Multiply
    half score = (((((((i_inner_j_inner_fused_outer_outer_outer_outer * 192) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)) / 384) * 392) + ((((i_inner_j_inner_fused_outer_outer_outer_outer * 192) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)) % 384)))
    score = hexp(score * (1.0 / hsqrt(score)));
    arr_scores[i_inner_j_inner_fused_outer_outer_outer_outer] = score;
  }
  // Now do reduce sum
  half sum = 0;
  for(int i=0; i<32; ++i){
    sum += arr_scores[i];
  }
  // warp reduce
  sum = warpReduce(sum);
  __shared__ half sum_warps[4];
  if(threadIdx.x==0){
    sum_warps[threadIdx.x/warpSize]=sum;
  }
  __syncthreads();
  sum = 0;
  for(int i=0; i<4; ++i){
    sum += sum_warps[i];
  }
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 32; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    compute[((((((((int)blockIdx.z) * 147456) + (((int)blockIdx.x) * 6144)) + (i_inner_j_inner_fused_outer_outer_outer_outer * 192)) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)))] = arr_scores[i_inner_j_inner_fused_outer_outer_outer_outer] / sum;
    // compute[((((((((int)blockIdx.z) * 147456) + (((int)blockIdx.x) * 6144)) + (i_inner_j_inner_fused_outer_outer_outer_outer * 192)) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)))] = compute_wmma_accumulator_shared[(((((((i_inner_j_inner_fused_outer_outer_outer_outer * 192) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)) / 384) * 392) + ((((i_inner_j_inner_fused_outer_outer_outer_outer * 192) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)) % 384)))];
  }
}
