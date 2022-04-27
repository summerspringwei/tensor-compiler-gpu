#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include "../../cuda_utils.h"

// dim3(6, 12, 12), dim3(32, 2, 1)
extern "C" __global__ void __launch_bounds__(64) query_key_matmul(half* __restrict__ placeholder, half* __restrict__ placeholder1, half* __restrict__ compute) {
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
extern "C" __global__ void __launch_bounds__(256) fused_mul_softmax(half* __restrict__ input, half* __restrict__ output){
  // Each warp reduce a row, each block reduce 8 rows
  const int kNumWarps = 8;
  __shared__ half shared_input[kNumWarps*384];
  const int warp_id = threadIdx.x / warpSize;
  const int row = blockIdx.x * kNumWarps + warp_id;
  // Load input to shared memory 
  const int kNumIters = 384 / 32;
  const int in_warp_index = threadIdx.x / warpSize;
  // Using vector, now the shared_input shape is kNumWarps * (384/vec_size)
  const int vec_size = 2;
  half scale = ((half)1.0 / hsqrt((__float2half)(64.0f)));
  half2 scale2; scale2.x=scale, scale2.y=scale;
  for(int i=0; i<kNumIters / vec_size; ++i){
    int col = (i * warpSize + in_warp_index);
    // Do the mul here
    h2exp(reinterpret_cast<half2*>(shared_input)[warp_id * 384 / vec_size + col]) = h2exp(__hmul2(reinterpret_cast<half2*>(input)[row*384/vec_size+col], scale2));
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
    reinterpret_cast<half2*>(output)[row*384/vec_size+col] = __h2div(reinterpret_cast<half2*>(shared_input)[warp_id*384/vec_size+col], sum2);
  }
}


extern "C" __global__ void __launch_bounds__(192) qeury_key_matmul_softmax(half* __restrict__ placeholder, half* __restrict__ placeholder1, half* __restrict__ compute) {
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
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 32; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    compute[((((((((int)blockIdx.z) * 147456) + (((int)blockIdx.x) * 6144)) + (i_inner_j_inner_fused_outer_outer_outer_outer * 192)) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)))] = compute_wmma_accumulator_shared[(((((((i_inner_j_inner_fused_outer_outer_outer_outer * 192) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)) / 384) * 392) + ((((i_inner_j_inner_fused_outer_outer_outer_outer * 192) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)) % 384)))];
  }
}


extern "C" __global__ void __launch_bounds__(192) fused_qeury_key_matmul_softmax(half* __restrict__ placeholder, half* __restrict__ placeholder1, half* __restrict__ compute) {
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
  #pragma unroll
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 32; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    // Multiply
    half score = compute_wmma_accumulator_shared[(i_inner_j_inner_fused_outer_outer_outer_outer / 2)  * 394 + (i_inner_j_inner_fused_outer_outer_outer_outer % 2) * 196 + (threadIdx.z) * 32 + threadIdx.x];
    arr_scores[i_inner_j_inner_fused_outer_outer_outer_outer] = hexp(score * ((half)1.0 / hsqrt((__float2half)(64.0f))));
  }
  // Now do reduce sum per thread
  const int kNumRows = 16;
  #pragma unroll
  for(int i=0; i<kNumRows; ++i){
    arr_scores[2*i] = arr_scores[2*i] + arr_scores[2*i+1];
    arr_scores[2*i] = warpReduceSum(arr_scores[2*i]);
  }
  // warp reduce between threads in a warp
  const int kNumWarps = 6;
  __shared__ half sum_warps[kNumRows * kNumWarps];
  #pragma unroll
  for(int i=0; i<kNumRows; ++i){
    if(threadIdx.x==0){
      sum_warps[i * kNumWarps + threadIdx.z]=arr_scores[2*i];
    }
  }
  __syncthreads();
  #pragma unroll
  for(int i=0; i<kNumRows; ++i){
    // Sum for a row
    half sum = 0;
    for(int j=0; j<kNumWarps; ++j){
      sum += sum_warps[i * kNumWarps + j];
    }
    compute_wmma_accumulator_shared[i*394 + threadIdx.z*blockDim.x*blockDim.y + threadIdx.x] = compute_wmma_accumulator_shared[i*394 + threadIdx.z*blockDim.x*blockDim.y + threadIdx.x] / sum;
    compute_wmma_accumulator_shared[i*394 + 196 + threadIdx.z*blockDim.x*blockDim.y + threadIdx.x] = compute_wmma_accumulator_shared[i*394 + 196 +threadIdx.z*blockDim.x*blockDim.y + threadIdx.x] / sum;
    compute[((((((((int)blockIdx.z) * 147456) + (((int)blockIdx.x) * 6144)) + (i * 192)) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)))] = compute_wmma_accumulator_shared[i*394 + threadIdx.z*blockDim.x*blockDim.y + threadIdx.x];
    compute[((((((((int)blockIdx.z) * 147456) + (((int)blockIdx.x) * 6144)) + ((i+1) * 192)) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)))] = compute_wmma_accumulator_shared[i*394 + 196 + threadIdx.z*blockDim.x*blockDim.y + threadIdx.x];
  }
  // for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 32; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
  //   compute[((((((((int)blockIdx.z) * 147456) + (((int)blockIdx.x) * 6144)) + (i_inner_j_inner_fused_outer_outer_outer_outer * 192)) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)))] = compute_wmma_accumulator_shared[(((((((i_inner_j_inner_fused_outer_outer_outer_outer * 192) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)) / 384) * 392) + ((((i_inner_j_inner_fused_outer_outer_outer_outer * 192) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)) % 384)))];
  // }
}



extern "C" __global__ void __launch_bounds__(192) fused_qeury_key_matmul_softmax_v2(half* __restrict__ placeholder, half* __restrict__ placeholder1, half* __restrict__ compute) {
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

  // Each warp compute a row
  const int kNumIter = 16 / 6 + 1;
  const int kWarpIter = 384 / 32;
  half sum[kNumIter];
  __shared__ half shared_sum[16];
  #pragma unroll
  for(int i=0; i<kNumIter; ++i){
    sum[i] = __float2half(0);
    int row = i * kNumIter + threadIdx.z;
    if(row>=16){
      continue;
    }
    #pragma unroll
    for(int j=0; j<kWarpIter;++j){
      int col = j * warpSize + threadIdx.x;
      compute_wmma_accumulator_shared[row*392+col] = hexp(compute_wmma_accumulator_shared[row*392+col] * ((half)1.0 / hsqrt((__float2half)(64.0f))));
      sum[i] += compute_wmma_accumulator_shared[row*392+col];
    }
    half warp_sum = warpReduceSum(sum[i]);
    if(threadIdx.x==0){
      shared_sum[row] = warp_sum;
    }
  }
  __syncthreads();
  #pragma unroll
  for(int i=0; i<kNumIter; ++i){
    int row = i * kNumIter + threadIdx.z;
    if(row>=16){
      continue;
    }
    #pragma unroll
    for(int j=0; j<kWarpIter;++j){
      int col = j * warpSize + threadIdx.x;
      compute[((((((((int)blockIdx.z) * 147456) + (((int)blockIdx.x) * 6144)) + (row * 192)) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)))] = compute_wmma_accumulator_shared[row*392+col] / shared_sum[row];
    }
  }
  // for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 32; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
  //   compute[((((((((int)blockIdx.z) * 147456) + (((int)blockIdx.x) * 6144)) + (i_inner_j_inner_fused_outer_outer_outer_outer * 192)) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)))] = compute_wmma_accumulator_shared[(((((((i_inner_j_inner_fused_outer_outer_outer_outer * 192) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)) / 384) * 392) + ((((i_inner_j_inner_fused_outer_outer_outer_outer * 192) + (((int)threadIdx.z) * 32)) + ((int)threadIdx.x)) % 384)))];
  // }
}