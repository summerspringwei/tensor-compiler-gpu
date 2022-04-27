#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <mma.h>

// dim3(1, 1, 256), dim3(32, 1, 1)
extern "C" __global__ void __launch_bounds__(32) swin_transformer_query_key(half* __restrict__ placeholder, half* __restrict__ placeholder1, half* __restrict__ compute) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> compute_wmma_accumulator[16];
  __shared__ half placeholder_shared[8192];
  __shared__ half placeholder_d_shared[8192];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> placeholder_shared_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> placeholder_d_shared_wmma_matrix_b[4];
  for (int i_c_outer_init = 0; i_c_outer_init < 4; ++i_c_outer_init) {
    for (int j_c_outer_init = 0; j_c_outer_init < 4; ++j_c_outer_init) {
      (void)nvcuda::wmma::fill_fragment(compute_wmma_accumulator[((i_c_outer_init * 4) + j_c_outer_init)], 0.000000e+00f);
    }
  }
  for (int ax1_ax2_fused_outer_outer_outer_outer = 0; ax1_ax2_fused_outer_outer_outer_outer < 32; ++ax1_ax2_fused_outer_outer_outer_outer) {
    ((uint1*)(placeholder_shared + ((((ax1_ax2_fused_outer_outer_outer_outer * 256) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)threadIdx.x) & 15) * 2)))))[0] = ((uint1*)(placeholder + ((((((int)blockIdx.z) * 2048) + (ax1_ax2_fused_outer_outer_outer_outer * 64)) + (((int)threadIdx.x) * 2)))))[0];
  }
  for (int ax1_ax2_fused_outer_outer_outer_outer1 = 0; ax1_ax2_fused_outer_outer_outer_outer1 < 32; ++ax1_ax2_fused_outer_outer_outer_outer1) {
    ((uint1*)(placeholder_d_shared + ((((ax1_ax2_fused_outer_outer_outer_outer1 * 256) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)threadIdx.x) & 15) * 2)))))[0] = ((uint1*)(placeholder1 + ((((((int)blockIdx.z) * 2048) + (ax1_ax2_fused_outer_outer_outer_outer1 * 64)) + (((int)threadIdx.x) * 2)))))[0];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
    for (int ax1_outer = 0; ax1_outer < 4; ++ax1_outer) {
      (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_a[ax1_outer], ((half *)placeholder_shared + (((ax1_outer * 2048) + (k_outer_inner * 16)))), 128);
    }
    for (int ax1_outer1 = 0; ax1_outer1 < 4; ++ax1_outer1) {
      (void)nvcuda::wmma::load_matrix_sync(placeholder_d_shared_wmma_matrix_b[ax1_outer1], ((half *)placeholder_d_shared + (((ax1_outer1 * 2048) + (k_outer_inner * 16)))), 128);
    }
    for (int i_c_outer = 0; i_c_outer < 4; ++i_c_outer) {
      for (int j_c_outer = 0; j_c_outer < 4; ++j_c_outer) {
        (void)nvcuda::wmma::mma_sync(compute_wmma_accumulator[((i_c_outer * 4) + j_c_outer)], placeholder_shared_wmma_matrix_a[i_c_outer], placeholder_d_shared_wmma_matrix_b[j_c_outer], compute_wmma_accumulator[((i_c_outer * 4) + j_c_outer)]);
      }
    }
  }
  __syncthreads();
  for (int ax1_outer_inner = 0; ax1_outer_inner < 4; ++ax1_outer_inner) {
    for (int ax2_outer_inner = 0; ax2_outer_inner < 4; ++ax2_outer_inner) {
      (void)nvcuda::wmma::store_matrix_sync(((half *)placeholder_shared + (((ax1_outer_inner * 1152) + (ax2_outer_inner * 16)))), compute_wmma_accumulator[((ax1_outer_inner * 4) + ax2_outer_inner)], 72, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 64; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    ((uint1*)(compute + ((((((int)blockIdx.z) * 4096) + (i_inner_j_inner_fused_outer_outer_outer_outer * 64)) + (((int)threadIdx.x) * 2)))))[0] = ((uint1*)(placeholder_shared + (((i_inner_j_inner_fused_outer_outer_outer_outer * 72) + (((int)threadIdx.x) * 2)))))[0];
  }
}

// const int batch_size=64, num_heads=4, seq_length=64, size_per_head = 32;
// dim3(batch_size*num_heads, 1, 1), dim3(seq_length, 1, 1)
template<const int batch_size, const int num_heads, const int seq_length, const int size_per_head>
__global__ void __launch_bounds__(seq_length) fused_mul_softmax(half* __restrict__ input, half* __restrict__ output){
  __shared__ half shared_input[seq_length * seq_length];
  half scale = ((half)1.0 / hsqrt((__float2half)(size_per_head)));
  half sum=0;
  #pragma unroll
  for(int i=0; i<seq_length; ++i){
    //shape: [size_per_head, seq_length]
    half tmp = hexp(input[blockIdx.x * seq_length * seq_length + threadIdx.x * seq_length + i] * scale);
    sum += tmp;
    shared_input[threadIdx.x * seq_length + i] = tmp;
  }
  __syncthreads();
  #pragma unroll
  for(int i=0; i<seq_length; ++i){
    output[blockIdx.x * seq_length * seq_length + threadIdx.x * seq_length + i] = shared_input[threadIdx.x * seq_length + i] / sum;
  }
}

// dim3(1, 1, 256), dim3(32, 1, 1)
extern "C" __global__ void __launch_bounds__(32) fused_swin_transformer_query_key_mul_softmax(half* __restrict__ placeholder, half* __restrict__ placeholder1, half* __restrict__ compute) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> compute_wmma_accumulator[16];
  __shared__ half placeholder_shared[8192];
  __shared__ half placeholder_d_shared[8192];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> placeholder_shared_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> placeholder_d_shared_wmma_matrix_b[4];
  for (int i_c_outer_init = 0; i_c_outer_init < 4; ++i_c_outer_init) {
    for (int j_c_outer_init = 0; j_c_outer_init < 4; ++j_c_outer_init) {
      (void)nvcuda::wmma::fill_fragment(compute_wmma_accumulator[((i_c_outer_init * 4) + j_c_outer_init)], 0.000000e+00f);
    }
  }
  for (int ax1_ax2_fused_outer_outer_outer_outer = 0; ax1_ax2_fused_outer_outer_outer_outer < 32; ++ax1_ax2_fused_outer_outer_outer_outer) {
    ((uint1*)(placeholder_shared + ((((ax1_ax2_fused_outer_outer_outer_outer * 256) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)threadIdx.x) & 15) * 2)))))[0] = ((uint1*)(placeholder + ((((((int)blockIdx.z) * 2048) + (ax1_ax2_fused_outer_outer_outer_outer * 64)) + (((int)threadIdx.x) * 2)))))[0];
  }
  for (int ax1_ax2_fused_outer_outer_outer_outer1 = 0; ax1_ax2_fused_outer_outer_outer_outer1 < 32; ++ax1_ax2_fused_outer_outer_outer_outer1) {
    ((uint1*)(placeholder_d_shared + ((((ax1_ax2_fused_outer_outer_outer_outer1 * 256) + ((((int)threadIdx.x) >> 4) * 128)) + ((((int)threadIdx.x) & 15) * 2)))))[0] = ((uint1*)(placeholder1 + ((((((int)blockIdx.z) * 2048) + (ax1_ax2_fused_outer_outer_outer_outer1 * 64)) + (((int)threadIdx.x) * 2)))))[0];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
    for (int ax1_outer = 0; ax1_outer < 4; ++ax1_outer) {
      (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_a[ax1_outer], ((half *)placeholder_shared + (((ax1_outer * 2048) + (k_outer_inner * 16)))), 128);
    }
    for (int ax1_outer1 = 0; ax1_outer1 < 4; ++ax1_outer1) {
      (void)nvcuda::wmma::load_matrix_sync(placeholder_d_shared_wmma_matrix_b[ax1_outer1], ((half *)placeholder_d_shared + (((ax1_outer1 * 2048) + (k_outer_inner * 16)))), 128);
    }
    for (int i_c_outer = 0; i_c_outer < 4; ++i_c_outer) {
      for (int j_c_outer = 0; j_c_outer < 4; ++j_c_outer) {
        (void)nvcuda::wmma::mma_sync(compute_wmma_accumulator[((i_c_outer * 4) + j_c_outer)], placeholder_shared_wmma_matrix_a[i_c_outer], placeholder_d_shared_wmma_matrix_b[j_c_outer], compute_wmma_accumulator[((i_c_outer * 4) + j_c_outer)]);
      }
    }
  }
  __syncthreads();
  for (int ax1_outer_inner = 0; ax1_outer_inner < 4; ++ax1_outer_inner) {
    for (int ax2_outer_inner = 0; ax2_outer_inner < 4; ++ax2_outer_inner) {
      (void)nvcuda::wmma::store_matrix_sync(((half *)placeholder_shared + (((ax1_outer_inner * 1152) + (ax2_outer_inner * 16)))), compute_wmma_accumulator[((ax1_outer_inner * 4) + ax2_outer_inner)], 72, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  half scale = ((half)1.0 / hsqrt((__float2half)(32.0f)));
  half2 scale2; scale2.x=scale, scale2.y=scale;
  half2 sum[2]; sum[0].x=0.0; sum[0].y=0.0;sum[1].x=0.0; sum[1].y=0.0;
  
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 2; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    const int row_idx = i_inner_j_inner_fused_outer_outer_outer_outer * 32 + threadIdx.x;
    for(int rk=0; rk<32;++rk){
      const int index = row_idx * 72 / 2 + rk;
      (reinterpret_cast<half2*>(placeholder_shared))[index] = h2exp(__hmul2((reinterpret_cast<half2*>(placeholder_shared))[index], scale2));
      sum[i_inner_j_inner_fused_outer_outer_outer_outer] += (reinterpret_cast<half2*>(placeholder_shared))[index];
    }
  }
  half2 sum2[2];// Used for vector mul
  sum2[0].x = sum[0].x + sum[0].y; sum2[0].y = sum2[0].x;
  sum2[1].x = sum[1].x + sum[1].y; sum2[1].y = sum2[1].x;
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 2; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    const int row_idx = i_inner_j_inner_fused_outer_outer_outer_outer * 32 + threadIdx.x;
    for(int rk=0; rk<32;++rk){
      const int index = row_idx * 72 / 2 + rk;
      reinterpret_cast<half2*>(placeholder_shared)[index] = __h2div(reinterpret_cast<half2*>(placeholder_shared)[index], sum2[i_inner_j_inner_fused_outer_outer_outer_outer]);
    }
  }
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 64; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    ((uint1*)(compute + ((((((int)blockIdx.z) * 4096) + (i_inner_j_inner_fused_outer_outer_outer_outer * 64)) + (((int)threadIdx.x) * 2)))))[0] = ((uint1*)(placeholder_shared + (((i_inner_j_inner_fused_outer_outer_outer_outer * 72) + (((int)threadIdx.x) * 2)))))[0];
  }
}

