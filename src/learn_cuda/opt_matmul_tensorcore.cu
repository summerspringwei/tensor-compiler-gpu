#include <mma.h>

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
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
