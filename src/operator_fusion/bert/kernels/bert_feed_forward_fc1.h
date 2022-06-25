
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <mma.h>


// // dim3(2, 96,1), dim3(32,2,2) change to dim3(192, 1, 1), dim3(128, 1, 1)
// extern "C" __global__ void __launch_bounds__(128) fc1(half* __restrict__ x, half* __restrict__ placeholder, half* __restrict__ T_dense) {
//   nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 32, 8, 16, half> T_dense_wmma_accumulator[2];
//   __shared__ half x_shared[8704];
//   __shared__ half placeholder_shared[4352];
//   const int blockIdx_x = blockIdx.x % 2;
//   const int blockIdx_y = blockIdx.x / 2;
//   const int threadIdx_x = threadIdx.x % 32;
//   const int threadIdx_y = (threadIdx.x / 32) / 2;
//   const int threadIdx_z = (threadIdx.x / 32) / 2;
//   nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 32, 8, 16, half, nvcuda::wmma::row_major> x_shared_wmma_matrix_a[1];
//   nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 32, 8, 16, half, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[2];
//   for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init) {
//     (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[j_c_outer_init], 0.000000e+00f);
//   }
//   for (int k_outer_outer = 0; k_outer_outer < 6; ++k_outer_outer) {
//     __syncthreads();
//     for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 8; ++ax0_ax1_fused_outer_outer_outer_outer) {
//       ((uint4*)(x_shared + ((((((ax0_ax1_fused_outer_outer_outer_outer * 1088) + (((int)threadIdx_z) * 544)) + (((int)threadIdx_y) * 272)) + ((((int)threadIdx_x) >> 4) * 136)) + ((((int)threadIdx_x) & 15) * 8)))))[0] = ((uint4*)(x + ((((((((((int)blockIdx_x) * 49152) + (ax0_ax1_fused_outer_outer_outer_outer * 6144)) + (((int)threadIdx_z) * 3072)) + (((int)threadIdx_y) * 1536)) + ((((int)threadIdx_x) >> 4) * 768)) + (k_outer_outer * 128)) + ((((int)threadIdx_x) & 15) * 8)))))[0];
//     }
//     for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0; ax0_ax1_fused_outer_outer_outer_outer1 < 4; ++ax0_ax1_fused_outer_outer_outer_outer1) {
//       ((uint4*)(placeholder_shared + ((((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) + (((int)threadIdx_z) * 544)) + (((int)threadIdx_y) * 272)) + ((((int)threadIdx_x) >> 4) * 136)) + ((((int)threadIdx_x) & 15) * 8)))))[0] = ((uint4*)(placeholder + ((((((((((int)blockIdx_y) * 24576) + (ax0_ax1_fused_outer_outer_outer_outer1 * 6144)) + (((int)threadIdx_z) * 3072)) + (((int)threadIdx_y) * 1536)) + ((((int)threadIdx_x) >> 4) * 768)) + (k_outer_outer * 128)) + ((((int)threadIdx_x) & 15) * 8)))))[0];
//     }
//     __syncthreads();
//     for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
//       (void)nvcuda::wmma::load_matrix_sync(x_shared_wmma_matrix_a[0], ((half *)x_shared + (((((int)threadIdx_y) * 4352) + (k_outer_inner * 16)))), 136);
//       for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
//         (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax0_outer], ((half *)placeholder_shared + ((((((int)threadIdx_z) * 2176) + (ax0_outer * 1088)) + (k_outer_inner * 16)))), 136);
//       }
//       for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
//         (void)nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[j_c_outer], x_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[j_c_outer], T_dense_wmma_accumulator[j_c_outer]);
//       }
//     }
//   }
//   __syncthreads();
//   for (int ax1_outer_inner = 0; ax1_outer_inner < 2; ++ax1_outer_inner) {
//     (void)nvcuda::wmma::store_matrix_sync(((half *)placeholder_shared + ((((((int)threadIdx_y) * 1280) + (((int)threadIdx_z) * 16)) + (ax1_outer_inner * 8)))), T_dense_wmma_accumulator[ax1_outer_inner], 40, nvcuda::wmma::mem_row_major);
//   }
//   __syncthreads();
//   // 196608=64*3072, 98304=32*3072, 49152=16*3072, 24576=8*3072, threadIdx.x/4*3072, blockIdx.y * 32
//   // Each block compute 64*32 output(blockIdx.x, blockIdx.y)
//   for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 2; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
//     ((uint4*)(T_dense + ((((((((((int)blockIdx_x) * 196608) + (i_inner_j_inner_fused_outer_outer_outer_outer * 98304)) + (((int)threadIdx_z) * 49152)) + (((int)threadIdx_y) * 24576)) + ((((int)threadIdx_x) >> 2) * 3072)) + (((int)blockIdx_y) * 32)) + ((((int)threadIdx_x) & 3) * 8)))))[0] = ((uint4*)(placeholder_shared + ((((((i_inner_j_inner_fused_outer_outer_outer_outer * 1280) + (((int)threadIdx_z) * 640)) + (((int)threadIdx_y) * 320)) + ((((int)threadIdx_x) >> 2) * 40)) + ((((int)threadIdx_x) & 3) * 8)))))[0];
//   }
// }


__global__ void __launch_bounds__(128) fc1_128_768_3072(half* __restrict__ x, half* __restrict__ placeholder, half* __restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 32, 8, 16, half> T_dense_wmma_accumulator[2];
  extern half __shared__ shared_buff[]; 
  half* x_shared = (half*)&shared_buff[0];
  half* placeholder_shared = (half*)&shared_buff[8704];
  // __shared__ half x_shared[8704];
  // __shared__ half placeholder_shared[4352];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 32, 8, 16, half, nvcuda::wmma::row_major> x_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 32, 8, 16, half, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[2];
  
  for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init) {
    (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[j_c_outer_init], 0.000000e+00f);
  }
  for (int k_outer_outer = 0; k_outer_outer < 6; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 8; ++ax0_ax1_fused_outer_outer_outer_outer) {
      ((uint4*)(x_shared + ((((((ax0_ax1_fused_outer_outer_outer_outer * 1088) + (((int)threadIdx.z) * 544)) + (((int)threadIdx.y) * 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)))))[0] = ((uint4*)(x + ((((((((((int)blockIdx.x) * 49152) + (ax0_ax1_fused_outer_outer_outer_outer * 6144)) + (((int)threadIdx.z) * 3072)) + (((int)threadIdx.y) * 1536)) + ((((int)threadIdx.x) >> 4) * 768)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) & 15) * 8)))))[0];
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0; ax0_ax1_fused_outer_outer_outer_outer1 < 4; ++ax0_ax1_fused_outer_outer_outer_outer1) {
      ((uint4*)(placeholder_shared + ((((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) + (((int)threadIdx.z) * 544)) + (((int)threadIdx.y) * 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)))))[0] = ((uint4*)(placeholder + ((((((((((int)blockIdx.y) * 24576) + (ax0_ax1_fused_outer_outer_outer_outer1 * 6144)) + (((int)threadIdx.z) * 3072)) + (((int)threadIdx.y) * 1536)) + ((((int)threadIdx.x) >> 4) * 768)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) & 15) * 8)))))[0];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      (void)nvcuda::wmma::load_matrix_sync(x_shared_wmma_matrix_a[0], ((half *)x_shared + (((((int)threadIdx.y) * 4352) + (k_outer_inner * 16)))), 136);
      for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
        (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax0_outer], ((half *)placeholder_shared + ((((((int)threadIdx.z) * 2176) + (ax0_outer * 1088)) + (k_outer_inner * 16)))), 136);
      }
      for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
        (void)nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[j_c_outer], x_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[j_c_outer], T_dense_wmma_accumulator[j_c_outer]);
      }
    }
  }
  __syncthreads();
  for (int ax1_outer_inner = 0; ax1_outer_inner < 2; ++ax1_outer_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((half *)placeholder_shared + ((((((int)threadIdx.y) * 1280) + (((int)threadIdx.z) * 16)) + (ax1_outer_inner * 8)))), T_dense_wmma_accumulator[ax1_outer_inner], 40, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 2; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    ((uint4*)(T_dense + ((((((((((int)blockIdx.x) * 196608) + (i_inner_j_inner_fused_outer_outer_outer_outer * 98304)) + (((int)threadIdx.z) * 49152)) + (((int)threadIdx.y) * 24576)) + ((((int)threadIdx.x) >> 2) * 3072)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 3) * 8)))))[0] = ((uint4*)(placeholder_shared + ((((((i_inner_j_inner_fused_outer_outer_outer_outer * 1280) + (((int)threadIdx.z) * 640)) + (((int)threadIdx.y) * 320)) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)))))[0];
  }
}
