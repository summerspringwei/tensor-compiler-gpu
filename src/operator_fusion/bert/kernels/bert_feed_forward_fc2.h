
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <mma.h>

// // dim3(4,24,1), dim3(32,4,1) to dim3(96,1,1), dim3(128,1,1)
// extern "C" __global__ void __launch_bounds__(128)
//     fc1(half *__restrict__ x, half *__restrict__ placeholder,
//         half *__restrict__ T_dense) {
//   nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, half>
//       T_dense_wmma_accumulator[1];
//   __shared__ half x_shared[4352];
//   __shared__ half placeholder_shared[4352];
//   const int blockIdx_x = blockIdx.x % 4;
//   const int blockIdx_y = blockIdx.x / 4;
//   const int threadIdx_x = threadIdx.x % 32;
//   const int threadIdx_y = threadIdx.x / 32;
//   nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, half,
//                          nvcuda::wmma::row_major>
//       x_shared_wmma_matrix_a[1];
//   nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, half,
//                          nvcuda::wmma::col_major>
//       placeholder_shared_wmma_matrix_b[1];
//   (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[0], 0.000000e+00f);
//   for (int k_outer_outer = 0; k_outer_outer < 24; ++k_outer_outer) {
//     __syncthreads();
//     for (int ax0_ax1_fused_outer_outer_outer_outer = 0;
//          ax0_ax1_fused_outer_outer_outer_outer < 4;
//          ++ax0_ax1_fused_outer_outer_outer_outer) {
//       ((uint4 *)(x_shared + (((((ax0_ax1_fused_outer_outer_outer_outer * 1088) +
//                                 (((int)threadIdx_y) * 272)) +
//                                ((((int)threadIdx_x) >> 4) * 136)) +
//                               ((((int)threadIdx_x) & 15) * 8)))))[0] =
//           ((uint4 *)(x +
//                      (((((((((int)blockIdx_x) * 98304) +
//                            (ax0_ax1_fused_outer_outer_outer_outer * 24576)) +
//                           (((int)threadIdx_y) * 6144)) +
//                          ((((int)threadIdx_x) >> 4) * 3072)) +
//                         (k_outer_outer * 128)) +
//                        ((((int)threadIdx_x) & 15) * 8)))))[0];
//     }
//     for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
//          ax0_ax1_fused_outer_outer_outer_outer1 < 4;
//          ++ax0_ax1_fused_outer_outer_outer_outer1) {
//       ((uint4 *)(placeholder_shared +
//                  (((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) +
//                      (((int)threadIdx_y) * 272)) +
//                     ((((int)threadIdx_x) >> 4) * 136)) +
//                    ((((int)threadIdx_x) & 15) * 8)))))[0] =
//           ((uint4 *)(placeholder +
//                      (((((((((int)blockIdx_y) * 98304) +
//                            (ax0_ax1_fused_outer_outer_outer_outer1 * 24576)) +
//                           (((int)threadIdx_y) * 6144)) +
//                          ((((int)threadIdx_x) >> 4) * 3072)) +
//                         (k_outer_outer * 128)) +
//                        ((((int)threadIdx_x) & 15) * 8)))))[0];
//     }
//     __syncthreads();
//     for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
//       (void)nvcuda::wmma::load_matrix_sync(
//           x_shared_wmma_matrix_a[0],
//           ((half *)x_shared +
//            (((((int)threadIdx_y) * 1088) + (k_outer_inner * 16)))),
//           136);
//       (void)nvcuda::wmma::load_matrix_sync(
//           placeholder_shared_wmma_matrix_b[0],
//           ((half *)placeholder_shared + ((k_outer_inner * 16))), 136);
//       (void)nvcuda::wmma::mma_sync(
//           T_dense_wmma_accumulator[0], x_shared_wmma_matrix_a[0],
//           placeholder_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[0]);
//     }
//   }
//   __syncthreads();
//   (void)nvcuda::wmma::store_matrix_sync(
//       ((half *)x_shared + ((((int)threadIdx_y) * 320))),
//       T_dense_wmma_accumulator[0], 40, nvcuda::wmma::mem_row_major);
//   __syncthreads();
//   // 24576=32*768, 6144=16*768, 768,
//   // Each block computes (32, 32)
//   ((uint4 *)(T_dense +
//              ((((((((int)blockIdx_x) * 24576) + (((int)threadIdx_y) * 6144)) +
//                  ((((int)threadIdx_x) >> 2) * 768)) +
//                 (((int)blockIdx_y) * 32)) +
//                ((((int)threadIdx_x) & 3) * 8)))))[0] =
//       ((uint4 *)(x_shared + ((((((int)threadIdx_y) * 320) +
//                                ((((int)threadIdx_x) >> 2) * 40)) +
//                               ((((int)threadIdx_x) & 3) * 8)))))[0];
// }

// dim3(4, 24,1), dim3(32,4,1)
extern "C" __global__ void __launch_bounds__(128) fc2(half* __restrict__ x, half* __restrict__ placeholder, half* __restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, half> T_dense_wmma_accumulator[1];
  extern half __shared__ shared_buff[]; 
  half* x_shared = (half*)&shared_buff[0];
  half* placeholder_shared = (half*)&shared_buff[4352];
  // __shared__ half x_shared[4352];
  // __shared__ half placeholder_shared[4352];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, half, nvcuda::wmma::row_major> x_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, half, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[1];
  (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[0], 0.000000e+00f);
  for (int k_outer_outer = 0; k_outer_outer < 24; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 4; ++ax0_ax1_fused_outer_outer_outer_outer) {
      ((uint4*)(x_shared + (((((ax0_ax1_fused_outer_outer_outer_outer * 1088) + (((int)threadIdx.y) * 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)))))[0] = ((uint4*)(x + (((((((((int)blockIdx.x) * 98304) + (ax0_ax1_fused_outer_outer_outer_outer * 24576)) + (((int)threadIdx.y) * 6144)) + ((((int)threadIdx.x) >> 4) * 3072)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) & 15) * 8)))))[0];
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0; ax0_ax1_fused_outer_outer_outer_outer1 < 4; ++ax0_ax1_fused_outer_outer_outer_outer1) {
      ((uint4*)(placeholder_shared + (((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) + (((int)threadIdx.y) * 272)) + ((((int)threadIdx.x) >> 4) * 136)) + ((((int)threadIdx.x) & 15) * 8)))))[0] = ((uint4*)(placeholder + (((((((((int)blockIdx.y) * 98304) + (ax0_ax1_fused_outer_outer_outer_outer1 * 24576)) + (((int)threadIdx.y) * 6144)) + ((((int)threadIdx.x) >> 4) * 3072)) + (k_outer_outer * 128)) + ((((int)threadIdx.x) & 15) * 8)))))[0];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      (void)nvcuda::wmma::load_matrix_sync(x_shared_wmma_matrix_a[0], ((half *)x_shared + (((((int)threadIdx.y) * 1088) + (k_outer_inner * 16)))), 136);
      (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[0], ((half *)placeholder_shared + ((k_outer_inner * 16))), 136);
      (void)nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[0], x_shared_wmma_matrix_a[0], placeholder_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[0]);
    }
  }
  __syncthreads();
  (void)nvcuda::wmma::store_matrix_sync(((half *)x_shared + ((((int)threadIdx.y) * 320))), T_dense_wmma_accumulator[0], 40, nvcuda::wmma::mem_row_major);
  __syncthreads();
  ((uint4*)(T_dense + ((((((((int)blockIdx.x) * 24576) + (((int)threadIdx.y) * 6144)) + ((((int)threadIdx.x) >> 2) * 768)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 3) * 8)))))[0] = ((uint4*)(x_shared + ((((((int)threadIdx.y) * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)))))[0];
}
