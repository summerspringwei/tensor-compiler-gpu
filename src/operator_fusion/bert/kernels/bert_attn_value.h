#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <mma.h>

// Computes (12, 128, 128) * (12, 64, 128) -> (12, 128, 64)
// dim3(8, 2,12), dim3(32,1,1)
extern "C" __global__ void __launch_bounds__(32)
    attn_value_matmul(half* __restrict__ x, half* __restrict__ placeholder,
                  half* __restrict__ compute) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half>
      compute_wmma_accumulator[2];
  __shared__ half x_shared[2176];
  __shared__ half placeholder_shared[4352];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half,
                         nvcuda::wmma::row_major>
      x_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half,
                         nvcuda::wmma::col_major>
      placeholder_shared_wmma_matrix_b[2];
  for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init) {
    (void)nvcuda::wmma::fill_fragment(compute_wmma_accumulator[j_c_outer_init],
                                      0.000000e+00f);
  }
  for (int ax1_ax2_fused_outer_outer_outer_outer = 0;
       ax1_ax2_fused_outer_outer_outer_outer < 8;
       ++ax1_ax2_fused_outer_outer_outer_outer) {
    ((uint4*)(x_shared + ((((ax1_ax2_fused_outer_outer_outer_outer * 272) +
                            ((((int)threadIdx_x) >> 4) * 136)) +
                           ((((int)threadIdx_x) & 15) * 8)))))[0] =
        ((uint4*)(x + (((((((int)blockIdx_z) * 16384) +
                          (((int)blockIdx_x) * 2048)) +
                         (ax1_ax2_fused_outer_outer_outer_outer * 256)) +
                        (((int)threadIdx_x) * 8)))))[0];
  }
  for (int ax1_ax2_fused_outer_outer_outer_outer1 = 0;
       ax1_ax2_fused_outer_outer_outer_outer1 < 16;
       ++ax1_ax2_fused_outer_outer_outer_outer1) {
    ((uint4*)(placeholder_shared +
              ((((ax1_ax2_fused_outer_outer_outer_outer1 * 272) +
                 ((((int)threadIdx_x) >> 4) * 136)) +
                ((((int)threadIdx_x) & 15) * 8)))))[0] =
        ((uint4*)(placeholder +
                  (((((((int)blockIdx_z) * 8192) + (((int)blockIdx_y) * 4096)) +
                     (ax1_ax2_fused_outer_outer_outer_outer1 * 256)) +
                    (((int)threadIdx_x) * 8)))))[0];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
    (void)nvcuda::wmma::load_matrix_sync(
        x_shared_wmma_matrix_a[0], ((half*)x_shared + ((k_outer_inner * 16))),
        136);
    for (int ax1_outer = 0; ax1_outer < 2; ++ax1_outer) {
      (void)nvcuda::wmma::load_matrix_sync(
          placeholder_shared_wmma_matrix_b[ax1_outer],
          ((half*)placeholder_shared +
           (((ax1_outer * 2176) + (k_outer_inner * 16)))),
          136);
    }
    for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
      (void)nvcuda::wmma::mma_sync(compute_wmma_accumulator[j_c_outer],
                                   x_shared_wmma_matrix_a[0],
                                   placeholder_shared_wmma_matrix_b[j_c_outer],
                                   compute_wmma_accumulator[j_c_outer]);
    }
  }
  __syncthreads();
  for (int ax2_outer_inner = 0; ax2_outer_inner < 2; ++ax2_outer_inner) {
    (void)nvcuda::wmma::store_matrix_sync(
        ((half*)x_shared + ((ax2_outer_inner * 16))),
        compute_wmma_accumulator[ax2_outer_inner], 32,
        nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0;
       i_inner_j_inner_fused_outer_outer_outer_outer < 2;
       ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    ((uint4*)(compute +
              (((((((((int)blockIdx_z) * 8192) + (((int)blockIdx_x) * 1024)) +
                   (i_inner_j_inner_fused_outer_outer_outer_outer * 512)) +
                  ((((int)threadIdx_x) >> 2) * 64)) +
                 (((int)blockIdx_y) * 32)) +
                ((((int)threadIdx_x) & 3) * 8)))))[0] =
        ((uint4*)(x_shared +
                  (((i_inner_j_inner_fused_outer_outer_outer_outer * 256) +
                    (((int)threadIdx_x) * 8)))))[0];
  }
}
