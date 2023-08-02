#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <mma.h>


// (128, 768) * (3*768, 768) -> (128, 3*768=2304)
// dim3(4, 36,1), dim3(32,2,1), each block computes (32, 64), each warp compute (32, 32)
extern "C" __global__ void __launch_bounds__(64)
    attn_qkv_matmul(half* __restrict__ x, half* __restrict__ placeholder,
                    half* __restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half>
      T_dense_wmma_accumulator[4];
  __shared__ half x_shared[4352];
  __shared__ half placeholder_shared[8704];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half,
                         nvcuda::wmma::row_major>
      x_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half,
                         nvcuda::wmma::col_major>
      placeholder_shared_wmma_matrix_b[4];
  for (int j_c_outer_init = 0; j_c_outer_init < 4; ++j_c_outer_init) {
    (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[j_c_outer_init],
                                      0.000000e+00f);
  }
  for (int k_outer_outer = 0; k_outer_outer < 6; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0;
         ax0_ax1_fused_outer_outer_outer_outer < 8;
         ++ax0_ax1_fused_outer_outer_outer_outer) {
      ((uint4*)(x_shared + (((((ax0_ax1_fused_outer_outer_outer_outer * 544) +
                               (((int)threadIdx.y) * 272)) +
                              ((((int)threadIdx.x) >> 4) * 136)) +
                             ((((int)threadIdx.x) & 15) * 8)))))[0] =
          ((uint4*)(x + (((((((((int)blockIdx.x) * 24576) +
                              (ax0_ax1_fused_outer_outer_outer_outer * 3072)) +
                             (((int)threadIdx.y) * 1536)) +
                            ((((int)threadIdx.x) >> 4) * 768)) +
                           (k_outer_outer * 128)) +
                          ((((int)threadIdx.x) & 15) * 8)))))[0];
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
         ax0_ax1_fused_outer_outer_outer_outer1 < 16;
         ++ax0_ax1_fused_outer_outer_outer_outer1) {
      ((uint4*)(placeholder_shared +
                (((((ax0_ax1_fused_outer_outer_outer_outer1 * 544) +
                    (((int)threadIdx.y) * 272)) +
                   ((((int)threadIdx.x) >> 4) * 136)) +
                  ((((int)threadIdx.x) & 15) * 8)))))[0] =
          ((uint4*)(placeholder +
                    (((((((((int)blockIdx.y) * 49152) +
                          (ax0_ax1_fused_outer_outer_outer_outer1 * 3072)) +
                         (((int)threadIdx.y) * 1536)) +
                        ((((int)threadIdx.x) >> 4) * 768)) +
                       (k_outer_outer * 128)) +
                      ((((int)threadIdx.x) & 15) * 8)))))[0];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      (void)nvcuda::wmma::load_matrix_sync(
          x_shared_wmma_matrix_a[0],
          ((half*)x_shared +
           (((((int)threadIdx.y) * 2176) + (k_outer_inner * 16)))),
          136);
      for (int ax0_outer = 0; ax0_outer < 4; ++ax0_outer) {
        (void)nvcuda::wmma::load_matrix_sync(
            placeholder_shared_wmma_matrix_b[ax0_outer],
            ((half*)placeholder_shared +
             (((ax0_outer * 2176) + (k_outer_inner * 16)))),
            136);
      }
      for (int j_c_outer = 0; j_c_outer < 4; ++j_c_outer) {
        (void)nvcuda::wmma::mma_sync(
            T_dense_wmma_accumulator[j_c_outer], x_shared_wmma_matrix_a[0],
            placeholder_shared_wmma_matrix_b[j_c_outer],
            T_dense_wmma_accumulator[j_c_outer]);
      }
    }
  }
  __syncthreads();
  for (int ax1_outer_inner = 0; ax1_outer_inner < 4; ++ax1_outer_inner) {
    (void)nvcuda::wmma::store_matrix_sync(
        ((half*)x_shared +
         (((((int)threadIdx.y) * 1152) + (ax1_outer_inner * 16)))),
        T_dense_wmma_accumulator[ax1_outer_inner], 72,
        nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  // 73728=32*2304, 18432=2304*8, 9216=2304*4, 
  // blockIdx.x compute 32 rows, blockIdx.y compute 64 cols
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0;
       i_inner_j_inner_fused_outer_outer_outer_outer < 4;
       ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    ((uint4*)(T_dense +
              (((((((((int)blockIdx.x) * 73728) +
                    (i_inner_j_inner_fused_outer_outer_outer_outer * 18432)) +
                   (((int)threadIdx.y) * 9216)) +
                  ((((int)threadIdx.x) >> 3) * 2304)) +
                 (((int)blockIdx.y) * 64)) +
                ((((int)threadIdx.x) & 7) * 8)))))[0] =
        ((uint4*)(x_shared +
                  (((((i_inner_j_inner_fused_outer_outer_outer_outer * 576) +
                      (((int)threadIdx.y) * 288)) +
                     ((((int)threadIdx.x) >> 3) * 72)) +
                    ((((int)threadIdx.x) & 7) * 8)))))[0];
  }
}







// Computes (12, 128, 128) * (12, 64, 128) -> (12, 128, 64)
// dim3(8, 2,12), dim3(32,1,1)
extern "C" __global__ void __launch_bounds__(32)
    attn_v_matmul(half* __restrict__ x, half* __restrict__ placeholder,
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
                            ((((int)threadIdx.x) >> 4) * 136)) +
                           ((((int)threadIdx.x) & 15) * 8)))))[0] =
        ((uint4*)(x + (((((((int)blockIdx.z) * 16384) +
                          (((int)blockIdx.x) * 2048)) +
                         (ax1_ax2_fused_outer_outer_outer_outer * 256)) +
                        (((int)threadIdx.x) * 8)))))[0];
  }
  for (int ax1_ax2_fused_outer_outer_outer_outer1 = 0;
       ax1_ax2_fused_outer_outer_outer_outer1 < 16;
       ++ax1_ax2_fused_outer_outer_outer_outer1) {
    ((uint4*)(placeholder_shared +
              ((((ax1_ax2_fused_outer_outer_outer_outer1 * 272) +
                 ((((int)threadIdx.x) >> 4) * 136)) +
                ((((int)threadIdx.x) & 15) * 8)))))[0] =
        ((uint4*)(placeholder +
                  (((((((int)blockIdx.z) * 8192) + (((int)blockIdx.y) * 4096)) +
                     (ax1_ax2_fused_outer_outer_outer_outer1 * 256)) +
                    (((int)threadIdx.x) * 8)))))[0];
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
              (((((((((int)blockIdx.z) * 8192) + (((int)blockIdx.x) * 1024)) +
                   (i_inner_j_inner_fused_outer_outer_outer_outer * 512)) +
                  ((((int)threadIdx.x) >> 2) * 64)) +
                 (((int)blockIdx.y) * 32)) +
                ((((int)threadIdx.x) & 3) * 8)))))[0] =
        ((uint4*)(x_shared +
                  (((i_inner_j_inner_fused_outer_outer_outer_outer * 256) +
                    (((int)threadIdx.x) * 8)))))[0];
  }
}













// (128, 768), (768, 768) -> (128, 768)
// dim3(4, 24,1), dim3(32,4,1)
extern "C" __global__ void __launch_bounds__(128)
    attn_fc(half* __restrict__ x, half* __restrict__ placeholder,
            half* __restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, half>
      T_dense_wmma_accumulator[1];
  __shared__ half x_shared[4352];
  __shared__ half placeholder_shared[4352];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, half,
                         nvcuda::wmma::row_major>
      x_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, half,
                         nvcuda::wmma::col_major>
      placeholder_shared_wmma_matrix_b[1];
  (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[0], 0.000000e+00f);
  for (int k_outer_outer = 0; k_outer_outer < 6; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0;
         ax0_ax1_fused_outer_outer_outer_outer < 4;
         ++ax0_ax1_fused_outer_outer_outer_outer) {
      ((uint4*)(x_shared + (((((ax0_ax1_fused_outer_outer_outer_outer * 1088) +
                               (((int)threadIdx.y) * 272)) +
                              ((((int)threadIdx.x) >> 4) * 136)) +
                             ((((int)threadIdx.x) & 15) * 8)))))[0] =
          ((uint4*)(x + (((((((((int)blockIdx.x) * 24576) +
                              (ax0_ax1_fused_outer_outer_outer_outer * 6144)) +
                             (((int)threadIdx.y) * 1536)) +
                            ((((int)threadIdx.x) >> 4) * 768)) +
                           (k_outer_outer * 128)) +
                          ((((int)threadIdx.x) & 15) * 8)))))[0];
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
         ax0_ax1_fused_outer_outer_outer_outer1 < 4;
         ++ax0_ax1_fused_outer_outer_outer_outer1) {
      ((uint4*)(placeholder_shared +
                (((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) +
                    (((int)threadIdx.y) * 272)) +
                   ((((int)threadIdx.x) >> 4) * 136)) +
                  ((((int)threadIdx.x) & 15) * 8)))))[0] =
          ((uint4*)(placeholder +
                    (((((((((int)blockIdx.y) * 24576) +
                          (ax0_ax1_fused_outer_outer_outer_outer1 * 6144)) +
                         (((int)threadIdx.y) * 1536)) +
                        ((((int)threadIdx.x) >> 4) * 768)) +
                       (k_outer_outer * 128)) +
                      ((((int)threadIdx.x) & 15) * 8)))))[0];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      (void)nvcuda::wmma::load_matrix_sync(
          x_shared_wmma_matrix_a[0],
          ((half*)x_shared +
           (((((int)threadIdx.y) * 1088) + (k_outer_inner * 16)))),
          136);
      (void)nvcuda::wmma::load_matrix_sync(
          placeholder_shared_wmma_matrix_b[0],
          ((half*)placeholder_shared + ((k_outer_inner * 16))), 136);
      (void)nvcuda::wmma::mma_sync(
          T_dense_wmma_accumulator[0], x_shared_wmma_matrix_a[0],
          placeholder_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[0]);
    }
  }
  __syncthreads();
  (void)nvcuda::wmma::store_matrix_sync(
      ((half*)x_shared + ((((int)threadIdx.y) * 256))),
      T_dense_wmma_accumulator[0], 32, nvcuda::wmma::mem_row_major);
  __syncthreads();
  ((uint4*)(T_dense +
            ((((((((int)blockIdx.x) * 24576) + (((int)threadIdx.y) * 6144)) +
                ((((int)threadIdx.x) >> 2) * 768)) +
               (((int)blockIdx.y) * 32)) +
              ((((int)threadIdx.x) & 3) * 8)))))[0] =
      ((uint4*)(x_shared +
                (((((int)threadIdx.y) * 256) + (((int)threadIdx.x) * 8)))))[0];
}
