// dim3(16, 16, 1), dim3(32, 1, 1)
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>
extern "C" __global__ void __launch_bounds__(128) module3_fc2_m2048_n512(half* __restrict__ x, half* __restrict__ weight, half* __restrict__ add, half* __restrict__ short_cut) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> T_dense_wmma_accumulator[2];
  __shared__ half reshape_permute_shared[2176];
  __shared__ half weight_shared[17408];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> reshape_permute_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> weight_shared_wmma_matrix_b[2];
  for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init) {
    (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[j_c_outer_init], 0.000000e+00f);
  }
  for (int k_outer_outer = 0; k_outer_outer < 4; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 4; ++ax0_ax1_fused_outer_outer_outer_outer) {
      int4 _1 = make_int4((((((((((int)blockIdx.x) * 8192) + (ax0_ax1_fused_outer_outer_outer_outer * 2048)) + ((((int)threadIdx.x) & 7) * 1024)) + (((int)threadIdx.z) * 512)) + (k_outer_outer * 4)) + (((int)threadIdx.x) >> 3)))+(256*0), (((((((((int)blockIdx.x) * 8192) + (ax0_ax1_fused_outer_outer_outer_outer * 2048)) + ((((int)threadIdx.x) & 7) * 1024)) + (((int)threadIdx.z) * 512)) + (k_outer_outer * 4)) + (((int)threadIdx.x) >> 3)))+(256*1), (((((((((int)blockIdx.x) * 8192) + (ax0_ax1_fused_outer_outer_outer_outer * 2048)) + ((((int)threadIdx.x) & 7) * 1024)) + (((int)threadIdx.z) * 512)) + (k_outer_outer * 4)) + (((int)threadIdx.x) >> 3)))+(256*2), (((((((((int)blockIdx.x) * 8192) + (ax0_ax1_fused_outer_outer_outer_outer * 2048)) + ((((int)threadIdx.x) & 7) * 1024)) + (((int)threadIdx.z) * 512)) + (k_outer_outer * 4)) + (((int)threadIdx.x) >> 3)))+(256*3));
      ((uint2*)(reshape_permute_shared + ((((ax0_ax1_fused_outer_outer_outer_outer * 544) + (((int)threadIdx.z) * 136)) + (((int)threadIdx.x) * 4)))))[0] = make_uint2(__pack_half2(x[_1.x],x[_1.y]),__pack_half2(x[_1.z],x[_1.w]));
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0; ax0_ax1_fused_outer_outer_outer_outer1 < 32; ++ax0_ax1_fused_outer_outer_outer_outer1) {
      ((uint2*)(weight_shared + ((((ax0_ax1_fused_outer_outer_outer_outer1 * 544) + (((int)threadIdx.z) * 136)) + (((int)threadIdx.x) * 4)))))[0] = ((uint2*)(weight + ((((((((int)blockIdx.y) * 65536) + (ax0_ax1_fused_outer_outer_outer_outer1 * 2048)) + (((int)threadIdx.z) * 512)) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)))))[0];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      (void)nvcuda::wmma::load_matrix_sync(reshape_permute_shared_wmma_matrix_a[0], ((half *)reshape_permute_shared + ((k_outer_inner * 16))), 136);
      for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
        (void)nvcuda::wmma::load_matrix_sync(weight_shared_wmma_matrix_b[ax0_outer], ((half *)weight_shared + ((((((int)threadIdx.z) * 4352) + (ax0_outer * 2176)) + (k_outer_inner * 16)))), 136);
      }
      for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
        (void)nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[j_c_outer], reshape_permute_shared_wmma_matrix_a[0], weight_shared_wmma_matrix_b[j_c_outer], T_dense_wmma_accumulator[j_c_outer]);
      }
    }
  }
  __syncthreads();
  for (int ax1_outer_inner = 0; ax1_outer_inner < 2; ++ax1_outer_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((half *)reshape_permute_shared + (((((int)threadIdx.z) * 32) + (ax1_outer_inner * 16)))), T_dense_wmma_accumulator[ax1_outer_inner], 136, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 4; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    ((uint2*)(T_dense + ((((((((int)blockIdx.x) * 8192) + (i_inner_j_inner_fused_outer_outer_outer_outer * 2048)) + (((int)threadIdx.z) * 512)) + (((int)blockIdx.y) * 128)) + (((int)threadIdx.x) * 4)))))[0] = ((uint2*)(reshape_permute_shared + ((((i_inner_j_inner_fused_outer_outer_outer_outer * 544) + (((int)threadIdx.z) * 136)) + (((int)threadIdx.x) * 4)))))[0];
  }
}