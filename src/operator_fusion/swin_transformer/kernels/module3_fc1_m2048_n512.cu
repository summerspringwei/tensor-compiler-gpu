// dim3(1, 1, 64), dim3(32, 2, 2)
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cuda/pipeline>
extern "C" __global__ void __launch_bounds__(32) default_function_kernel0(half* __restrict__ x, half* __restrict__ attn_v_weight, half* __restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, half> T_dense_wmma_accumulator[2];
  __shared__ half reshape_permute_shared[2176];
  __shared__ half attn_v_weight_shared[4352];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, half, nvcuda::wmma::row_major> reshape_permute_shared_wmma_matrix_a[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, half, nvcuda::wmma::col_major> weight_shared_wmma_matrix_b[1];
  for (int i_c_outer_init = 0; i_c_outer_init < 2; ++i_c_outer_init) {
    (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[i_c_outer_init], 0.000000e+00f);
  }
  for (int k_outer_outer = 0; k_outer_outer < 4; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 16; ++ax0_ax1_fused_outer_outer_outer_outer) {
      int4 _1 = make_int4(((((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) & 7) * 1024)) + (ax0_ax1_fused_outer_outer_outer_outer * 512)) + (k_outer_outer * 4)) + (((int)threadIdx.x) >> 3)))+(256*0), ((((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) & 7) * 1024)) + (ax0_ax1_fused_outer_outer_outer_outer * 512)) + (k_outer_outer * 4)) + (((int)threadIdx.x) >> 3)))+(256*1), ((((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) & 7) * 1024)) + (ax0_ax1_fused_outer_outer_outer_outer * 512)) + (k_outer_outer * 4)) + (((int)threadIdx.x) >> 3)))+(256*2), ((((((((int)blockIdx.x) * 8192) + ((((int)threadIdx.x) & 7) * 1024)) + (ax0_ax1_fused_outer_outer_outer_outer * 512)) + (k_outer_outer * 4)) + (((int)threadIdx.x) >> 3)))+(256*3));
      ((uint2*)(reshape_permute_shared + (((ax0_ax1_fused_outer_outer_outer_outer * 136) + (((int)threadIdx.x) * 4)))))[0] = make_uint2(__pack_half2(x[_1.x],x[_1.y]),__pack_half2(x[_1.z],x[_1.w]));
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0; ax0_ax1_fused_outer_outer_outer_outer1 < 32; ++ax0_ax1_fused_outer_outer_outer_outer1) {
      ((uint2*)(attn_v_weight_shared + (((ax0_ax1_fused_outer_outer_outer_outer1 * 136) + (((int)threadIdx.x) * 4)))))[0] = ((uint2*)(attn_v_weight + (((((((int)blockIdx.y) * 16384) + (ax0_ax1_fused_outer_outer_outer_outer1 * 512)) + (k_outer_outer * 128)) + (((int)threadIdx.x) * 4)))))[0];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
        (void)nvcuda::wmma::load_matrix_sync(reshape_permute_shared_wmma_matrix_a[ax0_outer], ((half *)reshape_permute_shared + (((ax0_outer * 1088) + (k_outer_inner * 16)))), 136);
      }
      (void)nvcuda::wmma::load_matrix_sync(weight_shared_wmma_matrix_b[0], ((half *)attn_v_weight_shared + ((k_outer_inner * 16))), 136);
      for (int i_c_outer = 0; i_c_outer < 2; ++i_c_outer) {
        (void)nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[i_c_outer], reshape_permute_shared_wmma_matrix_a[i_c_outer], weight_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[i_c_outer]);
      }
    }
  }
  __syncthreads();
  for (int ax0_outer_inner = 0; ax0_outer_inner < 2; ++ax0_outer_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((half *)reshape_permute_shared + ((ax0_outer_inner * 320))), T_dense_wmma_accumulator[ax0_outer_inner], 40, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 4; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    ((uint2*)(T_dense + ((((((((int)blockIdx.x) * 8192) + (i_inner_j_inner_fused_outer_outer_outer_outer * 2048)) + ((((int)threadIdx.x) >> 3) * 512)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 7) * 4)))))[0] = ((uint2*)(reshape_permute_shared + ((((i_inner_j_inner_fused_outer_outer_outer_outer * 160) + ((((int)threadIdx.x) >> 3) * 40)) + ((((int)threadIdx.x) & 7) * 4)))))[0];
  }
}
