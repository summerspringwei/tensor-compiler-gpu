// dim3(64, 1, 1), dim3(32, 2, 4)
extern "C" __global__ void __launch_bounds__(256) default_function_kernel0(half* __restrict__ x, half* __restrict__ weight, half* __restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, half> T_dense_wmma_accumulator[4];
  __shared__ half reshape_permute_shared[2560];
  __shared__ half weight_shared[8704];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, half, nvcuda::wmma::row_major> reshape_permute_shared_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, half, nvcuda::wmma::col_major> weight_shared_wmma_matrix_b[1];
  for (int i_c_outer_init = 0; i_c_outer_init < 4; ++i_c_outer_init) {
    (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[i_c_outer_init], 0.000000e+00f);
  }
  for (int k_outer_outer = 0; k_outer_outer < 4; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 8; ++ax0_ax1_fused_outer_outer_outer_outer) {
      reshape_permute_shared[(((((ax0_ax1_fused_outer_outer_outer_outer * 320) + (((int)threadIdx.z) * 80)) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)))] = x[(((((((((int)blockIdx.x) * 8192) + (ax0_ax1_fused_outer_outer_outer_outer * 1024)) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 128)) + (((int)threadIdx.x) * 16)) + k_outer_outer))];
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0; ax0_ax1_fused_outer_outer_outer_outer1 < 16; ++ax0_ax1_fused_outer_outer_outer_outer1) {
      weight_shared[(((((ax0_ax1_fused_outer_outer_outer_outer1 * 320) + (((int)threadIdx.z) * 80)) + (((int)threadIdx.y) * 40)) + ((int)threadIdx.x)))] = weight[((((((ax0_ax1_fused_outer_outer_outer_outer1 * 1024) + (((int)threadIdx.z) * 256)) + (((int)threadIdx.y) * 128)) + (k_outer_outer * 32)) + ((int)threadIdx.x)))];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
      for (int ax0_outer = 0; ax0_outer < 4; ++ax0_outer) {
        (void)nvcuda::wmma::load_matrix_sync(reshape_permute_shared_wmma_matrix_a[ax0_outer], ((half *)reshape_permute_shared + ((((((int)threadIdx.y) * 1280) + (ax0_outer * 320)) + (k_outer_inner * 16)))), 40);
      }
      (void)nvcuda::wmma::load_matrix_sync(weight_shared_wmma_matrix_b[0], ((half *)weight_shared + (((((int)threadIdx.z) * 1280) + (k_outer_inner * 16)))), 40);
      for (int i_c_outer = 0; i_c_outer < 4; ++i_c_outer) {
        (void)nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[i_c_outer], reshape_permute_shared_wmma_matrix_a[i_c_outer], weight_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[i_c_outer]);
      }
    }
  }
  __syncthreads();
  for (int ax0_outer_inner = 0; ax0_outer_inner < 4; ++ax0_outer_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((half *)weight_shared + ((((((int)threadIdx.y) * 4352) + (ax0_outer_inner * 1088)) + (((int)threadIdx.z) * 32)))), T_dense_wmma_accumulator[ax0_outer_inner], 136, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 32; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    T_dense[((((((((int)blockIdx.x) * 8192) + (i_inner_j_inner_fused_outer_outer_outer_outer * 256)) + (((int)threadIdx.z) * 64)) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)))] = weight_shared[((((i_inner_j_inner_fused_outer_outer_outer_outer * 272) + (((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) >> 7) * 136)) + ((((((int)threadIdx.z) * 64) + (((int)threadIdx.y) * 32)) + ((int)threadIdx.x)) & 127)))];
  }
}
