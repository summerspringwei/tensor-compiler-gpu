
// (128, 768), (768, 768) -> (128, 768)
// dim3(4, 24,1), dim3(32,4,1)
extern "C" __global__ void __launch_bounds__(128)
    attn_fc(half* __restrict__ x, half* __restrict__ placeholder,
            half* __restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, half>
      T_dense_wmma_accumulator[1];
  __shared__ half x_shared[4352];// 32*136
  __shared__ half placeholder_shared[4352];
  const int blockIdx_x = blockIdx.x % 4;
const int blockIdx_y = blockIdx.x / 4;
const int threadIdx_x = threadIdx.x % 32;
const int threadIdx_y = threadIdx.x / 32;
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, half,
                         nvcuda::wmma::row_major>
      x_shared_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, half,
                         nvcuda::wmma::col_major>
      placeholder_shared_wmma_matrix_b[1];
  (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[0], 0.000000e+00f);
  for (int k_outer_outer = 0; k_outer_outer < 6; ++k_outer_outer) {
    __syncthreads();
    // x_shared input: 32*128
    // 1088 = 8*136, 272=2*136, 24576=32*768, 6144=8*768, 1536=2*768
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0;
         ax0_ax1_fused_outer_outer_outer_outer < 4;
         ++ax0_ax1_fused_outer_outer_outer_outer) {
      ((uint4*)(x_shared + (((((ax0_ax1_fused_outer_outer_outer_outer * 1088) +
                               (((int)threadIdx_y) * 272)) +
                              ((((int)threadIdx_x) >> 4) * 136)) +
                             ((((int)threadIdx_x) & 15) * 8)))))[0] =
          ((uint4*)(x + (((((((((int)blockIdx_x) * 24576) +
                              (ax0_ax1_fused_outer_outer_outer_outer * 6144)) +
                             (((int)threadIdx_y) * 1536)) +
                            ((((int)threadIdx_x) >> 4) * 768)) +
                           (k_outer_outer * 128)) +
                          ((((int)threadIdx_x) & 15) * 8)))))[0];
    }
    // 1088=8*136, 272=2*136, placeholder_shared: 32*128
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
         ax0_ax1_fused_outer_outer_outer_outer1 < 4;
         ++ax0_ax1_fused_outer_outer_outer_outer1) {
      ((uint4*)(placeholder_shared +
                (((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) +
                    (((int)threadIdx_y) * 272)) +
                   ((((int)threadIdx_x) >> 4) * 136)) +
                  ((((int)threadIdx_x) & 15) * 8)))))[0] =
          ((uint4*)(placeholder +
                    (((((((((int)blockIdx_y) * 24576) +
                          (ax0_ax1_fused_outer_outer_outer_outer1 * 6144)) +
                         (((int)threadIdx_y) * 1536)) +
                        ((((int)threadIdx_x) >> 4) * 768)) +
                       (k_outer_outer * 128)) +
                      ((((int)threadIdx_x) & 15) * 8)))))[0];
    }
    __syncthreads();
    //1088=8*128
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      (void)nvcuda::wmma::load_matrix_sync(
          x_shared_wmma_matrix_a[0],
          ((half*)x_shared +
           (((((int)threadIdx_y) * 1088) + (k_outer_inner * 16)))),
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
      ((half*)x_shared + ((((int)threadIdx_y) * 256))),
      T_dense_wmma_accumulator[0], 32, nvcuda::wmma::mem_row_major);
  __syncthreads();
  ((uint4*)(T_dense +
            ((((((((int)blockIdx_x) * 24576) + (((int)threadIdx_y) * 6144)) +
                ((((int)threadIdx_x) >> 2) * 768)) +
               (((int)blockIdx_y) * 32)) +
              ((((int)threadIdx_x) & 3) * 8)))))[0] =
      ((uint4*)(x_shared +
                (((((int)threadIdx_y) * 256) + (((int)threadIdx_x) * 8)))))[0];
}
