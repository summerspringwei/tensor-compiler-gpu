
// ([4096, 512]) * ([128, 512]) -> [4096, 128]
#include <cuda_fp16.h>

#ifndef TVM_HELPER_FUNC
#define TVM_HELPER_FUNC
// Pack two half values.
static inline __device__ __host__ unsigned
__pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// fix undefined fp16 match function
static inline __device__ __host__ half hpow(half x, half y) {
  float tmp_x = __half2float(x);
  float tmp_y = __half2float(y);
  float result = powf(tmp_x, tmp_y);
  return __float2half(result);
}

static inline __device__ __host__ half htanh(half x) {
  float tmp_x = __half2float(x);
  float result = tanhf(tmp_x);
  return __float2half(result);
}
#endif
#include <mma.h>

// dim3(64, 2, 1), dim3(32, 2, 2)
// shared_memory: 18432
extern "C" __global__ void __launch_bounds__(128) feed_forward_fc2_m4096_n128_k512(half* __restrict__ x, half* __restrict__ weight, half* __restrict__ add, half* __restrict__ short_cut) {
  
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, half> T_dense_wmma_accumulator[4];
  // extern __shared__ float all_shared_mem[];
  // half* x_shared = (half*)all_shared_mem;
  // half* weight_shared = ((half*)all_shared_mem) + 4608;
  __shared__ half x_shared[4608];
  __shared__ half weight_shared[4608];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, half, nvcuda::wmma::row_major> x_shared_wmma_matrix_a[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, half, nvcuda::wmma::col_major> weight_shared_wmma_matrix_b[1];
  for (int i_c_outer_init = 0; i_c_outer_init < 4; ++i_c_outer_init) {
    (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[i_c_outer_init], 0.000000e+00f);
  }
  for (int k_outer_outer = 0; k_outer_outer < 8; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0; ax0_ax1_fused_outer_outer_outer_outer < 4; ++ax0_ax1_fused_outer_outer_outer_outer) {
      ((uint4*)(x_shared + ((((((ax0_ax1_fused_outer_outer_outer_outer * 1152) + (((int)threadIdx.z) * 576)) + (((int)threadIdx.y) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))))[0] = ((uint4*)(x + ((((((((((int)blockIdx.x) * 32768) + (ax0_ax1_fused_outer_outer_outer_outer * 8192)) + (((int)threadIdx.z) * 4096)) + (((int)threadIdx.y) * 2048)) + ((((int)threadIdx.x) >> 3) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 7) * 8)))))[0];
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0; ax0_ax1_fused_outer_outer_outer_outer1 < 4; ++ax0_ax1_fused_outer_outer_outer_outer1) {
      ((uint4*)(weight_shared + ((((((ax0_ax1_fused_outer_outer_outer_outer1 * 1152) + (((int)threadIdx.z) * 576)) + (((int)threadIdx.y) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))))[0] = ((uint4*)(weight + ((((((((((int)blockIdx.y) * 32768) + (ax0_ax1_fused_outer_outer_outer_outer1 * 8192)) + (((int)threadIdx.z) * 4096)) + (((int)threadIdx.y) * 2048)) + ((((int)threadIdx.x) >> 3) * 512)) + (k_outer_outer * 64)) + ((((int)threadIdx.x) & 7) * 8)))))[0];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
      for (int ax0_outer = 0; ax0_outer < 4; ++ax0_outer) {
        (void)nvcuda::wmma::load_matrix_sync(x_shared_wmma_matrix_a[ax0_outer], ((half *)x_shared + ((((((int)threadIdx.y) * 2304) + (ax0_outer * 576)) + (k_outer_inner * 16)))), 72);
      }
      (void)nvcuda::wmma::load_matrix_sync(weight_shared_wmma_matrix_b[0], ((half *)weight_shared + (((((int)threadIdx.z) * 2304) + (k_outer_inner * 16)))), 72);
      for (int i_c_outer = 0; i_c_outer < 4; ++i_c_outer) {
        (void)nvcuda::wmma::mma_sync(T_dense_wmma_accumulator[i_c_outer], x_shared_wmma_matrix_a[i_c_outer], weight_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[i_c_outer]);
      }
    }
  }
  __syncthreads();
  for (int ax0_outer_inner = 0; ax0_outer_inner < 4; ++ax0_outer_inner) {
    (void)nvcuda::wmma::store_matrix_sync(((half *)x_shared + ((((((int)threadIdx.y) * 2304) + (ax0_outer_inner * 576)) + (((int)threadIdx.z) * 32)))), T_dense_wmma_accumulator[ax0_outer_inner], 72, nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 4; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    uint4 _1;
      uint4 _2 = ((uint4*)(x_shared + ((((((i_inner_j_inner_fused_outer_outer_outer_outer * 1152) + (((int)threadIdx.z) * 576)) + (((int)threadIdx.y) * 288)) + ((((int)threadIdx.x) >> 3) * 72)) + ((((int)threadIdx.x) & 7) * 8)))))[0];
      uint4 _3 = ((uint4*)(short_cut + ((((((((((int)blockIdx.x) * 32768) + (i_inner_j_inner_fused_outer_outer_outer_outer * 8192)) + (((int)threadIdx.z) * 4096)) + (((int)threadIdx.y) * 2048)) + ((((int)threadIdx.x) >> 3) * 512)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)))))[0];
      ((half2*)(&(_1.x)))->x = (((half2*)(&(_2.x)))->x+((half2*)(&(_3.x)))->x);
      ((half2*)(&(_1.x)))->y = (((half2*)(&(_2.x)))->y+((half2*)(&(_3.x)))->y);
      ((half2*)(&(_1.y)))->x = (((half2*)(&(_2.y)))->x+((half2*)(&(_3.y)))->x);
      ((half2*)(&(_1.y)))->y = (((half2*)(&(_2.y)))->y+((half2*)(&(_3.y)))->y);
      ((half2*)(&(_1.z)))->x = (((half2*)(&(_2.z)))->x+((half2*)(&(_3.z)))->x);
      ((half2*)(&(_1.z)))->y = (((half2*)(&(_2.z)))->y+((half2*)(&(_3.z)))->y);
      ((half2*)(&(_1.w)))->x = (((half2*)(&(_2.w)))->x+((half2*)(&(_3.w)))->x);
      ((half2*)(&(_1.w)))->y = (((half2*)(&(_2.w)))->y+((half2*)(&(_3.w)))->y);
    ((uint4*)(add + ((((((((((int)blockIdx.x) * 8192) + (i_inner_j_inner_fused_outer_outer_outer_outer * 2048)) + (((int)threadIdx.z) * 1024)) + (((int)threadIdx.y) * 512)) + ((((int)threadIdx.x) >> 3) * 128)) + (((int)blockIdx.y) * 64)) + ((((int)threadIdx.x) & 7) * 8)))))[0] = _1;
  }
}

