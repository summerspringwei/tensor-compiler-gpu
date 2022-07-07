#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <mma.h>

#include "../../../cuda_utils.h"

// dim3(4, 4,12), dim3(32,1,1)
// Note we set blockDim.y to 8 to compute softmax 
extern "C" __global__ void __launch_bounds__(32) fused_query_key_matmul_softmax(half* __restrict__ x, half* __restrict__ placeholder, half* __restrict__ compute, half* __restrict__ sum) {
  int blockIdx_x = (blockIdx.x % 4);
  int blockIdx_y = ((blockIdx.x / 4) % 4);
  int blockIdx_z = (blockIdx.x / (4*4));
  int threadIdx_x = threadIdx.x % 32;
  int threadIdx_y = threadIdx.x / 32;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> compute_wmma_accumulator[4];
  // __shared__ half x_shared[4352];
  // __shared__ half placeholder_shared[4352];
  extern half __shared__ shared_buff_fused[]; 
  half* x_shared = (half*)&shared_buff_fused[0];
  half* placeholder_shared = (half*)&shared_buff_fused[4352];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> x_shared_wmma_matrix_a[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[2];
  for (int i_c_outer_init = 0; i_c_outer_init < 2; ++i_c_outer_init) {
    for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init) {
      (void)nvcuda::wmma::fill_fragment(compute_wmma_accumulator[((i_c_outer_init * 2) + j_c_outer_init)], 0.000000e+00f);
    }
  }
  for (int ax1_ax2_fused_outer_outer_outer_outer = 0; ax1_ax2_fused_outer_outer_outer_outer < 8; ++ax1_ax2_fused_outer_outer_outer_outer) {
    ((uint4*)(x_shared + ((((ax1_ax2_fused_outer_outer_outer_outer * 544) + ((((int)threadIdx_x) >> 3) * 136)) + ((((int)threadIdx_x) & 7) * 8)))))[0] = ((uint4*)(x + (((((((int)blockIdx_z) * 8192) + (((int)blockIdx_x) * 2048)) + (ax1_ax2_fused_outer_outer_outer_outer * 256)) + (((int)threadIdx_x) * 8)))))[0];
  }
  for (int ax1_ax2_fused_outer_outer_outer_outer1 = 0; ax1_ax2_fused_outer_outer_outer_outer1 < 8; ++ax1_ax2_fused_outer_outer_outer_outer1) {
    ((uint4*)(placeholder_shared + ((((ax1_ax2_fused_outer_outer_outer_outer1 * 544) + ((((int)threadIdx_x) >> 3) * 136)) + ((((int)threadIdx_x) & 7) * 8)))))[0] = ((uint4*)(placeholder + (((((((int)blockIdx_z) * 8192) + (((int)blockIdx_y) * 2048)) + (ax1_ax2_fused_outer_outer_outer_outer1 * 256)) + (((int)threadIdx_x) * 8)))))[0];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
    for (int ax1_outer = 0; ax1_outer < 2; ++ax1_outer) {
      (void)nvcuda::wmma::load_matrix_sync(x_shared_wmma_matrix_a[ax1_outer], ((half *)x_shared + (((ax1_outer * 2176) + (k_outer_inner * 16)))), 136);
    }
    for (int ax1_outer1 = 0; ax1_outer1 < 2; ++ax1_outer1) {
      (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax1_outer1], ((half *)placeholder_shared + (((ax1_outer1 * 2176) + (k_outer_inner * 16)))), 136);
    }
    for (int i_c_outer = 0; i_c_outer < 2; ++i_c_outer) {
      for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
        (void)nvcuda::wmma::mma_sync(compute_wmma_accumulator[((i_c_outer * 2) + j_c_outer)], x_shared_wmma_matrix_a[i_c_outer], placeholder_shared_wmma_matrix_b[j_c_outer], compute_wmma_accumulator[((i_c_outer * 2) + j_c_outer)]);
      }
    }
  }
  __syncthreads();
  // Stores 4x 16x16(32x32), 640=16*40
  for (int ax1_outer_inner = 0; ax1_outer_inner < 2; ++ax1_outer_inner) {
    for (int ax2_outer_inner = 0; ax2_outer_inner < 2; ++ax2_outer_inner) {
      (void)nvcuda::wmma::store_matrix_sync(((half *)x_shared + (((ax1_outer_inner * 640) + (ax2_outer_inner * 16)))), compute_wmma_accumulator[((ax1_outer_inner * 2) + ax2_outer_inner)], 40, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  // x_shared: 32 * 32
  // compute sum
  const int num_iter = 32 / blockDim.y;
  // const int num_iter = 4;
  half reg_sum = 0;
  const int x_shared_row_stride = 40;
  #pragma unroll
  for(int i=0; i<num_iter; ++i){
    int row = i * blockDim.y + threadIdx_y;
    half ele = hexp((x_shared + row * x_shared_row_stride + threadIdx_x)[0]);
    (x_shared + row * x_shared_row_stride + threadIdx_x)[0] = ele;
    reg_sum = warpReduceSum(ele);
    __syncthreads();
    if(threadIdx_x==0){
      atomicAdd(sum + blockIdx_z * 128 + blockIdx_x * 32 + row, reg_sum);
    }
    __syncthreads();
  }
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  __syncthreads();
  __threadfence();
  grid.sync();
  // -----------------------------------------------------
  for(int i=0; i<num_iter; ++i){
    int row = i * blockDim.y + threadIdx_y;
    reg_sum = sum[blockIdx_z * 128 + blockIdx_x * x_shared_row_stride + row];
    (x_shared + row * x_shared_row_stride + threadIdx_x)[0] = (x_shared + row * x_shared_row_stride + threadIdx_x)[0] / reg_sum;
  }
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 4; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    ((uint4*)(compute + (((((((((int)blockIdx_z) * 16384) + (((int)blockIdx_x) * 4096)) + (i_inner_j_inner_fused_outer_outer_outer_outer * 1024)) + ((((int)threadIdx_x) >> 2) * 128)) + (((int)blockIdx_y) * 32)) + ((((int)threadIdx_x) & 3) * 8)))))[0] = 
    ((uint4*)(x_shared + ((((i_inner_j_inner_fused_outer_outer_outer_outer * 320) + ((((int)threadIdx_x) >> 2) * 40)) + ((((int)threadIdx_x) & 3) * 8)))))[0];
  }
}



// dim3(4, 4,12), dim3(32,1,1)
// Note we set blockDim.y to 8 to compute softmax faster
extern "C" __global__ void __launch_bounds__(32) fused_query_key_matmul_softmax_v2(half* __restrict__ x, half* __restrict__ placeholder, half* __restrict__ compute, half* __restrict__ sum) {
  int blockIdx_x = (blockIdx.x % 4);
  int blockIdx_y = ((blockIdx.x / 4) % 4);
  int blockIdx_z = (blockIdx.x / (4*4));
  int threadIdx_x = threadIdx.x % 32;
  int threadIdx_y = threadIdx.x / 32;
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> compute_wmma_accumulator[4];
  // __shared__ half x_shared[4352];
  // __shared__ half placeholder_shared[4352];
  extern half __shared__ shared_buff_fused[]; 
  half* x_shared = (half*)&shared_buff_fused[0];
  half* placeholder_shared = (half*)&shared_buff_fused[4352];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> x_shared_wmma_matrix_a[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[2];
  if(threadIdx_y==0){
  for (int i_c_outer_init = 0; i_c_outer_init < 2; ++i_c_outer_init) {
    for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init) {
      (void)nvcuda::wmma::fill_fragment(compute_wmma_accumulator[((i_c_outer_init * 2) + j_c_outer_init)], 0.000000e+00f);
    }
  }
  for (int ax1_ax2_fused_outer_outer_outer_outer = 0; ax1_ax2_fused_outer_outer_outer_outer < 8; ++ax1_ax2_fused_outer_outer_outer_outer) {
    ((uint4*)(x_shared + ((((ax1_ax2_fused_outer_outer_outer_outer * 544) + ((((int)threadIdx_x) >> 3) * 136)) + ((((int)threadIdx_x) & 7) * 8)))))[0] = ((uint4*)(x + (((((((int)blockIdx_z) * 8192) + (((int)blockIdx_x) * 2048)) + (ax1_ax2_fused_outer_outer_outer_outer * 256)) + (((int)threadIdx_x) * 8)))))[0];
  }
  for (int ax1_ax2_fused_outer_outer_outer_outer1 = 0; ax1_ax2_fused_outer_outer_outer_outer1 < 8; ++ax1_ax2_fused_outer_outer_outer_outer1) {
    ((uint4*)(placeholder_shared + ((((ax1_ax2_fused_outer_outer_outer_outer1 * 544) + ((((int)threadIdx_x) >> 3) * 136)) + ((((int)threadIdx_x) & 7) * 8)))))[0] = ((uint4*)(placeholder + (((((((int)blockIdx_z) * 8192) + (((int)blockIdx_y) * 2048)) + (ax1_ax2_fused_outer_outer_outer_outer1 * 256)) + (((int)threadIdx_x) * 8)))))[0];
  }
  }
  __syncthreads();
  if(threadIdx_y==0){
  for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
    for (int ax1_outer = 0; ax1_outer < 2; ++ax1_outer) {
      (void)nvcuda::wmma::load_matrix_sync(x_shared_wmma_matrix_a[ax1_outer], ((half *)x_shared + (((ax1_outer * 2176) + (k_outer_inner * 16)))), 136);
    }
    for (int ax1_outer1 = 0; ax1_outer1 < 2; ++ax1_outer1) {
      (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax1_outer1], ((half *)placeholder_shared + (((ax1_outer1 * 2176) + (k_outer_inner * 16)))), 136);
    }
    for (int i_c_outer = 0; i_c_outer < 2; ++i_c_outer) {
      for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
        (void)nvcuda::wmma::mma_sync(compute_wmma_accumulator[((i_c_outer * 2) + j_c_outer)], x_shared_wmma_matrix_a[i_c_outer], placeholder_shared_wmma_matrix_b[j_c_outer], compute_wmma_accumulator[((i_c_outer * 2) + j_c_outer)]);
      }
    }
  }
  }
  __syncthreads();
  // Stores 4x 16x16(32x32), 640=16*40
  if(threadIdx_y==0){
  for (int ax1_outer_inner = 0; ax1_outer_inner < 2; ++ax1_outer_inner) {
    for (int ax2_outer_inner = 0; ax2_outer_inner < 2; ++ax2_outer_inner) {
      (void)nvcuda::wmma::store_matrix_sync(((half *)x_shared + (((ax1_outer_inner * 640) + (ax2_outer_inner * 16)))), compute_wmma_accumulator[((ax1_outer_inner * 2) + ax2_outer_inner)], 40, nvcuda::wmma::mem_row_major);
    }
  }
  }
  __syncthreads();
  // x_shared: 32 * 32
  // compute sum
  const int num_iter = 32 / 8;
  // const int num_iter = 4;
  half reg_sum = 0;
  const int x_shared_row_stride = 40;
  #pragma unroll
  for(int i=0; i<num_iter; ++i){
    int row = i * blockDim.y + threadIdx_y;
    half ele = hexp((x_shared + row * x_shared_row_stride + threadIdx_x)[0]);
    (x_shared + row * x_shared_row_stride + threadIdx_x)[0] = ele;
    reg_sum = warpReduceSum(ele);
    __syncthreads();
    if(threadIdx_x==0){
      atomicAdd(sum + blockIdx_z * 128 + blockIdx_x * 32 + row, reg_sum);
    }
    __syncthreads();
  }
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  __syncthreads();
  __threadfence();
  grid.sync();
  // -----------------------------------------------------
  #pragma unroll
  for(int i=0; i<num_iter; ++i){
    int row = i * blockDim.y + threadIdx_y;
    reg_sum = sum[blockIdx_z * 128 + blockIdx_x * x_shared_row_stride + row];
    (x_shared + row * x_shared_row_stride + threadIdx_x)[0] = (x_shared + row * x_shared_row_stride + threadIdx_x)[0] / reg_sum;
  }
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 4; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    ((uint4*)(compute + (((((((((int)blockIdx_z) * 16384) + (((int)blockIdx_x) * 4096)) + (i_inner_j_inner_fused_outer_outer_outer_outer * 1024)) + ((((int)threadIdx_x) >> 2) * 128)) + (((int)blockIdx_y) * 32)) + ((((int)threadIdx_x) & 3) * 8)))))[0] = 
    ((uint4*)(x_shared + ((((i_inner_j_inner_fused_outer_outer_outer_outer * 320) + ((((int)threadIdx_x) >> 2) * 40)) + ((((int)threadIdx_x) & 3) * 8)))))[0];
  }
}



// dim3(4, 4,12), dim3(32,1,1)
// Note we set blockDim.y to 8 to compute softmax 
extern "C" __global__ void __launch_bounds__(32) fused_query_key_matmul_softmax_v3(
  half* __restrict__ x, half* __restrict__ placeholder, half* __restrict__ compute, float* __restrict__ sum) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> compute_wmma_accumulator[4];
  // __shared__ half x_shared[4352];
  // __shared__ half placeholder_shared[4352];
  int blockIdx_x = (blockIdx.x % 4);
  int blockIdx_y = ((blockIdx.x / 4) % 4);
  int blockIdx_z = (blockIdx.x / (4*4));
  int threadIdx_x = threadIdx.x % 32;
  int threadIdx_y = threadIdx.x / 32;
  extern half __shared__ shared_buff_fused[]; 
  half* x_shared = (half*)&shared_buff_fused[0];
  half* placeholder_shared = (half*)&shared_buff_fused[4352];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> x_shared_wmma_matrix_a[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[2];
  for (int i_c_outer_init = 0; i_c_outer_init < 2; ++i_c_outer_init) {
    for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init) {
      (void)nvcuda::wmma::fill_fragment(compute_wmma_accumulator[((i_c_outer_init * 2) + j_c_outer_init)], 0.000000e+00f);
    }
  }
  for (int ax1_ax2_fused_outer_outer_outer_outer = 0; ax1_ax2_fused_outer_outer_outer_outer < 8; ++ax1_ax2_fused_outer_outer_outer_outer) {
    ((uint4*)(x_shared + ((((ax1_ax2_fused_outer_outer_outer_outer * 544) + ((((int)threadIdx_x) >> 3) * 136)) + ((((int)threadIdx_x) & 7) * 8)))))[0] = ((uint4*)(x + (((((((int)blockIdx_z) * 8192) + (((int)blockIdx_x) * 2048)) + (ax1_ax2_fused_outer_outer_outer_outer * 256)) + (((int)threadIdx_x) * 8)))))[0];
  }
  for (int ax1_ax2_fused_outer_outer_outer_outer1 = 0; ax1_ax2_fused_outer_outer_outer_outer1 < 8; ++ax1_ax2_fused_outer_outer_outer_outer1) {
    ((uint4*)(placeholder_shared + ((((ax1_ax2_fused_outer_outer_outer_outer1 * 544) + ((((int)threadIdx_x) >> 3) * 136)) + ((((int)threadIdx_x) & 7) * 8)))))[0] = ((uint4*)(placeholder + (((((((int)blockIdx_z) * 8192) + (((int)blockIdx_y) * 2048)) + (ax1_ax2_fused_outer_outer_outer_outer1 * 256)) + (((int)threadIdx_x) * 8)))))[0];
  }
  __syncthreads();
  for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
    for (int ax1_outer = 0; ax1_outer < 2; ++ax1_outer) {
      (void)nvcuda::wmma::load_matrix_sync(x_shared_wmma_matrix_a[ax1_outer], ((half *)x_shared + (((ax1_outer * 2176) + (k_outer_inner * 16)))), 136);
    }
    for (int ax1_outer1 = 0; ax1_outer1 < 2; ++ax1_outer1) {
      (void)nvcuda::wmma::load_matrix_sync(placeholder_shared_wmma_matrix_b[ax1_outer1], ((half *)placeholder_shared + (((ax1_outer1 * 2176) + (k_outer_inner * 16)))), 136);
    }
    for (int i_c_outer = 0; i_c_outer < 2; ++i_c_outer) {
      for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
        (void)nvcuda::wmma::mma_sync(compute_wmma_accumulator[((i_c_outer * 2) + j_c_outer)], x_shared_wmma_matrix_a[i_c_outer], placeholder_shared_wmma_matrix_b[j_c_outer], compute_wmma_accumulator[((i_c_outer * 2) + j_c_outer)]);
      }
    }
  }
  __syncthreads();
  // Stores 4x 16x16(32x32), 640=16*40
  for (int ax1_outer_inner = 0; ax1_outer_inner < 2; ++ax1_outer_inner) {
    for (int ax2_outer_inner = 0; ax2_outer_inner < 2; ++ax2_outer_inner) {
      (void)nvcuda::wmma::store_matrix_sync(((half *)x_shared + (((ax1_outer_inner * 640) + (ax2_outer_inner * 16)))), compute_wmma_accumulator[((ax1_outer_inner * 2) + ax2_outer_inner)], 40, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  // x_shared: 32 * 32
  // compute reduce sum
  const int x_shared_row_stride = 40;

  // const int num_iter = 32;
  // half reg_sum = 0;
  // #pragma unroll
  // for(int i=0; i<num_iter; ++i){
  //   reg_sum += x_shared[threadIdx_x * x_shared_row_stride + i];
  // }
  {
    // each thread compute half2
    const int num_iter = 32 / (sizeof(half2)/sizeof(half));
    float reg_sum = 0.0;
    half2 norm_factor(half(1.0/8), half(1.0/8)); // 1/sqrt(64) = 1/8
    #pragma unroll
    for(int i=0; i<num_iter; ++i){
      auto tmp = ((half2*)(x_shared + threadIdx_x * x_shared_row_stride + i * 2))[0];
      // Do normalization: attention_scores = attention_scores / math.sqrt(self.attention_head_size)
      tmp = tmp * norm_factor;
      tmp = h2exp(tmp);
      ((half2*)(x_shared + threadIdx_x * x_shared_row_stride + i * 2))[0] = tmp;
      reg_sum += (__half2float(tmp.x) + __half2float(tmp.y));
    }
    atomicAdd(sum + blockIdx_z * 128 + blockIdx_x * 32 + threadIdx_x, reg_sum);
  }
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  __syncthreads();
  __threadfence();
  grid.sync();
  // -----------------------------------------------------
  {
    const int num_iter = 32;
    #pragma unroll
    for(int i=0; i<num_iter; ++i){ 
      int row = i;
      auto reg_sum = sum[blockIdx_z * 128 + blockIdx_x * 32 + row];
      (x_shared + row * x_shared_row_stride + threadIdx_x)[0] = __float2half(__half2float((x_shared + row * x_shared_row_stride + threadIdx_x)[0]) / reg_sum);
    }
  }
  __syncthreads();
  // 16384=12
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 4; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    ((uint4*)(compute + (((((((((int)blockIdx_z) * 16384) + (((int)blockIdx_x) * 4096)) + (i_inner_j_inner_fused_outer_outer_outer_outer * 1024)) + ((((int)threadIdx_x) >> 2) * 128)) + (((int)blockIdx_y) * 32)) + ((((int)threadIdx_x) & 3) * 8)))))[0] = 
    ((uint4*)(x_shared + ((((i_inner_j_inner_fused_outer_outer_outer_outer * 320) + ((((int)threadIdx_x) >> 2) * 40)) + ((((int)threadIdx_x) & 3) * 8)))))[0];
  }
}
