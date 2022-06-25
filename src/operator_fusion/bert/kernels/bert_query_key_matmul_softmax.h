#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <mma.h>

#include "../../../cuda_utils.h"

// dim3(4, 4,12), dim3(32,1,1)
// Note we set blockDim.y to 8 to compute softmax 
extern "C" __global__ void __launch_bounds__(32) fused_query_key_matmul_softmax(half* __restrict__ x, half* __restrict__ placeholder, half* __restrict__ compute, half* __restrict__ sum) {
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
    ((uint4*)(x_shared + ((((ax1_ax2_fused_outer_outer_outer_outer * 544) + ((((int)threadIdx.x) >> 3) * 136)) + ((((int)threadIdx.x) & 7) * 8)))))[0] = ((uint4*)(x + (((((((int)blockIdx.z) * 8192) + (((int)blockIdx.x) * 2048)) + (ax1_ax2_fused_outer_outer_outer_outer * 256)) + (((int)threadIdx.x) * 8)))))[0];
  }
  for (int ax1_ax2_fused_outer_outer_outer_outer1 = 0; ax1_ax2_fused_outer_outer_outer_outer1 < 8; ++ax1_ax2_fused_outer_outer_outer_outer1) {
    ((uint4*)(placeholder_shared + ((((ax1_ax2_fused_outer_outer_outer_outer1 * 544) + ((((int)threadIdx.x) >> 3) * 136)) + ((((int)threadIdx.x) & 7) * 8)))))[0] = ((uint4*)(placeholder + (((((((int)blockIdx.z) * 8192) + (((int)blockIdx.y) * 2048)) + (ax1_ax2_fused_outer_outer_outer_outer1 * 256)) + (((int)threadIdx.x) * 8)))))[0];
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
    int row = i * blockDim.y + threadIdx.y;
    half ele = hexp((x_shared + row * x_shared_row_stride + threadIdx.x)[0]);
    (x_shared + row * x_shared_row_stride + threadIdx.x)[0] = ele;
    reg_sum = warpReduceSum(ele);
    __syncthreads();
    if(threadIdx.x==0){
      atomicAdd(sum + blockIdx.z * 128 + blockIdx.x * 32 + row, reg_sum);
    }
    __syncthreads();
  }
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  __syncthreads();
  __threadfence();
  grid.sync();
  // -----------------------------------------------------
  for(int i=0; i<num_iter; ++i){
    int row = i * blockDim.y + threadIdx.y;
    reg_sum = sum[blockIdx.z * 128 + blockIdx.x * x_shared_row_stride + row];
    (x_shared + row * x_shared_row_stride + threadIdx.x)[0] = (x_shared + row * x_shared_row_stride + threadIdx.x)[0] / reg_sum;
  }
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 4; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    ((uint4*)(compute + (((((((((int)blockIdx.z) * 16384) + (((int)blockIdx.x) * 4096)) + (i_inner_j_inner_fused_outer_outer_outer_outer * 1024)) + ((((int)threadIdx.x) >> 2) * 128)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 3) * 8)))))[0] = 
    ((uint4*)(x_shared + ((((i_inner_j_inner_fused_outer_outer_outer_outer * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)))))[0];
  }
}



// dim3(4, 4,12), dim3(32,1,1)
// Note we set blockDim.y to 8 to compute softmax faster
extern "C" __global__ void __launch_bounds__(32) fused_query_key_matmul_softmax_v2(half* __restrict__ x, half* __restrict__ placeholder, half* __restrict__ compute, half* __restrict__ sum) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> compute_wmma_accumulator[4];
  // __shared__ half x_shared[4352];
  // __shared__ half placeholder_shared[4352];
  extern half __shared__ shared_buff_fused[]; 
  half* x_shared = (half*)&shared_buff_fused[0];
  half* placeholder_shared = (half*)&shared_buff_fused[4352];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> x_shared_wmma_matrix_a[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[2];
  if(threadIdx.y==0){
  for (int i_c_outer_init = 0; i_c_outer_init < 2; ++i_c_outer_init) {
    for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init) {
      (void)nvcuda::wmma::fill_fragment(compute_wmma_accumulator[((i_c_outer_init * 2) + j_c_outer_init)], 0.000000e+00f);
    }
  }
  for (int ax1_ax2_fused_outer_outer_outer_outer = 0; ax1_ax2_fused_outer_outer_outer_outer < 8; ++ax1_ax2_fused_outer_outer_outer_outer) {
    ((uint4*)(x_shared + ((((ax1_ax2_fused_outer_outer_outer_outer * 544) + ((((int)threadIdx.x) >> 3) * 136)) + ((((int)threadIdx.x) & 7) * 8)))))[0] = ((uint4*)(x + (((((((int)blockIdx.z) * 8192) + (((int)blockIdx.x) * 2048)) + (ax1_ax2_fused_outer_outer_outer_outer * 256)) + (((int)threadIdx.x) * 8)))))[0];
  }
  for (int ax1_ax2_fused_outer_outer_outer_outer1 = 0; ax1_ax2_fused_outer_outer_outer_outer1 < 8; ++ax1_ax2_fused_outer_outer_outer_outer1) {
    ((uint4*)(placeholder_shared + ((((ax1_ax2_fused_outer_outer_outer_outer1 * 544) + ((((int)threadIdx.x) >> 3) * 136)) + ((((int)threadIdx.x) & 7) * 8)))))[0] = ((uint4*)(placeholder + (((((((int)blockIdx.z) * 8192) + (((int)blockIdx.y) * 2048)) + (ax1_ax2_fused_outer_outer_outer_outer1 * 256)) + (((int)threadIdx.x) * 8)))))[0];
  }
  }
  __syncthreads();
  if(threadIdx.y==0){
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
  if(threadIdx.y==0){
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
    int row = i * blockDim.y + threadIdx.y;
    half ele = hexp((x_shared + row * x_shared_row_stride + threadIdx.x)[0]);
    (x_shared + row * x_shared_row_stride + threadIdx.x)[0] = ele;
    reg_sum = warpReduceSum(ele);
    __syncthreads();
    if(threadIdx.x==0){
      atomicAdd(sum + blockIdx.z * 128 + blockIdx.x * 32 + row, reg_sum);
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
    int row = i * blockDim.y + threadIdx.y;
    reg_sum = sum[blockIdx.z * 128 + blockIdx.x * x_shared_row_stride + row];
    (x_shared + row * x_shared_row_stride + threadIdx.x)[0] = (x_shared + row * x_shared_row_stride + threadIdx.x)[0] / reg_sum;
  }
  __syncthreads();
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 4; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    ((uint4*)(compute + (((((((((int)blockIdx.z) * 16384) + (((int)blockIdx.x) * 4096)) + (i_inner_j_inner_fused_outer_outer_outer_outer * 1024)) + ((((int)threadIdx.x) >> 2) * 128)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 3) * 8)))))[0] = 
    ((uint4*)(x_shared + ((((i_inner_j_inner_fused_outer_outer_outer_outer * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)))))[0];
  }
}



// dim3(4, 4,12), dim3(32,1,1)
// Note we set blockDim.y to 8 to compute softmax 
extern "C" __global__ void __launch_bounds__(32) fused_query_key_matmul_softmax_v3(half* __restrict__ x, half* __restrict__ placeholder, half* __restrict__ compute, half* __restrict__ sum) {
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
    ((uint4*)(x_shared + ((((ax1_ax2_fused_outer_outer_outer_outer * 544) + ((((int)threadIdx.x) >> 3) * 136)) + ((((int)threadIdx.x) & 7) * 8)))))[0] = ((uint4*)(x + (((((((int)blockIdx.z) * 8192) + (((int)blockIdx.x) * 2048)) + (ax1_ax2_fused_outer_outer_outer_outer * 256)) + (((int)threadIdx.x) * 8)))))[0];
  }
  for (int ax1_ax2_fused_outer_outer_outer_outer1 = 0; ax1_ax2_fused_outer_outer_outer_outer1 < 8; ++ax1_ax2_fused_outer_outer_outer_outer1) {
    ((uint4*)(placeholder_shared + ((((ax1_ax2_fused_outer_outer_outer_outer1 * 544) + ((((int)threadIdx.x) >> 3) * 136)) + ((((int)threadIdx.x) & 7) * 8)))))[0] = ((uint4*)(placeholder + (((((((int)blockIdx.z) * 8192) + (((int)blockIdx.y) * 2048)) + (ax1_ax2_fused_outer_outer_outer_outer1 * 256)) + (((int)threadIdx.x) * 8)))))[0];
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
  const int num_iter = 32;
  half reg_sum = 0;
  const int x_shared_row_stride = 40;
  #pragma unroll
  for(int i=0; i<num_iter; ++i){
    reg_sum += x_shared[threadIdx.x * x_shared_row_stride + i];
  }
  atomicAdd(sum + blockIdx.z * 128 + blockIdx.x * 32 + threadIdx.x, reg_sum);
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  __syncthreads();
  __threadfence();
  grid.sync();
  // -----------------------------------------------------
  // half2 reg_sum_h2;
  // #pragma unroll
  // for(int i=0; i<num_iter / 2; ++i){ // each thread compute half2
  //   int row = (i << 1) + (threadIdx.x >> 4);
  //   int col = (threadIdx.x & 0xf);
  //   reg_sum = sum[blockIdx.z * 128 + blockIdx.x * x_shared_row_stride + row];
  //   reg_sum_h2.x=reg_sum; reg_sum_h2.y=reg_sum;
  //   reinterpret_cast<half2*>(x_shared + row * x_shared_row_stride + (col<<1))[0] = reinterpret_cast<half2*>(x_shared + row * x_shared_row_stride + (col<<1))[0]/reg_sum_h2;
  //   // ((half2*)(x_shared + row * x_shared_row_stride + col*2))[0] = ((half2*)(x_shared + row * x_shared_row_stride + col*2))[0] / reg_sum_h2;
  // }

  #pragma unroll
  for(int i=0; i<num_iter; ++i){ // each thread compute half2
    int row = i;
    reg_sum = sum[blockIdx.z * 128 + blockIdx.x * 32 + row];
    (x_shared + row * x_shared_row_stride + threadIdx.x)[0] = (x_shared + row * x_shared_row_stride + threadIdx.x)[0] / reg_sum;
  }
  __syncthreads();
  // 16384=12
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 4; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    ((uint4*)(compute + (((((((((int)blockIdx.z) * 16384) + (((int)blockIdx.x) * 4096)) + (i_inner_j_inner_fused_outer_outer_outer_outer * 1024)) + ((((int)threadIdx.x) >> 2) * 128)) + (((int)blockIdx.y) * 32)) + ((((int)threadIdx.x) & 3) * 8)))))[0] = 
    ((uint4*)(x_shared + ((((i_inner_j_inner_fused_outer_outer_outer_outer * 320) + ((((int)threadIdx.x) >> 2) * 40)) + ((((int)threadIdx.x) & 3) * 8)))))[0];
  }
}
