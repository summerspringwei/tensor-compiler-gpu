#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <mma.h>

// (128, 768) * (3*768, 768) -> (128, 3*768=2304)
// dim3(4, 36,1), dim3(32,2,1), each block computes (32, 64), each warp compute (32, 32)
extern "C" __global__ void __launch_bounds__(64)
    fused_attn_qkv_matmul_transpose_kernel(const half* __restrict__ x, const half* __restrict__ placeholder,
                    half* __restrict__ T_dense, 
                    half* __restrict__ query, half* __restrict__ key, half* __restrict__ value) {
  int blockIdx_x = (blockIdx.x % 4);
  int blockIdx_y = (blockIdx.x / 4);
  int threadIdx_x = threadIdx.x % 32;
  int threadIdx_y = threadIdx.x / 32;
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half>
      T_dense_wmma_accumulator[4];
  extern half __shared__ shared_buff_fused[]; 
  half* x_shared = (half*)&shared_buff_fused[0]; 
  half* placeholder_shared = (half*)&shared_buff_fused[4352];
  // __shared__ half x_shared[4352];
  // __shared__ half placeholder_shared[8704];
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
                               (((int)threadIdx_y) * 272)) +
                              ((((int)threadIdx_x) >> 4) * 136)) +
                             ((((int)threadIdx_x) & 15) * 8)))))[0] =
          ((uint4*)(x + (((((((((int)blockIdx_x) * 24576) +
                              (ax0_ax1_fused_outer_outer_outer_outer * 3072)) +
                             (((int)threadIdx_y) * 1536)) +
                            ((((int)threadIdx_x) >> 4) * 768)) +
                           (k_outer_outer * 128)) +
                          ((((int)threadIdx_x) & 15) * 8)))))[0];
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
         ax0_ax1_fused_outer_outer_outer_outer1 < 16;
         ++ax0_ax1_fused_outer_outer_outer_outer1) {
      ((uint4*)(placeholder_shared +
                (((((ax0_ax1_fused_outer_outer_outer_outer1 * 544) +
                    (((int)threadIdx_y) * 272)) +
                   ((((int)threadIdx_x) >> 4) * 136)) +
                  ((((int)threadIdx_x) & 15) * 8)))))[0] =
          ((uint4*)(placeholder +
                    (((((((((int)blockIdx_y) * 49152) +
                          (ax0_ax1_fused_outer_outer_outer_outer1 * 3072)) +
                         (((int)threadIdx_y) * 1536)) +
                        ((((int)threadIdx_x) >> 4) * 768)) +
                       (k_outer_outer * 128)) +
                      ((((int)threadIdx_x) & 15) * 8)))))[0];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      (void)nvcuda::wmma::load_matrix_sync(
          x_shared_wmma_matrix_a[0],
          ((half*)x_shared +
           (((((int)threadIdx_y) * 2176) + (k_outer_inner * 16)))),
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
         (((((int)threadIdx_y) * 1152) + (ax1_outer_inner * 16)))),
        T_dense_wmma_accumulator[ax1_outer_inner], 72,
        nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  // 73728=32*2304, 18432=2304*8, 9216=2304*4, 
  // blockIdx_x compute 32 rows, blockIdx_y compute 64 cols
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0;
       i_inner_j_inner_fused_outer_outer_outer_outer < 4;
       ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    ((uint4*)(T_dense +
              (((((((((int)blockIdx_x) * 73728) +
                    (i_inner_j_inner_fused_outer_outer_outer_outer * 18432)) +
                   (((int)threadIdx_y) * 9216)) +
                  ((((int)threadIdx_x) >> 3) * 2304)) +
                 (((int)blockIdx_y) * 64)) +
                ((((int)threadIdx_x) & 7) * 8)))))[0] =
        ((uint4*)(x_shared +
                  (((((i_inner_j_inner_fused_outer_outer_outer_outer * 576) +
                      (((int)threadIdx_y) * 288)) +
                     ((((int)threadIdx_x) >> 3) * 72)) +
                    ((((int)threadIdx_x) & 7) * 8)))))[0];
  }

  pipe.producer_acquire();
  // Do transpose for query and key here
  // Now (128, 768) -> reshape (128, 12, 64) -> transpose (12, 128, 64)
  // (row, col) -> (row, num_head, hidden)
  // Using vector, we have 64 threads, each iteration save 8 rows (64*sizeof(float4)/sizeof(half)/64) = 8
  if(blockIdx_y < 12){
    for(int i=0; i<4; ++i){
      int s_row = i * 8 + (threadIdx_y * 32 + threadIdx_x) / 8;
      int s_col = ((threadIdx_y * 32 + threadIdx_x) % 8) * 8;
      int s_addr = s_row * 72 + s_col;
      int g_row = blockIdx_x * 32 + s_row;
      int g_col = blockIdx_y * 64 + s_col;
      int num_head = (g_col >> 6); // num_head = col / 64
      int hidden = g_col & 0x3f; // hidden = col % 64;
      int g_addr = num_head * 128 * 64 + g_row * 64 + hidden;
      cuda::memcpy_async(((query + g_addr)), ((x_shared + s_addr)), shape, pipe);
    }
  }
  
  if(blockIdx_y >=12 && blockIdx_y <24){
    for(int i=0; i<4; ++i){
      int s_row = i * 8 + (threadIdx_y * 32 + threadIdx_x) / 8;
      int s_col = ((threadIdx_y * 32 + threadIdx_x) % 8) * 8;
      int s_addr = s_row * 72 + s_col;
      int g_row = blockIdx_x * 32 + s_row;
      int g_col = (blockIdx_y-12) * 64 + s_col;
      int num_head = (g_col >> 6); // num_head = col / 64
      int hidden = g_col & 0x3f; // hidden = col % 64;
      int g_addr = num_head * 128 * 64 + g_row * 64 + hidden;
      cuda::memcpy_async(((key + g_addr)), ((x_shared + s_addr)), shape, pipe);
    }
  }

  // Do transpose for value
  // Now (128, 768) -> reshape (128, 12, 64) -> transpose (12, 64, 128)
  // (row, col) -> (row, num_head, hidden)
  // Using vector, we have 64 threads, each iteration save 8 rows (64*sizeof(float4)/sizeof(half)/64) = 8
  if(blockIdx_y >= 24 && blockIdx_y < 36){
    int tidx = (threadIdx_y * 32 + threadIdx_x);
    for(int i=0; i<32; i++){
      int s_row = i;
      int s_col = tidx;
      int s_addr = i * 72 + s_col;
      int g_row = blockIdx_x * 32 + s_row;
      int g_col = (blockIdx_y-24) * 64 + s_col;
      int num_head = (g_col >> 6); // num_head = col / 64
      int hidden = g_col & 0x3f; // hidden = col % 64;
      int g_addr = num_head * 128 * 64 + hidden * 128 + g_row;
      cuda::memcpy_async(((value + g_addr)), ((x_shared + s_addr)), sizeof(half), pipe);
    }
  }

  pipe.producer_commit();
  pipe.consumer_wait();
  pipe.consumer_release();
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  __syncthreads();
  __threadfence();
  grid.sync();
}




// (128, 768) * (3*768, 768) -> (128, 3*768=2304)
// dim3(4, 36,1), dim3(32,2,1), each block computes (32, 64), each warp compute (32, 32)
extern "C" __global__ void __launch_bounds__(64)
    fused_attn_qkv_matmul_transpose_kernel_v2(const half* __restrict__ x, const half* __restrict__ placeholder,
                    half* __restrict__ T_dense, 
                    half* __restrict__ query, half* __restrict__ key, half* __restrict__ value) {
  int blockIdx_x = (blockIdx.x % 4);
  int blockIdx_y = (blockIdx.x / 4);
  int threadIdx_x = threadIdx.x % 32;
  int threadIdx_y = threadIdx.x / 32;
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half>
      T_dense_wmma_accumulator[4];
  extern half __shared__ shared_buff_fused[]; 
  half* x_shared = (half*)&shared_buff_fused[0]; 
  half* placeholder_shared = (half*)&shared_buff_fused[4352];
  // __shared__ half x_shared[4352];
  // __shared__ half placeholder_shared[8704];
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
                               (((int)threadIdx_y) * 272)) +
                              ((((int)threadIdx_x) >> 4) * 136)) +
                             ((((int)threadIdx_x) & 15) * 8)))))[0] =
          ((uint4*)(x + (((((((((int)blockIdx_x) * 24576) +
                              (ax0_ax1_fused_outer_outer_outer_outer * 3072)) +
                             (((int)threadIdx_y) * 1536)) +
                            ((((int)threadIdx_x) >> 4) * 768)) +
                           (k_outer_outer * 128)) +
                          ((((int)threadIdx_x) & 15) * 8)))))[0];
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
         ax0_ax1_fused_outer_outer_outer_outer1 < 16;
         ++ax0_ax1_fused_outer_outer_outer_outer1) {
      ((uint4*)(placeholder_shared +
                (((((ax0_ax1_fused_outer_outer_outer_outer1 * 544) +
                    (((int)threadIdx_y) * 272)) +
                   ((((int)threadIdx_x) >> 4) * 136)) +
                  ((((int)threadIdx_x) & 15) * 8)))))[0] =
          ((uint4*)(placeholder +
                    (((((((((int)blockIdx_y) * 49152) +
                          (ax0_ax1_fused_outer_outer_outer_outer1 * 3072)) +
                         (((int)threadIdx_y) * 1536)) +
                        ((((int)threadIdx_x) >> 4) * 768)) +
                       (k_outer_outer * 128)) +
                      ((((int)threadIdx_x) & 15) * 8)))))[0];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      (void)nvcuda::wmma::load_matrix_sync(
          x_shared_wmma_matrix_a[0],
          ((half*)x_shared +
           (((((int)threadIdx_y) * 2176) + (k_outer_inner * 16)))),
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
         (((((int)threadIdx_y) * 1152) + (ax1_outer_inner * 16)))),
        T_dense_wmma_accumulator[ax1_outer_inner], 72,
        nvcuda::wmma::mem_row_major);
  }
  __syncthreads();
  // 73728=32*2304, 18432=2304*8, 9216=2304*4, 
  // blockIdx_x compute 32 rows, blockIdx_y compute 64 cols
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0;
       i_inner_j_inner_fused_outer_outer_outer_outer < 4;
       ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    ((uint4*)(T_dense +
              (((((((((int)blockIdx_x) * 73728) +
                    (i_inner_j_inner_fused_outer_outer_outer_outer * 18432)) +
                   (((int)threadIdx_y) * 9216)) +
                  ((((int)threadIdx_x) >> 3) * 2304)) +
                 (((int)blockIdx_y) * 64)) +
                ((((int)threadIdx_x) & 7) * 8)))))[0] =
        ((uint4*)(x_shared +
                  (((((i_inner_j_inner_fused_outer_outer_outer_outer * 576) +
                      (((int)threadIdx_y) * 288)) +
                     ((((int)threadIdx_x) >> 3) * 72)) +
                    ((((int)threadIdx_x) & 7) * 8)))))[0];
  }

  pipe.producer_acquire();
  // Do transpose for query and key here
  // Now (128, 768) -> reshape (128, 12, 64) -> transpose (12, 128, 64)
  // (row, col) -> (row, num_head, hidden)
  // Using vector, we have 64 threads, each iteration save 8 rows (64*sizeof(float4)/sizeof(half)/64) = 8
  // Do arithmetic simplify
  if(blockIdx_y < 12){
    // for(int i=0; i<4; ++i){
    //   int s_row = i * 8 + (threadIdx_y * 32 + threadIdx_x) / 8;
    //   int s_col = ((threadIdx_y * 32 + threadIdx_x) % 8) * 8;
    //   int s_addr = s_row * 72 + s_col;
    //   int g_row = blockIdx_x * 32 + s_row;
    //   int g_col = blockIdx_y * 64 + s_col;
    //   int num_head = (g_col >> 6); // num_head = col / 64
    //   int hidden = g_col & 0x3f; // hidden = col % 64;
    //   int g_addr = num_head * 128 * 64 + g_row * 64 + hidden;
    //   cuda::memcpy_async(((query + g_addr)), ((x_shared + s_addr)), shape, pipe);
    // }
    const int tidx = (threadIdx_y * 32 + threadIdx_x);
    const int tidx_div_8 = tidx >> 3;
    const int tidx_mod_8_mul_8 = (tidx & 0x7) << 3;
    for(int i=0; i<4; ++i){
      int s_row = (i << 3) + tidx_div_8;
      int s_col = tidx_mod_8_mul_8;
      int s_addr = s_row * 72 + s_col;
      int g_row = (blockIdx_x << 5) + s_row;
      int num_head = blockIdx_y + (s_col >> 6);
      int hidden = s_col & 0x3f;
      int g_addr = (num_head << 13) + (g_row << 6) + hidden;
      cuda::memcpy_async(((query + g_addr)), ((x_shared + s_addr)), shape, pipe);
    }
  }
  
  if(blockIdx_y >=12 && blockIdx_y <24){
    const int tidx = (threadIdx_y * 32 + threadIdx_x);
    const int tidx_div_8 = tidx >> 3;
    const int tidx_mod_8_mul_8 = (tidx & 0x7) << 3;
    for(int i=0; i<4; ++i){
      int s_row = (i << 3) + tidx_div_8;
      int s_col = tidx_mod_8_mul_8;
      int s_addr = s_row * 72 + s_col;
      int g_row = (blockIdx_x << 5) + s_row;
      int num_head = (blockIdx_y - 12) + (s_col >> 6);
      int hidden = s_col & 0x3f;
      int g_addr = (num_head << 13) + (g_row << 6) + hidden;
      cuda::memcpy_async(((key + g_addr)), ((x_shared + s_addr)), shape, pipe);
    }
  }

  // Do transpose for value
  // Now (128, 768) -> reshape (128, 12, 64) -> transpose (12, 64, 128)
  // (row, col) -> (row, num_head, hidden)
  // Using vector, we have 64 threads, each iteration save 8 rows (64*sizeof(float4)/sizeof(half)/64) = 8
  if(blockIdx_y >= 24 && blockIdx_y < 36){
    // int s_row = i;
    // int s_col = tidx;
    // int s_addr = i * 72 + s_col;
    // int g_row = (blockIdx_x << 5) + s_row;
    // int g_col = ((blockIdx_y - 24) << 6) + s_col;
    // int num_head = (g_col >> 6); // num_head = col / 64
    // int hidden = g_col & 0x3f; // hidden = col % 64;
    // int g_addr = (num_head << 13) + (hidden << 7) + g_row;
    int tidx = (threadIdx_y * 32 + threadIdx_x);
    for(int i=0; i<32; i++){
      int s_row = i;
      int s_col = tidx;
      int s_addr = i * 72 + s_col;
      int g_row = (blockIdx_x << 5) + s_row;
      int num_head = (blockIdx_y - 24) + (s_col >> 6);
      int hidden = s_col & 0x3f; // hidden = col % 64;
      int g_addr = (num_head << 13) + (hidden << 7) + g_row;
      cuda::memcpy_async(((value + g_addr)), ((x_shared + s_addr)), sizeof(half), pipe);
    }
  }

  pipe.producer_commit();
  pipe.consumer_wait();
  pipe.consumer_release();
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  __syncthreads();
  __threadfence();
  grid.sync();
}
