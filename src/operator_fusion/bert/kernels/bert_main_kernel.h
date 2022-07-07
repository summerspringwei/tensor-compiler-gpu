#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <mma.h>

#include "../../../cuda_utils.h"


extern "C" __global__ void __launch_bounds__(128)
    bert_attn_kernel(const half* __restrict__ x, const half* __restrict__ placeholder,
                    half* __restrict__ T_dense, 
                    half* __restrict__ query, half* __restrict__ key, half* __restrict__ value,
                    half* __restrict__ query_key_output, float* __restrict__ sum,
                    half* __restrict__ attn_value_output, half* __restrict__ attn_fc_weight,
                    half* attn_fc_output, float* __restrict__ variance, half eps, half gama, half beta,
                    half* t_attn_value_output, half* t_attn_fc_weight, 
                    half* t_attn_fc_output_tmp, half* t_attn_fc_output, half* ptr_inter_attn_fc_output) {
  extern half __shared__ shared_buff_fused[];
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  // (128, 768) * (3*768, 768) -> (128, 3*768=2304)
  // each block computes (32, 64), each warp compute (32, 32)
  // Begin of fused_attn_qkv_matmul_transpose_kernel_v2, dim3(4, 36,1), dim3(32,2,1)
  if(blockIdx.x < 4*36){
    int blockIdx_x = (blockIdx.x % 4);
    int blockIdx_y = (blockIdx.x / 4);
    int threadIdx_x = threadIdx.x % 32;
    int threadIdx_y = threadIdx.x / 32;
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half>
        T_dense_wmma_accumulator[4];
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
    if(threadIdx.x < 32 * 2){
      for (int j_c_outer_init = 0; j_c_outer_init < 4; ++j_c_outer_init) {
        (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[j_c_outer_init],
                                          0.000000e+00f);
      }
    }
    for (int k_outer_outer = 0; k_outer_outer < 6; ++k_outer_outer) {
      __syncthreads();
      if(threadIdx.x < 32 * 2){
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
      }
      __syncthreads();
      if(threadIdx.x < 32 * 2){
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
    }
    __syncthreads();
    if(threadIdx.x < 32 * 2){
      for (int ax1_outer_inner = 0; ax1_outer_inner < 4; ++ax1_outer_inner) {
        (void)nvcuda::wmma::store_matrix_sync(
            ((half*)x_shared +
            (((((int)threadIdx_y) * 1152) + (ax1_outer_inner * 16)))),
            T_dense_wmma_accumulator[ax1_outer_inner], 72,
            nvcuda::wmma::mem_row_major);
      }
    }
    __syncthreads();
    if(threadIdx.x < 32 * 2){
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
    }

    pipe.producer_acquire();
    // Do transpose for query and key here
    // Now (128, 768) -> reshape (128, 12, 64) -> transpose (12, 128, 64)
    // (row, col) -> (row, num_head, hidden)
    // Using vector, we have 64 threads, each iteration save 8 rows (64*sizeof(float4)/sizeof(half)/64) = 8
    // Do arithmetic simplify 
    if(blockIdx_y < 12 && threadIdx.x < 32 * 2){
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
    
    if(blockIdx_y >=12 && blockIdx_y <24 && threadIdx.x < 32 * 2){
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
    if(blockIdx_y >= 24 && blockIdx_y < 36 && threadIdx.x < 32 * 2){
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
    __syncthreads();
  } // end of fused_attn_qkv_matmul_transpose_kernel_v2

  
  grid.sync();

  // Begin of fused_query_key_matmul_softmax, dim3(4,4,12), dim3(32,1,1)
  if(blockIdx.x < 192){
    // __shared__ half x_shared[4352];
    // __shared__ half placeholder_shared[4352];
    int blockIdx_x = (blockIdx.x % 4);
    int blockIdx_y = ((blockIdx.x / 4) % 4);
    int blockIdx_z = (blockIdx.x / (4*4));
    int threadIdx_x = threadIdx.x % 32;
    int threadIdx_y = threadIdx.x / 32;
    half* x_shared = (half*)&shared_buff_fused[0];
    half* placeholder_shared = (half*)&shared_buff_fused[4352];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> compute_wmma_accumulator[4];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> x_shared_wmma_matrix_a[2];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::col_major> placeholder_shared_wmma_matrix_b[2];
    if(threadIdx.x < 32){
      for (int i_c_outer_init = 0; i_c_outer_init < 2; ++i_c_outer_init) {
        for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init) {
          (void)nvcuda::wmma::fill_fragment(compute_wmma_accumulator[((i_c_outer_init * 2) + j_c_outer_init)], 0.000000e+00f);
        }
      }
      for (int ax1_ax2_fused_outer_outer_outer_outer = 0; ax1_ax2_fused_outer_outer_outer_outer < 8; ++ax1_ax2_fused_outer_outer_outer_outer) {
        ((uint4*)(x_shared + ((((ax1_ax2_fused_outer_outer_outer_outer * 544) + ((((int)threadIdx_x) >> 3) * 136)) + ((((int)threadIdx_x) & 7) * 8)))))[0] = 
          ((uint4*)(query + (((((((int)blockIdx_z) * 8192) + (((int)blockIdx_x) * 2048)) + (ax1_ax2_fused_outer_outer_outer_outer * 256)) + (((int)threadIdx_x) * 8)))))[0];
      }
      for (int ax1_ax2_fused_outer_outer_outer_outer1 = 0; ax1_ax2_fused_outer_outer_outer_outer1 < 8; ++ax1_ax2_fused_outer_outer_outer_outer1) {
        ((uint4*)(placeholder_shared + ((((ax1_ax2_fused_outer_outer_outer_outer1 * 544) + ((((int)threadIdx_x) >> 3) * 136)) + ((((int)threadIdx_x) & 7) * 8)))))[0] = 
          ((uint4*)(key + (((((((int)blockIdx_z) * 8192) + (((int)blockIdx_y) * 2048)) + (ax1_ax2_fused_outer_outer_outer_outer1 * 256)) + (((int)threadIdx_x) * 8)))))[0];
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
    }
    __syncthreads();
    // Stores 4x 16x16(32x32), 640=16*40
    if(threadIdx.x < 32){
      for (int ax1_outer_inner = 0; ax1_outer_inner < 2; ++ax1_outer_inner) {
        for (int ax2_outer_inner = 0; ax2_outer_inner < 2; ++ax2_outer_inner) {
          (void)nvcuda::wmma::store_matrix_sync(((half *)x_shared + (((ax1_outer_inner * 640) + (ax2_outer_inner * 16)))), compute_wmma_accumulator[((ax1_outer_inner * 2) + ax2_outer_inner)], 40, nvcuda::wmma::mem_row_major);
        }
      }
    }
    __syncthreads();
    // x_shared: 32 * 32
    // compute reduce sum
    const int x_shared_row_stride = 40;
    if(threadIdx.x < 32){
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
    __syncthreads();
  }

  grid.sync();

  if(blockIdx.x < 192){
    const int x_shared_row_stride = 40;
    half* x_shared = (half*)&shared_buff_fused[0];
    int blockIdx_x = (blockIdx.x % 4);
    int blockIdx_y = ((blockIdx.x / 4) % 4);
    int blockIdx_z = (blockIdx.x / (4*4));
    int threadIdx_x = threadIdx.x % 32;
    int threadIdx_y = threadIdx.x / 32;
    if(threadIdx.x < 32){
      const int num_iter = 32;
      #pragma unroll
      for(int i=0; i<num_iter; ++i){ 
        int row = i;
        auto reg_sum = sum[blockIdx_z * 128 + blockIdx_x * 32 + row];
        (x_shared + row * x_shared_row_stride + threadIdx_x)[0] = __float2half(__half2float((x_shared + row * x_shared_row_stride + threadIdx_x)[0]) / reg_sum);
      }
    }
    __syncthreads();
    if(threadIdx.x < 32){
      for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0; i_inner_j_inner_fused_outer_outer_outer_outer < 4; ++i_inner_j_inner_fused_outer_outer_outer_outer) {
        ((uint4*)(query_key_output + (((((((((int)blockIdx_z) * 16384) + (((int)blockIdx_x) * 4096)) + (i_inner_j_inner_fused_outer_outer_outer_outer * 1024)) + ((((int)threadIdx_x) >> 2) * 128)) + (((int)blockIdx_y) * 32)) + ((((int)threadIdx_x) & 3) * 8)))))[0] = 
        ((uint4*)(x_shared + ((((i_inner_j_inner_fused_outer_outer_outer_outer * 320) + ((((int)threadIdx_x) >> 2) * 40)) + ((((int)threadIdx_x) & 3) * 8)))))[0];
      }
    }
    __syncthreads();
  } // End of fused_query_key_matmul_softmax, dim3(4,4,12), dim3(32,1,1)

  grid.sync();

  // Begin of attn_value_matmul, dim3(8, 2,12), dim3(32,1,1)
  if(blockIdx.x < 192){
    const int blockIdx_x = (blockIdx.x % 8);
    const int blockIdx_y = ((blockIdx.x / 8) % 2);
    const int blockIdx_z = (blockIdx.x / (2*8));
    const int threadIdx_x = threadIdx.x % 32;
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
    pipe.producer_acquire();
    half* x_shared = &(shared_buff_fused[0]);
    half* placeholder_shared = &(shared_buff_fused[2176]);
    // __shared__ half x_shared[2176];
    // __shared__ half placeholder_shared[4352];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half>
        compute_wmma_accumulator[2];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half,
                          nvcuda::wmma::row_major>
        x_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half,
                          nvcuda::wmma::col_major>
        placeholder_shared_wmma_matrix_b[2];
    if(threadIdx.x < 32){
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
            ((uint4*)(query_key_output + (((((((int)blockIdx_z) * 16384) +
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
            ((uint4*)(value +
                      (((((((int)blockIdx_z) * 8192) + (((int)blockIdx_y) * 4096)) +
                        (ax1_ax2_fused_outer_outer_outer_outer1 * 256)) +
                        (((int)threadIdx_x) * 8)))))[0];
      }
    }
    __syncthreads();
    if(threadIdx.x < 32){
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
    }
    __syncthreads();
    if(threadIdx.x<32){
      for (int ax2_outer_inner = 0; ax2_outer_inner < 2; ++ax2_outer_inner) {
        (void)nvcuda::wmma::store_matrix_sync(
            ((half*)x_shared + ((ax2_outer_inner * 16))),
            compute_wmma_accumulator[ax2_outer_inner], 32,
            nvcuda::wmma::mem_row_major);
      }
    }
    __syncthreads();
    if(threadIdx.x<32){
      // Do transpose (12, 128, 64) to (128, 12, 64), reshape to (128, 768)
      // x_shared shape: (16, 32) x (8, 2) = (128, 64)
      const int vec_size = (sizeof(float4)/sizeof(half));
      const int num_iter = 16*32/vec_size/32;
      for(int i=0; i<num_iter; ++i){
        int s_row = i*8 + (threadIdx_x / 4);
        int s_col = (threadIdx_x % 4) * vec_size;
        int s_addr = s_row * 32 + s_col;
        int g_row = (blockIdx_x * 16 + s_row);
        int g_col = (blockIdx_z * 64 + blockIdx_y * 32 + s_col);
        int g_addr = g_row * 12 * 64 + g_col;
        cuda::memcpy_async((float4*)(attn_value_output + g_addr), 
          (float4*)(x_shared + s_addr), shape, pipe);
      }
      pipe.producer_commit();
      pipe.consumer_wait();
      pipe.consumer_release();
      // Results before transpose
      // 8192=128*64, 1024=16*64, 
      // for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0;
      //     i_inner_j_inner_fused_outer_outer_outer_outer < 2;
      //     ++i_inner_j_inner_fused_outer_outer_outer_outer) {
      //   ((uint4*)(attn_value_output +
      //             (((((((((int)blockIdx_z) * 8192) + (((int)blockIdx_x) * 1024)) +
      //                 (i_inner_j_inner_fused_outer_outer_outer_outer * 512)) +
      //                 ((((int)threadIdx_x) >> 2) * 64)) +
      //               (((int)blockIdx_y) * 32)) +
      //               ((((int)threadIdx_x) & 3) * 8)))))[0] =
      //       ((uint4*)(x_shared +
      //                 (((i_inner_j_inner_fused_outer_outer_outer_outer * 256) +
      //                   (((int)threadIdx_x) * 8)))))[0];
      // }
    }
    __syncthreads();
  }// Begin of attn_value_matmul, dim3(8, 2,12), dim3(32,1,1)
  grid.sync();
  __syncthreads();
  // (128, 768), (768, 768) -> (128, 768)
  // Begin of attn_fc, dim3(4, 24,1), dim3(32,4,1)
  //half* __restrict__ attn_value_output, half* __restrict__ attn_fc_weight,
  if(blockIdx.x < 96){
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, half>
      T_dense_wmma_accumulator[1];
      half* x_shared = &(shared_buff_fused[0]);
      half* placeholder_shared = &(shared_buff_fused[4352]);
      half* tmp_shared = &(shared_buff_fused[8704]);
  // __shared__ half x_shared[4352];// 32*136
  // __shared__ half placeholder_shared[4352];
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
    __threadfence();
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
          ((uint4*)(attn_value_output + (((((((((int)blockIdx_x) * 24576) +
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
          ((uint4*)(attn_fc_weight +
                    (((((((((int)blockIdx_y) * 24576) +
                          (ax0_ax1_fused_outer_outer_outer_outer1 * 6144)) +
                         (((int)threadIdx_y) * 1536)) +
                        ((((int)threadIdx_x) >> 4) * 768)) +
                       (k_outer_outer * 128)) +
                      ((((int)threadIdx_x) & 15) * 8)))))[0];
    }
    __threadfence();
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
          //Testing
      __threadfence();
      __syncthreads();
    (void)nvcuda::wmma::store_matrix_sync(
        ((half*)tmp_shared + ((((int)threadIdx_y) * 256))),
        T_dense_wmma_accumulator[0], 32, nvcuda::wmma::mem_row_major);
      __threadfence();
      __syncthreads();
      if(blockIdx.x<96){
        printf("A: block!%d, %d!thread!%d, %d!^%d,%d^->%f %f\n", 
          blockIdx_x, blockIdx_y,
          threadIdx_x, threadIdx_y, k_outer_outer, k_outer_inner,
          __half2float(x_shared[threadIdx_y*32+threadIdx_x]),
          __half2float(x_shared[136+threadIdx_y*32+threadIdx_x]));
        auto ga1 = (attn_value_output + ((((int)blockIdx_x) * 24576)) +
                             ((k_outer_outer * 128) +threadIdx_y*32+threadIdx_x))[0];
        auto ga2 = (attn_value_output + ((((int)blockIdx_x) * 24576)) + 768 +
                             ((k_outer_outer * 128) +threadIdx_y*32+threadIdx_x))[0];
        printf("GA: block!%d, %d!thread!%d, %d!^%d,%d^->%f, %f \n", 
          blockIdx_x, blockIdx_y,
          threadIdx_x, threadIdx_y, k_outer_outer, k_outer_inner,
          __half2float(ga1), __half2float(ga1));
        printf("B: block!%d, %d!thread!%d, %d!^%d,%d^->%f %f\n", 
          blockIdx_x, blockIdx_y,
          threadIdx_x, threadIdx_y, k_outer_outer, k_outer_inner,
          __half2float(placeholder_shared[threadIdx_y*32+threadIdx_x]),
          __half2float(placeholder_shared[136+threadIdx_y*32+threadIdx_x]));
        printf("C: block!%d, %d!thread!%d, %d!^%d,%d^->%f \n", 
          blockIdx_x, blockIdx_y,
          threadIdx_x, threadIdx_y, k_outer_outer, k_outer_inner,
          __half2float(tmp_shared[threadIdx_y*32+threadIdx_x]));
      }
    }
  }
  __threadfence();
  __syncthreads();
  (void)nvcuda::wmma::store_matrix_sync(
      ((half*)placeholder_shared + ((((int)threadIdx_y) * 256))),
      T_dense_wmma_accumulator[0], 32, nvcuda::wmma::mem_row_major);
  __threadfence();
  __syncthreads();
  ((uint4*)(ptr_inter_attn_fc_output +
            ((((((((int)blockIdx_x) * 24576) + (((int)threadIdx_y) * 6144)) +
                ((((int)threadIdx_x) >> 2) * 768)) +
               (((int)blockIdx_y) * 32)) +
              ((((int)threadIdx_x) & 3) * 8)))))[0] =
      ((uint4*)(placeholder_shared +
                (((((int)threadIdx_y) * 256) + (((int)threadIdx_x) * 8)))))[0];
    printf("FC: block!%d, %d!thread!%d, %d!->%f \n", 
          blockIdx_x, blockIdx_y,
          threadIdx_x, threadIdx_y,
          __half2float(tmp_shared[threadIdx_y*32+threadIdx_x]));
    __threadfence();
  __syncthreads();
  }
  __threadfence();
  __syncthreads();
  // if(blockIdx.x < 96){
  //   // cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  //   const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  //   const int blockIdx_x = blockIdx.x % 4;
  //   const int blockIdx_y = blockIdx.x / 4;
  //   const int threadIdx_x = threadIdx.x % 32;
  //   const int threadIdx_y = threadIdx.x / 32;
  //   nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, half>
  //     T_dense_wmma_accumulator[1];
  //   half* x_shared = &(shared_buff_fused[0]);
  //   half* placeholder_shared = &(shared_buff_fused[4352]);
  //   half* short_cut_shared = &(shared_buff_fused[8704]);
  //   // __shared__ half x_shared[4352];
  //   // __shared__ half placeholder_shared[4352];
  //   nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, half,
  //                         nvcuda::wmma::row_major>
  //       x_shared_wmma_matrix_a[1];
  //   nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, half,
  //                         nvcuda::wmma::col_major>
  //       placeholder_shared_wmma_matrix_b[1];
  //   (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[0], 0.000000e+00f);
  //   for (int k_outer_outer = 0; k_outer_outer < 6; ++k_outer_outer) {
  //     __syncthreads();
  //     // if(threadIdx.x < 128){
  //       for (int ax0_ax1_fused_outer_outer_outer_outer = 0;
  //           ax0_ax1_fused_outer_outer_outer_outer < 4;
  //           ++ax0_ax1_fused_outer_outer_outer_outer) {
  //         ((uint4*)(x_shared + (((((ax0_ax1_fused_outer_outer_outer_outer * 1088) +
  //                                 (((int)threadIdx_y) * 272)) +
  //                                 ((((int)threadIdx_x) >> 4) * 136)) +
  //                               ((((int)threadIdx_x) & 15) * 8)))))[0] =
  //             ((uint4*)(attn_value_output + (((((((((int)blockIdx_x) * 24576) +
  //                                 (ax0_ax1_fused_outer_outer_outer_outer * 6144)) +
  //                               (((int)threadIdx_y) * 1536)) +
  //                               ((((int)threadIdx_x) >> 4) * 768)) +
  //                             (k_outer_outer * 128)) +
  //                             ((((int)threadIdx_x) & 15) * 8)))))[0];
  //       }
  //       for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
  //           ax0_ax1_fused_outer_outer_outer_outer1 < 4;
  //           ++ax0_ax1_fused_outer_outer_outer_outer1) {
  //         ((uint4*)(placeholder_shared +
  //                   (((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) +
  //                       (((int)threadIdx_y) * 272)) +
  //                     ((((int)threadIdx_x) >> 4) * 136)) +
  //                     ((((int)threadIdx_x) & 15) * 8)))))[0] =
  //             ((uint4*)(attn_fc_weight +
  //                       (((((((((int)blockIdx_y) * 24576) +
  //                             (ax0_ax1_fused_outer_outer_outer_outer1 * 6144)) +
  //                           (((int)threadIdx_y) * 1536)) +
  //                           ((((int)threadIdx_x) >> 4) * 768)) +
  //                         (k_outer_outer * 128)) +
  //                         ((((int)threadIdx_x) & 15) * 8)))))[0];
  //       }
  //     // }
  //     __syncthreads();
  //     // if(threadIdx.x < 128){
  //       // For next pipe Short_cut_add
  //       // if(k_outer_outer == 5){
  //       //   pipe.producer_acquire();
  //       //   cuda::memcpy_async((short_cut_shared +
  //       //                   (((((int)threadIdx_y) * 256) + (((int)threadIdx_x) * 8)))),
  //       //                   (x +
  //       //               ((((((((int)blockIdx_x) * 24576) + (((int)threadIdx_y) * 6144)) +
  //       //                   ((((int)threadIdx_x) >> 2) * 768)) +
  //       //                 (((int)blockIdx_y) * 32)) +
  //       //                 ((((int)threadIdx_x) & 3) * 8)))), shape, pipe);
  //       //   pipe.producer_commit();
  //       // }
  //       for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
  //         (void)nvcuda::wmma::load_matrix_sync(
  //             x_shared_wmma_matrix_a[0],
  //             ((half*)x_shared +
  //             (((((int)threadIdx_y) * 1088) + (k_outer_inner * 16)))),
  //             136);
  //         (void)nvcuda::wmma::load_matrix_sync(
  //             placeholder_shared_wmma_matrix_b[0],
  //             ((half*)placeholder_shared + ((k_outer_inner * 16))), 136);
  //         (void)nvcuda::wmma::mma_sync(
  //             T_dense_wmma_accumulator[0], x_shared_wmma_matrix_a[0],
  //             placeholder_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[0]);
  //       }
  //     // }
  //   }
  //   __syncthreads();
    
  //   // if(threadIdx.x < 128){
  //     // x_shared shape is (32, 32) =  (4x8, 32)
  //     (void)nvcuda::wmma::store_matrix_sync(
  //         ((half*)x_shared + ((((int)threadIdx_y) * 256))),
  //         T_dense_wmma_accumulator[0], 32, nvcuda::wmma::mem_row_major);
  //     __syncthreads();
  //   ((uint4*)(ptr_inter_attn_fc_output +
  //           ((((((((int)blockIdx_x) * 24576) + (((int)threadIdx_y) * 6144)) +
  //               ((((int)threadIdx_x) >> 2) * 768)) +
  //              (((int)blockIdx_y) * 32)) +
  //             ((((int)threadIdx_x) & 3) * 8)))))[0] =
  //     ((uint4*)(x_shared +
  //               (((((int)threadIdx_y) * 256) + (((int)threadIdx_x) * 8)))))[0];
  //   // }
  //   __syncthreads();

  //   // Do Short cut add
  //   // shared shape (32, 32), each thread compute half2, 16 threads compute a row, 128 threads compute 8 row
  //   if(threadIdx.x < 128){
  //     // pipe.consumer_wait();
  //     // pipe.consumer_release();
  //     const int num_iter = 32 / 8;
  //     for(int i=0; i<num_iter;++i){
  //       // int row = i * 8 + threadIdx_y * 2 + (threadIdx_x / 16);
  //       // int col = (threadIdx_x % 16) * 2;
  //       // int s_addr = row * 32 + col;
  //       int row = (i << 3) + (threadIdx_y << 1) + (threadIdx_x >> 4);
  //       int col = (threadIdx_x & 0xf) << 1;
  //       int s_addr = (row << 5) + col;
  //       ((half2*)(x_shared+s_addr))[0] = ((half2*)(x_shared+s_addr))[0] + ((half2*)(short_cut_shared+s_addr))[0];
  //     }
  //   }
  //   __syncthreads();

  //   // Do Layer norm
  //   // 1. sum
  //   const int x_shared_row_stride = 32;
  //   if(threadIdx.x < 32){
  //     // each thread compute half2
  //     const int num_iter = 32 / (sizeof(half2)/sizeof(half));
  //     float reg_sum = 0.0;
  //     #pragma unroll
  //     for(int i=0; i<num_iter; ++i){
  //       auto tmp = ((half2*)(x_shared + threadIdx_x * x_shared_row_stride + i * 2))[0];
  //       reg_sum += (__half2float(tmp.x) + __half2float(tmp.y));
  //     }
  //     atomicAdd(sum + blockIdx_x * 32 + threadIdx.x, reg_sum);
  //   }
  //   __syncthreads();
  //   }
    // 2. grid.sync()
    grid.sync();
  if(blockIdx.x < 96){
    const int blockIdx_x = blockIdx.x % 4;
    const int blockIdx_y = blockIdx.x / 4;
    const int threadIdx_x = threadIdx.x % 32;
    const int threadIdx_y = threadIdx.x / 32;
    half* x_shared = &(shared_buff_fused[0]);
    
    // 3. compute variance
    const int x_shared_row_stride = 32;
    if(threadIdx.x < 32){
      float reduce_sum = 0;
      const int num_iter = 32 / (sizeof(half2)/sizeof(half));
      float avg = ((sum + blockIdx_x * 32 + threadIdx_x)[0]) / 768;
      #pragma unroll
      for(int i=0; i<num_iter; ++i){
        auto tmp = ((half2*)(x_shared + threadIdx_x * x_shared_row_stride + i * 2))[0];
        float delt_x = (__half2float(tmp.x) - avg);
        float delt_y = (__half2float(tmp.y) - avg);
        reduce_sum += (delt_x * delt_x + delt_y * delt_y);
      }
      atomicAdd(variance + blockIdx_x * 32 + threadIdx_x, reduce_sum);
    }
    __syncthreads();
  }
  grid.sync();
  if(blockIdx.x < 96){
    const int blockIdx_x = blockIdx.x % 4;
    const int blockIdx_y = blockIdx.x / 4;
    const int threadIdx_x = threadIdx.x % 32;
    const int threadIdx_y = threadIdx.x / 32;
    half* x_shared = &(shared_buff_fused[0]);
    // // 4. normalize
    if(threadIdx.x < 128){
      const int num_iter = 32 / (128*2/32);
      half2 gama2(gama, gama);
      half2 beta2(beta, beta);
      for(int i=0; i<num_iter;++i){
        int row = (i << 3) + (threadIdx_y << 1) + (threadIdx_x >> 4);
        half avg = __float2half(((sum + blockIdx_x * 32 + row)[0]) / 768);
        half reciprocal_variance = __float2half( 1.0 / (sqrt(((variance + blockIdx_x * 32 + row)[0]) / 768.0)  + __half2float(eps)));
        half2 avg2(avg, avg);
        half2 reciprocal_variance2(reciprocal_variance, reciprocal_variance);
        int col = (threadIdx_x & 0xf) << 1;
        int s_addr = (row << 5) + col;
        ((half2*)(x_shared+s_addr))[0] = (((half2*)(x_shared+s_addr))[0] - avg2) * reciprocal_variance2 * gama2 + beta2;
      }
    }
    __syncthreads();
    // 24576=32*768, 6144=8*768
    if(threadIdx.x < 128){
      ((uint4*)(attn_fc_output +
                ((((((((int)blockIdx_x) * 24576) + (((int)threadIdx_y) * 6144)) +
                    ((((int)threadIdx_x) >> 2) * 768)) +
                  (((int)blockIdx_y) * 32)) +
                  ((((int)threadIdx_x) & 3) * 8)))))[0] =
          ((uint4*)(x_shared +
                    (((((int)threadIdx_y) * 256) + (((int)threadIdx_x) * 8)))))[0];
    }
  }// End of fused attn_fc + add + LayerNorm, dim3(4, 24,1), dim3(32,4,1)

}

