
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <mma.h>
#include <cooperative_groups/memcpy_async.h>
#include <cuda/pipeline>

// dim3(192, 1, 1), dim3(128, 1, 1)
// Do double buffer
extern "C" __global__ void __launch_bounds__(128)
    fused_fc_fc_v4(half *__restrict__ x, half *__restrict__ placeholder,
                half *__restrict__ T_dense, half *__restrict__ placeholder2,
                half *__restrict__ T_dense2, float* sum, float* variance, half eps, half gama, half beta) {
  extern half __shared__ shared_buff_fused[]; 
  half* x_shared = (half*)&shared_buff_fused[0]; // 8706=64*136
  half* placeholder_shared = (half*)&shared_buff_fused[8704];// 4352=32*136
  half* x_shared2 = (half*)&shared_buff_fused[8704+4352]; // 8706=64*136
  half* placeholder_shared2 = (half*)&shared_buff_fused[8704+4352+8704];// 4352=32*136

  half* double_buffer_x_shared[] = {x_shared, x_shared2};
  half* double_buffer_placeholder_shared2[] = {placeholder_shared, placeholder_shared2};
  half* pipe1_placeholder_shared = (half*)&shared_buff_fused[4352];

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  // __shared__ half x_shared[8704];
  // __shared__ half placeholder_shared[4352];
  {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 32, 8, 16, half>
        T_dense_wmma_accumulator[2];
    const int blockIdx_x = blockIdx.x % 2;
    const int blockIdx_y = blockIdx.x / 2;
    const int threadIdx_x = threadIdx.x % 32;
    const int threadIdx_y = (threadIdx.x / 32) % 2;
    const int threadIdx_z = (threadIdx.x / 32) / 2;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 32, 8, 16, half,
                           nvcuda::wmma::row_major>
        x_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 32, 8, 16, half,
                           nvcuda::wmma::col_major>
        placeholder_shared_wmma_matrix_b[2];
    for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init) {
      (void)nvcuda::wmma::fill_fragment(
          T_dense_wmma_accumulator[j_c_outer_init], 0.000000e+00f);
    }
    // Set up double buffer
    pipe.producer_acquire();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0;
           ax0_ax1_fused_outer_outer_outer_outer < 8;
           ++ax0_ax1_fused_outer_outer_outer_outer) {
      cuda::memcpy_async((double_buffer_x_shared[0] +
                   ((((((ax0_ax1_fused_outer_outer_outer_outer * 1088) +
                        (((int)threadIdx_z) * 544)) +
                       (((int)threadIdx_y) * 272)) +
                      ((((int)threadIdx_x) >> 4) * 136)) +
                     ((((int)threadIdx_x) & 15) * 8)))), 
                     (x +
                       ((((((((((int)blockIdx_x) * 49152) +
                              (ax0_ax1_fused_outer_outer_outer_outer * 6144)) +
                             (((int)threadIdx_z) * 3072)) +
                            (((int)threadIdx_y) * 1536)) +
                           ((((int)threadIdx_x) >> 4) * 768)) +
                          (/*k_outer_outer*/ 0 * 128)) +
                         ((((int)threadIdx_x) & 15) * 8)))), shape, pipe);
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
           ax0_ax1_fused_outer_outer_outer_outer1 < 4;
           ++ax0_ax1_fused_outer_outer_outer_outer1) {
      cuda::memcpy_async((double_buffer_placeholder_shared2[0] +
                   ((((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) +
                        (((int)threadIdx_z) * 544)) +
                       (((int)threadIdx_y) * 272)) +
                      ((((int)threadIdx_x) >> 4) * 136)) +
                     ((((int)threadIdx_x) & 15) * 8)))), 
                     (placeholder +
                       ((((((((((int)blockIdx_y) * 24576) +
                              (ax0_ax1_fused_outer_outer_outer_outer1 * 6144)) +
                             (((int)threadIdx_z) * 3072)) +
                            (((int)threadIdx_y) * 1536)) +
                           ((((int)threadIdx_x) >> 4) * 768)) +
                          (/*k_outer_outer*/ 0 * 128)) +
                         ((((int)threadIdx_x) & 15) * 8)))), shape, pipe);
    }
    pipe.producer_commit();
    // Each time load k 768/6=128 elements
    for (int k_outer_outer = 1; k_outer_outer < 6; ++k_outer_outer) {
      __syncthreads();
      pipe.producer_acquire();
      for (int ax0_ax1_fused_outer_outer_outer_outer = 0;
           ax0_ax1_fused_outer_outer_outer_outer < 8;
           ++ax0_ax1_fused_outer_outer_outer_outer) {
      cuda::memcpy_async((double_buffer_x_shared[k_outer_outer % 2] +
                   ((((((ax0_ax1_fused_outer_outer_outer_outer * 1088) +
                        (((int)threadIdx_z) * 544)) +
                       (((int)threadIdx_y) * 272)) +
                      ((((int)threadIdx_x) >> 4) * 136)) +
                     ((((int)threadIdx_x) & 15) * 8)))), 
                     (x +
                       ((((((((((int)blockIdx_x) * 49152) +
                              (ax0_ax1_fused_outer_outer_outer_outer * 6144)) +
                             (((int)threadIdx_z) * 3072)) +
                            (((int)threadIdx_y) * 1536)) +
                           ((((int)threadIdx_x) >> 4) * 768)) +
                          (k_outer_outer * 128)) +
                         ((((int)threadIdx_x) & 15) * 8)))), shape, pipe);
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
           ax0_ax1_fused_outer_outer_outer_outer1 < 4;
           ++ax0_ax1_fused_outer_outer_outer_outer1) {
      cuda::memcpy_async((double_buffer_placeholder_shared2[k_outer_outer % 2] +
                   ((((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) +
                        (((int)threadIdx_z) * 544)) +
                       (((int)threadIdx_y) * 272)) +
                      ((((int)threadIdx_x) >> 4) * 136)) +
                     ((((int)threadIdx_x) & 15) * 8)))), 
                     (placeholder +
                       ((((((((((int)blockIdx_y) * 24576) +
                              (ax0_ax1_fused_outer_outer_outer_outer1 * 6144)) +
                             (((int)threadIdx_z) * 3072)) +
                            (((int)threadIdx_y) * 1536)) +
                           ((((int)threadIdx_x) >> 4) * 768)) +
                          (k_outer_outer * 128)) +
                         ((((int)threadIdx_x) & 15) * 8)))), shape, pipe);
      }
      pipe.producer_commit();
      pipe.consumer_wait();
      __syncthreads();
      for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
        (void)nvcuda::wmma::load_matrix_sync(
            x_shared_wmma_matrix_a[0],
            ((half *)double_buffer_x_shared[(k_outer_outer-1) % 2] +
             (((((int)threadIdx_y) * 4352) + (k_outer_inner * 16)))),
            136);
        for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
          (void)nvcuda::wmma::load_matrix_sync(
              placeholder_shared_wmma_matrix_b[ax0_outer],
              ((half *)double_buffer_placeholder_shared2[(k_outer_outer-1) % 2] +
               ((((((int)threadIdx_z) * 2176) + (ax0_outer * 1088)) +
                 (k_outer_inner * 16)))),
              136);
        }
        for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
          (void)nvcuda::wmma::mma_sync(
              T_dense_wmma_accumulator[j_c_outer], x_shared_wmma_matrix_a[0],
              placeholder_shared_wmma_matrix_b[j_c_outer],
              T_dense_wmma_accumulator[j_c_outer]);
        }
      }
      pipe.consumer_release();
    }
    // Drain the double buffer
    pipe.consumer_wait();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      (void)nvcuda::wmma::load_matrix_sync(
          x_shared_wmma_matrix_a[0],
          ((half *)double_buffer_x_shared[(6-1) % 2] +
            (((((int)threadIdx_y) * 4352) + (k_outer_inner * 16)))),
          136);
      for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
        (void)nvcuda::wmma::load_matrix_sync(
            placeholder_shared_wmma_matrix_b[ax0_outer],
            ((half *)double_buffer_placeholder_shared2[(6-1) % 2] +
              ((((((int)threadIdx_z) * 2176) + (ax0_outer * 1088)) +
                (k_outer_inner * 16)))),
            136);
      }
      for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
        (void)nvcuda::wmma::mma_sync(
            T_dense_wmma_accumulator[j_c_outer], x_shared_wmma_matrix_a[0],
            placeholder_shared_wmma_matrix_b[j_c_outer],
            T_dense_wmma_accumulator[j_c_outer]);
      }
    }
    pipe.consumer_release();

    // For next fc
    if(blockIdx.x < 96){
      const int blockIdx_x = blockIdx.x % 4;
      const int blockIdx_y = blockIdx.x / 4;
      const int threadIdx_x = threadIdx.x % 32;
      const int threadIdx_y = threadIdx.x / 32;
      pipe.producer_acquire();
      for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
            ax0_ax1_fused_outer_outer_outer_outer1 < 4;
            ++ax0_ax1_fused_outer_outer_outer_outer1) {
          cuda::memcpy_async((pipe1_placeholder_shared +
                    (((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) +
                        (((int)threadIdx_y) * 272)) +
                        ((((int)threadIdx_x) >> 4) * 136)) +
                      ((((int)threadIdx_x) & 15) * 8)))),
              (placeholder2 +
                        (((((((((int)blockIdx_y) * 98304) +
                              (ax0_ax1_fused_outer_outer_outer_outer1 * 24576)) +
                              (((int)threadIdx_y) * 6144)) +
                            ((((int)threadIdx_x) >> 4) * 3072)) +
                            (0 * 128)) +
                          ((((int)threadIdx_x) & 15) * 8)))), shape, pipe);
      }
      pipe.producer_commit();
    }

    __syncthreads();
    for (int ax1_outer_inner = 0; ax1_outer_inner < 2; ++ax1_outer_inner) {
      (void)nvcuda::wmma::store_matrix_sync(
          ((half *)placeholder_shared +
           ((((((int)threadIdx_y) * 1280) + (((int)threadIdx_z) * 16)) +
             (ax1_outer_inner * 8)))),
          T_dense_wmma_accumulator[ax1_outer_inner], 40,
          nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    // 196608=64*3072, 98304=32*3072, 49152=16*3072, 24576=8*3072,
    // threadIdx.x/4*3072, blockIdx.y * 32 Each block compute 64*32
    // output(blockIdx.x, blockIdx.y)
    for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0;
         i_inner_j_inner_fused_outer_outer_outer_outer < 2;
         ++i_inner_j_inner_fused_outer_outer_outer_outer) {
      ((uint4 *)(T_dense +
                 ((((((((((int)blockIdx_x) * 196608) +
                        (i_inner_j_inner_fused_outer_outer_outer_outer *
                         98304)) +
                       (((int)threadIdx_z) * 49152)) +
                      (((int)threadIdx_y) * 24576)) +
                     ((((int)threadIdx_x) >> 2) * 3072)) +
                    (((int)blockIdx_y) * 32)) +
                   ((((int)threadIdx_x) & 3) * 8)))))[0] =
          ((uint4 *)(placeholder_shared +
                     ((((((i_inner_j_inner_fused_outer_outer_outer_outer *
                           1280) +
                          (((int)threadIdx_z) * 640)) +
                         (((int)threadIdx_y) * 320)) +
                        ((((int)threadIdx_x) >> 2) * 40)) +
                       ((((int)threadIdx_x) & 3) * 8)))))[0];
    }
    __syncthreads();
  } // end of fc1

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  __syncthreads();
  __threadfence();
  grid.sync();
  
  if(blockIdx.x < 96){
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, half>
        T_dense_wmma_accumulator[1];
    const int blockIdx_x = blockIdx.x % 4;
    const int blockIdx_y = blockIdx.x / 4;
    const int threadIdx_x = threadIdx.x % 32;
    const int threadIdx_y = threadIdx.x / 32;

    // Load input for next fc
    pipe.producer_acquire();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0;
          ax0_ax1_fused_outer_outer_outer_outer < 4;
          ++ax0_ax1_fused_outer_outer_outer_outer) {
        cuda::memcpy_async((x_shared +
                  (((((ax0_ax1_fused_outer_outer_outer_outer * 1088) +
                      (((int)threadIdx_y) * 272)) +
                      ((((int)threadIdx_x) >> 4) * 136)) +
                    ((((int)threadIdx_x) & 15) * 8)))), 
                    (T_dense +
                      (((((((((int)blockIdx_x) * 98304) +
                            (ax0_ax1_fused_outer_outer_outer_outer * 24576)) +
                            (((int)threadIdx_y) * 6144)) +
                          ((((int)threadIdx_x) >> 4) * 3072)) +
                          (0 * 128)) +
                        ((((int)threadIdx_x) & 15) * 8)))), shape, pipe);
    }
    __syncthreads();
    pipe.producer_commit();

    pipe.consumer_wait();// Wait for weight
    pipe.consumer_wait();// Wait for input
    pipe.consumer_release();
    pipe.consumer_release();
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, half,
                           nvcuda::wmma::row_major>
        x_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, half,
                           nvcuda::wmma::col_major>
        placeholder_shared_wmma_matrix_b[1];
    (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[0],
                                      0.000000e+00f);
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      (void)nvcuda::wmma::load_matrix_sync(
          x_shared_wmma_matrix_a[0],
          ((half *)x_shared +
            (((((int)threadIdx_y) * 1088) + (k_outer_inner * 16)))),
          136);
      (void)nvcuda::wmma::load_matrix_sync(
          placeholder_shared_wmma_matrix_b[0],
          ((half *)pipe1_placeholder_shared + ((k_outer_inner * 16))), 136);
      (void)nvcuda::wmma::mma_sync(
          T_dense_wmma_accumulator[0], x_shared_wmma_matrix_a[0],
          placeholder_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[0]);
    }

    for (int k_outer_outer = 1; k_outer_outer < 24; ++k_outer_outer) {
      __syncthreads();
      for (int ax0_ax1_fused_outer_outer_outer_outer = 0;
           ax0_ax1_fused_outer_outer_outer_outer < 4;
           ++ax0_ax1_fused_outer_outer_outer_outer) {
        ((uint4 *)(x_shared +
                   (((((ax0_ax1_fused_outer_outer_outer_outer * 1088) +
                       (((int)threadIdx_y) * 272)) +
                      ((((int)threadIdx_x) >> 4) * 136)) +
                     ((((int)threadIdx_x) & 15) * 8)))))[0] =
            ((uint4 *)(T_dense +
                       (((((((((int)blockIdx_x) * 98304) +
                             (ax0_ax1_fused_outer_outer_outer_outer * 24576)) +
                            (((int)threadIdx_y) * 6144)) +
                           ((((int)threadIdx_x) >> 4) * 3072)) +
                          (k_outer_outer * 128)) +
                         ((((int)threadIdx_x) & 15) * 8)))))[0];
      }
      for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
           ax0_ax1_fused_outer_outer_outer_outer1 < 4;
           ++ax0_ax1_fused_outer_outer_outer_outer1) {
        ((uint4 *)(pipe1_placeholder_shared +
                   (((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) +
                       (((int)threadIdx_y) * 272)) +
                      ((((int)threadIdx_x) >> 4) * 136)) +
                     ((((int)threadIdx_x) & 15) * 8)))))[0] =
            ((uint4 *)(placeholder2 +
                       (((((((((int)blockIdx_y) * 98304) +
                             (ax0_ax1_fused_outer_outer_outer_outer1 * 24576)) +
                            (((int)threadIdx_y) * 6144)) +
                           ((((int)threadIdx_x) >> 4) * 3072)) +
                          (k_outer_outer * 128)) +
                         ((((int)threadIdx_x) & 15) * 8)))))[0];
      }
      __syncthreads();
      for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
        (void)nvcuda::wmma::load_matrix_sync(
            x_shared_wmma_matrix_a[0],
            ((half *)x_shared +
             (((((int)threadIdx_y) * 1088) + (k_outer_inner * 16)))),
            136);
        (void)nvcuda::wmma::load_matrix_sync(
            placeholder_shared_wmma_matrix_b[0],
            ((half *)pipe1_placeholder_shared + ((k_outer_inner * 16))), 136);
        (void)nvcuda::wmma::mma_sync(
            T_dense_wmma_accumulator[0], x_shared_wmma_matrix_a[0],
            placeholder_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[0]);
      }
    }
    // For next pipe
    // Do `src = src + self.dropout2(src2)`, Note we omit dropout here
    // Now T_dense2 is src2, x is src, so we load src to shared memory ahead of time
    pipe.producer_acquire();
    cuda::memcpy_async(((uint4 *)(placeholder_shared + ((((((int)threadIdx_y) * 320) +
                                 ((((int)threadIdx_x) >> 2) * 40)) +
                                ((((int)threadIdx_x) & 3) * 8))))), 
                    ((uint4 *)(x +
               ((((((((int)blockIdx_x) * 24576) + (((int)threadIdx_y) * 6144)) +
                   ((((int)threadIdx_x) >> 2) * 768)) +
                  (((int)blockIdx_y) * 32)) +
                 ((((int)threadIdx_x) & 3) * 8))))), shape, pipe);
    __syncthreads();

    (void)nvcuda::wmma::store_matrix_sync(
        ((half *)x_shared + ((((int)threadIdx_y) * 320))),
        T_dense_wmma_accumulator[0], 40, nvcuda::wmma::mem_row_major);
    pipe.producer_commit();
    pipe.consumer_wait();
    __syncthreads();

    
    // Compute Add, we have 128 threads, each thread compute half2, need iter 32*32/2/128=4
    const int x_shared_row_stride = 40;
    const int add_num_iter = 32 * 32 / 128;
    #pragma unroll
    for(int i=0; i<add_num_iter; ++i){
      int row = (i << 2) + (threadIdx.x >> 5); // row = threadIdx.x / 32; each block process 128/32=4 rows
      int col = threadIdx.x & 0x1f; //col = threadIdx.x % 32;
      int offset = row * x_shared_row_stride + col;
      x_shared[offset] = x_shared[offset] + placeholder_shared[offset];
    }
    pipe.consumer_release();
    __syncthreads();

    // Do self.norm(src)
    // 1. compute sum
    if(threadIdx.x < 32){
      float reduce_sum = 0;
      #pragma unroll
      for(int i=0; i<16; ++i){
        auto tmp = ((half2*)(x_shared + threadIdx.x * x_shared_row_stride + i*2))[0];
        reduce_sum += (__half2float(tmp.x) + __half2float(tmp.y));
      }
      atomicAdd(sum + blockIdx_x * 32 + threadIdx.x, reduce_sum / 768);
    }
    __syncthreads();
    __threadfence();
  }
  grid.sync();
  if(blockIdx.x < 96){
    const int blockIdx_x = blockIdx.x % 4;
    const int blockIdx_y = blockIdx.x / 4;
    const int threadIdx_x = threadIdx.x % 32;
    const int threadIdx_y = threadIdx.x / 32;
    const int x_shared_row_stride = 40;
    const int add_num_iter = 32 * 32 / 128;
    
    // Compute variance
    if(threadIdx.x < 32){
      half avg = __float2half(sum[blockIdx_x * 32 + threadIdx.x]);
      half2 avg2(avg, avg);
      // Compute variace
      float reduce_sum = 0;
      #pragma unroll
      for(int i=0; i<16; ++i){
        auto delt = ((half2*)(x_shared + threadIdx.x * x_shared_row_stride + i*2))[0] - avg2;
        float2 delt_f = __half22float2(delt);
        reduce_sum += (delt_f.x * delt_f.x + delt_f.y * delt_f.y);
      }
      atomicAdd(variance + blockIdx_x * 32 + threadIdx.x, reduce_sum / 768);
    }
    __syncthreads();
    __threadfence();
  }
  grid.sync();
  if(blockIdx.x < 96){
    const int blockIdx_x = blockIdx.x % 4;
    const int blockIdx_y = blockIdx.x / 4;
    const int threadIdx_x = threadIdx.x % 32;
    const int threadIdx_y = threadIdx.x / 32;
    
    const int add_num_iter = 32 * 32 / 128;
    #pragma unroll
    for(int i=0; i<add_num_iter; ++i){
      int row = (i << 2) + (threadIdx.x >> 5); // row = threadIdx.x / 32; each block process 128/32=4 rows
      int col = threadIdx.x & 0x1f; //col = threadIdx.x % 32;
      int g_row = blockIdx_x * 32 + row;
      half avg = __float2half(sum[g_row]);
      half reciprocal_vairance = half(1) / __float2half(sqrt(variance[g_row] + __half2float(eps)));
      int offset = row * 40 + col;
      x_shared[offset] = (x_shared[offset] - avg) * reciprocal_vairance * gama + beta;
    }
    __syncthreads();

    // 24576=32*768, 6144=16*768, 768,
    // Each block computes (32, 32)
    ((uint4 *)(T_dense2 +
               ((((((((int)blockIdx_x) * 24576) + (((int)threadIdx_y) * 6144)) +
                   ((((int)threadIdx_x) >> 2) * 768)) +
                  (((int)blockIdx_y) * 32)) +
                 ((((int)threadIdx_x) & 3) * 8)))))[0] =
        ((uint4 *)(x_shared + ((((((int)threadIdx_y) * 320) +
                                 ((((int)threadIdx_x) >> 2) * 40)) +
                                ((((int)threadIdx_x) & 3) * 8)))))[0];
  } // end of fc2
}











// dim3(192, 1, 1), dim3(128, 1, 1)
// Do double buffer with difference pipeline API, 
extern "C" __global__ void __launch_bounds__(128)
    fused_fc_fc_v5(half *__restrict__ x, half *__restrict__ placeholder,
                half *__restrict__ T_dense, half *__restrict__ placeholder2,
                half *__restrict__ T_dense2, float* sum, float* variance, half eps, half gama, half beta) {
  extern half __shared__ shared_buff_fused[]; 
  half* x_shared = (half*)&shared_buff_fused[0]; // 8706=64*136
  half* placeholder_shared = (half*)&shared_buff_fused[8704];// 4352=32*136
  half* x_shared2 = (half*)&shared_buff_fused[8704+4352]; // 8706=64*136
  half* placeholder_shared2 = (half*)&shared_buff_fused[8704+4352+8704];// 4352=32*136

  half* double_buffer_x_shared[] = {x_shared, x_shared2};
  half* double_buffer_placeholder_shared2[] = {placeholder_shared, placeholder_shared2};
  half* pipe1_placeholder_shared = (half*)&shared_buff_fused[4352];

  auto block = cooperative_groups::this_thread_block();
  const size_t stages_count = 2;
  __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;
  auto pipe = cuda::make_pipeline(block, &shared_state);
  const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  // __shared__ half x_shared[8704];
  // __shared__ half placeholder_shared[4352];
  {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 32, 8, 16, half>
        T_dense_wmma_accumulator[2];
    const int blockIdx_x = blockIdx.x % 2;
    const int blockIdx_y = blockIdx.x / 2;
    const int threadIdx_x = threadIdx.x % 32;
    const int threadIdx_y = (threadIdx.x / 32) % 2;
    const int threadIdx_z = (threadIdx.x / 32) / 2;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 32, 8, 16, half,
                           nvcuda::wmma::row_major>
        x_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 32, 8, 16, half,
                           nvcuda::wmma::col_major>
        placeholder_shared_wmma_matrix_b[2];
    for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init) {
      (void)nvcuda::wmma::fill_fragment(
          T_dense_wmma_accumulator[j_c_outer_init], 0.000000e+00f);
    }
    // Set up double buffer
    pipe.producer_acquire();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0;
           ax0_ax1_fused_outer_outer_outer_outer < 8;
           ++ax0_ax1_fused_outer_outer_outer_outer) {
      cuda::memcpy_async(block, (double_buffer_x_shared[0] +
                   ((((((ax0_ax1_fused_outer_outer_outer_outer * 1088) +
                        (((int)threadIdx_z) * 544)) +
                       (((int)threadIdx_y) * 272)) +
                      ((((int)threadIdx_x) >> 4) * 136)) +
                     ((((int)threadIdx_x) & 15) * 8)))), 
                     (x +
                       ((((((((((int)blockIdx_x) * 49152) +
                              (ax0_ax1_fused_outer_outer_outer_outer * 6144)) +
                             (((int)threadIdx_z) * 3072)) +
                            (((int)threadIdx_y) * 1536)) +
                           ((((int)threadIdx_x) >> 4) * 768)) +
                          (/*k_outer_outer*/ 0 * 128)) +
                         ((((int)threadIdx_x) & 15) * 8)))), shape, pipe);
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
           ax0_ax1_fused_outer_outer_outer_outer1 < 4;
           ++ax0_ax1_fused_outer_outer_outer_outer1) {
      cuda::memcpy_async(block, (double_buffer_placeholder_shared2[0] +
                   ((((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) +
                        (((int)threadIdx_z) * 544)) +
                       (((int)threadIdx_y) * 272)) +
                      ((((int)threadIdx_x) >> 4) * 136)) +
                     ((((int)threadIdx_x) & 15) * 8)))), 
                     (placeholder +
                       ((((((((((int)blockIdx_y) * 24576) +
                              (ax0_ax1_fused_outer_outer_outer_outer1 * 6144)) +
                             (((int)threadIdx_z) * 3072)) +
                            (((int)threadIdx_y) * 1536)) +
                           ((((int)threadIdx_x) >> 4) * 768)) +
                          (/*k_outer_outer*/ 0 * 128)) +
                         ((((int)threadIdx_x) & 15) * 8)))), shape, pipe);
    }
    pipe.producer_commit();
    // Each time load k 768/6=128 elements
    for (int k_outer_outer = 1; k_outer_outer < 6; ++k_outer_outer) {
      __syncthreads();
      pipe.producer_acquire();
      for (int ax0_ax1_fused_outer_outer_outer_outer = 0;
           ax0_ax1_fused_outer_outer_outer_outer < 8;
           ++ax0_ax1_fused_outer_outer_outer_outer) {
      cuda::memcpy_async(block, (double_buffer_x_shared[k_outer_outer % 2] +
                   ((((((ax0_ax1_fused_outer_outer_outer_outer * 1088) +
                        (((int)threadIdx_z) * 544)) +
                       (((int)threadIdx_y) * 272)) +
                      ((((int)threadIdx_x) >> 4) * 136)) +
                     ((((int)threadIdx_x) & 15) * 8)))), 
                     (x +
                       ((((((((((int)blockIdx_x) * 49152) +
                              (ax0_ax1_fused_outer_outer_outer_outer * 6144)) +
                             (((int)threadIdx_z) * 3072)) +
                            (((int)threadIdx_y) * 1536)) +
                           ((((int)threadIdx_x) >> 4) * 768)) +
                          (k_outer_outer * 128)) +
                         ((((int)threadIdx_x) & 15) * 8)))), shape, pipe);
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
           ax0_ax1_fused_outer_outer_outer_outer1 < 4;
           ++ax0_ax1_fused_outer_outer_outer_outer1) {
      cuda::memcpy_async(block, (double_buffer_placeholder_shared2[k_outer_outer % 2] +
                   ((((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) +
                        (((int)threadIdx_z) * 544)) +
                       (((int)threadIdx_y) * 272)) +
                      ((((int)threadIdx_x) >> 4) * 136)) +
                     ((((int)threadIdx_x) & 15) * 8)))), 
                     (placeholder +
                       ((((((((((int)blockIdx_y) * 24576) +
                              (ax0_ax1_fused_outer_outer_outer_outer1 * 6144)) +
                             (((int)threadIdx_z) * 3072)) +
                            (((int)threadIdx_y) * 1536)) +
                           ((((int)threadIdx_x) >> 4) * 768)) +
                          (k_outer_outer * 128)) +
                         ((((int)threadIdx_x) & 15) * 8)))), shape, pipe);
      }
      pipe.producer_commit();
      pipe.consumer_wait();
      __syncthreads();
      for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
        (void)nvcuda::wmma::load_matrix_sync(
            x_shared_wmma_matrix_a[0],
            ((half *)double_buffer_x_shared[(k_outer_outer-1) % 2] +
             (((((int)threadIdx_y) * 4352) + (k_outer_inner * 16)))),
            136);
        for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
          (void)nvcuda::wmma::load_matrix_sync(
              placeholder_shared_wmma_matrix_b[ax0_outer],
              ((half *)double_buffer_placeholder_shared2[(k_outer_outer-1) % 2] +
               ((((((int)threadIdx_z) * 2176) + (ax0_outer * 1088)) +
                 (k_outer_inner * 16)))),
              136);
        }
        for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
          (void)nvcuda::wmma::mma_sync(
              T_dense_wmma_accumulator[j_c_outer], x_shared_wmma_matrix_a[0],
              placeholder_shared_wmma_matrix_b[j_c_outer],
              T_dense_wmma_accumulator[j_c_outer]);
        }
      }
      pipe.consumer_release();
    }
    // Drain the double buffer
    pipe.consumer_wait();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      (void)nvcuda::wmma::load_matrix_sync(
          x_shared_wmma_matrix_a[0],
          ((half *)double_buffer_x_shared[(6-1) % 2] +
            (((((int)threadIdx_y) * 4352) + (k_outer_inner * 16)))),
          136);
      for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
        (void)nvcuda::wmma::load_matrix_sync(
            placeholder_shared_wmma_matrix_b[ax0_outer],
            ((half *)double_buffer_placeholder_shared2[(6-1) % 2] +
              ((((((int)threadIdx_z) * 2176) + (ax0_outer * 1088)) +
                (k_outer_inner * 16)))),
            136);
      }
      for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
        (void)nvcuda::wmma::mma_sync(
            T_dense_wmma_accumulator[j_c_outer], x_shared_wmma_matrix_a[0],
            placeholder_shared_wmma_matrix_b[j_c_outer],
            T_dense_wmma_accumulator[j_c_outer]);
      }
    }
    pipe.consumer_release();

    // For next fc
    if(blockIdx.x < 96){
      const int blockIdx_x = blockIdx.x % 4;
      const int blockIdx_y = blockIdx.x / 4;
      const int threadIdx_x = threadIdx.x % 32;
      const int threadIdx_y = threadIdx.x / 32;
      pipe.producer_acquire();
      for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
            ax0_ax1_fused_outer_outer_outer_outer1 < 4;
            ++ax0_ax1_fused_outer_outer_outer_outer1) {
          cuda::memcpy_async(block, (pipe1_placeholder_shared +
                    (((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) +
                        (((int)threadIdx_y) * 272)) +
                        ((((int)threadIdx_x) >> 4) * 136)) +
                      ((((int)threadIdx_x) & 15) * 8)))),
              (placeholder2 +
                        (((((((((int)blockIdx_y) * 98304) +
                              (ax0_ax1_fused_outer_outer_outer_outer1 * 24576)) +
                              (((int)threadIdx_y) * 6144)) +
                            ((((int)threadIdx_x) >> 4) * 3072)) +
                            (0 * 128)) +
                          ((((int)threadIdx_x) & 15) * 8)))), shape, pipe);
      }
      pipe.producer_commit();
    }

    __syncthreads();
    for (int ax1_outer_inner = 0; ax1_outer_inner < 2; ++ax1_outer_inner) {
      (void)nvcuda::wmma::store_matrix_sync(
          ((half *)placeholder_shared +
           ((((((int)threadIdx_y) * 1280) + (((int)threadIdx_z) * 16)) +
             (ax1_outer_inner * 8)))),
          T_dense_wmma_accumulator[ax1_outer_inner], 40,
          nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    // 196608=64*3072, 98304=32*3072, 49152=16*3072, 24576=8*3072,
    // threadIdx.x/4*3072, blockIdx.y * 32 Each block compute 64*32
    // output(blockIdx.x, blockIdx.y)
    for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0;
         i_inner_j_inner_fused_outer_outer_outer_outer < 2;
         ++i_inner_j_inner_fused_outer_outer_outer_outer) {
      ((uint4 *)(T_dense +
                 ((((((((((int)blockIdx_x) * 196608) +
                        (i_inner_j_inner_fused_outer_outer_outer_outer *
                         98304)) +
                       (((int)threadIdx_z) * 49152)) +
                      (((int)threadIdx_y) * 24576)) +
                     ((((int)threadIdx_x) >> 2) * 3072)) +
                    (((int)blockIdx_y) * 32)) +
                   ((((int)threadIdx_x) & 3) * 8)))))[0] =
          ((uint4 *)(placeholder_shared +
                     ((((((i_inner_j_inner_fused_outer_outer_outer_outer *
                           1280) +
                          (((int)threadIdx_z) * 640)) +
                         (((int)threadIdx_y) * 320)) +
                        ((((int)threadIdx_x) >> 2) * 40)) +
                       ((((int)threadIdx_x) & 3) * 8)))))[0];
    }
    __syncthreads();
  } // end of fc1

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  __syncthreads();
  __threadfence();
  grid.sync();
  
  if(blockIdx.x < 96){
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, half>
        T_dense_wmma_accumulator[1];
    const int blockIdx_x = blockIdx.x % 4;
    const int blockIdx_y = blockIdx.x / 4;
    const int threadIdx_x = threadIdx.x % 32;
    const int threadIdx_y = threadIdx.x / 32;

    pipe.producer_acquire();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0;
          ax0_ax1_fused_outer_outer_outer_outer < 4;
          ++ax0_ax1_fused_outer_outer_outer_outer) {
        cuda::memcpy_async(block, (x_shared +
                  (((((ax0_ax1_fused_outer_outer_outer_outer * 1088) +
                      (((int)threadIdx_y) * 272)) +
                      ((((int)threadIdx_x) >> 4) * 136)) +
                    ((((int)threadIdx_x) & 15) * 8)))), 
                    (T_dense +
                      (((((((((int)blockIdx_x) * 98304) +
                            (ax0_ax1_fused_outer_outer_outer_outer * 24576)) +
                            (((int)threadIdx_y) * 6144)) +
                          ((((int)threadIdx_x) >> 4) * 3072)) +
                          (0 * 128)) +
                        ((((int)threadIdx_x) & 15) * 8)))), shape, pipe);
    }
    __syncthreads();
    pipe.producer_commit();

    pipe.consumer_wait();// Wait for weight
    pipe.consumer_wait();// Wait for input
    pipe.consumer_release();
    pipe.consumer_release();
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, half,
                           nvcuda::wmma::row_major>
        x_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, half,
                           nvcuda::wmma::col_major>
        placeholder_shared_wmma_matrix_b[1];
    (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[0],
                                      0.000000e+00f);
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      (void)nvcuda::wmma::load_matrix_sync(
          x_shared_wmma_matrix_a[0],
          ((half *)x_shared +
            (((((int)threadIdx_y) * 1088) + (k_outer_inner * 16)))),
          136);
      (void)nvcuda::wmma::load_matrix_sync(
          placeholder_shared_wmma_matrix_b[0],
          ((half *)pipe1_placeholder_shared + ((k_outer_inner * 16))), 136);
      (void)nvcuda::wmma::mma_sync(
          T_dense_wmma_accumulator[0], x_shared_wmma_matrix_a[0],
          placeholder_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[0]);
    }

    for (int k_outer_outer = 1; k_outer_outer < 24; ++k_outer_outer) {
      __syncthreads();
      for (int ax0_ax1_fused_outer_outer_outer_outer = 0;
           ax0_ax1_fused_outer_outer_outer_outer < 4;
           ++ax0_ax1_fused_outer_outer_outer_outer) {
        ((uint4 *)(x_shared +
                   (((((ax0_ax1_fused_outer_outer_outer_outer * 1088) +
                       (((int)threadIdx_y) * 272)) +
                      ((((int)threadIdx_x) >> 4) * 136)) +
                     ((((int)threadIdx_x) & 15) * 8)))))[0] =
            ((uint4 *)(T_dense +
                       (((((((((int)blockIdx_x) * 98304) +
                             (ax0_ax1_fused_outer_outer_outer_outer * 24576)) +
                            (((int)threadIdx_y) * 6144)) +
                           ((((int)threadIdx_x) >> 4) * 3072)) +
                          (k_outer_outer * 128)) +
                         ((((int)threadIdx_x) & 15) * 8)))))[0];
      }
      for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
           ax0_ax1_fused_outer_outer_outer_outer1 < 4;
           ++ax0_ax1_fused_outer_outer_outer_outer1) {
        ((uint4 *)(pipe1_placeholder_shared +
                   (((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) +
                       (((int)threadIdx_y) * 272)) +
                      ((((int)threadIdx_x) >> 4) * 136)) +
                     ((((int)threadIdx_x) & 15) * 8)))))[0] =
            ((uint4 *)(placeholder2 +
                       (((((((((int)blockIdx_y) * 98304) +
                             (ax0_ax1_fused_outer_outer_outer_outer1 * 24576)) +
                            (((int)threadIdx_y) * 6144)) +
                           ((((int)threadIdx_x) >> 4) * 3072)) +
                          (k_outer_outer * 128)) +
                         ((((int)threadIdx_x) & 15) * 8)))))[0];
      }
      __syncthreads();
      for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
        (void)nvcuda::wmma::load_matrix_sync(
            x_shared_wmma_matrix_a[0],
            ((half *)x_shared +
             (((((int)threadIdx_y) * 1088) + (k_outer_inner * 16)))),
            136);
        (void)nvcuda::wmma::load_matrix_sync(
            placeholder_shared_wmma_matrix_b[0],
            ((half *)pipe1_placeholder_shared + ((k_outer_inner * 16))), 136);
        (void)nvcuda::wmma::mma_sync(
            T_dense_wmma_accumulator[0], x_shared_wmma_matrix_a[0],
            placeholder_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[0]);
      }
    }
    // For next pipe
    // Do `src = src + self.dropout2(src2)`, Note we omit dropout here
    // Now T_dense2 is src2, x is src, so we load src to shared memory ahead of time
    pipe.producer_acquire();
    cuda::memcpy_async(block, ((uint4 *)(placeholder_shared + ((((((int)threadIdx_y) * 320) +
                                 ((((int)threadIdx_x) >> 2) * 40)) +
                                ((((int)threadIdx_x) & 3) * 8))))), 
                    ((uint4 *)(x +
               ((((((((int)blockIdx_x) * 24576) + (((int)threadIdx_y) * 6144)) +
                   ((((int)threadIdx_x) >> 2) * 768)) +
                  (((int)blockIdx_y) * 32)) +
                 ((((int)threadIdx_x) & 3) * 8))))), shape, pipe);
    __syncthreads();

    (void)nvcuda::wmma::store_matrix_sync(
        ((half *)x_shared + ((((int)threadIdx_y) * 320))),
        T_dense_wmma_accumulator[0], 40, nvcuda::wmma::mem_row_major);
    pipe.producer_commit();
    pipe.consumer_wait();
    __syncthreads();

    
    // Compute Add, we have 128 threads, each thread compute half2, need iter 32*32/2/128=4
    const int x_shared_row_stride = 40;
    const int add_num_iter = 32 * 32 / 128;
    #pragma unroll
    for(int i=0; i<add_num_iter; ++i){
      int row = (i << 2) + (threadIdx.x >> 5); // row = threadIdx.x / 32; each block process 128/32=4 rows
      int col = threadIdx.x & 0x1f; //col = threadIdx.x % 32;
      int offset = row * x_shared_row_stride + col;
      x_shared[offset] = x_shared[offset] + placeholder_shared[offset];
    }
    pipe.consumer_release();
    __syncthreads();

    // Do self.norm(src)
    // 1. compute sum
    if(threadIdx.x < 32){
      float reduce_sum = 0;
      #pragma unroll
      for(int i=0; i<16; ++i){
        auto tmp = ((half2*)(x_shared + threadIdx.x * x_shared_row_stride + i*2))[0];
        reduce_sum += (__half2float(tmp.x) + __half2float(tmp.y));
      }
      atomicAdd(sum + blockIdx_x * 32 + threadIdx.x, reduce_sum / 768);
    }
    __syncthreads();
    __threadfence();
  }
  grid.sync();
  if(blockIdx.x < 96){
    const int blockIdx_x = blockIdx.x % 4;
    const int blockIdx_y = blockIdx.x / 4;
    const int threadIdx_x = threadIdx.x % 32;
    const int threadIdx_y = threadIdx.x / 32;
    const int x_shared_row_stride = 40;
    const int add_num_iter = 32 * 32 / 128;
    
    // Compute variance
    if(threadIdx.x < 32){
      half avg = __float2half(sum[blockIdx_x * 32 + threadIdx.x]);
      half2 avg2(avg, avg);
      // Compute variace
      float reduce_sum = 0;
      #pragma unroll
      for(int i=0; i<16; ++i){
        auto delt = ((half2*)(x_shared + threadIdx.x * x_shared_row_stride + i*2))[0] - avg2;
        float2 delt_f = __half22float2(delt);
        reduce_sum += (delt_f.x * delt_f.x + delt_f.y * delt_f.y);
      }
      atomicAdd(variance + blockIdx_x * 32 + threadIdx.x, reduce_sum / 768);
    }
    __syncthreads();
    __threadfence();
  }
  grid.sync();
  if(blockIdx.x < 96){
    const int blockIdx_x = blockIdx.x % 4;
    const int blockIdx_y = blockIdx.x / 4;
    const int threadIdx_x = threadIdx.x % 32;
    const int threadIdx_y = threadIdx.x / 32;
    
    const int add_num_iter = 32 * 32 / 128;
    #pragma unroll
    for(int i=0; i<add_num_iter; ++i){
      int row = (i << 2) + (threadIdx.x >> 5); // row = threadIdx.x / 32; each block process 128/32=4 rows
      int col = threadIdx.x & 0x1f; //col = threadIdx.x % 32;
      int g_row = blockIdx_x * 32 + row;
      half avg = __float2half(sum[g_row]);
      half reciprocal_vairance = half(1) / __float2half(sqrt(variance[g_row] + __half2float(eps)));
      int offset = row * 40 + col;
      x_shared[offset] = (x_shared[offset] - avg) * reciprocal_vairance * gama + beta;
    }
    __syncthreads();

    // 24576=32*768, 6144=16*768, 768,
    // Each block computes (32, 32)
    ((uint4 *)(T_dense2 +
               ((((((((int)blockIdx_x) * 24576) + (((int)threadIdx_y) * 6144)) +
                   ((((int)threadIdx_x) >> 2) * 768)) +
                  (((int)blockIdx_y) * 32)) +
                 ((((int)threadIdx_x) & 3) * 8)))))[0] =
        ((uint4 *)(x_shared + ((((((int)threadIdx_y) * 320) +
                                 ((((int)threadIdx_x) >> 2) * 40)) +
                                ((((int)threadIdx_x) & 3) * 8)))))[0];
  } // end of fc2
}




// dim3(192, 1, 1), dim3(128, 1, 1)
// Do double buffer with smaller shared memory,
// use (8704+8704+4352) * sizeof(half) shared memory
extern "C" __global__ void __launch_bounds__(128)
    fused_fc_fc_v6(half *__restrict__ x, half *__restrict__ placeholder,
                half *__restrict__ T_dense, half *__restrict__ placeholder2,
                half *__restrict__ T_dense2, float* sum, float* variance, half eps, half gama, half beta) {
  extern half __shared__ shared_buff_fused[]; 
  half* x_shared = (half*)&shared_buff_fused[0]; // 8706=64*136
  half* placeholder_shared = (half*)&shared_buff_fused[8704];// 4352=32*136
  half* x_shared2 = (half*)&shared_buff_fused[8704+4352]; // 8706=64*136
  half* double_buffer_x_shared[] = {x_shared, x_shared2};
  half* pipe1_placeholder_shared = (half*)&shared_buff_fused[4352];

  half* fc2_x_shared = (half*)&shared_buff_fused[0];
  half* fc2_x_placeholder_shared= (half*)&shared_buff_fused[4352];
  half* fc2_x_shared2 = (half*)&shared_buff_fused[4352+4352];
  half* fc2_x_placeholder_shared2 = (half*)&shared_buff_fused[4352+4352+4352];
  half* fc2_double_buffer_x_shared[] = {fc2_x_shared, fc2_x_shared2};
  half* fc2_double_buffer_placeholder_shared[] = {fc2_x_placeholder_shared, fc2_x_placeholder_shared2};

  auto block = cooperative_groups::this_thread_block();
  const size_t stages_count = 2;
  __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope::thread_scope_block,
        stages_count
    > shared_state;
  auto pipe = cuda::make_pipeline(block, &shared_state);
  const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  // __shared__ half x_shared[8704];
  // __shared__ half placeholder_shared[4352];
  {
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 32, 8, 16, half>
        T_dense_wmma_accumulator[2];
    const int blockIdx_x = blockIdx.x % 2;
    const int blockIdx_y = blockIdx.x / 2;
    const int threadIdx_x = threadIdx.x % 32;
    const int threadIdx_y = (threadIdx.x / 32) % 2;
    const int threadIdx_z = (threadIdx.x / 32) / 2;
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 32, 8, 16, half,
                           nvcuda::wmma::row_major>
        x_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 32, 8, 16, half,
                           nvcuda::wmma::col_major>
        placeholder_shared_wmma_matrix_b[2];
    for (int j_c_outer_init = 0; j_c_outer_init < 2; ++j_c_outer_init) {
      (void)nvcuda::wmma::fill_fragment(
          T_dense_wmma_accumulator[j_c_outer_init], 0.000000e+00f);
    }
    // Set up double buffer
    pipe.producer_acquire();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0;
           ax0_ax1_fused_outer_outer_outer_outer < 8;
           ++ax0_ax1_fused_outer_outer_outer_outer) {
      cuda::memcpy_async(block, (double_buffer_x_shared[0] +
                   ((((((ax0_ax1_fused_outer_outer_outer_outer * 1088) +
                        (((int)threadIdx_z) * 544)) +
                       (((int)threadIdx_y) * 272)) +
                      ((((int)threadIdx_x) >> 4) * 136)) +
                     ((((int)threadIdx_x) & 15) * 8)))), 
                     (x +
                       ((((((((((int)blockIdx_x) * 49152) +
                              (ax0_ax1_fused_outer_outer_outer_outer * 6144)) +
                             (((int)threadIdx_z) * 3072)) +
                            (((int)threadIdx_y) * 1536)) +
                           ((((int)threadIdx_x) >> 4) * 768)) +
                          (/*k_outer_outer*/ 0 * 128)) +
                         ((((int)threadIdx_x) & 15) * 8)))), shape, pipe);
    }
    pipe.producer_commit();

    // Each time load k 768/6=128 elements
    for (int k_outer_outer = 1; k_outer_outer < 6; ++k_outer_outer) {
      __syncthreads();
      pipe.producer_acquire();
      for (int ax0_ax1_fused_outer_outer_outer_outer = 0;
           ax0_ax1_fused_outer_outer_outer_outer < 8;
           ++ax0_ax1_fused_outer_outer_outer_outer) {
        cuda::memcpy_async(block, (double_buffer_x_shared[k_outer_outer % 2] +
                   ((((((ax0_ax1_fused_outer_outer_outer_outer * 1088) +
                        (((int)threadIdx_z) * 544)) +
                       (((int)threadIdx_y) * 272)) +
                      ((((int)threadIdx_x) >> 4) * 136)) +
                     ((((int)threadIdx_x) & 15) * 8)))), 
                     (x +
                       ((((((((((int)blockIdx_x) * 49152) +
                              (ax0_ax1_fused_outer_outer_outer_outer * 6144)) +
                             (((int)threadIdx_z) * 3072)) +
                            (((int)threadIdx_y) * 1536)) +
                           ((((int)threadIdx_x) >> 4) * 768)) +
                          (k_outer_outer * 128)) +
                         ((((int)threadIdx_x) & 15) * 8)))), shape, pipe);
      }
      pipe.producer_commit();
      for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
           ax0_ax1_fused_outer_outer_outer_outer1 < 4;
           ++ax0_ax1_fused_outer_outer_outer_outer1) {
        ((uint4 *)(placeholder_shared +
                   ((((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) +
                        (((int)threadIdx_z) * 544)) +
                       (((int)threadIdx_y) * 272)) +
                      ((((int)threadIdx_x) >> 4) * 136)) +
                     ((((int)threadIdx_x) & 15) * 8)))))[0] =
            ((uint4 *)(placeholder +
                       ((((((((((int)blockIdx_y) * 24576) +
                              (ax0_ax1_fused_outer_outer_outer_outer1 * 6144)) +
                             (((int)threadIdx_z) * 3072)) +
                            (((int)threadIdx_y) * 1536)) +
                           ((((int)threadIdx_x) >> 4) * 768)) +
                          ((k_outer_outer-1) * 128)) +
                         ((((int)threadIdx_x) & 15) * 8)))))[0];
      }
      pipe.consumer_wait();
      __syncthreads();
      for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
        (void)nvcuda::wmma::load_matrix_sync(
            x_shared_wmma_matrix_a[0],
            ((half *)double_buffer_x_shared[(k_outer_outer-1) % 2] +
             (((((int)threadIdx_y) * 4352) + (k_outer_inner * 16)))),
            136);
        for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
          (void)nvcuda::wmma::load_matrix_sync(
              placeholder_shared_wmma_matrix_b[ax0_outer],
              ((half *)placeholder_shared +
               ((((((int)threadIdx_z) * 2176) + (ax0_outer * 1088)) +
                 (k_outer_inner * 16)))),
              136);
        }
        for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
          (void)nvcuda::wmma::mma_sync(
              T_dense_wmma_accumulator[j_c_outer], x_shared_wmma_matrix_a[0],
              placeholder_shared_wmma_matrix_b[j_c_outer],
              T_dense_wmma_accumulator[j_c_outer]);
        }
      }
      pipe.consumer_release();
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
           ax0_ax1_fused_outer_outer_outer_outer1 < 4;
           ++ax0_ax1_fused_outer_outer_outer_outer1) {
        ((uint4 *)(placeholder_shared +
                   ((((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) +
                        (((int)threadIdx_z) * 544)) +
                       (((int)threadIdx_y) * 272)) +
                      ((((int)threadIdx_x) >> 4) * 136)) +
                     ((((int)threadIdx_x) & 15) * 8)))))[0] =
            ((uint4 *)(placeholder +
                       ((((((((((int)blockIdx_y) * 24576) +
                              (ax0_ax1_fused_outer_outer_outer_outer1 * 6144)) +
                             (((int)threadIdx_z) * 3072)) +
                            (((int)threadIdx_y) * 1536)) +
                           ((((int)threadIdx_x) >> 4) * 768)) +
                          ((6-1) * 128)) +
                         ((((int)threadIdx_x) & 15) * 8)))))[0];
      }
    // Drain the double buffer
    pipe.consumer_wait();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      (void)nvcuda::wmma::load_matrix_sync(
          x_shared_wmma_matrix_a[0],
          ((half *)double_buffer_x_shared[(6-1) % 2] +
            (((((int)threadIdx_y) * 4352) + (k_outer_inner * 16)))),
          136);
      for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
        (void)nvcuda::wmma::load_matrix_sync(
            placeholder_shared_wmma_matrix_b[ax0_outer],
            ((half *)placeholder_shared +
              ((((((int)threadIdx_z) * 2176) + (ax0_outer * 1088)) +
                (k_outer_inner * 16)))),
            136);
      }
      for (int j_c_outer = 0; j_c_outer < 2; ++j_c_outer) {
        (void)nvcuda::wmma::mma_sync(
            T_dense_wmma_accumulator[j_c_outer], x_shared_wmma_matrix_a[0],
            placeholder_shared_wmma_matrix_b[j_c_outer],
            T_dense_wmma_accumulator[j_c_outer]);
      }
    }
    pipe.consumer_release();

    // For next fc
    if(blockIdx.x < 96){
      const int blockIdx_x = blockIdx.x % 4;
      const int blockIdx_y = blockIdx.x / 4;
      const int threadIdx_x = threadIdx.x % 32;
      const int threadIdx_y = threadIdx.x / 32;
      pipe.producer_acquire();
      for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
            ax0_ax1_fused_outer_outer_outer_outer1 < 4;
            ++ax0_ax1_fused_outer_outer_outer_outer1) {
          cuda::memcpy_async(block, (pipe1_placeholder_shared +
                    (((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) +
                        (((int)threadIdx_y) * 272)) +
                        ((((int)threadIdx_x) >> 4) * 136)) +
                      ((((int)threadIdx_x) & 15) * 8)))),
              (placeholder2 +
                        (((((((((int)blockIdx_y) * 98304) +
                              (ax0_ax1_fused_outer_outer_outer_outer1 * 24576)) +
                              (((int)threadIdx_y) * 6144)) +
                            ((((int)threadIdx_x) >> 4) * 3072)) +
                            (0 * 128)) +
                          ((((int)threadIdx_x) & 15) * 8)))), shape, pipe);
      }
      pipe.producer_commit();
    }

    __syncthreads();
    for (int ax1_outer_inner = 0; ax1_outer_inner < 2; ++ax1_outer_inner) {
      (void)nvcuda::wmma::store_matrix_sync(
          ((half *)placeholder_shared +
           ((((((int)threadIdx_y) * 1280) + (((int)threadIdx_z) * 16)) +
             (ax1_outer_inner * 8)))),
          T_dense_wmma_accumulator[ax1_outer_inner], 40,
          nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    // 196608=64*3072, 98304=32*3072, 49152=16*3072, 24576=8*3072,
    // threadIdx.x/4*3072, blockIdx.y * 32 Each block compute 64*32
    // output(blockIdx.x, blockIdx.y)
    for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0;
         i_inner_j_inner_fused_outer_outer_outer_outer < 2;
         ++i_inner_j_inner_fused_outer_outer_outer_outer) {
      ((uint4 *)(T_dense +
                 ((((((((((int)blockIdx_x) * 196608) +
                        (i_inner_j_inner_fused_outer_outer_outer_outer *
                         98304)) +
                       (((int)threadIdx_z) * 49152)) +
                      (((int)threadIdx_y) * 24576)) +
                     ((((int)threadIdx_x) >> 2) * 3072)) +
                    (((int)blockIdx_y) * 32)) +
                   ((((int)threadIdx_x) & 3) * 8)))))[0] =
          ((uint4 *)(placeholder_shared +
                     ((((((i_inner_j_inner_fused_outer_outer_outer_outer *
                           1280) +
                          (((int)threadIdx_z) * 640)) +
                         (((int)threadIdx_y) * 320)) +
                        ((((int)threadIdx_x) >> 2) * 40)) +
                       ((((int)threadIdx_x) & 3) * 8)))))[0];
    }
    __syncthreads();
  } // end of fc1

  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  __syncthreads();
  __threadfence();
  grid.sync();
  
  if(blockIdx.x < 96){
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 8, 32, 16, half>
        T_dense_wmma_accumulator[1];
    const int blockIdx_x = blockIdx.x % 4;
    const int blockIdx_y = blockIdx.x / 4;
    const int threadIdx_x = threadIdx.x % 32;
    const int threadIdx_y = threadIdx.x / 32;

    pipe.producer_acquire();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0;
          ax0_ax1_fused_outer_outer_outer_outer < 4;
          ++ax0_ax1_fused_outer_outer_outer_outer) {
        cuda::memcpy_async(block, (x_shared +
                  (((((ax0_ax1_fused_outer_outer_outer_outer * 1088) +
                      (((int)threadIdx_y) * 272)) +
                      ((((int)threadIdx_x) >> 4) * 136)) +
                    ((((int)threadIdx_x) & 15) * 8)))), 
                    (T_dense +
                      (((((((((int)blockIdx_x) * 98304) +
                            (ax0_ax1_fused_outer_outer_outer_outer * 24576)) +
                            (((int)threadIdx_y) * 6144)) +
                          ((((int)threadIdx_x) >> 4) * 3072)) +
                          (0 * 128)) +
                        ((((int)threadIdx_x) & 15) * 8)))), shape, pipe);
    }
    pipe.producer_commit();
    __syncthreads();

    pipe.consumer_wait();// Wait for weight
    pipe.consumer_wait();// Wait for input
    pipe.consumer_release();
    pipe.consumer_release();
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 8, 32, 16, half,
                           nvcuda::wmma::row_major>
        x_shared_wmma_matrix_a[1];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 8, 32, 16, half,
                           nvcuda::wmma::col_major>
        placeholder_shared_wmma_matrix_b[1];
    (void)nvcuda::wmma::fill_fragment(T_dense_wmma_accumulator[0],
                                      0.000000e+00f);
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      (void)nvcuda::wmma::load_matrix_sync(
          x_shared_wmma_matrix_a[0],
          ((half *)x_shared +
            (((((int)threadIdx_y) * 1088) + (k_outer_inner * 16)))),
          136);
      (void)nvcuda::wmma::load_matrix_sync(
          placeholder_shared_wmma_matrix_b[0],
          ((half *)pipe1_placeholder_shared + ((k_outer_inner * 16))), 136);
      (void)nvcuda::wmma::mma_sync(
          T_dense_wmma_accumulator[0], x_shared_wmma_matrix_a[0],
          placeholder_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[0]);
    }
    // Set up double buffer
    pipe.producer_acquire();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0;
          ax0_ax1_fused_outer_outer_outer_outer < 4;
          ++ax0_ax1_fused_outer_outer_outer_outer) {
        cuda::memcpy_async(block, (fc2_double_buffer_x_shared[0] +
                  (((((ax0_ax1_fused_outer_outer_outer_outer * 1088) +
                      (((int)threadIdx_y) * 272)) +
                      ((((int)threadIdx_x) >> 4) * 136)) +
                    ((((int)threadIdx_x) & 15) * 8)))), 
                    (T_dense +
                      (((((((((int)blockIdx_x) * 98304) +
                            (ax0_ax1_fused_outer_outer_outer_outer * 24576)) +
                            (((int)threadIdx_y) * 6144)) +
                          ((((int)threadIdx_x) >> 4) * 3072)) +
                          (1 * 128)) +
                        ((((int)threadIdx_x) & 15) * 8)))), shape, pipe);
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
            ax0_ax1_fused_outer_outer_outer_outer1 < 4;
            ++ax0_ax1_fused_outer_outer_outer_outer1) {
        cuda::memcpy_async(block, (fc2_double_buffer_placeholder_shared[0] +
                  (((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) +
                      (((int)threadIdx_y) * 272)) +
                      ((((int)threadIdx_x) >> 4) * 136)) +
                    ((((int)threadIdx_x) & 15) * 8)))),
            (placeholder2 +
                      (((((((((int)blockIdx_y) * 98304) +
                            (ax0_ax1_fused_outer_outer_outer_outer1 * 24576)) +
                            (((int)threadIdx_y) * 6144)) +
                          ((((int)threadIdx_x) >> 4) * 3072)) +
                          (1 * 128)) +
                        ((((int)threadIdx_x) & 15) * 8)))), shape, pipe);
    }
    pipe.producer_commit();

    for (int k_outer_outer = 2; k_outer_outer < 24; ++k_outer_outer) {
      __syncthreads();
      pipe.producer_acquire();
      for (int ax0_ax1_fused_outer_outer_outer_outer = 0;
            ax0_ax1_fused_outer_outer_outer_outer < 4;
            ++ax0_ax1_fused_outer_outer_outer_outer) {
          cuda::memcpy_async(block, (fc2_double_buffer_x_shared[(k_outer_outer-1)%2] +
                    (((((ax0_ax1_fused_outer_outer_outer_outer * 1088) +
                        (((int)threadIdx_y) * 272)) +
                        ((((int)threadIdx_x) >> 4) * 136)) +
                      ((((int)threadIdx_x) & 15) * 8)))), 
                      (T_dense +
                        (((((((((int)blockIdx_x) * 98304) +
                              (ax0_ax1_fused_outer_outer_outer_outer * 24576)) +
                              (((int)threadIdx_y) * 6144)) +
                            ((((int)threadIdx_x) >> 4) * 3072)) +
                            (k_outer_outer * 128)) +
                          ((((int)threadIdx_x) & 15) * 8)))), shape, pipe);
      }
      for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
              ax0_ax1_fused_outer_outer_outer_outer1 < 4;
              ++ax0_ax1_fused_outer_outer_outer_outer1) {
          cuda::memcpy_async(block, (fc2_double_buffer_placeholder_shared[(k_outer_outer-1)%2] +
                    (((((ax0_ax1_fused_outer_outer_outer_outer1 * 1088) +
                        (((int)threadIdx_y) * 272)) +
                        ((((int)threadIdx_x) >> 4) * 136)) +
                      ((((int)threadIdx_x) & 15) * 8)))),
              (placeholder2 +
                        (((((((((int)blockIdx_y) * 98304) +
                              (ax0_ax1_fused_outer_outer_outer_outer1 * 24576)) +
                              (((int)threadIdx_y) * 6144)) +
                            ((((int)threadIdx_x) >> 4) * 3072)) +
                            (k_outer_outer * 128)) +
                          ((((int)threadIdx_x) & 15) * 8)))), shape, pipe);
      }
      pipe.producer_commit();
      pipe.consumer_wait();
      __syncthreads();
      for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
        (void)nvcuda::wmma::load_matrix_sync(
            x_shared_wmma_matrix_a[0],
            ((half *)fc2_double_buffer_x_shared[k_outer_outer%2] +
             (((((int)threadIdx_y) * 1088) + (k_outer_inner * 16)))),
            136);
        (void)nvcuda::wmma::load_matrix_sync(
            placeholder_shared_wmma_matrix_b[0],
            ((half *)fc2_double_buffer_placeholder_shared[k_outer_outer%2] + ((k_outer_inner * 16))), 136);
        (void)nvcuda::wmma::mma_sync(
            T_dense_wmma_accumulator[0], x_shared_wmma_matrix_a[0],
            placeholder_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[0]);
      }
      pipe.consumer_release();
    }

    // Drain the double buffer
    pipe.consumer_wait();
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 8; ++k_outer_inner) {
      (void)nvcuda::wmma::load_matrix_sync(
          x_shared_wmma_matrix_a[0],
          ((half *)fc2_double_buffer_x_shared[(24-1)%2] +
            (((((int)threadIdx_y) * 1088) + (k_outer_inner * 16)))),
          136);
      (void)nvcuda::wmma::load_matrix_sync(
          placeholder_shared_wmma_matrix_b[0],
          ((half *)fc2_double_buffer_placeholder_shared[(24-1)%2] + ((k_outer_inner * 16))), 136);
      (void)nvcuda::wmma::mma_sync(
          T_dense_wmma_accumulator[0], x_shared_wmma_matrix_a[0],
          placeholder_shared_wmma_matrix_b[0], T_dense_wmma_accumulator[0]);
    }
    pipe.consumer_release();
    // For next pipe
    // Do `src = src + self.dropout2(src2)`, Note we omit dropout here
    // Now T_dense2 is src2, x is src, so we load src to shared memory ahead of time
    pipe.producer_acquire();
    cuda::memcpy_async(block, ((uint4 *)(placeholder_shared + ((((((int)threadIdx_y) * 320) +
                                 ((((int)threadIdx_x) >> 2) * 40)) +
                                ((((int)threadIdx_x) & 3) * 8))))), 
                    ((uint4 *)(x +
               ((((((((int)blockIdx_x) * 24576) + (((int)threadIdx_y) * 6144)) +
                   ((((int)threadIdx_x) >> 2) * 768)) +
                  (((int)blockIdx_y) * 32)) +
                 ((((int)threadIdx_x) & 3) * 8))))), shape, pipe);
    pipe.producer_commit();
    __syncthreads();

    (void)nvcuda::wmma::store_matrix_sync(
        ((half *)x_shared + ((((int)threadIdx_y) * 320))),
        T_dense_wmma_accumulator[0], 40, nvcuda::wmma::mem_row_major);
    
    pipe.consumer_wait();
    __syncthreads();

    
    // Compute Add, we have 128 threads, each thread compute half2, need iter 32*32/2/128=4
    const int x_shared_row_stride = 40;
    const int add_num_iter = 32 * 32 / 128;
    #pragma unroll
    for(int i=0; i<add_num_iter; ++i){
      int row = (i << 2) + (threadIdx.x >> 5); // row = threadIdx.x / 32; each block process 128/32=4 rows
      int col = threadIdx.x & 0x1f; //col = threadIdx.x % 32;
      int offset = row * x_shared_row_stride + col;
      x_shared[offset] = x_shared[offset] + placeholder_shared[offset];
    }
    pipe.consumer_release();
    __syncthreads();

    // Do self.norm(src)
    // 1. compute sum
    if(threadIdx.x < 32){
      float reduce_sum = 0;
      #pragma unroll
      for(int i=0; i<16; ++i){
        auto tmp = ((half2*)(x_shared + threadIdx.x * x_shared_row_stride + i*2))[0];
        reduce_sum += (__half2float(tmp.x) + __half2float(tmp.y));
      }
      atomicAdd(sum + blockIdx_x * 32 + threadIdx.x, reduce_sum / 768);
    }
    __syncthreads();
    __threadfence();
  }
  grid.sync();
  if(blockIdx.x < 96){
    const int blockIdx_x = blockIdx.x % 4;
    const int blockIdx_y = blockIdx.x / 4;
    const int threadIdx_x = threadIdx.x % 32;
    const int threadIdx_y = threadIdx.x / 32;
    const int x_shared_row_stride = 40;
    const int add_num_iter = 32 * 32 / 128;
    
    // Compute variance
    if(threadIdx.x < 32){
      half avg = __float2half(sum[blockIdx_x * 32 + threadIdx.x]);
      half2 avg2(avg, avg);
      // Compute variace
      float reduce_sum = 0;
      #pragma unroll
      for(int i=0; i<16; ++i){
        auto delt = ((half2*)(x_shared + threadIdx.x * x_shared_row_stride + i*2))[0] - avg2;
        float2 delt_f = __half22float2(delt);
        reduce_sum += (delt_f.x * delt_f.x + delt_f.y * delt_f.y);
      }
      atomicAdd(variance + blockIdx_x * 32 + threadIdx.x, reduce_sum / 768);
    }
    __syncthreads();
    __threadfence();
  }
  grid.sync();
  if(blockIdx.x < 96){
    const int blockIdx_x = blockIdx.x % 4;
    const int blockIdx_y = blockIdx.x / 4;
    const int threadIdx_x = threadIdx.x % 32;
    const int threadIdx_y = threadIdx.x / 32;
    
    const int add_num_iter = 32 * 32 / 128;
    #pragma unroll
    for(int i=0; i<add_num_iter; ++i){
      int row = (i << 2) + (threadIdx.x >> 5); // row = threadIdx.x / 32; each block process 128/32=4 rows
      int col = threadIdx.x & 0x1f; //col = threadIdx.x % 32;
      int g_row = blockIdx_x * 32 + row;
      half avg = __float2half(sum[g_row]);
      half reciprocal_vairance = half(1) / __float2half(sqrt(variance[g_row] + __half2float(eps)));
      int offset = row * 40 + col;
      x_shared[offset] = (x_shared[offset] - avg) * reciprocal_vairance * gama + beta;
    }
    __syncthreads();

    // 24576=32*768, 6144=16*768, 768,
    // Each block computes (32, 32)
    ((uint4 *)(T_dense2 +
               ((((((((int)blockIdx_x) * 24576) + (((int)threadIdx_y) * 6144)) +
                   ((((int)threadIdx_x) >> 2) * 768)) +
                  (((int)blockIdx_y) * 32)) +
                 ((((int)threadIdx_x) & 3) * 8)))))[0] =
        ((uint4 *)(x_shared + ((((((int)threadIdx_y) * 320) +
                                 ((((int)threadIdx_x) >> 2) * 40)) +
                                ((((int)threadIdx_x) & 3) * 8)))))[0];
  } // end of fc2
}
