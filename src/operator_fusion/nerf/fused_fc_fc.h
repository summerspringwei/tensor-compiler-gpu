

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <mma.h>

// Check whether all the weight is right
// const int check_weight_num_iter = tile_size_n * tile_size_k / k_block_size;
// for(int i=0; i<check_weight_num_iter; ++i){
//   int offset = i*k_block_size + threadIdx.y * warpSize + threadIdx.x;
//   auto ele = ((half*)gemm_weight_shared)[offset];
//   if(ele != (half)1){
//     printf("97: blockIdx.x %d, on %d, ok %d, offset %d, value %f\n", blockIdx.x, on, 0, offset, __half2float(ele));
//     ((half*)gemm_weight_shared)[offset] = (half)1;
//   }
// }

// dim3(128*4, 1, 1), dim3(32, 4, 1)
const int N = 256;
const int K = 256;
const int k_block_size_x = 32;
const int k_block_size_y = 4;
const int k_block_size = k_block_size_x * k_block_size_y;
const int warpSize = 32;
extern "C" __global__ void __launch_bounds__(128)
    fused_fc_fc(half *__restrict__ input, half *__restrict__ weight1,
                half *__restrict__ weight2, half *__restrict__ weight3,
                half *__restrict__ weight4, half *__restrict__ output) {
  half *weight_arr[] = {weight1, weight2, weight3, weight4};
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half>
      output_wmma_accumulator[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half,
                         nvcuda::wmma::row_major>
      input_wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half,
                         nvcuda::wmma::row_major>
      weight_wmma_matrix_b[2];
  for (int i_c_outer_init = 0; i_c_outer_init < 2; ++i_c_outer_init) {
    (void)nvcuda::wmma::fill_fragment(output_wmma_accumulator[i_c_outer_init],
                                      0.000000e+00f);
  }
  // Can be configure
  const int tile_size_m = 16;
  const int tile_size_n = 128;
  const int tile_size_k = 32;
  __shared__ half input_shared[tile_size_m * K];
  __shared__ half buffer_1_weight_shared[tile_size_n * tile_size_k];
  __shared__ half buffer_2_weight_shared[tile_size_n * tile_size_k];
  __shared__ half output_shared[tile_size_m * K];
  half *input_shared_ptr = input_shared;
  half *output_shared_ptr = output_shared;

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  pipe.producer_acquire();
  // First load all input to shared memory
  const int vec_size = sizeof(float4) / sizeof(half);
  const int warpId = threadIdx.y;
  const int load_input_2_shared_num_iter =
      tile_size_m * K / vec_size / k_block_size_x / k_block_size_y;
  for (int i = 0; i < load_input_2_shared_num_iter; ++i) {
    int offset =
        (i * k_block_size + threadIdx.y * warpSize + threadIdx.x) * vec_size;
    cuda::memcpy_async(input_shared_ptr + offset,
                       input + (blockIdx.x * tile_size_m * K) + offset, shape, pipe);
  }

  half* weight_ptr;
  for (int w = 0; w < 4; ++w) {
    weight_ptr = weight_arr[w];
    // Now start load to shared and compute in double buffer
    const int n_num_iter = N / tile_size_n;
    const int k_num_iter = K / tile_size_k;
    for (int on = 0; on < n_num_iter; ++on) { 
      // init accumulator
      #pragma unroll
      for (int i_c_outer_init = 0; i_c_outer_init < 2; ++i_c_outer_init) {
        (void)nvcuda::wmma::fill_fragment(output_wmma_accumulator[i_c_outer_init],
                                          0.000000e+00f);
      }
      // Load first weight
      half *load_weight_shared = buffer_1_weight_shared,
           *gemm_weight_shared = buffer_2_weight_shared;
      const int load_weight_num_iter =
          tile_size_n * tile_size_k / vec_size / k_block_size;
      // For each thread the row and col index in the tiled shared weight
      // Load to [128, 32], now load to shared [128, 4], global [256, 64]
      const int vec_tile_size_k = tile_size_k / vec_size;
      const int vec_K = K / vec_size;
      #pragma unroll
      for (int i = 0; i < load_weight_num_iter; ++i) {
        int row = i * (k_block_size/vec_tile_size_k) + ((threadIdx.y*warpSize+threadIdx.x)/vec_tile_size_k);
        int col = (threadIdx.y*warpSize+threadIdx.x)%vec_tile_size_k;
        int g_row = on*tile_size_n + row;
        int g_col = 0*vec_tile_size_k+col;
        int s_offset = (row*vec_tile_size_k+col)*vec_size;
        int g_offset = (g_row*vec_K+g_col)*vec_size;
        cuda::memcpy_async(gemm_weight_shared + s_offset, weight_ptr + g_offset,
                           shape, pipe);
      }
      pipe.producer_commit();
      pipe.consumer_wait();
      __syncthreads();
      
      for (int ok = 1; ok < k_num_iter; ++ok) {
        // Load second weight
        for (int i = 0; i < load_weight_num_iter; ++i) {
          int row = i * (k_block_size/vec_tile_size_k) + ((threadIdx.y*warpSize+threadIdx.x)/vec_tile_size_k);
          int col = (threadIdx.y*warpSize+threadIdx.x)%vec_tile_size_k;
          int g_row = on*tile_size_n + row;
          int g_col = ok*vec_tile_size_k+col;
          int s_offset = (row*vec_tile_size_k+col)*vec_size;
          int g_offset = (g_row*vec_K+g_col)*vec_size;
          cuda::memcpy_async(load_weight_shared + s_offset, weight_ptr + g_offset,
                            shape, pipe);
        }
        // Do gemm
        const int inner_num_iter = tile_size_k / 16;
        #pragma unroll
        for (int ik = 0; ik < inner_num_iter; ++ik) {
          (void)nvcuda::wmma::load_matrix_sync(
              input_wmma_matrix_a[0],
              ((half *)input_shared_ptr) + ok * tile_size_k + ik * 16, K);
          (void)nvcuda::wmma::load_matrix_sync(
              weight_wmma_matrix_b[0],
              ((half *)gemm_weight_shared +
               (0 * k_block_size_y * 16 + warpId * 16) * tile_size_k + ik * 16),
              tile_size_k);
          (void)nvcuda::wmma::load_matrix_sync(
              weight_wmma_matrix_b[1],
              ((half *)gemm_weight_shared +
               (1 * k_block_size_y * 16 + warpId * 16) * tile_size_k + ik * 16),
              tile_size_k);
          (void)nvcuda::wmma::mma_sync(
              output_wmma_accumulator[0], input_wmma_matrix_a[0],
              weight_wmma_matrix_b[0], output_wmma_accumulator[0]);
          (void)nvcuda::wmma::mma_sync(
              output_wmma_accumulator[1], input_wmma_matrix_a[0],
              weight_wmma_matrix_b[1], output_wmma_accumulator[1]);
        }
        pipe.producer_commit();
        pipe.consumer_wait();
        __syncthreads();
        // Swap pointers
        if (threadIdx.x == 0 && threadIdx.y == 0) {
          half *tmp_weight = load_weight_shared;
          load_weight_shared = gemm_weight_shared;
          gemm_weight_shared = tmp_weight;
        }
        __syncthreads();
      }
      pipe.consumer_release();
      // Do the last gemm
      const int inner_num_iter = tile_size_k / 16;
      for (int ik = 0; ik < inner_num_iter; ++ik) {
        (void)nvcuda::wmma::load_matrix_sync(
            input_wmma_matrix_a[0],
            ((half *)input_shared_ptr) + (k_num_iter - 1) * tile_size_k +
                ik * 16,
            K);
        (void)nvcuda::wmma::load_matrix_sync(
            weight_wmma_matrix_b[0],
            ((half *)gemm_weight_shared +
             (0 * k_block_size_y * 16 + warpId * 16) * tile_size_k + ik * 16),
            tile_size_k);
        (void)nvcuda::wmma::load_matrix_sync(
            weight_wmma_matrix_b[1],
            ((half *)gemm_weight_shared +
             (1 * k_block_size_y * 16 + warpId * 16) * tile_size_k + ik * 16),
            tile_size_k);
        (void)nvcuda::wmma::mma_sync(
            output_wmma_accumulator[0], input_wmma_matrix_a[0],
            weight_wmma_matrix_b[0], output_wmma_accumulator[0]);
        (void)nvcuda::wmma::mma_sync(
            output_wmma_accumulator[1], input_wmma_matrix_a[0],
            weight_wmma_matrix_b[1], output_wmma_accumulator[1]);
      }
      __syncthreads();
      // Store back to output shared
      (void)nvcuda::wmma::store_matrix_sync(
          ((half *)output_shared_ptr + on * tile_size_n +
           0 * k_block_size_y * 16 + warpId * 16),
          output_wmma_accumulator[0], K, nvcuda::wmma::mem_row_major);
      (void)nvcuda::wmma::store_matrix_sync(
          ((half *)output_shared_ptr + on * tile_size_n +
           1 * k_block_size_y * 16 + warpId * 16),
          output_wmma_accumulator[1], K, nvcuda::wmma::mem_row_major);
      __syncthreads();
    }
    // Do the activation for all outputs
    const int act_vec_size = sizeof(half2) / sizeof(half);
    const int act_input_shared_num_iter =
        tile_size_m * K / act_vec_size / k_block_size_x / k_block_size_y;
    #pragma unroll
    for (int i = 0; i < act_input_shared_num_iter; ++i) {
      int offset = (i * k_block_size + threadIdx.y * warpSize + threadIdx.x);
      half2 ele = ((half2 *)output_shared_ptr + offset)[0];
      if (ele.x < (half)0) {
        ele.x = (half)0;
      }
      if (ele.y < (half)0) {
        ele.y = (half)0;
      }
      ((half2 *)output_shared_ptr + offset)[0] = ele;
    }
    __syncthreads();
    __threadfence_block();
    // Swap shared input and output ptr
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      half *tmp_ptr = input_shared_ptr;
      input_shared_ptr = output_shared_ptr;
      output_shared_ptr = tmp_ptr;
    }
    __syncthreads();
  }
  // Write output from shared memory to global memory (16*256/8/128=4)
  int store_output_num_iters =
      tile_size_m * K / (sizeof(float4) / sizeof(half)) / k_block_size;
  #pragma unroll
  for (int i = 0; i < store_output_num_iters; ++i) {
    int offset = i * k_block_size + threadIdx.y * warpSize + threadIdx.x;
    ((float4 *)output +
     (blockIdx.x * tile_size_m * K / (sizeof(float4) / sizeof(half)) +
      offset))[0] = ((float4 *)output_shared_ptr + offset)[0];
  }
}