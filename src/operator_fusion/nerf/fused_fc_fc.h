

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
//     printf("97: blockIdx.x %d, on %d, ok %d, offset %d, value %f\n",
//     blockIdx.x, on, 0, offset, __half2float(ele));
//     ((half*)gemm_weight_shared)[offset] = (half)1;
//   }
// }

// dim3(128*4, 1, 1), dim3(32, 4, 1)

extern "C" __global__ void __launch_bounds__(128)
    fused_fc_fc(half *__restrict__ input, half *__restrict__ weight1,
                half *__restrict__ weight2, half *__restrict__ weight3,
                half *__restrict__ weight4, half *__restrict__ output) {
  const int N = 256;
  const int K = 256;
  const int k_block_size_x = 32;
  const int k_block_size_y = 4;
  const int k_block_size = k_block_size_x * k_block_size_y;
  const int warpSize = 32;
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
                       input + (blockIdx.x * tile_size_m * K) + offset, shape,
                       pipe);
  }

  half *weight_ptr;
  const int num_layers = 1;
  for (int w = 0; w < num_layers; ++w) {
    weight_ptr = weight_arr[w];
    // Now start load to shared and compute in double buffer
    const int n_num_iter = N / tile_size_n;
    const int k_num_iter = K / tile_size_k;
    for (int on = 0; on < n_num_iter; ++on) {
// init accumulator
#pragma unroll
      for (int i_c_outer_init = 0; i_c_outer_init < 2; ++i_c_outer_init) {
        (void)nvcuda::wmma::fill_fragment(
            output_wmma_accumulator[i_c_outer_init], 0.000000e+00f);
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
        int row = i * (k_block_size / vec_tile_size_k) +
                  ((threadIdx.y * warpSize + threadIdx.x) / vec_tile_size_k);
        int col = (threadIdx.y * warpSize + threadIdx.x) % vec_tile_size_k;
        int g_row = on * tile_size_n + row;
        int g_col = 0 * vec_tile_size_k + col;
        int s_offset = (row * vec_tile_size_k + col) * vec_size;
        int g_offset = (g_row * vec_K + g_col) * vec_size;
        cuda::memcpy_async(gemm_weight_shared + s_offset, weight_ptr + g_offset,
                           shape, pipe);
      }
      pipe.producer_commit();
      pipe.consumer_wait();
      __syncthreads();

      for (int ok = 1; ok < k_num_iter; ++ok) {
        // Load second weight
        for (int i = 0; i < load_weight_num_iter; ++i) {
          int row = i * (k_block_size / vec_tile_size_k) +
                    ((threadIdx.y * warpSize + threadIdx.x) / vec_tile_size_k);
          int col = (threadIdx.y * warpSize + threadIdx.x) % vec_tile_size_k;
          int g_row = on * tile_size_n + row;
          int g_col = ok * vec_tile_size_k + col;
          int s_offset = (row * vec_tile_size_k + col) * vec_size;
          int g_offset = (g_row * vec_K + g_col) * vec_size;
          cuda::memcpy_async(load_weight_shared + s_offset,
                             weight_ptr + g_offset, shape, pipe);
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
    if (num_layers != 1) {
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        half *tmp_ptr = input_shared_ptr;
        input_shared_ptr = output_shared_ptr;
        output_shared_ptr = tmp_ptr;
      }
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

extern "C" __global__ void __launch_bounds__(128)
    fused_fc_fc_v2(half *__restrict__ input, half *__restrict__ weight1,
                   half *__restrict__ weight2, half *__restrict__ weight3,
                   half *__restrict__ weight4, half *__restrict__ output) {
  const int N = 256;
  const int K = 256;
  const int k_block_size_x = 32;
  const int k_block_size_y = 4;
  const int k_block_size = k_block_size_x * k_block_size_y;
  const int warpSize = 32;
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
                       input + (blockIdx.x * tile_size_m * K) + offset, shape,
                       pipe);
  }

  half *weight_ptr;
  const int num_layers = 1;
  for (int w = 0; w < num_layers; ++w) {
    weight_ptr = weight_arr[w];
    // Now start load to shared and compute in double buffer
    const int n_num_iter = N / tile_size_n;
    const int k_num_iter = K / tile_size_k;
    for (int on = 0; on < n_num_iter; ++on) {
// init accumulator
#pragma unroll
      for (int i_c_outer_init = 0; i_c_outer_init < 2; ++i_c_outer_init) {
        (void)nvcuda::wmma::fill_fragment(
            output_wmma_accumulator[i_c_outer_init], 0.000000e+00f);
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
        int row = i * (k_block_size / vec_tile_size_k) +
                  ((threadIdx.y * warpSize + threadIdx.x) / vec_tile_size_k);
        int col = (threadIdx.y * warpSize + threadIdx.x) % vec_tile_size_k;
        int g_row = on * tile_size_n + row;
        int g_col = 0 * vec_tile_size_k + col;
        int s_offset = (row * vec_tile_size_k + col) * vec_size;
        int g_offset = (g_row * vec_K + g_col) * vec_size;
        cuda::memcpy_async(gemm_weight_shared + s_offset, weight_ptr + g_offset,
                           shape, pipe);
      }

      for (int ok = 1; ok < k_num_iter; ++ok) {
        // Load second weight
        pipe.producer_commit();
        pipe.consumer_wait();
        __syncthreads();
        for (int i = 0; i < load_weight_num_iter; ++i) {
          int row = i * (k_block_size / vec_tile_size_k) +
                    ((threadIdx.y * warpSize + threadIdx.x) / vec_tile_size_k);
          int col = (threadIdx.y * warpSize + threadIdx.x) % vec_tile_size_k;
          int g_row = on * tile_size_n + row;
          int g_col = ok * vec_tile_size_k + col;
          int s_offset = (row * vec_tile_size_k + col) * vec_size;
          int g_offset = (g_row * vec_K + g_col) * vec_size;
          cuda::memcpy_async(load_weight_shared + s_offset,
                             weight_ptr + g_offset, shape, pipe);
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
        __syncthreads();
        // Swap pointers
        if (threadIdx.x == 0 && threadIdx.y == 0) {
          half *tmp_weight = load_weight_shared;
          load_weight_shared = gemm_weight_shared;
          gemm_weight_shared = tmp_weight;
        }
      }
      pipe.producer_commit();
      pipe.consumer_wait();
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
    // const int act_vec_size = sizeof(half2) / sizeof(half);
    // const int act_input_shared_num_iter =
    //     tile_size_m * K / act_vec_size / k_block_size_x / k_block_size_y;
    // #pragma unroll
    // for (int i = 0; i < act_input_shared_num_iter; ++i) {
    //   int offset = (i * k_block_size + threadIdx.y * warpSize + threadIdx.x);
    //   half2 ele = ((half2 *)output_shared_ptr + offset)[0];
    //   if (ele.x < (half)0) {
    //     ele.x = (half)0;
    //   }
    //   if (ele.y < (half)0) {
    //     ele.y = (half)0;
    //   }
    //   ((half2 *)output_shared_ptr + offset)[0] = ele;
    // }
    __syncthreads();
    __threadfence_block();
    // Swap shared input and output ptr
    if (num_layers != 1) {
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        half *tmp_ptr = input_shared_ptr;
        input_shared_ptr = output_shared_ptr;
        output_shared_ptr = tmp_ptr;
      }
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

// dim(108*4, 1, 1), dim3(32, 8, 1)
extern "C" __global__ void __launch_bounds__(256)
    fused_fc_fc_v3(half *__restrict__ input, half *__restrict__ weight1,
                   half *__restrict__ weight2, half *__restrict__ weight3,
                   half *__restrict__ weight4, half *__restrict__ output) {
  const int N = 256;
  const int K = 256;
  const int k_block_size_x = 32;
  const int k_block_size_y = 8;
  const int k_block_size = k_block_size_x * k_block_size_y;
  const int warpSize = 32;
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
  const int tile_size_n = 256;
  const int tile_size_k = 16;
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
  // 16 * 256 / 8 / 256 = 2
  const int load_input_2_shared_num_iter =
      tile_size_m * K / vec_size / k_block_size_x / k_block_size_y;
  for (int i = 0; i < load_input_2_shared_num_iter; ++i) {
    int offset =
        (i * k_block_size + threadIdx.y * warpSize + threadIdx.x) * vec_size;
    cuda::memcpy_async(input_shared_ptr + offset,
                       input + (blockIdx.x * tile_size_m * K) + offset, shape,
                       pipe);
  }

  half *weight_ptr;
  const int num_layers = 1;
  for (int w = 0; w < num_layers; ++w) {
    weight_ptr = weight_arr[w];
    // Now start load to shared and compute in double buffer
    const int n_num_iter = N / tile_size_n;
    const int k_num_iter = K / tile_size_k;
    for (int on = 0; on < n_num_iter; ++on) {
// init accumulator
#pragma unroll
      for (int i_c_outer_init = 0; i_c_outer_init < 2; ++i_c_outer_init) {
        (void)nvcuda::wmma::fill_fragment(
            output_wmma_accumulator[i_c_outer_init], 0.000000e+00f);
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
        int row = i * (k_block_size / vec_tile_size_k) +
                  ((threadIdx.y * warpSize + threadIdx.x) / vec_tile_size_k);
        int col = (threadIdx.y * warpSize + threadIdx.x) % vec_tile_size_k;
        int g_row = on * tile_size_n + row;
        int g_col = 0 * vec_tile_size_k + col;
        int s_offset = (row * vec_tile_size_k + col) * vec_size;
        int g_offset = (g_row * vec_K + g_col) * vec_size;
        cuda::memcpy_async(gemm_weight_shared + s_offset, weight_ptr + g_offset,
                           shape, pipe);
      }

      for (int ok = 1; ok < k_num_iter; ++ok) {
        // Load second weight
        pipe.producer_commit();
        pipe.consumer_wait();
        __syncthreads();
        for (int i = 0; i < load_weight_num_iter; ++i) {
          int row = i * (k_block_size / vec_tile_size_k) +
                    ((threadIdx.y * warpSize + threadIdx.x) / vec_tile_size_k);
          int col = (threadIdx.y * warpSize + threadIdx.x) % vec_tile_size_k;
          int g_row = on * tile_size_n + row;
          int g_col = ok * vec_tile_size_k + col;
          int s_offset = (row * vec_tile_size_k + col) * vec_size;
          int g_offset = (g_row * vec_K + g_col) * vec_size;
          cuda::memcpy_async(load_weight_shared + s_offset,
                             weight_ptr + g_offset, shape, pipe);
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
        __syncthreads();
        // Swap pointers
        if (threadIdx.x == 0 && threadIdx.y == 0) {
          half *tmp_weight = load_weight_shared;
          load_weight_shared = gemm_weight_shared;
          gemm_weight_shared = tmp_weight;
        }
      }
      pipe.producer_commit();
      pipe.consumer_wait();
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
    __threadfence_block();
    // Swap shared input and output ptr
    if (num_layers != 1) {
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        half *tmp_ptr = input_shared_ptr;
        input_shared_ptr = output_shared_ptr;
        output_shared_ptr = tmp_ptr;
      }
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

// dim3(54, 2, 1), dim3(32, 4, 2)
extern "C" __global__ void __launch_bounds__(256)
    tvm_matmul(half *__restrict__ x, half *__restrict__ placeholder,
               half *__restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half>
      T_dense_wmma_accumulator[8];
  __shared__ half x_shared[5120];
  __shared__ half placeholder_shared[17408];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half,
                         nvcuda::wmma::row_major>
      x_shared_wmma_matrix_a[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half,
                         nvcuda::wmma::col_major>
      placeholder_shared_wmma_matrix_b[4];
  for (int i_c_outer_init = 0; i_c_outer_init < 2; ++i_c_outer_init) {
    for (int j_c_outer_init = 0; j_c_outer_init < 4; ++j_c_outer_init) {
      (void)nvcuda::wmma::fill_fragment(
          T_dense_wmma_accumulator[((i_c_outer_init * 4) + j_c_outer_init)],
          0.000000e+00f);
    }
  }
  // inner K 为32, x_shared应该是[128, 40], placeholder_shared应该是[256, 68]
  for (int k_outer_outer = 0; k_outer_outer < 8; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0;
         ax0_ax1_fused_outer_outer_outer_outer < 4;
         ++ax0_ax1_fused_outer_outer_outer_outer) {
      ((uint2 *)(x_shared +
                 ((((((ax0_ax1_fused_outer_outer_outer_outer * 1280) +
                      (((int)threadIdx.z) * 640)) +
                     (((int)threadIdx.y) * 160)) +
                    ((((int)threadIdx.x) >> 3) * 40)) +
                   ((((int)threadIdx.x) & 7) * 4)))))[0] =
          ((uint2 *)(x +
                     ((((((((((int)blockIdx.x) * 32768) +
                            (ax0_ax1_fused_outer_outer_outer_outer * 8192)) +
                           (((int)threadIdx.z) * 4096)) +
                          (((int)threadIdx.y) * 1024)) +
                         ((((int)threadIdx.x) >> 3) * 256)) +
                        (k_outer_outer * 32)) +
                       ((((int)threadIdx.x) & 7) * 4)))))[0];
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
         ax0_ax1_fused_outer_outer_outer_outer1 < 4;
         ++ax0_ax1_fused_outer_outer_outer_outer1) {
      ((uint2 *)(placeholder_shared +
                 ((((((ax0_ax1_fused_outer_outer_outer_outer1 * 1280) +
                      (((int)threadIdx.z) * 640)) +
                     (((int)threadIdx.y) * 160)) +
                    ((((int)threadIdx.x) >> 3) * 40)) +
                   ((((int)threadIdx.x) & 7) * 4)))))[0] =
          ((uint2 *)(placeholder +
                     ((((((((((int)blockIdx.y) * 32768) +
                            (ax0_ax1_fused_outer_outer_outer_outer1 * 8192)) +
                           (((int)threadIdx.z) * 4096)) +
                          (((int)threadIdx.y) * 1024)) +
                         ((((int)threadIdx.x) >> 3) * 256)) +
                        (k_outer_outer * 32)) +
                       ((((int)threadIdx.x) & 7) * 4)))))[0];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 2; ++k_outer_inner) {
      for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
        (void)nvcuda::wmma::load_matrix_sync(
            x_shared_wmma_matrix_a[ax0_outer],
            ((half *)x_shared +
             ((((((int)threadIdx.y) * 1280) + (ax0_outer * 640)) +
               (k_outer_inner * 16)))),
            40);
      }
      for (int ax0_outer1 = 0; ax0_outer1 < 4; ++ax0_outer1) {
        (void)nvcuda::wmma::load_matrix_sync(
            placeholder_shared_wmma_matrix_b[ax0_outer1],
            ((half *)placeholder_shared +
             ((((((int)threadIdx.z) * 2560) + (ax0_outer1 * 640)) +
               (k_outer_inner * 16)))),
            40);
      }
      for (int i_c_outer = 0; i_c_outer < 2; ++i_c_outer) {
        for (int j_c_outer = 0; j_c_outer < 4; ++j_c_outer) {
          (void)nvcuda::wmma::mma_sync(
              T_dense_wmma_accumulator[((i_c_outer * 4) + j_c_outer)],
              x_shared_wmma_matrix_a[i_c_outer],
              placeholder_shared_wmma_matrix_b[j_c_outer],
              T_dense_wmma_accumulator[((i_c_outer * 4) + j_c_outer)]);
        }
      }
    }
  }
  __syncthreads();
  for (int ax0_outer_inner = 0; ax0_outer_inner < 2; ++ax0_outer_inner) {
    for (int ax1_outer_inner = 0; ax1_outer_inner < 4; ++ax1_outer_inner) {
      (void)nvcuda::wmma::store_matrix_sync(
          ((half *)placeholder_shared +
           (((((((int)threadIdx.y) * 4352) + (ax0_outer_inner * 2176)) +
              (((int)threadIdx.z) * 64)) +
             (ax1_outer_inner * 16)))),
          T_dense_wmma_accumulator[((ax0_outer_inner * 4) + ax1_outer_inner)],
          136, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  // 32768/256=128, 每个block算了[128, 256],
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0;
       i_inner_j_inner_fused_outer_outer_outer_outer < 16;
       ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    ((uint2 *)(T_dense +
               (((((((((int)blockIdx.x) * 32768) +
                     (i_inner_j_inner_fused_outer_outer_outer_outer * 2048)) +
                    (((int)threadIdx.z) * 1024)) +
                   (((int)threadIdx.y) * 256)) +
                  (((int)blockIdx.y) * 128)) +
                 (((int)threadIdx.x) * 4)))))[0] =
        ((uint2 *)(placeholder_shared +
                   (((((i_inner_j_inner_fused_outer_outer_outer_outer * 1088) +
                       (((int)threadIdx.z) * 544)) +
                      (((int)threadIdx.y) * 136)) +
                     (((int)threadIdx.x) * 4)))))[0];
  }
}