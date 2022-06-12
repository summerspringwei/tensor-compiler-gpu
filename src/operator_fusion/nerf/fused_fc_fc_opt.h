#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <mma.h>

// dim(108*4, 1, 1), dim3(32, 8, 1)
extern "C" __global__ void __launch_bounds__(256)
    fused_fc_fc_v4(half *__restrict__ input, half *__restrict__ weight1,
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
  __shared__ half buffer_input_output_shared[tile_size_m * K * 2];
  __shared__ half buffer_weight_shared[tile_size_n * tile_size_k * 2];
  half *weight_shared_ptr[2];
  weight_shared_ptr[0] = buffer_weight_shared;
  weight_shared_ptr[1] = buffer_weight_shared + tile_size_n * tile_size_k;

  half *input_output_shared_ptr[2];
  input_output_shared_ptr[0] = buffer_input_output_shared;
  input_output_shared_ptr[1] = buffer_input_output_shared + tile_size_m * K;

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  pipe.producer_acquire();
  // First load all input to shared memory
  const int vec_size = sizeof(float4) / sizeof(half);
  const int warpId = threadIdx.y;
  const int tidx = (threadIdx.y * warpSize + threadIdx.x);
  const int load_input_2_shared_num_iter =
      tile_size_m * K / vec_size / k_block_size_x / k_block_size_y;
  const int input_block_offset = (blockIdx.x * tile_size_m * K);
#pragma unroll
  for (int i = 0; i < load_input_2_shared_num_iter; ++i) {
    int offset = (i * k_block_size + threadIdx.y * warpSize + threadIdx.x) << 3;
    cuda::memcpy_async(input_output_shared_ptr[0] + offset,
                       input + input_block_offset + offset, shape, pipe);
  }

  half *weight_ptr;
  const int num_layers = 1;
  for (int w = 0; w < num_layers; ++w) {
    weight_ptr = weight_arr[w];
    // Now start load to shared and compute in double buffer
    const int n_num_iter = N / tile_size_n;
    const int k_num_iter = K / tile_size_k;
    static_assert(k_num_iter > 2);
    
    for (int on = 0; on < n_num_iter; ++on) {
// init accumulator
#pragma unroll
      for (int i_c_outer_init = 0; i_c_outer_init < 2; ++i_c_outer_init) {
        (void)nvcuda::wmma::fill_fragment(
            output_wmma_accumulator[i_c_outer_init], 0.000000e+00f);
      }
      // Load first weight
      const int load_weight_num_iter =
          tile_size_n * tile_size_k / vec_size / k_block_size;
      // For each thread the row and col index in the tiled shared weight
      // Load to [128, 32], now load to shared [128, 4], global [256, 64]
      const int vec_tile_size_k = tile_size_k / vec_size;
      const int vec_K = K / vec_size;
#pragma unroll
      for (int i = 0; i < load_weight_num_iter; ++i) {
        int row = i * (k_block_size / vec_tile_size_k) + (tidx >> 1);
        int col = tidx & 0x1;
        int g_row = on * tile_size_n + row;
        int g_col = 0 * vec_tile_size_k + col;
        int s_offset = (row * vec_tile_size_k + col) << 3;
        int g_offset = (g_row * vec_K + g_col) << 3;
        cuda::memcpy_async(weight_shared_ptr[0] + s_offset,
                           weight_ptr + g_offset, shape, pipe);
      }
#pragma unroll
      for (int ok = 1; ok < k_num_iter; ++ok) {
        // Load second weight
        pipe.producer_commit();
        pipe.consumer_wait();
        __syncthreads();
#pragma unroll
        for (int i = 0; i < load_weight_num_iter; ++i) {
          int row = i * (k_block_size >> 1) + (tidx >> 1);
          int col = tidx & 0x1;
          int g_row = on * tile_size_n + row;
          int g_col = ok * vec_tile_size_k + col;
          int s_offset = (row * vec_tile_size_k + col) << 3;
          int g_offset = (g_row * vec_K + g_col) << 3;
          cuda::memcpy_async(weight_shared_ptr[ok & 0x1] + s_offset,
                             weight_ptr + g_offset, shape, pipe);
        }
        // Do gemm
        const int inner_num_iter = tile_size_k / 16;
#pragma unroll
        for (int ik = 0; ik < inner_num_iter; ++ik) {
          (void)nvcuda::wmma::load_matrix_sync(
              input_wmma_matrix_a[0],
              ((half *)input_output_shared_ptr[w & 0x1]) + ok * tile_size_k +
                  ik * 16,
              K);
          (void)nvcuda::wmma::load_matrix_sync(
              weight_wmma_matrix_b[0],
              ((half *)weight_shared_ptr[(ok - 1) & 0x1] +
               (0 * k_block_size_y * 16 + warpId * 16) * tile_size_k + ik * 16),
              tile_size_k);
          (void)nvcuda::wmma::load_matrix_sync(
              weight_wmma_matrix_b[1],
              ((half *)weight_shared_ptr[(ok - 1) & 0x1] +
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
      }
      pipe.producer_commit();
      pipe.consumer_wait();
      pipe.consumer_release();
      // Do the last gemm
      const int inner_num_iter = tile_size_k / 16;
      for (int ik = 0; ik < inner_num_iter; ++ik) {
        (void)nvcuda::wmma::load_matrix_sync(
            input_wmma_matrix_a[0],
            ((half *)input_output_shared_ptr[w & 0x1]) +
                (k_num_iter - 1) * tile_size_k + ik * 16,
            K);
        (void)nvcuda::wmma::load_matrix_sync(
            weight_wmma_matrix_b[0],
            ((half *)weight_shared_ptr[(k_num_iter - 1) & 0x1] +
             (0 * k_block_size_y * 16 + warpId * 16) * tile_size_k + ik * 16),
            tile_size_k);
        (void)nvcuda::wmma::load_matrix_sync(
            weight_wmma_matrix_b[1],
            ((half *)weight_shared_ptr[(k_num_iter - 1) & 0x1] +
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
          ((half *)input_output_shared_ptr[(w + 1) & 0x1] + on * tile_size_n +
           0 * k_block_size_y * 16 + warpId * 16),
          output_wmma_accumulator[0], K, nvcuda::wmma::mem_row_major);
      (void)nvcuda::wmma::store_matrix_sync(
          ((half *)input_output_shared_ptr[(w + 1) & 0x1] + on * tile_size_n +
           1 * k_block_size_y * 16 + warpId * 16),
          output_wmma_accumulator[1], K, nvcuda::wmma::mem_row_major);
      __syncthreads();
    }
    __syncthreads();
  }
  // Write output from shared memory to global memory (16*256/8/128=4)
  int store_output_num_iters =
      tile_size_m * K / (sizeof(float4) / sizeof(half)) / k_block_size;
#pragma unroll
  for (int i = 0; i < store_output_num_iters; ++i) {
    int offset = i * k_block_size + tidx;
    ((float4 *)output +
     (blockIdx.x * tile_size_m * K / (sizeof(float4) / sizeof(half)) +
      offset))[0] =
        ((float4 *)(input_output_shared_ptr[num_layers & 0x1]) + offset)[0];
  }
}


// Note, input shape changed to (108*4*32)
// dim(108*4, 1, 1), dim3(32, 8, 1)
extern "C" __global__ void __launch_bounds__(256)
    fused_fc_fc_v4_reuse_input_shared(half *__restrict__ input, half *__restrict__ weight1,
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
      output_wmma_accumulator[4];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half,
                         nvcuda::wmma::row_major>
      input_wmma_matrix_a[2];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half,
                         nvcuda::wmma::row_major>
      weight_wmma_matrix_b[2];

  // Can be configure
  const int tile_size_m = 32;
  const int tile_size_n = 256;
  const int tile_size_k = 16;
  __shared__ half buffer_input_output_shared[tile_size_m * K];
  __shared__ half buffer_weight_shared[tile_size_n * tile_size_k * 2];
  half *weight_shared_ptr[2];
  weight_shared_ptr[0] = buffer_weight_shared;
  weight_shared_ptr[1] = buffer_weight_shared + tile_size_n * tile_size_k;

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  pipe.producer_acquire();
  // First load all input to shared memory
  const int vec_size = sizeof(float4) / sizeof(half);
  const int warpId = threadIdx.y;
  const int tidx = (threadIdx.y * warpSize + threadIdx.x);
  const int load_input_2_shared_num_iter =
      tile_size_m * K / vec_size / k_block_size_x / k_block_size_y;
  const int input_block_offset = (blockIdx.x * tile_size_m * K);
#pragma unroll
  for (int i = 0; i < load_input_2_shared_num_iter; ++i) {
    int offset = (i * k_block_size + tidx) << 3;
    cuda::memcpy_async(buffer_input_output_shared + offset,
                       input + input_block_offset + offset, shape, pipe);
  }

  half *weight_ptr;
  const int num_layers = 4;
  for (int w = 0; w < num_layers; ++w) {
    weight_ptr = weight_arr[w];
    // Now start load to shared and compute in double buffer
    const int n_num_iter = N / tile_size_n;
    const int k_num_iter = K / tile_size_k;
    static_assert(k_num_iter > 2);
    
    for (int on = 0; on < n_num_iter; ++on) {
// init accumulator
#pragma unroll
      for (int i_c_outer_init = 0; i_c_outer_init < 4; ++i_c_outer_init) {
        (void)nvcuda::wmma::fill_fragment(
            output_wmma_accumulator[i_c_outer_init], 0.000000e+00f);
      }
      // Load first weight
      const int load_weight_num_iter =
          tile_size_n * tile_size_k / vec_size / k_block_size;
      const int vec_tile_size_k = tile_size_k / vec_size; // 2
      const int vec_K = K / vec_size;
      int t_row = tidx >> 1;
      int t_col = tidx & 0x1;
#pragma unroll
      for (int i = 0; i < load_weight_num_iter; ++i) {
        int row = i * (k_block_size / vec_tile_size_k) + t_row;
        int col = t_col;
        int g_row = on * tile_size_n + row;
        int g_col = 0 * vec_tile_size_k + col;
        int s_offset = (row * vec_tile_size_k + col) << 3;
        int g_offset = (g_row * vec_K + g_col) << 3;
        cuda::memcpy_async(weight_shared_ptr[0] + s_offset,
                           weight_ptr + g_offset, shape, pipe);
      }
#pragma unroll
      for (int ok = 1; ok < k_num_iter; ++ok) {
        // Load second weight
        pipe.producer_commit();
        pipe.consumer_wait();
        __syncthreads();
#pragma unroll
        for (int i = 0; i < load_weight_num_iter; ++i) {
          int row = i * (k_block_size / vec_tile_size_k) + t_row;
          int col = t_col;
          int g_row = on * tile_size_n + row;
          int g_col = ok * vec_tile_size_k + col;
          int s_offset = (row * vec_tile_size_k + col) << 3;
          int g_offset = (g_row * vec_K + g_col) << 3;
          cuda::memcpy_async(weight_shared_ptr[ok & 0x1] + s_offset,
                             weight_ptr + g_offset, shape, pipe);
        }
        // Do gemm
        const int inner_num_iter = tile_size_k / 16;
#pragma unroll
        for (int ik = 0; ik < inner_num_iter; ++ik) {
          (void)nvcuda::wmma::load_matrix_sync(
              input_wmma_matrix_a[0],
              ((half *)buffer_input_output_shared) + ok * tile_size_k +
                  ik * 16,
              K);
          (void)nvcuda::wmma::load_matrix_sync(
              weight_wmma_matrix_b[0],
              ((half *)weight_shared_ptr[(ok - 1) & 0x1] +
               (0 * k_block_size_y * 16 + warpId * 16) * tile_size_k + ik * 16),
              tile_size_k);
          (void)nvcuda::wmma::mma_sync(
              output_wmma_accumulator[0], input_wmma_matrix_a[0],
              weight_wmma_matrix_b[0], output_wmma_accumulator[0]);
          (void)nvcuda::wmma::load_matrix_sync(
              input_wmma_matrix_a[1],
              ((half *)buffer_input_output_shared) + 16 * K + ok * tile_size_k +
                  ik * 16,
              K);
          (void)nvcuda::wmma::mma_sync(
              output_wmma_accumulator[2], input_wmma_matrix_a[1],
              weight_wmma_matrix_b[0], output_wmma_accumulator[2]);
          (void)nvcuda::wmma::load_matrix_sync(
              weight_wmma_matrix_b[1],
              ((half *)weight_shared_ptr[(ok - 1) & 0x1] +
               (1 * k_block_size_y * 16 + warpId * 16) * tile_size_k + ik * 16),
              tile_size_k);
          (void)nvcuda::wmma::mma_sync(
              output_wmma_accumulator[1], input_wmma_matrix_a[0],
              weight_wmma_matrix_b[1], output_wmma_accumulator[1]);
          (void)nvcuda::wmma::mma_sync(
              output_wmma_accumulator[3], input_wmma_matrix_a[1],
              weight_wmma_matrix_b[1], output_wmma_accumulator[3]);
        }
        __syncthreads();
      }
      pipe.producer_commit();
      pipe.consumer_wait();
      pipe.consumer_release();
      // Do the last gemm
      const int inner_num_iter = tile_size_k / 16;
      for (int ik = 0; ik < inner_num_iter; ++ik) {
        (void)nvcuda::wmma::load_matrix_sync(
            input_wmma_matrix_a[0],
            ((half *)buffer_input_output_shared) +
                (k_num_iter - 1) * tile_size_k + ik * 16,
            K);
        (void)nvcuda::wmma::load_matrix_sync(
            weight_wmma_matrix_b[0],
            ((half *)weight_shared_ptr[(k_num_iter - 1) & 0x1] +
             (0 * k_block_size_y * 16 + warpId * 16) * tile_size_k + ik * 16),
            tile_size_k);
        (void)nvcuda::wmma::mma_sync(
              output_wmma_accumulator[0], input_wmma_matrix_a[0],
              weight_wmma_matrix_b[0], output_wmma_accumulator[0]);
        (void)nvcuda::wmma::load_matrix_sync(
            input_wmma_matrix_a[1],
            ((half *)buffer_input_output_shared) +
                16 * K + (k_num_iter - 1) * tile_size_k + ik * 16,
            K);
        (void)nvcuda::wmma::mma_sync(
              output_wmma_accumulator[2], input_wmma_matrix_a[1],
              weight_wmma_matrix_b[0], output_wmma_accumulator[2]);
        (void)nvcuda::wmma::load_matrix_sync(
            weight_wmma_matrix_b[1],
            ((half *)weight_shared_ptr[(k_num_iter - 1) & 0x1] +
             (1 * k_block_size_y * 16 + warpId * 16) * tile_size_k + ik * 16),
            tile_size_k);
        (void)nvcuda::wmma::mma_sync(
              output_wmma_accumulator[1], input_wmma_matrix_a[0],
              weight_wmma_matrix_b[1], output_wmma_accumulator[1]);
        (void)nvcuda::wmma::mma_sync(
            output_wmma_accumulator[3], input_wmma_matrix_a[1],
            weight_wmma_matrix_b[1], output_wmma_accumulator[3]);
      }
      // Store back to output shared
      (void)nvcuda::wmma::store_matrix_sync(
          ((half *)buffer_input_output_shared + on * tile_size_n +
           0 * k_block_size_y * 16 + warpId * 16),
          output_wmma_accumulator[0], K, nvcuda::wmma::mem_row_major);
      (void)nvcuda::wmma::store_matrix_sync(
          ((half *)buffer_input_output_shared + on * tile_size_n +
           1 * k_block_size_y * 16 + warpId * 16),
          output_wmma_accumulator[1], K, nvcuda::wmma::mem_row_major);
      (void)nvcuda::wmma::store_matrix_sync(
          ((half *)buffer_input_output_shared + 16 * K + on * tile_size_n +
           0 * k_block_size_y * 16 + warpId * 16),
          output_wmma_accumulator[3], K, nvcuda::wmma::mem_row_major);
      (void)nvcuda::wmma::store_matrix_sync(
          ((half *)buffer_input_output_shared + 16 * K + on * tile_size_n +
           1 * k_block_size_y * 16 + warpId * 16),
          output_wmma_accumulator[4], K, nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
  }
  // Write output from shared memory to global memory (16*256/8/128=4)
  int store_output_num_iters =
      tile_size_m * K / (sizeof(float4) / sizeof(half)) / k_block_size;
#pragma unroll
  for (int i = 0; i < store_output_num_iters; ++i) {
    int offset = i * k_block_size + tidx;
    ((float4 *)output +
     (blockIdx.x * tile_size_m * K / (sizeof(float4) / sizeof(half)) +
      offset))[0] =
        ((float4 *)(buffer_input_output_shared) + offset)[0];
  }
}


// dim(108*4, 1, 1), dim3(32, 8, 1)
// We pad the input shared memory and try to avoid bank conflict
extern "C" __global__ void __launch_bounds__(256)
    fused_fc_fc_v5(half *__restrict__ input, half *__restrict__ weight1,
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
  const int pad_K = K + 8;
  const int pad_tile_size_k = tile_size_k + 8;
  __shared__ half buffer_input_output_shared[tile_size_m * pad_K * 2];
  __shared__ half buffer_weight_shared[tile_size_n * pad_tile_size_k * 2];
  half *weight_shared_ptr[2];
  weight_shared_ptr[0] = buffer_weight_shared;
  weight_shared_ptr[1] = buffer_weight_shared + tile_size_n * pad_tile_size_k;

  half *input_output_shared_ptr[2];
  input_output_shared_ptr[0] = buffer_input_output_shared;
  input_output_shared_ptr[1] = buffer_input_output_shared + tile_size_m * pad_K;

  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  pipe.producer_acquire();
  // First load all input to shared memory
  const int vec_size = sizeof(float4) / sizeof(half);
  const int warpId = threadIdx.y;
  const int tidx = (threadIdx.y * warpSize + threadIdx.x);
  const int load_input_2_shared_num_iter =
      tile_size_m * K / vec_size / k_block_size_x / k_block_size_y;
  const int input_block_offset = (blockIdx.x * tile_size_m * K);
  // [16, 256+16] to [16, (256+16)/8]
  // 256 threads, each load 8 elements,
  int row = tidx / (K/vec_size);
  int col = tidx % (K/vec_size);
#pragma unroll
  for (int i = 0; i < load_input_2_shared_num_iter; ++i) {
    // int offset = (i * k_block_size + threadIdx.y * warpSize + threadIdx.x) << 3;
    int s_offset = ((i * (k_block_size / (K / vec_size)) + row) * (pad_K/vec_size) + col) * vec_size;
    int g_offset = ((i * (k_block_size / (K / vec_size)) + row) * K + col) * vec_size;
    cuda::memcpy_async(input_output_shared_ptr[0] + s_offset,
                       input + input_block_offset + g_offset, shape, pipe);
  }

  half *weight_ptr;
  const int num_layers = 1;
  for (int w = 0; w < num_layers; ++w) {
    weight_ptr = weight_arr[w];
    // Now start load to shared and compute in double buffer
    const int n_num_iter = N / tile_size_n;
    const int k_num_iter = K / tile_size_k;
    static_assert(k_num_iter > 2);
    
    for (int on = 0; on < n_num_iter; ++on) {
// init accumulator
#pragma unroll
      for (int i_c_outer_init = 0; i_c_outer_init < 2; ++i_c_outer_init) {
        (void)nvcuda::wmma::fill_fragment(
            output_wmma_accumulator[i_c_outer_init], 0.000000e+00f);
      }
      // Load first weight
      const int load_weight_num_iter =
          tile_size_n * tile_size_k / vec_size / k_block_size;
      // For each thread the row and col index in the tiled shared weight
      // Load to [256, 16], now load to shared [256, 2], global [256, 64]
      const int vec_tile_size_k = tile_size_k / vec_size;
      const int pad_vec_tile_size_k = pad_tile_size_k / vec_size;
      const int vec_K = K / vec_size;
#pragma unroll
      for (int i = 0; i < load_weight_num_iter; ++i) {
        int row = i * (k_block_size / vec_tile_size_k) + (tidx >> 1);
        int col = tidx & 0x1;
        int g_row = on * tile_size_n + row;
        int g_col = 0 * vec_tile_size_k + col;
        int s_offset = (row * pad_vec_tile_size_k + col) << 3;
        int g_offset = (g_row * vec_K + g_col) << 3;
        cuda::memcpy_async(weight_shared_ptr[0] + s_offset,
                           weight_ptr + g_offset, shape, pipe);
      }
#pragma unroll
      for (int ok = 1; ok < k_num_iter; ++ok) {
        // Load second weight
        pipe.producer_commit();
        pipe.consumer_wait();
        __syncthreads();
#pragma unroll
        for (int i = 0; i < load_weight_num_iter; ++i) {
          int row = i * (k_block_size >> 1) + (tidx >> 1);
          int col = tidx & 0x1;
          int g_row = on * tile_size_n + row;
          int g_col = ok * vec_tile_size_k + col;
          int s_offset = (row * pad_vec_tile_size_k + col) << 3;
          int g_offset = (g_row * vec_K + g_col) << 3;
          cuda::memcpy_async(weight_shared_ptr[ok & 0x1] + s_offset,
                             weight_ptr + g_offset, shape, pipe);
        }
        // Do gemm
        const int inner_num_iter = tile_size_k / 16;
#pragma unroll
        for (int ik = 0; ik < inner_num_iter; ++ik) {
          (void)nvcuda::wmma::load_matrix_sync(
              input_wmma_matrix_a[0],
              ((half *)input_output_shared_ptr[w & 0x1]) + ok * tile_size_k +
                  ik * 16,
              pad_K);
          (void)nvcuda::wmma::load_matrix_sync(
              weight_wmma_matrix_b[0],
              ((half *)weight_shared_ptr[(ok - 1) & 0x1] +
               (0 * k_block_size_y * 16 + warpId * 16) * pad_vec_tile_size_k + ik * 16),
              pad_vec_tile_size_k);
          (void)nvcuda::wmma::load_matrix_sync(
              weight_wmma_matrix_b[1],
              ((half *)weight_shared_ptr[(ok - 1) & 0x1] +
               (1 * k_block_size_y * 16 + warpId * 16) * pad_vec_tile_size_k + ik * 16),
              pad_vec_tile_size_k);
          (void)nvcuda::wmma::mma_sync(
              output_wmma_accumulator[0], input_wmma_matrix_a[0],
              weight_wmma_matrix_b[0], output_wmma_accumulator[0]);
          (void)nvcuda::wmma::mma_sync(
              output_wmma_accumulator[1], input_wmma_matrix_a[0],
              weight_wmma_matrix_b[1], output_wmma_accumulator[1]);
        }
        __syncthreads();
      }
      pipe.producer_commit();
      pipe.consumer_wait();
      pipe.consumer_release();
      // Do the last gemm
      const int inner_num_iter = tile_size_k / 16;
      for (int ik = 0; ik < inner_num_iter; ++ik) {
        (void)nvcuda::wmma::load_matrix_sync(
            input_wmma_matrix_a[0],
            ((half *)input_output_shared_ptr[w & 0x1]) +
                (k_num_iter - 1) * tile_size_k + ik * 16,
            pad_K);
        (void)nvcuda::wmma::load_matrix_sync(
            weight_wmma_matrix_b[0],
            ((half *)weight_shared_ptr[(k_num_iter - 1) & 0x1] +
             (0 * k_block_size_y * 16 + warpId * 16) * pad_vec_tile_size_k + ik * 16),
            pad_vec_tile_size_k);
        (void)nvcuda::wmma::load_matrix_sync(
            weight_wmma_matrix_b[1],
            ((half *)weight_shared_ptr[(k_num_iter - 1) & 0x1] +
             (1 * k_block_size_y * 16 + warpId * 16) * pad_vec_tile_size_k + ik * 16),
            pad_vec_tile_size_k);
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
          ((half *)input_output_shared_ptr[(w + 1) & 0x1] + on * tile_size_n +
           0 * k_block_size_y * 16 + warpId * 16),
          output_wmma_accumulator[0], K, nvcuda::wmma::mem_row_major);
      (void)nvcuda::wmma::store_matrix_sync(
          ((half *)input_output_shared_ptr[(w + 1) & 0x1] + on * tile_size_n +
           1 * k_block_size_y * 16 + warpId * 16),
          output_wmma_accumulator[1], K, nvcuda::wmma::mem_row_major);
      __syncthreads();
    }
    __syncthreads();
  }
  // Write output from shared memory to global memory (16*256/8/128=4)
  int store_output_num_iters =
      tile_size_m * K / (sizeof(float4) / sizeof(half)) / k_block_size;
  row = tidx / (K / vec_size);
  col = tidx % (K / vec_size);
#pragma unroll
  for (int i = 0; i < store_output_num_iters; ++i) {
    int s_offset = row * (pad_K / vec_size) + col;
    int g_offset = i * k_block_size + tidx;
    ((float4 *)output +
     (blockIdx.x * tile_size_m * K / (sizeof(float4) / sizeof(half)) +
      g_offset))[0] =
        ((float4 *)(input_output_shared_ptr[num_layers & 0x1]) + s_offset)[0];
  }
}
