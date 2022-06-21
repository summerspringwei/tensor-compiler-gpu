#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <mma.h>

// A:[108*4*16], B: [256, 256]
//<54, 4, 1>,<32, 4, 1>
__global__ void __launch_bounds__(128)
    tvm_fc(half *__restrict__ x, half *__restrict__ placeholder,
           half *__restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half>
      T_dense_wmma_accumulator[8];
  extern half __shared__ data_buffer[];
  half *x_shared = (half *)&data_buffer[0];
  half *placeholder_shared = (half *)&data_buffer[9216];
  // x_shared: 128*64 (128*72)
  // __shared__ half x_shared[9216];
  // placeholder_shared 64*64, (64*72)
  // __shared__ half placeholder_shared[4608];
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
  // Tile at K
  for (int k_outer_outer = 0; k_outer_outer < 4; ++k_outer_outer) {
    __syncthreads();
    // A: 128*256
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0;
         ax0_ax1_fused_outer_outer_outer_outer < 8;
         ++ax0_ax1_fused_outer_outer_outer_outer) {
      ((uint4 *)(x_shared + (((((ax0_ax1_fused_outer_outer_outer_outer * 1152) +
                                (((int)threadIdx.y) * 288)) +
                               ((((int)threadIdx.x) >> 3) * 72)) +
                              ((((int)threadIdx.x) & 7) * 8)))))[0] =
          ((uint4 *)(x + (((((((((int)blockIdx.x) * 32768) +
                               (ax0_ax1_fused_outer_outer_outer_outer * 4096)) +
                              (((int)threadIdx.y) * 1024)) +
                             ((((int)threadIdx.x) >> 3) * 256)) +
                            (k_outer_outer * 64)) +
                           ((((int)threadIdx.x) & 7) * 8)))))[0];
    }
    // B_shared: 1152(16*72), 288=4*72
    // B: 16384(64*256), 4096(16*256), 1024=4*256, threadIdx.x/8 * 256,
    // (threadIdx.x % 8)*8
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
         ax0_ax1_fused_outer_outer_outer_outer1 < 4;
         ++ax0_ax1_fused_outer_outer_outer_outer1) {
      ((uint4 *)(placeholder_shared +
                 (((((ax0_ax1_fused_outer_outer_outer_outer1 * 1152) +
                     (((int)threadIdx.y) * 288)) +
                    ((((int)threadIdx.x) >> 3) * 72)) +
                   ((((int)threadIdx.x) & 7) * 8)))))[0] =
          ((uint4 *)(placeholder +
                     (((((((((int)blockIdx.y) * 16384) +
                           (ax0_ax1_fused_outer_outer_outer_outer1 * 4096)) +
                          (((int)threadIdx.y) * 1024)) +
                         ((((int)threadIdx.x) >> 3) * 256)) +
                        (k_outer_outer * 64)) +
                       ((((int)threadIdx.x) & 7) * 8)))))[0];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
      for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
        (void)nvcuda::wmma::load_matrix_sync(
            x_shared_wmma_matrix_a[ax0_outer],
            ((half *)x_shared +
             ((((((int)threadIdx.y) * 2304) + (ax0_outer * 1152)) +
               (k_outer_inner * 16)))),
            72);
      }
      for (int ax0_outer1 = 0; ax0_outer1 < 4; ++ax0_outer1) {
        (void)nvcuda::wmma::load_matrix_sync(
            placeholder_shared_wmma_matrix_b[ax0_outer1],
            ((half *)placeholder_shared +
             (((ax0_outer1 * 1152) + (k_outer_inner * 16)))),
            72);
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
  // ldg is 72, 2304=32*72, so each warp compute 32 rows;
  // 1152 = 16*72, so each axi_outer_inner compute 16 row
  // ax1_outer_inner is from 0 to 3, so compute 64 result
  for (int ax0_outer_inner = 0; ax0_outer_inner < 2; ++ax0_outer_inner) {
    for (int ax1_outer_inner = 0; ax1_outer_inner < 4; ++ax1_outer_inner) {
      (void)nvcuda::wmma::store_matrix_sync(
          ((half *)x_shared +
           ((((((int)threadIdx.y) * 2304) + (ax0_outer_inner * 1152)) +
             (ax1_outer_inner * 16)))),
          T_dense_wmma_accumulator[((ax0_outer_inner * 4) + ax1_outer_inner)],
          72, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  // 32768 = [128*256], note blockIdx.y *64, so each block computes [128*64]
  // elements
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0;
       i_inner_j_inner_fused_outer_outer_outer_outer < 8;
       ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    ((uint4 *)(T_dense +
               (((((((((int)blockIdx.x) * 32768) +
                     (i_inner_j_inner_fused_outer_outer_outer_outer * 4096)) +
                    (((int)threadIdx.y) * 1024)) +
                   ((((int)threadIdx.x) >> 3) * 256)) +
                  (((int)blockIdx.y) * 64)) +
                 ((((int)threadIdx.x) & 7) * 8)))))[0] =
        ((uint4 *)(x_shared +
                   (((((i_inner_j_inner_fused_outer_outer_outer_outer * 1152) +
                       (((int)threadIdx.y) * 288)) +
                      ((((int)threadIdx.x) >> 3) * 72)) +
                     ((((int)threadIdx.x) & 7) * 8)))))[0];
  }
}

// A:[108*4*16], B: [256, 256]
//<54, 4, 1>,<32, 4, 1>
__global__ void __launch_bounds__(128)
    tvm_fused_fc(half *__restrict__ x, half *__restrict__ placeholder,
                 half *__restrict__ placeholder1, half *__restrict__ T_dense) {
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half>
      T_dense_wmma_accumulator[8];
  extern half __shared__ data_buffer[];
  half *x_shared = (half *)&data_buffer[0];
  half *placeholder_shared = (half *)&data_buffer[9216];
  // __shared__ half x_shared[9216];
  // __shared__ half placeholder_shared[4608];
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
  for (int k_outer_outer = 0; k_outer_outer < 4; ++k_outer_outer) {
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0;
         ax0_ax1_fused_outer_outer_outer_outer < 8;
         ++ax0_ax1_fused_outer_outer_outer_outer) {
      ((uint4 *)(x_shared + (((((ax0_ax1_fused_outer_outer_outer_outer * 1152) +
                                (((int)threadIdx.y) * 288)) +
                               ((((int)threadIdx.x) >> 3) * 72)) +
                              ((((int)threadIdx.x) & 7) * 8)))))[0] =
          ((uint4 *)(x + (((((((((int)blockIdx.x) * 32768) +
                               (ax0_ax1_fused_outer_outer_outer_outer * 4096)) +
                              (((int)threadIdx.y) * 1024)) +
                             ((((int)threadIdx.x) >> 3) * 256)) +
                            (k_outer_outer * 64)) +
                           ((((int)threadIdx.x) & 7) * 8)))))[0];
    }
    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
         ax0_ax1_fused_outer_outer_outer_outer1 < 4;
         ++ax0_ax1_fused_outer_outer_outer_outer1) {
      ((uint4 *)(placeholder_shared +
                 (((((ax0_ax1_fused_outer_outer_outer_outer1 * 1152) +
                     (((int)threadIdx.y) * 288)) +
                    ((((int)threadIdx.x) >> 3) * 72)) +
                   ((((int)threadIdx.x) & 7) * 8)))))[0] =
          ((uint4 *)(placeholder +
                     (((((((((int)blockIdx.y) * 16384) +
                           (ax0_ax1_fused_outer_outer_outer_outer1 * 4096)) +
                          (((int)threadIdx.y) * 1024)) +
                         ((((int)threadIdx.x) >> 3) * 256)) +
                        (k_outer_outer * 64)) +
                       ((((int)threadIdx.x) & 7) * 8)))))[0];
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
      for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
        (void)nvcuda::wmma::load_matrix_sync(
            x_shared_wmma_matrix_a[ax0_outer],
            ((half *)x_shared +
             ((((((int)threadIdx.y) * 2304) + (ax0_outer * 1152)) +
               (k_outer_inner * 16)))),
            72);
      }
      for (int ax0_outer1 = 0; ax0_outer1 < 4; ++ax0_outer1) {
        (void)nvcuda::wmma::load_matrix_sync(
            placeholder_shared_wmma_matrix_b[ax0_outer1],
            ((half *)placeholder_shared +
             (((ax0_outer1 * 1152) + (k_outer_inner * 16)))),
            72);
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
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  pipe.producer_acquire();
  // [For pipe 1] Load weight for next fc, to reuse the output in shared memory,
  // We reuse the coresponding weight
  const int vec_size = sizeof(float4) / sizeof(half);
  for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
       ax0_ax1_fused_outer_outer_outer_outer1 < 4;
       ++ax0_ax1_fused_outer_outer_outer_outer1) {
    cuda::memcpy_async(
        (placeholder_shared +
         (((((ax0_ax1_fused_outer_outer_outer_outer1 * 1152) +
             (((int)threadIdx.y) * 288)) +
            ((((int)threadIdx.x) >> 3) * 72)) +
           ((((int)threadIdx.x) & 7) * 8)))),
        (placeholder1 + (((((((((int)blockIdx.y) * 16384) +
                              (ax0_ax1_fused_outer_outer_outer_outer1 * 4096)) +
                             (((int)threadIdx.y) * 1024)) +
                            ((((int)threadIdx.x) >> 3) * 256)) +
                           (/*k_outer_outer*/ blockIdx.y * 64)) +
                          ((((int)threadIdx.x) & 7) * 8)))),
        shape, pipe);
  }
  __syncthreads();
  // [For pipe0]
  // ldg is 72, 2304=32*72, so each warp compute 32 rows;
  // 1152 = 16*72, so each axi_outer_inner compute 16 row
  // ax1_outer_inner is from 0 to 3, so compute 64 result
  for (int ax0_outer_inner = 0; ax0_outer_inner < 2; ++ax0_outer_inner) {
    for (int ax1_outer_inner = 0; ax1_outer_inner < 4; ++ax1_outer_inner) {
      (void)nvcuda::wmma::store_matrix_sync(
          ((half *)x_shared +
           ((((((int)threadIdx.y) * 2304) + (ax0_outer_inner * 1152)) +
             (ax1_outer_inner * 16)))),
          T_dense_wmma_accumulator[((ax0_outer_inner * 4) + ax1_outer_inner)],
          72, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  // 32768 = [128*256], note blockIdx.y *64, so each block computes [128*64]
  // elements [For pipe 0]
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0;
       i_inner_j_inner_fused_outer_outer_outer_outer < 8;
       ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    float4 tmp =
        ((float4 *)(x_shared +
                    (((((i_inner_j_inner_fused_outer_outer_outer_outer * 1152) +
                        (((int)threadIdx.y) * 288)) +
                       ((((int)threadIdx.x) >> 3) * 72)) +
                      ((((int)threadIdx.x) & 7) * 8)))))[0];
    // half2 x = ((half2*)(&(tmp.x)))[0];
    // if(__half2float(x.x)-2.560547 > 0.1){
    //   printf("282: <%d, %d, %d>, <%d, %d, %d> %f\n", blockIdx.x, blockIdx.y,
    //   blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, __half2float(x.x));
    // }
    ((float4 *)(T_dense +
                (((((((((int)blockIdx.x) * 32768) +
                      (i_inner_j_inner_fused_outer_outer_outer_outer * 4096)) +
                     (((int)threadIdx.y) * 1024)) +
                    ((((int)threadIdx.x) >> 3) * 256)) +
                   (((int)blockIdx.y) * 64)) +
                  ((((int)threadIdx.x) & 7) * 8)))))[0] = tmp;
  }
  //[For Pipe 1]
  // same tiling and block/thread mapping with original implementation
  // Do gemm
  for (int i_c_outer_init = 0; i_c_outer_init < 2; ++i_c_outer_init) {
    for (int j_c_outer_init = 0; j_c_outer_init < 4; ++j_c_outer_init) {
      (void)nvcuda::wmma::fill_fragment(
          T_dense_wmma_accumulator[((i_c_outer_init * 4) + j_c_outer_init)],
          0.000000e+00f);
    }
  }
  __syncthreads();
  // For pipe 1
  pipe.producer_commit();
  pipe.consumer_wait();
  for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
    for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
      (void)nvcuda::wmma::load_matrix_sync(
          x_shared_wmma_matrix_a[ax0_outer],
          ((half *)x_shared +
           ((((((int)threadIdx.y) * 2304) + (ax0_outer * 1152)) +
             (k_outer_inner * 16)))),
          72);
    }
    for (int ax0_outer1 = 0; ax0_outer1 < 4; ++ax0_outer1) {
      (void)nvcuda::wmma::load_matrix_sync(
          placeholder_shared_wmma_matrix_b[ax0_outer1],
          ((half *)placeholder_shared +
           (((ax0_outer1 * 1152) + (k_outer_inner * 16)))),
          72);
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
  // Now let pipe0's output store to global memory
  __syncthreads();
  __threadfence();
  grid.sync();

  // Load other tiled weights
  for (int k_outer_outer = 0; k_outer_outer < 4; ++k_outer_outer) {
    // Skip as the output cached in shared memory has done the gemm
    if (k_outer_outer == blockIdx.y) {
      continue;
    }
    __syncthreads();
    for (int ax0_ax1_fused_outer_outer_outer_outer = 0;
         ax0_ax1_fused_outer_outer_outer_outer < 8;
         ++ax0_ax1_fused_outer_outer_outer_outer) {
      cuda::memcpy_async(
          (x_shared + (((((ax0_ax1_fused_outer_outer_outer_outer * 1152) +
                          (((int)threadIdx.y) * 288)) +
                         ((((int)threadIdx.x) >> 3) * 72)) +
                        ((((int)threadIdx.x) & 7) * 8)))),
          (T_dense + (((((((((int)blockIdx.x) * 32768) +
                           (ax0_ax1_fused_outer_outer_outer_outer * 4096)) +
                          (((int)threadIdx.y) * 1024)) +
                         ((((int)threadIdx.x) >> 3) * 256)) +
                        (k_outer_outer * 64)) +
                       ((((int)threadIdx.x) & 7) * 8)))),
          shape, pipe);
    }

    for (int ax0_ax1_fused_outer_outer_outer_outer1 = 0;
         ax0_ax1_fused_outer_outer_outer_outer1 < 4;
         ++ax0_ax1_fused_outer_outer_outer_outer1) {
      cuda::memcpy_async(
          (placeholder_shared +
           (((((ax0_ax1_fused_outer_outer_outer_outer1 * 1152) +
               (((int)threadIdx.y) * 288)) +
              ((((int)threadIdx.x) >> 3) * 72)) +
             ((((int)threadIdx.x) & 7) * 8)))),
          (placeholder1 +
           (((((((((int)blockIdx.y) * 16384) +
                 (ax0_ax1_fused_outer_outer_outer_outer1 * 4096)) +
                (((int)threadIdx.y) * 1024)) +
               ((((int)threadIdx.x) >> 3) * 256)) +
              (k_outer_outer * 64)) +
             ((((int)threadIdx.x) & 7) * 8)))),
          shape, pipe);
    }
    __syncthreads();
    pipe.producer_commit();
    pipe.consumer_wait();
    for (int k_outer_inner = 0; k_outer_inner < 4; ++k_outer_inner) {
      for (int ax0_outer = 0; ax0_outer < 2; ++ax0_outer) {
        (void)nvcuda::wmma::load_matrix_sync(
            x_shared_wmma_matrix_a[ax0_outer],
            ((half *)x_shared +
             ((((((int)threadIdx.y) * 2304) + (ax0_outer * 1152)) +
               (k_outer_inner * 16)))),
            72);
      }
      for (int ax0_outer1 = 0; ax0_outer1 < 4; ++ax0_outer1) {
        (void)nvcuda::wmma::load_matrix_sync(
            placeholder_shared_wmma_matrix_b[ax0_outer1],
            ((half *)placeholder_shared +
             (((ax0_outer1 * 1152) + (k_outer_inner * 16)))),
            72);
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
  } // End of outer K
  __syncthreads();
  for (int ax0_outer_inner = 0; ax0_outer_inner < 2; ++ax0_outer_inner) {
    for (int ax1_outer_inner = 0; ax1_outer_inner < 4; ++ax1_outer_inner) {
      (void)nvcuda::wmma::store_matrix_sync(
          ((half *)x_shared +
           ((((((int)threadIdx.y) * 2304) + (ax0_outer_inner * 1152)) +
             (ax1_outer_inner * 16)))),
          T_dense_wmma_accumulator[((ax0_outer_inner * 4) + ax1_outer_inner)],
          72, nvcuda::wmma::mem_row_major);
    }
  }
  __syncthreads();
  // <6, 2, 0>, <6, 1, 0> 2.560547
  for (int i_inner_j_inner_fused_outer_outer_outer_outer = 0;
       i_inner_j_inner_fused_outer_outer_outer_outer < 8;
       ++i_inner_j_inner_fused_outer_outer_outer_outer) {
    float4 tmp =
        ((float4 *)(x_shared +
                    (((((i_inner_j_inner_fused_outer_outer_outer_outer * 1152) +
                        (((int)threadIdx.y) * 288)) +
                       ((((int)threadIdx.x) >> 3) * 72)) +
                      ((((int)threadIdx.x) & 7) * 8)))))[0];
    ((float4 *)(T_dense +
                (((((((((int)blockIdx.x) * 32768) +
                      (i_inner_j_inner_fused_outer_outer_outer_outer * 4096)) +
                     (((int)threadIdx.y) * 1024)) +
                    ((((int)threadIdx.x) >> 3) * 256)) +
                   (((int)blockIdx.y) * 64)) +
                  ((((int)threadIdx.x) & 7) * 8)))))[0] = tmp;
  }
}
