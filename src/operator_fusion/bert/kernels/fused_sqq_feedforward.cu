
#include "bert.h"
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <mma.h>

using namespace fuselage::experiments::networks::bert;

__global__ void fused_sqq_feedforward(half* __restrict__ attn_fc_output,
                                half* __restrict__ feed_forward_fc1_weight,
                                half* __restrict__ feed_forward_fc1_output,
                                half* __restrict__ feed_forward_fc2_weight,
                                half* __restrict__ feed_forward_fc2_output,
                                half eps, half h_gama, half h_beta,
                                float * query_key_softmax_sum,
                                float* layer_norm_variance,
                                int64_t* profile_grid_clock){
  using namespace nvcuda;
  extern __shared__ half all_shared_mem[];
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  int clock_idx = 0;
  unsigned int c = 0;
  const int warpIdx = threadIdx.x >> 5;
  
  profile_grid_clock[clock_idx * 108 * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;

  if(blockIdx.x < 96){  
    const int kWarpRowTiles=kGemmK5WarpRowTiles;
    const int kWarpColTiles=kGemmK5WarpColTiles;
    const int M=kHiddenSize * kHiddenDim;
    const int N=kSeqLength;
    const int K=kHiddenDim;
    const int B=1;

    enum {
        kBlockRowTiles = kBlockRowWarps * kWarpRowTiles,
        kBlockColTiles = kBlockColWarps * kWarpColTiles,
    };

    half *matrix_a_shared[kStage], *matrix_b_shared[kStage];
    half *acc_shared;

    matrix_a_shared[0] = all_shared_mem;
    matrix_a_shared[1] =
        all_shared_mem +
        kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[2] =
        all_shared_mem +
        2 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);

    matrix_b_shared[0] =
        all_shared_mem +
        3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_b_shared[1] =
        all_shared_mem +
        3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
        kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);
    matrix_b_shared[2] =
        all_shared_mem +
        3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
        2 * kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);

    acc_shared = all_shared_mem;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::col_major>
        wmma_matrix_a[kWarpRowTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::col_major>
        wmma_matrix_b[kWarpColTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                           half>
        wmma_accumulator[kWarpColTiles * kWarpRowTiles];

    const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
    const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;
    const int batch_stride =
        (N / kBlockColTiles / kWmmaN) * (M / kBlockRowTiles / kWmmaM);
    const int batched_id = blockIdx.x / batch_stride;
    const int row_block_id =
        blockIdx.x % batch_stride % (M / kBlockRowTiles / kWmmaM);
    const int col_block_id =
        blockIdx.x % batch_stride / (M / kBlockRowTiles / kWmmaM);

#pragma unroll
    for (int col = 0; col < kWarpColTiles; ++col) {
#pragma unroll
        for (int row = 0; row < kWarpRowTiles; ++row) {
            nvcuda::wmma::fill_fragment(
                wmma_accumulator[col * kWarpRowTiles + row], 0.0f);
        }
    }

    enum {
        kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kLoadALanesPerRow =
            kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
        kLoadAColsPerIter = kThreads / kLoadALanesPerRow,

        kLoadBLanesPerRow = kWmmaK * kChunkK / (sizeof(float4) / sizeof(half)),
        kLoadBColsPerIter = kThreads / kLoadBLanesPerRow,

        kStoreCLanesPerRow = kLoadALanesPerRow,
        kStoreCColsPerIter = kLoadAColsPerIter,
    };

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
    int stage = 0;
    int k_loop = 0;

    const int a_dst_stride =
        kLoadAColsPerIter * (kWmmaM * kBlockRowTiles + kInputSkew);
    const int a_src_stride = kLoadAColsPerIter * M;

    const int b_dst_stride =
        kLoadBColsPerIter * (kWmmaK * kChunkK + kInputSkew);
    const int b_src_stride = kLoadBColsPerIter * K;

    // Prologue
#pragma unroll
    for (int s = 0; s < kStage - 1; ++s) {
        pipe.producer_acquire();
        half *a_dst_base = matrix_a_shared[(stage + s) % kStage] +
                           threadIdx.x / kLoadALanesPerRow *
                               (kWmmaM * kBlockRowTiles + kInputSkew) +
                           (threadIdx.x & (kLoadALanesPerRow - 1)) *
                               sizeof(float4) / sizeof(half);

        const half *a_src_base = feed_forward_fc1_weight + batched_id * K * M +
                                 row_block_id * kBlockRowTiles * kWmmaM +
                                 ((k_loop + s) * kChunkK * kWmmaK +
                                  threadIdx.x / kLoadALanesPerRow) *
                                     M +
                                 (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                     (sizeof(float4) / sizeof(half));

        half *b_dst_base =
            matrix_b_shared[(stage + s) % kStage] +
            threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
            (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) /
                sizeof(half);

        const half *b_src_base = attn_fc_output + batched_id * N * K +
                                 (k_loop + s) * kChunkK * kWmmaK +
                                 (col_block_id * kBlockColTiles * kWmmaN +
                                  threadIdx.x / kLoadBLanesPerRow) *
                                     K +
                                 (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                                     (sizeof(float4) / sizeof(half));

#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
            cuda::memcpy_async(a_dst_base + i * a_dst_stride,
                               a_src_base + i * a_src_stride, shape, pipe);
        }

#pragma unroll
        for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
            cuda::memcpy_async(b_dst_base + i * b_dst_stride,
                               b_src_base + i * b_src_stride, shape, pipe);
        }
        pipe.producer_commit();
    }

    // Soft pipeline
#pragma unroll
    for (; k_loop < (K / kChunkK / kWmmaK) - (kStage - 1); ++k_loop) {
        pipe.producer_acquire();

        half *a_dst_base = matrix_a_shared[(stage + kStage - 1) % kStage] +
                           threadIdx.x / kLoadALanesPerRow *
                               (kWmmaM * kBlockRowTiles + kInputSkew) +
                           (threadIdx.x & (kLoadALanesPerRow - 1)) *
                               sizeof(float4) / sizeof(half);

        const half *a_src_base = feed_forward_fc1_weight + batched_id * K * M +
                                 row_block_id * kBlockRowTiles * kWmmaM +
                                 ((k_loop + kStage - 1) * kChunkK * kWmmaK +
                                  threadIdx.x / kLoadALanesPerRow) *
                                     M +
                                 (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                     (sizeof(float4) / sizeof(half));

        half *b_dst_base =
            matrix_b_shared[(stage + kStage - 1) % kStage] +
            threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
            (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) /
                sizeof(half);

        const half *b_src_base = attn_fc_output + batched_id * N * K +
                                 (k_loop + kStage - 1) * kChunkK * kWmmaK +
                                 (col_block_id * kBlockColTiles * kWmmaN +
                                  threadIdx.x / kLoadBLanesPerRow) *
                                     K +
                                 (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                                     (sizeof(float4) / sizeof(half));

#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
            cuda::memcpy_async(a_dst_base + i * a_dst_stride,
                               a_src_base + i * a_src_stride, shape, pipe);
        }

#pragma unroll
        for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
            cuda::memcpy_async(b_dst_base + i * b_dst_stride,
                               b_src_base + i * b_src_stride, shape, pipe);
        }
        pipe.producer_commit();

        pipe.consumer_wait();
        __syncthreads();
        pipe.consumer_release();

#pragma unroll
        for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
            for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[tile_m],
                    (matrix_a_shared[stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
            }
#pragma unroll
            for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_b[tile_n],
                    (matrix_b_shared[stage] +
                     (col_warp_id * kWarpColTiles + tile_n) * kWmmaN *
                         (kChunkK * kWmmaK + kInputSkew) +
                     tile_k * kWmmaK),
                    kChunkK * kWmmaK + kInputSkew);
            }
#pragma unroll
            for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
#pragma unroll
                for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
                    nvcuda::wmma::mma_sync(
                        wmma_accumulator[tile_m + tile_n * kWarpRowTiles],
                        wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
                        wmma_accumulator[tile_m + tile_n * kWarpRowTiles]);
                }
            }
        }
        stage = (stage + 1) % kStage;
    }

    // Epilogue
#pragma unroll
    for (int s = kStage - 1; s >= 1; --s) {
        k_loop = (K / kChunkK / kWmmaK) - s;
        pipe.consumer_wait();
        __syncthreads();
        pipe.consumer_release();

#pragma unroll
        for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
            for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[tile_m],
                    (matrix_a_shared[stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
            }
#pragma unroll
            for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_b[tile_n],
                    (matrix_b_shared[stage] +
                     (col_warp_id * kWarpColTiles + tile_n) * kWmmaN *
                         (kChunkK * kWmmaK + kInputSkew) +
                     tile_k * kWmmaK),
                    kChunkK * kWmmaK + kInputSkew);
            }
#pragma unroll
            for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
#pragma unroll
                for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
                    nvcuda::wmma::mma_sync(
                        wmma_accumulator[tile_m + tile_n * kWarpRowTiles],
                        wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
                        wmma_accumulator[tile_m + tile_n * kWarpRowTiles]);
                }
            }
        }
        stage = (stage + 1) % kStage;
    }

#pragma unroll
    for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
#pragma unroll
        for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
            nvcuda::wmma::store_matrix_sync(
                acc_shared +
                    (col_warp_id * kWarpColTiles + tile_n) * kWmmaK *
                        (kBlockRowTiles * kWmmaM + kAccSkew) +
                    (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM,
                wmma_accumulator[tile_n * kWarpRowTiles + tile_m],
                (kBlockRowTiles * kWmmaM + kAccSkew),
                nvcuda::wmma::mem_col_major);
        }
    }

    __syncthreads();

    const int c_dst_stride = kStoreCColsPerIter * M;
    const int c_src_stride =
        kStoreCColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);
    // Shared sizeL (6*16, 8*16)
    // Do activation (Relu)
    const int kComputeRowsPerIter = 128 * (sizeof(half2) / sizeof(half)) / (kBlockRowTiles * kWmmaM);
    for(int i=0; i<kBlockColTiles*kWmmaN / kComputeRowsPerIter; ++i){
        int row = i * kComputeRowsPerIter + (threadIdx.x >> 6);
        int col = (threadIdx.x & 63) * 2;
        int idx = row * (kBlockRowTiles * kWmmaM + kAccSkew) + col;
        half2 value = ((half2*)(acc_shared + idx))[0];
        if(value.x<half(0)){
            value.x = half(0);
        }if(value.y<half(0)){
            value.y = half(0);
        }
        ((half2*)(acc_shared + idx))[0] = value;
    }
    __syncthreads();
    half *c_dst_base = feed_forward_fc1_output + batched_id * N * M +
                       row_block_id * kBlockRowTiles * kWmmaM +
                       (col_block_id * kBlockColTiles * kWmmaN +
                        threadIdx.x / kStoreCLanesPerRow) *
                           M +
                       (threadIdx.x & (kStoreCLanesPerRow - 1)) *
                           sizeof(float4) / sizeof(half);
    
    half *c_src_base = acc_shared +
                       threadIdx.x / kStoreCLanesPerRow *
                           (kBlockRowTiles * kWmmaM + kAccSkew) +
                       (threadIdx.x & (kStoreCLanesPerRow - 1)) *
                           sizeof(float4) / sizeof(half);

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
        *reinterpret_cast<float4 *>(c_dst_base + i * c_dst_stride) =
            *reinterpret_cast<float4 *>(c_src_base + i * c_src_stride);
    }
    } // End of feed_forward_fc1 + relu
    grid.sync();
    
  profile_grid_clock[clock_idx * 108 * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
    // Begin of feed_forward_fc2 + shor_cuda  Add
    if(blockIdx.x < 72){

    half *matrix_a_shared[kStage], *matrix_b_shared[kStage];
    half *acc_shared;

    matrix_a_shared[0] = all_shared_mem;
    matrix_a_shared[1] =
        all_shared_mem + kGemmK6BlockSliceKTiles * kWmmaK *
                             (kGemmK6BlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[2] =
        all_shared_mem + 2 * kGemmK6BlockSliceKTiles * kWmmaK *
                             (kGemmK6BlockRowTiles * kWmmaM + kInputSkew);

    matrix_b_shared[0] =
        all_shared_mem + 3 * kGemmK6BlockSliceKTiles * kWmmaK *
                             (kGemmK6BlockRowTiles * kWmmaM + kInputSkew);
    matrix_b_shared[1] = all_shared_mem +
                         3 * kGemmK6BlockSliceKTiles * kWmmaK *
                             (kGemmK6BlockRowTiles * kWmmaM + kInputSkew) +
                         kGemmK6BlockColTiles * kWmmaN *
                             (kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew);
    matrix_b_shared[2] = all_shared_mem +
                         3 * kGemmK6BlockSliceKTiles * kWmmaK *
                             (kGemmK6BlockRowTiles * kWmmaM + kInputSkew) +
                         2 * kGemmK6BlockColTiles * kWmmaN *
                             (kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew);

    acc_shared = all_shared_mem;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::col_major>
        wmma_matrix_a[kGemmK6BlockRowTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::col_major>
        wmma_matrix_b[kGemmK6BlockColTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                           half>
        wmma_accumulator[kGemmK6BlockRowTiles * kGemmK6BlockColTiles];

    const int slicek_warp_id = threadIdx.x / kWarpSize;
    const int row_block_id =
        blockIdx.x % (kHiddenDim / kGemmK6BlockRowTiles / kWmmaM);
    const int col_block_id =
        blockIdx.x / (kHiddenDim / kGemmK6BlockRowTiles / kWmmaM);

#pragma unroll
    for (int col = 0; col < kGemmK6BlockColTiles; ++col) {
#pragma unroll
        for (int row = 0; row < kGemmK6BlockRowTiles; ++row) {
            nvcuda::wmma::fill_fragment(
                wmma_accumulator[col * kGemmK6BlockRowTiles + row], 0.0f);
        }
    }

    enum {
        kThreads = kGemmK6BlockSliceKTiles * kWarpSize,
        kLoadALanesPerRow =
            kWmmaM * kGemmK6BlockRowTiles / (sizeof(float4) / sizeof(half)),
        kLoadAColsPerIter = kThreads / kLoadALanesPerRow,

        kLoadBLanesPerRow =
            kWmmaK * kGemmK6BlockSliceKTiles / (sizeof(float4) / sizeof(half)),
        kLoadBColsPerIter = kThreads / kLoadBLanesPerRow,

        kReduceCLanesPerRow =
            kWmmaM * kGemmK6BlockRowTiles / (sizeof(half2) / sizeof(half)),
        kReduceCColsPerIter = kThreads / kReduceCLanesPerRow,

        kStoreCLanesPerRow = kLoadALanesPerRow,
        kStoreCColsPerIter = kLoadAColsPerIter,
    };

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
    int stage = 0;
    int k_loop = 0;

    const int a_dst_stride =
        kLoadAColsPerIter * (kWmmaM * kGemmK6BlockRowTiles + kInputSkew);
    const int a_src_stride = kLoadAColsPerIter * kHiddenDim;

    const int b_dst_stride =
        kLoadBColsPerIter * (kWmmaK * kGemmK6BlockSliceKTiles + kInputSkew);
    const int b_src_stride = kLoadBColsPerIter * kHiddenDim * kHiddenSize;

    // Prologue
#pragma unroll
    for (int s = 0; s < kStage - 1; ++s) {
        pipe.producer_acquire();
        half *a_dst_base = matrix_a_shared[(stage + s) % kStage] +
                           threadIdx.x / kLoadALanesPerRow *
                               (kWmmaM * kGemmK6BlockRowTiles + kInputSkew) +
                           (threadIdx.x & (kLoadALanesPerRow - 1)) *
                               sizeof(float4) / sizeof(half);

        const half *a_src_base =
            feed_forward_fc2_weight + row_block_id * kGemmK6BlockRowTiles * kWmmaM +
            ((k_loop + s) * kGemmK6BlockSliceKTiles * kWmmaK +
             threadIdx.x / kLoadALanesPerRow) *
                kHiddenDim +
            (threadIdx.x & (kLoadALanesPerRow - 1)) *
                (sizeof(float4) / sizeof(half));

        half *b_dst_base = matrix_b_shared[(stage + s) % kStage] +
                           threadIdx.x / kLoadBLanesPerRow *
                               (kWmmaK * kGemmK6BlockSliceKTiles + kInputSkew) +
                           (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                               sizeof(float4) / sizeof(half);

        const half *b_src_base =
            feed_forward_fc1_output + (k_loop + s) * kGemmK6BlockSliceKTiles * kWmmaK +
            (col_block_id * kGemmK6BlockColTiles * kWmmaN +
             threadIdx.x / kLoadBLanesPerRow) *
                kHiddenDim * kHiddenSize +
            (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                (sizeof(float4) / sizeof(half));

#pragma unroll
        for (int i = 0;
             i < kGemmK6BlockSliceKTiles * kWmmaK / kLoadAColsPerIter; ++i) {
            cuda::memcpy_async(a_dst_base + i * a_dst_stride,
                               a_src_base + i * a_src_stride, shape, pipe);
        }

#pragma unroll
        for (int i = 0; i < kGemmK6BlockColTiles * kWmmaN / kLoadBColsPerIter;
             ++i) {
            cuda::memcpy_async(b_dst_base + i * b_dst_stride,
                               b_src_base + i * b_src_stride, shape, pipe);
        }
        pipe.producer_commit();
    }

    // Soft pipeline
#pragma unroll
    for (; k_loop <
           (kHiddenDim * kHiddenSize / kGemmK6BlockSliceKTiles / kWmmaK) -
               (kStage - 1);
         ++k_loop) {
        pipe.producer_acquire();

        half *a_dst_base = matrix_a_shared[(stage + kStage - 1) % kStage] +
                           threadIdx.x / kLoadALanesPerRow *
                               (kWmmaM * kGemmK6BlockRowTiles + kInputSkew) +
                           (threadIdx.x & (kLoadALanesPerRow - 1)) *
                               sizeof(float4) / sizeof(half);

        const half *a_src_base =
            feed_forward_fc2_weight + row_block_id * kGemmK6BlockRowTiles * kWmmaM +
            ((k_loop + kStage - 1) * kGemmK6BlockSliceKTiles * kWmmaK +
             threadIdx.x / kLoadALanesPerRow) *
                kHiddenDim +
            (threadIdx.x & (kLoadALanesPerRow - 1)) *
                (sizeof(float4) / sizeof(half));

        half *b_dst_base = matrix_b_shared[(stage + kStage - 1) % kStage] +
                           threadIdx.x / kLoadBLanesPerRow *
                               (kWmmaK * kGemmK6BlockSliceKTiles + kInputSkew) +
                           (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                               sizeof(float4) / sizeof(half);

        const half *b_src_base =
            feed_forward_fc1_output +
            (k_loop + kStage - 1) * kGemmK6BlockSliceKTiles * kWmmaK +
            (col_block_id * kGemmK6BlockColTiles * kWmmaN +
             threadIdx.x / kLoadBLanesPerRow) *
                kHiddenDim * kHiddenSize +
            (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                (sizeof(float4) / sizeof(half));

#pragma unroll
        for (int i = 0;
             i < kGemmK6BlockSliceKTiles * kWmmaK / kLoadAColsPerIter; ++i) {
            cuda::memcpy_async(a_dst_base + i * a_dst_stride,
                               a_src_base + i * a_src_stride, shape, pipe);
        }

#pragma unroll
        for (int i = 0; i < kGemmK6BlockColTiles * kWmmaN / kLoadBColsPerIter;
             ++i) {
            cuda::memcpy_async(b_dst_base + i * b_dst_stride,
                               b_src_base + i * b_src_stride, shape, pipe);
        }
        pipe.producer_commit();

        pipe.consumer_wait();
        __syncthreads();
        pipe.consumer_release();

#pragma unroll
        for (int tile_m = 0; tile_m < kGemmK6BlockRowTiles; ++tile_m) {
            nvcuda::wmma::load_matrix_sync(
                wmma_matrix_a[tile_m],
                (matrix_a_shared[stage] +
                 slicek_warp_id * kWmmaK *
                     (kGemmK6BlockRowTiles * kWmmaM + kInputSkew) +
                 tile_m * kWmmaM),
                kGemmK6BlockRowTiles * kWmmaM + kInputSkew);
        }
#pragma unroll
        for (int tile_n = 0; tile_n < kGemmK6BlockColTiles; ++tile_n) {
            nvcuda::wmma::load_matrix_sync(
                wmma_matrix_b[tile_n],
                (matrix_b_shared[stage] +
                 tile_n * kWmmaN *
                     (kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew) +
                 slicek_warp_id * kWmmaK),
                kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew);
        }
#pragma unroll
        for (int tile_m = 0; tile_m < kGemmK6BlockRowTiles; ++tile_m) {
#pragma unroll
            for (int tile_n = 0; tile_n < kGemmK6BlockColTiles; ++tile_n) {
                nvcuda::wmma::mma_sync(
                    wmma_accumulator[tile_m + tile_n * kGemmK6BlockRowTiles],
                    wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
                    wmma_accumulator[tile_m + tile_n * kGemmK6BlockRowTiles]);
            }
        }
        stage = (stage + 1) % kStage;
    }

#pragma unroll
    for (int s = kStage - 1; s >= 1; --s) {
        k_loop =
            (kHiddenDim * kHiddenSize / kGemmK6BlockSliceKTiles / kWmmaK) - s;
        pipe.consumer_wait();
        __syncthreads();
        pipe.consumer_release();

#pragma unroll
        for (int tile_m = 0; tile_m < kGemmK6BlockRowTiles; ++tile_m) {
            nvcuda::wmma::load_matrix_sync(
                wmma_matrix_a[tile_m],
                (matrix_a_shared[stage] +
                 slicek_warp_id * kWmmaK *
                     (kGemmK6BlockRowTiles * kWmmaM + kInputSkew) +
                 tile_m * kWmmaM),
                kGemmK6BlockRowTiles * kWmmaM + kInputSkew);
        }
#pragma unroll
        for (int tile_n = 0; tile_n < kGemmK6BlockColTiles; ++tile_n) {
            nvcuda::wmma::load_matrix_sync(
                wmma_matrix_b[tile_n],
                (matrix_b_shared[stage] +
                 tile_n * kWmmaN *
                     (kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew) +
                 slicek_warp_id * kWmmaK),
                kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew);
        }
#pragma unroll
        for (int tile_m = 0; tile_m < kGemmK6BlockRowTiles; ++tile_m) {
#pragma unroll
            for (int tile_n = 0; tile_n < kGemmK6BlockColTiles; ++tile_n) {
                nvcuda::wmma::mma_sync(
                    wmma_accumulator[tile_m + tile_n * kGemmK6BlockRowTiles],
                    wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
                    wmma_accumulator[tile_m + tile_n * kGemmK6BlockRowTiles]);
            }
        }
        stage = (stage + 1) % kStage;
    }

#pragma unroll
    for (int tile_m = 0; tile_m < kGemmK6BlockRowTiles; ++tile_m) {
#pragma unroll
        for (int tile_n = 0; tile_n < kGemmK6BlockColTiles; ++tile_n) {
            nvcuda::wmma::store_matrix_sync(
                acc_shared +
                    (slicek_warp_id * kGemmK6BlockColTiles + tile_n) * kWmmaN *
                        (kGemmK6BlockRowTiles * kWmmaM + kAccSkew) +
                    tile_m * kWmmaM,
                wmma_accumulator[tile_n * kGemmK6BlockRowTiles + tile_m],
                (kGemmK6BlockRowTiles * kWmmaM + kAccSkew),
                nvcuda::wmma::mem_col_major);
        }
    }

    __syncthreads();

    const int c_reduce_stride =
        kReduceCColsPerIter * (kGemmK6BlockRowTiles * kWmmaM + kAccSkew);
    const int c_reduce_k_stride = kGemmK6BlockColTiles * kWmmaN *
                                  (kGemmK6BlockRowTiles * kWmmaM + kAccSkew) *
                                  sizeof(half) / sizeof(half2);
    half *c_reduce_base = acc_shared +
                          threadIdx.x / kReduceCLanesPerRow *
                              (kGemmK6BlockRowTiles * kWmmaM + kAccSkew) +
                          (threadIdx.x & (kReduceCLanesPerRow - 1)) *
                              sizeof(half2) / sizeof(half);
#pragma unroll
    for (int i = 0; i < kGemmK6BlockColTiles * kWmmaN / kReduceCColsPerIter;
         ++i) {
        half2 *c_reduce_src =
            reinterpret_cast<half2 *>(c_reduce_base + i * c_reduce_stride);
#pragma unroll
        for (int k = 1; k < kGemmK6BlockSliceKTiles; ++k) {
            *c_reduce_src += *(c_reduce_src + k * c_reduce_k_stride);
        }
    }
    __syncthreads();
    // Short cut
    const int N = kSeqLength;
    const int M = kHiddenDim;
    half* short_cut_add_shared = acc_shared + 
        ((kGemmK6BlockColTiles * kWmmaN) * (kGemmK6BlockRowTiles * kWmmaM + kAccSkew));
    const int batch_stride =
        (N / kGemmK6BlockColTiles / kWmmaN) * (M / kGemmK6BlockRowTiles / kWmmaM);
    const int batched_id = blockIdx.x / batch_stride;

    uint64_t attn_fc_offset = batched_id * N * M +
                       row_block_id * kGemmK6BlockRowTiles * kWmmaM +
                       (col_block_id * kGemmK6BlockColTiles * kWmmaN +
                        threadIdx.x / kStoreCLanesPerRow) *
                           M +
                       (threadIdx.x & (kStoreCLanesPerRow - 1)) *
                           sizeof(float4) / sizeof(half);
    uint64_t shared_attn_fc_offset = threadIdx.x / kStoreCLanesPerRow *
                           (kGemmK6BlockRowTiles * kWmmaM + kAccSkew) +
                       (threadIdx.x & (kStoreCLanesPerRow - 1)) *
                           sizeof(float4) / sizeof(half);
    const int c_dst_stride = kStoreCColsPerIter * M;
    const int c_src_stride =
        kStoreCColsPerIter * (kGemmK6BlockRowTiles * kWmmaM + kAccSkew);
    
    half* src_base = attn_fc_output + attn_fc_offset;
    half* short_cut_add_shared_base = short_cut_add_shared + shared_attn_fc_offset;
    // Load src to short_cut_add_shared
    pipe.producer_acquire();
    for (int i = 0; i < kGemmK6BlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
        cuda::memcpy_async((float4*)(short_cut_add_shared_base + i * c_src_stride),
                            (float4*)(src_base + i * c_dst_stride), shape, pipe);
    }
    pipe.producer_commit();
    pipe.consumer_wait();
    __syncthreads();
    pipe.consumer_release();
    
    
    // Compute short_cut_add, shape:(64, 64+8), we have 128 threads
    uint64_t global_col_offset = batched_id * N + col_block_id * kGemmK6BlockColTiles * kWmmaN;
    // Init reduce sum global memory
    float* arr_shared[] = {query_key_softmax_sum, layer_norm_variance};
    arr_shared[(threadIdx.x >> 6)][threadIdx.x & 63] = 0;
    // if(threadIdx.x < 64){
    //     query_key_softmax_sum[global_col_offset + (threadIdx.x & 63)] = 0;
    // }else{
    //     layer_norm_variance[global_col_offset + (threadIdx.x & 63)] = 0;
    // }
    __syncthreads();
    // const int kComputeRowsPerIter = 128 * sizeof(half2) / sizeof(half) / (kGemmK6BlockRowTiles * kWmmaM);
    // for(int i=0; i<(kGemmK6BlockColTiles * kWmmaN / kComputeRowsPerIter); ++i){
    //     int row = i * kComputeRowsPerIter + (threadIdx.x >> 5);
    //     int col = (threadIdx.x & 31) * (sizeof(half2)/sizeof(half));
    //     int idx = row * (kGemmK6BlockRowTiles * kWmmaM + kAccSkew) + col;
    //     half2 value = ((half2*)(acc_shared + idx))[0] + ((half2*)(short_cut_add_shared + idx))[0];
        
    //     ((half2*)(acc_shared + idx))[0] = value;
    //     half2 value_x_2 = value * value;
    //     atomicAdd(query_key_softmax_sum + global_col_offset + row, (__half2float(value_x_2.x) + __half2float(value_x_2.y)));
    //     atomicAdd(layer_norm_variance + global_col_offset + row, (__half2float(value.x) + __half2float(value.y)));
    //     // half2 reduce_sum_x_2 = warpReduceSumHalf2((value*value));
    //     // __syncthreads();
    //     // half2 reduce_sum_x = warpReduceSumHalf2(value);
    //     // __syncthreads();
    //     // if((threadIdx.x & 31) == 0){
    //     //     atomicAdd(query_key_softmax_sum + global_col_offset + row, (__half2float(reduce_sum_x_2.x) + __half2float(reduce_sum_x_2.y)));
    //     //     atomicAdd(layer_norm_variance + global_col_offset + row, (__half2float(reduce_sum_x.x) + __half2float(reduce_sum_x.y)));
    //     // }
    //     // __syncthreads();
    // }
    // if(threadIdx.x < 64){
    //     half2 reduce_sum_x_2 = half2(0, 0), reduce_sum_x = half2(0, 0);
    //     const int row_stride = (kGemmK6BlockRowTiles * kWmmaM + kAccSkew);
    //     for(int i=0; i<kGemmK6BlockColTiles * kWmmaN; ++i){
    //         half2 value = ((half2*)acc_shared + threadIdx.x * row_stride + i * 2)[0];
    //         half2 value_x_2 = value * value;
    //         reduce_sum_x += value;
    //         reduce_sum_x_2 += value_x_2;
    //     }
    //     atomicAdd(query_key_softmax_sum + global_col_offset + threadIdx.x, (__half2float(reduce_sum_x_2.x) + __half2float(reduce_sum_x_2.y)));
    //     atomicAdd(layer_norm_variance + global_col_offset + threadIdx.x, (__half2float(reduce_sum_x.x) + __half2float(reduce_sum_x.y)));
    // }
    // __syncthreads();
        float sum_x = 0, sum_x2=0;
        const int kVecSize = sizeof(half2) / sizeof(half);
        const int kNumRowTiles = kThreads / (kGemmK6BlockColTiles * kWmmaN);
        const int offset = (threadIdx.x >> 6) << 5;
        const int cmp_shared_stride = threadIdx.x * (kGemmK6BlockRowTiles * kWmmaM) + offset;
        for(int i=0; i<(kGemmK6BlockRowTiles * kWmmaM / kNumRowTiles / kVecSize); ++i){
            half2 value = ((half2*)(acc_shared + cmp_shared_stride + (i << 1)))[0];
            float2 value_f = __half22float2(value);
            sum_x += (value_f.x+value_f.y);
            sum_x2 += (value_f.x * value_f.x + value_f.y * value_f.y);
        }
        const int g_idx = row_block_id * (kGemmK6BlockColTiles * kWmmaN) + (threadIdx.x & 63);
        atomicAdd(query_key_softmax_sum + g_idx, sum_x);
        atomicAdd(layer_norm_variance + g_idx, sum_x2);
    }
    grid.sync();
    
  profile_grid_clock[clock_idx * 108 * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
    if(blockIdx.x < 72){
    enum {
        kThreads = kGemmK6BlockSliceKTiles * kWarpSize,
        kLoadALanesPerRow =
            kWmmaM * kGemmK6BlockRowTiles / (sizeof(float4) / sizeof(half)),
        kLoadAColsPerIter = kThreads / kLoadALanesPerRow,

        kLoadBLanesPerRow =
            kWmmaK * kGemmK6BlockSliceKTiles / (sizeof(float4) / sizeof(half)),
        kLoadBColsPerIter = kThreads / kLoadBLanesPerRow,

        kReduceCLanesPerRow =
            kWmmaM * kGemmK6BlockRowTiles / (sizeof(half2) / sizeof(half)),
        kReduceCColsPerIter = kThreads / kReduceCLanesPerRow,

        kStoreCLanesPerRow = kLoadALanesPerRow,
        kStoreCColsPerIter = kLoadAColsPerIter,
    };
    half* acc_shared = all_shared_mem;
    const int N = kSeqLength;
    const int M = kHiddenDim;
    half* short_cut_add_shared = acc_shared + 
        ((kGemmK6BlockColTiles * kWmmaN) * (kGemmK6BlockRowTiles * kWmmaM + kAccSkew));
    const int batch_stride =
        (N / kGemmK6BlockColTiles / kWmmaN) * (M / kGemmK6BlockRowTiles / kWmmaM);
    const int batched_id = blockIdx.x / batch_stride;
    const int row_block_id =
        blockIdx.x % (kHiddenDim / kGemmK6BlockRowTiles / kWmmaM);
    const int col_block_id =
        blockIdx.x / (kHiddenDim / kGemmK6BlockRowTiles / kWmmaM);
    uint64_t attn_fc_offset = batched_id * N * M +
                       row_block_id * kGemmK6BlockRowTiles * kWmmaM +
                       (col_block_id * kGemmK6BlockColTiles * kWmmaN +
                        threadIdx.x / kStoreCLanesPerRow) *
                           M +
                       (threadIdx.x & (kStoreCLanesPerRow - 1)) *
                           sizeof(float4) / sizeof(half);
    uint64_t shared_attn_fc_offset = threadIdx.x / kStoreCLanesPerRow *
                           (kGemmK6BlockRowTiles * kWmmaM + kAccSkew) +
                       (threadIdx.x & (kStoreCLanesPerRow - 1)) *
                           sizeof(float4) / sizeof(half);
    const int c_dst_stride = kStoreCColsPerIter * M;
    const int c_src_stride =
        kStoreCColsPerIter * (kGemmK6BlockRowTiles * kWmmaM + kAccSkew);
    
    // Compute short cut add and layer norm variance
    uint64_t global_col_offset = batched_id * N + col_block_id * kGemmK6BlockColTiles * kWmmaN;
    const int kThreadsPerBlock = 128;
    const int kComputeRowsPerIter = kThreadsPerBlock * sizeof(half2) / sizeof(half) / (kGemmK6BlockRowTiles * kWmmaM);
    for(int i=0; i<(kGemmK6BlockColTiles * kWmmaN / kComputeRowsPerIter); ++i){
        int row = i * kComputeRowsPerIter + (threadIdx.x >> 5);
        int col = (threadIdx.x & 31) * (sizeof(half2)/sizeof(half));
        int idx = row * (kGemmK6BlockRowTiles * kWmmaM + kAccSkew) + col;
        half2 value = ((half2*)(acc_shared + idx))[0];
        float sum_x_2 = query_key_softmax_sum[global_col_offset + row];
        float sum_x = layer_norm_variance[global_col_offset + row];
        half standard_deviation = __float2half(sqrt((sum_x_2 - (sum_x * sum_x)/kHiddenDim) / kHiddenDim + __half2float(eps)));
        half mean = __float2half(sum_x / kHiddenDim);
        half2 norm;
        norm.x = ((value.x - mean) / standard_deviation) * h_gama + h_beta;
        norm.y = ((value.y - mean) / standard_deviation) * h_gama + h_beta;
        ((half2*)(acc_shared + idx))[0] = norm;
        __syncthreads();
    }
    

    half *c_dst_base = feed_forward_fc2_output + row_block_id * kGemmK6BlockRowTiles * kWmmaM +
                       (col_block_id * kGemmK6BlockColTiles * kWmmaN +
                        threadIdx.x / kStoreCLanesPerRow) *
                           kHiddenDim +
                       (threadIdx.x & (kStoreCLanesPerRow - 1)) *
                           sizeof(float4) / sizeof(half);
    half *c_src_base = acc_shared +
                       threadIdx.x / kStoreCLanesPerRow *
                           (kGemmK6BlockRowTiles * kWmmaM + kAccSkew) +
                       (threadIdx.x & (kStoreCLanesPerRow - 1)) *
                           sizeof(float4) / sizeof(half);

#pragma unroll
    for (int i = 0; i < kGemmK6BlockColTiles * kWmmaN / kStoreCColsPerIter;
         ++i) {
        *reinterpret_cast<float4 *>(c_dst_base + i * c_dst_stride) =
            *reinterpret_cast<float4 *>(c_src_base + i * c_src_stride);
    }
    }

}