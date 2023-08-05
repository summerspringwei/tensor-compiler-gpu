#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <mma.h>

__global__ void debug_feed_forward_fc1(half* __restrict__ feed_forward_fc1_weight,
                                half* __restrict__ attn_fc_output,
                                half* __restrict__ feed_forward_fc1_output){
    using namespace nvcuda;
    
    extern __shared__ half all_shared_mem[];
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
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
    // use 3*64*132*sizeof(half)=49.5KB
    const size_t feed_forward_fc1_weight_shared_mem_offset = 3 * 64 * (128+8);
    matrix_a_shared[0] = all_shared_mem + feed_forward_fc1_weight_shared_mem_offset;
    matrix_a_shared[1] =
        all_shared_mem + feed_forward_fc1_weight_shared_mem_offset +
        kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[2] =
        all_shared_mem + feed_forward_fc1_weight_shared_mem_offset +
        2 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);

    matrix_b_shared[0] =
        all_shared_mem;
    matrix_b_shared[1] =
        all_shared_mem +
        kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);
    matrix_b_shared[2] =
        all_shared_mem +
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

#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
            cuda::memcpy_async(a_dst_base + i * a_dst_stride,
                               a_src_base + i * a_src_stride, shape, pipe);
        }
//         half *b_dst_base =
//             matrix_b_shared[(stage + s) % kStage] +
//             threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
//             (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) /
//                 sizeof(half);

//         const half *b_src_base = attn_fc_output + batched_id * N * K +
//                                  (k_loop + s) * kChunkK * kWmmaK +
//                                  (col_block_id * kBlockColTiles * kWmmaN +
//                                   threadIdx.x / kLoadBLanesPerRow) *
//                                      K +
//                                  (threadIdx.x & (kLoadBLanesPerRow - 1)) *
//                                      (sizeof(float4) / sizeof(half));

// #pragma unroll
//         for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
//             cuda::memcpy_async(b_dst_base + i * b_dst_stride,
//                                b_src_base + i * b_src_stride, shape, pipe);
//         }
        pipe.producer_commit();
    }

#pragma unroll
    for (int s = 0; s < kStage - 1; ++s) {
        pipe.producer_acquire();
//         half *a_dst_base = matrix_a_shared[(stage + s) % kStage] +
//                            threadIdx.x / kLoadALanesPerRow *
//                                (kWmmaM * kBlockRowTiles + kInputSkew) +
//                            (threadIdx.x & (kLoadALanesPerRow - 1)) *
//                                sizeof(float4) / sizeof(half);

//         const half *a_src_base = feed_forward_fc1_weight + batched_id * K * M +
//                                  row_block_id * kBlockRowTiles * kWmmaM +
//                                  ((k_loop + s) * kChunkK * kWmmaK +
//                                   threadIdx.x / kLoadALanesPerRow) *
//                                      M +
//                                  (threadIdx.x & (kLoadALanesPerRow - 1)) *
//                                      (sizeof(float4) / sizeof(half));

// #pragma unroll
//         for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
//             cuda::memcpy_async(a_dst_base + i * a_dst_stride,
//                                a_src_base + i * a_src_stride, shape, pipe);
//         }
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
        for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
            cuda::memcpy_async(b_dst_base + i * b_dst_stride,
                               b_src_base + i * b_src_stride, shape, pipe);
        }
        pipe.producer_commit();
    }

pipe.consumer_wait();
        __syncthreads();
        pipe.consumer_release();
int stage_count = 0;
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
        if(stage_count>0){
            pipe.consumer_wait();
            __syncthreads();
            pipe.consumer_release();
        }
        stage_count--;
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
pipe.consumer_wait();
        __syncthreads();
        pipe.consumer_release();
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
    int col = (threadIdx.x & 63) * 2;
    const int row_offset = (threadIdx.x >> 6);
    const int kComputeRowsPerIter = 128 * (sizeof(half2) / sizeof(half)) / (kBlockRowTiles * kWmmaM);
    for(int i=0; i<kBlockColTiles*kWmmaN / kComputeRowsPerIter; ++i){
        int row = i * kComputeRowsPerIter + row_offset;
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
    


#include "bert.h"
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <mma.h>


// qkv matmul shared memory: 87552, blocks 96
// gemm_k2 matmul shared memory: 149504, blocks 108
// gemm_k3 shared memory 55296, blocks 72
// gemm_k4 shared memory 55296, blocks 72
// gemm_k5 shared memory 93696, blocks 96
// gemm_k6 shared memory 55296, blocks 72
using namespace fuselage::experiments::networks::bert;

__global__ void debug_fused_sqq_bert_pipelined(const half *__restrict__ qkv_weight, 
                                half *__restrict__ src,
                                const half *__restrict__ qkv_bias,
                                half *__restrict__ qkv_output,
                                half *__restrict__ query_key_output,
                                half *__restrict__ query_key_mask,
                                float * query_key_softmax_sum,
                                half *__restrict__ attn_value_output,
                                half* __restrict__ attn_fc_weight,
                                half* __restrict__ attn_fc_output,
                                float* attn_layer_norm_sum,
                                float* attn_layer_norm_variance,
                                half eps, half h_gama, half h_beta,
                                half* __restrict__ feed_forward_fc1_weight,
                                half* __restrict__ feed_forward_fc1_output,
                                half* __restrict__ feed_forward_fc2_weight,
                                half* __restrict__ feed_forward_fc2_output,
                                float* feed_forward_layernorm_sum,
                                float* feed_forward_layernorm_variance,
                                int64_t* profile_grid_clock,
                                // Pointers from pytorch
                                half* ptr_t_attn_fc_output,
                                half* ptr_t_attn_fc_short_cut_add
                                ){
  using namespace nvcuda;
  extern __shared__ half all_shared_mem[];
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  int pipeline_stack = 0;
  int clock_idx = 0;
  unsigned int c = 0;
  const int warpIdx = threadIdx.x >> 5;
  
  profile_grid_clock[clock_idx * 108 * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
  
  // Begin of fused QKV matmul
  if(blockIdx.x < 96){
    enum {
        kBlockRowTiles = kBlockRowWarps * kGemmK1WarpRowTiles,
        kBlockColTiles = kBlockColWarps * kGemmK1WarpColTiles,
    };

    half *matrix_a_shared[3][kStage], *matrix_b_shared[kStage];
    half *acc_shared;

    matrix_a_shared[0][0] = all_shared_mem;
    matrix_a_shared[0][1] =
        all_shared_mem +
        kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[0][2] =
        all_shared_mem +
        2 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[1][0] =
        matrix_a_shared[0][0] +
        3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[1][1] =
        matrix_a_shared[0][1] +
        3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[1][2] =
        matrix_a_shared[0][2] +
        3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[2][0] =
        matrix_a_shared[1][0] +
        3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[2][1] =
        matrix_a_shared[1][1] +
        3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[2][2] =
        matrix_a_shared[1][2] +
        3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);

    matrix_b_shared[0] =
        all_shared_mem +
        9 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_b_shared[1] =
        all_shared_mem +
        9 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
        kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);
    matrix_b_shared[2] =
        all_shared_mem +
        9 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
        2 * kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);

    acc_shared = all_shared_mem;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::col_major>
        wmma_matrix_a[3][kGemmK1WarpRowTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::col_major>
        wmma_matrix_b[kGemmK1WarpColTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                           half>
        wmma_accumulator[3][kGemmK1WarpColTiles * kGemmK1WarpRowTiles];

    const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
    const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;
    const int row_block_id =
        blockIdx.x % (kHiddenDim / kBlockRowTiles / kWmmaM);
    const int col_block_id =
        blockIdx.x / (kHiddenDim / kBlockRowTiles / kWmmaM);

#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int col = 0; col < kGemmK1WarpColTiles; ++col) {
#pragma unroll
            for (int row = 0; row < kGemmK1WarpRowTiles; ++row) {
                nvcuda::wmma::fill_fragment(
                    wmma_accumulator[i][col * kGemmK1WarpRowTiles + row], 0.0f);
            }
        }
    }

    enum {
        kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kLoadALanesPerRow =
            kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
        kLoadAColsPerIter = kThreads / kLoadALanesPerRow,

        kLoadBLanesPerRow = kWmmaK * kChunkK / (sizeof(float4) / sizeof(half)),
        kLoadBColsPerIter = kThreads / kLoadBLanesPerRow,

        kAddBiasLanesPerRow =
            kWmmaM * kBlockRowTiles / (sizeof(half2) / sizeof(half)),
        kAddBiasColsPerIter = kThreads / kAddBiasLanesPerRow,

        kStoreCLanesPerRow = kLoadALanesPerRow,
        kStoreCColsPerIter = kLoadAColsPerIter,
    };

    

    const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
    int stage = 0;
    int k_loop = 0;

    const int a_dst_stride =
        kLoadAColsPerIter * (kWmmaM * kBlockRowTiles + kInputSkew);
    const int a_src_stride = kLoadAColsPerIter * kHiddenDim;

    const int b_dst_stride =
        kLoadBColsPerIter * (kWmmaK * kChunkK + kInputSkew);
    const int b_src_stride = kLoadBColsPerIter * kHiddenDim;

#pragma unroll
    for (int s = 0; s < kStage - 1; ++s) {
        pipe.producer_acquire();
        half *a_dst_base_0 = matrix_a_shared[0][(stage + s) % kStage] +
                             threadIdx.x / kLoadALanesPerRow *
                                 (kWmmaM * kBlockRowTiles + kInputSkew) +
                             (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                 sizeof(float4) / sizeof(half);
        half *a_dst_base_1 =
            a_dst_base_0 +
            3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
        half *a_dst_base_2 =
            a_dst_base_0 +
            6 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);

        const half *a_src_base_0 = qkv_weight +
                                   row_block_id * kBlockRowTiles * kWmmaM +
                                   ((k_loop + s) * kChunkK * kWmmaK +
                                    threadIdx.x / kLoadALanesPerRow) *
                                       kHiddenDim +
                                   (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                       (sizeof(float4) / sizeof(half));
        const half *a_src_base_1 = a_src_base_0 + kHiddenDim * kHiddenDim;
        const half *a_src_base_2 = a_src_base_1 + kHiddenDim * kHiddenDim;

        half *b_dst_base =
            matrix_b_shared[(stage + s) % kStage] +
            threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
            (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) /
                sizeof(half);

        const half *b_src_base = src + (k_loop + s) * kChunkK * kWmmaK +
                                 (col_block_id * kBlockColTiles * kWmmaN +
                                  threadIdx.x / kLoadBLanesPerRow) *
                                     kHiddenDim +
                                 (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                                     (sizeof(float4) / sizeof(half));

#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
            cuda::memcpy_async(a_dst_base_0 + i * a_dst_stride,
                               a_src_base_0 + i * a_src_stride, shape, pipe);
        }
#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
            cuda::memcpy_async(a_dst_base_1 + i * a_dst_stride,
                               a_src_base_1 + i * a_src_stride, shape, pipe);
        }
#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
            cuda::memcpy_async(a_dst_base_2 + i * a_dst_stride,
                               a_src_base_2 + i * a_src_stride, shape, pipe);
        }

#pragma unroll
        for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
            cuda::memcpy_async(b_dst_base + i * b_dst_stride,
                               b_src_base + i * b_src_stride, shape, pipe);
        }
        pipe.producer_commit();
        
    }

#pragma unroll
    for (; k_loop < (kHiddenDim / kChunkK / kWmmaK) - (kStage - 1); ++k_loop) {
        pipe.producer_acquire();

        half *a_dst_base_0 = matrix_a_shared[0][(stage + kStage - 1) % kStage] +
                             threadIdx.x / kLoadALanesPerRow *
                                 (kWmmaM * kBlockRowTiles + kInputSkew) +
                             (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                 sizeof(float4) / sizeof(half);
        half *a_dst_base_1 =
            a_dst_base_0 +
            3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
        half *a_dst_base_2 =
            a_dst_base_0 +
            6 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
        const half *a_src_base_0 = qkv_weight +
                                   row_block_id * kBlockRowTiles * kWmmaM +
                                   ((k_loop + kStage - 1) * kChunkK * kWmmaK +
                                    threadIdx.x / kLoadALanesPerRow) *
                                       kHiddenDim +
                                   (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                       (sizeof(float4) / sizeof(half));
        const half *a_src_base_1 = a_src_base_0 + kHiddenDim * kHiddenDim;
        const half *a_src_base_2 = a_src_base_1 + kHiddenDim * kHiddenDim;

        half *b_dst_base =
            matrix_b_shared[(stage + kStage - 1) % kStage] +
            threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
            (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) /
                sizeof(half);

        const half *b_src_base = src +
                                 (k_loop + kStage - 1) * kChunkK * kWmmaK +
                                 (col_block_id * kBlockColTiles * kWmmaN +
                                  threadIdx.x / kLoadBLanesPerRow) *
                                     kHiddenDim +
                                 (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                                     (sizeof(float4) / sizeof(half));

#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
            cuda::memcpy_async(a_dst_base_0 + i * a_dst_stride,
                               a_src_base_0 + i * a_src_stride, shape, pipe);
        }
#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
            cuda::memcpy_async(a_dst_base_1 + i * a_dst_stride,
                               a_src_base_1 + i * a_src_stride, shape, pipe);
        }
#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
            cuda::memcpy_async(a_dst_base_2 + i * a_dst_stride,
                               a_src_base_2 + i * a_src_stride, shape, pipe);
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
            for (int tile_m = 0; tile_m < kGemmK1WarpRowTiles; ++tile_m) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[0][tile_m],
                    (matrix_a_shared[0][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[1][tile_m],
                    (matrix_a_shared[1][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[2][tile_m],
                    (matrix_a_shared[2][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
            }
#pragma unroll
            for (int tile_n = 0; tile_n < kGemmK1WarpColTiles; ++tile_n) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_b[tile_n],
                    (matrix_b_shared[stage] +
                     (col_warp_id * kGemmK1WarpColTiles + tile_n) * kWmmaN *
                         (kChunkK * kWmmaK + kInputSkew) +
                     tile_k * kWmmaK),
                    kChunkK * kWmmaK + kInputSkew);
            }
#pragma unroll
            for (int tile_m = 0; tile_m < kGemmK1WarpRowTiles; ++tile_m) {
#pragma unroll
                for (int i = 0; i < 3; ++i) {
#pragma unroll
                    for (int tile_n = 0; tile_n < kGemmK1WarpColTiles;
                         ++tile_n) {
                        nvcuda::wmma::mma_sync(
                            wmma_accumulator[i][tile_m +
                                                tile_n * kGemmK1WarpRowTiles],
                            wmma_matrix_a[i][tile_m], wmma_matrix_b[tile_n],
                            wmma_accumulator[i][tile_m +
                                                tile_n * kGemmK1WarpRowTiles]);
                    }
                }
            }
        }
        stage = (stage + 1) % kStage;
    }

#pragma unroll
    for (int s = kStage - 1; s >= 1; --s) {
        k_loop = (kHiddenDim / kChunkK / kWmmaK) - s;
        pipe.consumer_wait();
        __syncthreads();
        pipe.consumer_release();

#pragma unroll
        for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
            for (int tile_m = 0; tile_m < kGemmK1WarpRowTiles; ++tile_m) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[0][tile_m],
                    (matrix_a_shared[0][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[1][tile_m],
                    (matrix_a_shared[1][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[2][tile_m],
                    (matrix_a_shared[2][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
            }
#pragma unroll
            for (int tile_n = 0; tile_n < kGemmK1WarpColTiles; ++tile_n) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_b[tile_n],
                    (matrix_b_shared[stage] +
                     (col_warp_id * kGemmK1WarpColTiles + tile_n) * kWmmaN *
                         (kChunkK * kWmmaK + kInputSkew) +
                     tile_k * kWmmaK),
                    kChunkK * kWmmaK + kInputSkew);
            }
#pragma unroll
            for (int tile_m = 0; tile_m < kGemmK1WarpRowTiles; ++tile_m) {
#pragma unroll
                for (int i = 0; i < 3; ++i) {
#pragma unroll
                    for (int tile_n = 0; tile_n < kGemmK1WarpColTiles;
                         ++tile_n) {
                        nvcuda::wmma::mma_sync(
                            wmma_accumulator[i][tile_m +
                                                tile_n * kGemmK1WarpRowTiles],
                            wmma_matrix_a[i][tile_m], wmma_matrix_b[tile_n],
                            wmma_accumulator[i][tile_m +
                                                tile_n * kGemmK1WarpRowTiles]);
                    }
                }
            }
        }
        stage = (stage + 1) % kStage;
    }

#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int tile_n = 0; tile_n < kGemmK1WarpColTiles; ++tile_n) {
#pragma unroll
            for (int tile_m = 0; tile_m < kGemmK1WarpRowTiles; ++tile_m) {
                nvcuda::wmma::store_matrix_sync(
                    acc_shared +
                        i * kBlockColTiles * kWmmaN *
                            (kBlockRowTiles * kWmmaM + kAccSkew) +
                        (col_warp_id * kGemmK1WarpColTiles + tile_n) * kWmmaK *
                            (kBlockRowTiles * kWmmaM + kAccSkew) +
                        (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM,
                    wmma_accumulator[i][tile_n * kGemmK1WarpRowTiles + tile_m],
                    (kBlockRowTiles * kWmmaM + kAccSkew),
                    nvcuda::wmma::mem_col_major);
            }
        }
    }

    __syncthreads();

    const int bias_stride =
        kAddBiasColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);
    half *bias_dst_base = acc_shared +
                          threadIdx.x / kAddBiasLanesPerRow *
                              (kBlockRowTiles * kWmmaM + kAccSkew) +
                          (threadIdx.x & (kAddBiasLanesPerRow - 1)) *
                              sizeof(half2) / sizeof(half);
    const half *bias_src_base = qkv_bias + row_block_id * kBlockRowTiles * kWmmaM +
                                (threadIdx.x & (kAddBiasLanesPerRow - 1)) *
                                    sizeof(half2) / sizeof(half);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
#pragma unroll
        for (int i = 0; i < kBlockColTiles * kWmmaN / kAddBiasColsPerIter;
             ++i) {
            *reinterpret_cast<half2 *>(
                bias_dst_base +
                j * kBlockColTiles * kWmmaN *
                    (kBlockRowTiles * kWmmaM + kAccSkew) +
                i * bias_stride) +=
                __ldg(reinterpret_cast<const half2 *>(bias_src_base +
                                                      j * kHiddenDim));
        }
    }

    __syncthreads();

    const int c_dst_stride = kStoreCColsPerIter * kHeadSize;
    const int c_src_stride =
        kStoreCColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);

    half *c_dst_base =
        qkv_output +
        (row_block_id / 2) * 2 * kBlockRowTiles * kWmmaM * kSeqLength +
        (row_block_id % 2) * kBlockRowTiles * kWmmaM +
        (col_block_id * kBlockColTiles * kWmmaN +
         threadIdx.x / kStoreCLanesPerRow) *
            kHeadSize +
        (threadIdx.x & (kStoreCLanesPerRow - 1)) * sizeof(float4) /
            sizeof(half);
    half *c_src_base = acc_shared +
                       threadIdx.x / kStoreCLanesPerRow *
                           (kBlockRowTiles * kWmmaM + kAccSkew) +
                       (threadIdx.x & (kStoreCLanesPerRow - 1)) *
                           sizeof(float4) / sizeof(half);

#pragma unroll
    for (int j = 0; j < 3; ++j) {
#pragma unroll
        for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
            #ifndef REWRITE_RESULT_DEBUG
            *reinterpret_cast<float4 *>(c_dst_base + i * c_dst_stride +
                                        j * kHiddenDim * kSeqLength) =
                *reinterpret_cast<float4 *>(
                    c_src_base + i * c_src_stride +
                    j * kBlockColTiles * kWmmaN *
                        (kBlockRowTiles * kWmmaM + kAccSkew));
            #endif
        }
    }
  } // End of fused QKV matmul
/* ----------------------------------------------- */
 
  grid.sync();
  
  profile_grid_clock[clock_idx * 108 * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
  // Begin of Query-Key bmm
  if(blockIdx.x < 108){
    const half* __restrict__ query = qkv_output;
    const half* __restrict__ key = query + BertScaleParams::kBatchSize * BertScaleParams::kSeqLength * BertScaleParams::kHiddenDim;
    enum {
        kBlockRowTiles = kBlockRowWarps * kGemmK2WarpRowTiles,
        kBlockColTiles = kBlockColWarps * kGemmK2WarpColTiles,
    };
    // Occupies shared memory [0: 128*72+128*72*sizeof(half)]
    half *matrix_a_shared = all_shared_mem;

    half *matrix_b_shared =
        matrix_a_shared + kBlockRowTiles * kWmmaM * (kHeadSize + kInputSkew);

    half *acc_shared = all_shared_mem;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::row_major>
        wmma_matrix_a[kGemmK2WarpRowTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::col_major>
        wmma_matrix_b[kGemmK2WarpColTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                           half>
        wmma_accumulator[kGemmK2WarpColTiles * kGemmK2WarpRowTiles];

    const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
    const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;
    const int batch_stride = (kSeqLength / kBlockColTiles / kWmmaN) *
                             (kSeqLength / kBlockRowTiles / kWmmaM);
    const int batched_id = blockIdx.x / batch_stride;
    const int row_block_id =
        blockIdx.x % batch_stride % (kSeqLength / kBlockRowTiles / kWmmaM);
    const int col_block_id =
        blockIdx.x % batch_stride / (kSeqLength / kBlockRowTiles / kWmmaM);

#pragma unroll
    for (int col = 0; col < kGemmK2WarpColTiles; ++col) {
#pragma unroll
        for (int row = 0; row < kGemmK2WarpRowTiles; ++row) {
            nvcuda::wmma::fill_fragment(
                wmma_accumulator[col * kGemmK2WarpRowTiles + row], 0.0f);
        }
    }

    enum {
        kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kLoadLanesPerRow = kHeadSize / (sizeof(float4) / sizeof(half)),
        kLoadColsPerIter = kThreads / kLoadLanesPerRow,

        kStoreLanesPerRow =
            kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
        kStoreColsPerIter = kThreads / kStoreLanesPerRow,
    };

    const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));

    pipe.producer_acquire();
#pragma unroll
    for (int i = 0; i < kBlockRowTiles * kWmmaM / kLoadColsPerIter; ++i) {
        cuda::memcpy_async(
            reinterpret_cast<float4 *>(
                matrix_a_shared +
                (i * kLoadColsPerIter + threadIdx.x / kLoadLanesPerRow) *
                    (kHeadSize + kInputSkew) +
                (threadIdx.x & (kLoadLanesPerRow - 1)) * sizeof(float4) /
                    sizeof(half)),
            reinterpret_cast<const float4 *>(
                key + batched_id * kSeqLength * kHeadSize +
                (row_block_id * kBlockRowTiles * kWmmaM + i * kLoadColsPerIter +
                 threadIdx.x / kLoadLanesPerRow) *
                    kHeadSize +
                (threadIdx.x & (kLoadLanesPerRow - 1)) *
                    (sizeof(float4) / sizeof(half))),
            shape, pipe);
    }

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadColsPerIter; ++i) {
        cuda::memcpy_async(
            reinterpret_cast<float4 *>(
                matrix_b_shared +
                (i * kLoadColsPerIter + threadIdx.x / kLoadLanesPerRow) *
                    (kHeadSize + kInputSkew) +
                (threadIdx.x & (kLoadLanesPerRow - 1)) * sizeof(float4) /
                    sizeof(half)),
            reinterpret_cast<const float4 *>(
                query + batched_id * kSeqLength * kHeadSize +
                (col_block_id * kBlockColTiles * kWmmaN + i * kLoadColsPerIter +
                 threadIdx.x / kLoadLanesPerRow) *
                    kHeadSize +
                (threadIdx.x & (kLoadLanesPerRow - 1)) *
                    (sizeof(float4) / sizeof(half))),
            shape, pipe);
    }
    pipe.producer_commit();
    pipe.consumer_wait();
    __syncthreads();
    pipe.consumer_release();

__syncthreads();

#pragma unroll
    for (int tile_k = 0; tile_k < kHeadSize / kWmmaK; ++tile_k) {
#pragma unroll
        for (int tile_m = 0; tile_m < kGemmK2WarpRowTiles; ++tile_m) {
            nvcuda::wmma::load_matrix_sync(
                wmma_matrix_a[tile_m],
                (matrix_a_shared +
                 (row_warp_id * kGemmK2WarpRowTiles + tile_m) * kWmmaM *
                     (kHeadSize + kInputSkew) +
                 tile_k * kWmmaK),
                kHeadSize + kInputSkew);
        }
#pragma unroll
        for (int tile_n = 0; tile_n < kGemmK2WarpColTiles; ++tile_n) {
            nvcuda::wmma::load_matrix_sync(
                wmma_matrix_b[tile_n],
                (matrix_b_shared +
                 (col_warp_id * kGemmK2WarpColTiles + tile_n) * kWmmaN *
                     (kHeadSize + kInputSkew) +
                 tile_k * kWmmaK),
                kHeadSize + kInputSkew);
        }
#pragma unroll
        for (int tile_m = 0; tile_m < kGemmK2WarpRowTiles; ++tile_m) {
#pragma unroll
            for (int tile_n = 0; tile_n < kGemmK2WarpColTiles; ++tile_n) {
                nvcuda::wmma::mma_sync(
                    wmma_accumulator[tile_m + tile_n * kGemmK2WarpRowTiles],
                    wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
                    wmma_accumulator[tile_m + tile_n * kGemmK2WarpRowTiles]);
            }
        }
    }
    __syncthreads();


#pragma unroll
    for (int tile_n = 0; tile_n < kGemmK2WarpColTiles; ++tile_n) {
#pragma unroll
        for (int tile_m = 0; tile_m < kGemmK2WarpRowTiles; ++tile_m) {
            nvcuda::wmma::store_matrix_sync(
                acc_shared +
                    (col_warp_id * kGemmK2WarpColTiles + tile_n) * kWmmaK *
                        (kBlockRowTiles * kWmmaM + kAccSkew) +
                    (row_warp_id * kGemmK2WarpRowTiles + tile_m) * kWmmaM,
                wmma_accumulator[tile_n * kGemmK2WarpRowTiles + tile_m],
                (kBlockRowTiles * kWmmaM + kAccSkew),
                nvcuda::wmma::mem_col_major);
        }
    }

    __syncthreads();
    // Each acc_shared contains (128, 128) elements
    // fused sqrt(hidden_size) + Softmax_reduce_sum
    // Now all values are in shared memory, need to find the layout in shared memory
    // The shared memory uses the same layout as global memory
    const uint64_t attn_mask_base_idx = batched_id * kSeqLength * kSeqLength + 
      (col_block_id * kBlockColTiles * kWmmaN) * kSeqLength + 
      row_block_id * kBlockRowTiles * kWmmaM + threadIdx.x * kSeqLength;
    const int row_shared_size = (kBlockRowTiles * kWmmaM + kAccSkew);
    const int col_shared_size = (kBlockColTiles * kWmmaN);
    float softmax_sum = 0;
    half scale = half(1.0) / hsqrt(__float2half(kHiddenDim));
    half2 scale_h2(scale, scale);
    const int kHalf2Vec = sizeof(half2) / sizeof(half);

    // Now we let one thread to compute half # of elements of the row
    // const int reduce_shared_stride = (threadIdx.x & 63) * row_shared_size + ((threadIdx.x >> 6) << 5);
    const int kSplitNum = 128 / (kBlockColTiles * kWmmaN);
    const int reduce_shared_stride = threadIdx.x * row_shared_size;
    for(int i=0; i<kBlockRowTiles * kWmmaM / kHalf2Vec; ++i){
        int idx = reduce_shared_stride + i * kHalf2Vec;
        auto scaled_acc = ((half2*)(acc_shared + idx))[0] * scale_h2;
        auto mask_h2 = ((half2*)(query_key_mask + attn_mask_base_idx + i * kHalf2Vec))[0];
        auto new_attn_value = h2exp(scaled_acc + mask_h2);
        ((half2*)(acc_shared + idx))[0] = new_attn_value;
        softmax_sum += (__half2float(new_attn_value.x) + __half2float(new_attn_value.y));
    }
    atomicAdd(query_key_softmax_sum + batched_id * kSeqLength + col_block_id *  (kBlockColTiles * kWmmaN) + threadIdx.x, softmax_sum);
    __syncthreads();
  }
  grid.sync();
  
  profile_grid_clock[clock_idx * 108 * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;



// Begin of Load value weight pipeline
if(blockIdx.x < 72){
    enum {
        kBlockRowTiles = kBlockRowWarps * kGemmK3WarpRowTiles,
        kBlockColTiles = kBlockColWarps * kGemmK3WarpColTiles,
    };
    half* value = qkv_output + 2 * kSeqLength * kHiddenDim;
    
    half *matrix_a_shared[kStage];
    // Three stage for matrix 
    const size_t query_key_shared_offset = 128*72+128*72;
    matrix_a_shared[0] = all_shared_mem + query_key_shared_offset;
    matrix_a_shared[1] =
        all_shared_mem + query_key_shared_offset +
        kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[2] =
        all_shared_mem + query_key_shared_offset +
        2 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);

    const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
    const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;
    const int batch_stride = kSeqLength / kBlockColTiles / kWmmaN;
    const int batched_id = blockIdx.x / batch_stride;
    const int col_block_id = blockIdx.x % batch_stride;

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

    const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
    int stage = 0;
    int k_loop = 0;

    const int a_dst_stride =
        kLoadAColsPerIter * (kWmmaM * kBlockRowTiles + kInputSkew);
    const int a_src_stride = kLoadAColsPerIter * kHeadSize;

    // Prologue
#pragma unroll
    for (int s = 0; s < kStage - 1; ++s) {
        pipe.producer_acquire();
        half *a_dst_base = matrix_a_shared[(stage + s) % kStage] +
                           threadIdx.x / kLoadALanesPerRow *
                               (kWmmaM * kBlockRowTiles + kInputSkew) +
                           (threadIdx.x & (kLoadALanesPerRow - 1)) *
                               sizeof(float4) / sizeof(half);

        const half *a_src_base = value +
                                 batched_id * kSeqLength * kHeadSize +
                                 ((k_loop + s) * kChunkK * kWmmaK +
                                  threadIdx.x / kLoadALanesPerRow) *
                                     kHeadSize +
                                 (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                     (sizeof(float4) / sizeof(half));

#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
            cuda::memcpy_async(a_dst_base + i * a_dst_stride,
                               a_src_base + i * a_src_stride, shape, pipe);
        }
        pipe.producer_commit();
    }
}// End of Load value weight pipeline

  // Shared memory size: 128*(128+8)*sizeof(half)
  if(blockIdx.x<108){
    enum {
        kBlockRowTiles = kBlockRowWarps * kGemmK2WarpRowTiles,
        kBlockColTiles = kBlockColWarps * kGemmK2WarpColTiles,
    };
    const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
    const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;
    const int batch_stride = (kSeqLength / kBlockColTiles / kWmmaN) *
                             (kSeqLength / kBlockRowTiles / kWmmaM);
    const int batched_id = blockIdx.x / batch_stride;
    const int row_block_id =
        blockIdx.x % batch_stride % (kSeqLength / kBlockRowTiles / kWmmaM);
    const int col_block_id =
        blockIdx.x % batch_stride / (kSeqLength / kBlockRowTiles / kWmmaM);
    enum {
        kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kLoadLanesPerRow = kHeadSize / (sizeof(float4) / sizeof(half)),
        kLoadColsPerIter = kThreads / kLoadLanesPerRow,

        kStoreLanesPerRow =
            kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
        kStoreColsPerIter = kThreads / kStoreLanesPerRow,
    };
    half *acc_shared = all_shared_mem;
    // Do query-key-softmax normalization
    const int kHalf2Vec = 2;
    const int row_shared_size = (kBlockRowTiles * kWmmaM + kAccSkew);
    const int col_shared_size = (kBlockColTiles * kWmmaN);
    float* softmax_sum_base_ptr = query_key_softmax_sum + batched_id * kSeqLength + 
        col_block_id * kBlockColTiles * kWmmaN;
    const int kNormalizePerIter = kThreads * kHalf2Vec / (kBlockRowTiles * kWmmaM);
    const int row_shared_offset = (threadIdx.x >> 6);
    const int idx_offset = (threadIdx.x & 63) * kHalf2Vec;
    float* softmax_shared = (float*)(acc_shared + col_shared_size * row_shared_size);
    // Load to softmax to shared
    softmax_shared[threadIdx.x] = softmax_sum_base_ptr[threadIdx.x];
    __syncthreads();
    for(int i=0; i<kBlockColTiles * kWmmaN / kNormalizePerIter; ++i){
      const int row_shared = i * kNormalizePerIter + row_shared_offset;
      int idx = row_shared * row_shared_size + idx_offset;
    //   float softmax_sum = (softmax_sum_base_ptr + row_shared)[0];
      float softmax_sum = (softmax_shared + row_shared)[0];
      auto attn_value = __half22float2(((half2*)(acc_shared + idx))[0]);
      half2 normalized;
      normalized.x = __float2half(attn_value.x / softmax_sum);
      normalized.y = __float2half(attn_value.y / softmax_sum);
      ((half2*)(acc_shared + idx))[0] = normalized;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreColsPerIter; ++i) {
        #ifndef REWRITE_RESULT_DEBUG
        *reinterpret_cast<float4 *>(
            query_key_output + batched_id * kSeqLength * kSeqLength +
            row_block_id * kBlockRowTiles * kWmmaM +
            (col_block_id * kBlockColTiles * kWmmaN + i * kStoreColsPerIter +
             threadIdx.x / kStoreLanesPerRow) *
                kSeqLength +
            (threadIdx.x & (kStoreLanesPerRow - 1)) * sizeof(float4) /
                sizeof(half)) =
            *reinterpret_cast<float4 *>(
                acc_shared +
                (i * kStoreColsPerIter + threadIdx.x / kStoreLanesPerRow) *
                    (kBlockRowTiles * kWmmaM + kAccSkew) +
                (threadIdx.x & (kStoreLanesPerRow - 1)) * sizeof(float4) /
                    sizeof(half));
        #endif
    }
  }
  /* ------------------------------------------------------------- */
  grid.sync();
  
  profile_grid_clock[clock_idx * 108 * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
  
  // Begin of attn_value
  if(blockIdx.x < 72){
    enum {
        kBlockRowTiles = kBlockRowWarps * kGemmK3WarpRowTiles,
        kBlockColTiles = kBlockColWarps * kGemmK3WarpColTiles,
    };
    half* value = qkv_output + 2 * kSeqLength * kHiddenDim;
    
    // Shared memory useage: [0: 3*64*72*sizeof(half)], [128*72+128*72*sizeof(half), (128*72+128*72 + 3*64*72)*sizeof(half), ], [0, 64*72*sizeof(half)]
    // [0, 27KB], [36KB, 63KB], [0, 9KB]
    half *matrix_a_shared[kStage], *matrix_b_shared[kStage];
    half *acc_shared;
    // Three stage for matrix 
    const size_t query_key_shared_offset = 128*72+128*72;
    matrix_a_shared[0] = all_shared_mem + query_key_shared_offset;
    matrix_a_shared[1] =
        all_shared_mem + query_key_shared_offset +
        kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[2] =
        all_shared_mem + query_key_shared_offset +
        2 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);

    matrix_b_shared[0] =
        all_shared_mem ;
    matrix_b_shared[1] =
        all_shared_mem +
        kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);
    matrix_b_shared[2] =
        all_shared_mem +
        2 * kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);

    acc_shared = all_shared_mem;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::col_major>
        wmma_matrix_a[kGemmK3WarpRowTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::col_major>
        wmma_matrix_b[kGemmK3WarpColTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                           half>
        wmma_accumulator[kGemmK3WarpColTiles * kGemmK3WarpRowTiles];

    const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
    const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;
    const int batch_stride = kSeqLength / kBlockColTiles / kWmmaN;
    const int batched_id = blockIdx.x / batch_stride;
    const int col_block_id = blockIdx.x % batch_stride;

#pragma unroll
    for (int col = 0; col < kGemmK3WarpColTiles; ++col) {
#pragma unroll
        for (int row = 0; row < kGemmK3WarpRowTiles; ++row) {
            nvcuda::wmma::fill_fragment(
                wmma_accumulator[col * kGemmK3WarpRowTiles + row], 0.0f);
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

    const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
    int stage = 0;
    int k_loop = 0;

    const int a_dst_stride =
        kLoadAColsPerIter * (kWmmaM * kBlockRowTiles + kInputSkew);
    const int a_src_stride = kLoadAColsPerIter * kHeadSize;

    const int b_dst_stride =
        kLoadBColsPerIter * (kWmmaK * kChunkK + kInputSkew);
    const int b_src_stride = kLoadBColsPerIter * kSeqLength;


    // Prologue
#pragma unroll
    for (int s = 0; s < kStage - 1; ++s) {
        pipe.producer_acquire();

        half *b_dst_base =
            matrix_b_shared[(stage + s) % kStage] +
            threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
            (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) /
                sizeof(half);

        const half *b_src_base = query_key_output +
                                 batched_id * kSeqLength * kSeqLength +
                                 (k_loop + s) * kChunkK * kWmmaK +
                                 (col_block_id * kBlockColTiles * kWmmaN +
                                  threadIdx.x / kLoadBLanesPerRow) *
                                     kSeqLength +
                                 (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                                     (sizeof(float4) / sizeof(half));
#pragma unroll
        for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
            cuda::memcpy_async(b_dst_base + i * b_dst_stride,
                               b_src_base + i * b_src_stride, shape, pipe);
        }
        pipe.producer_commit();
    }
// Wait for value stage 0 
pipe.consumer_wait();
__syncthreads();
pipe.consumer_release();
// wait for value stage 1
pipe.consumer_wait();
__syncthreads();
pipe.consumer_release();


    int debug_k_count = 0;
    // Soft pipeline
#pragma unroll
    for (; k_loop < (kSeqLength / kChunkK / kWmmaK) - (kStage - 1); ++k_loop) {
        pipe.producer_acquire();

        half *a_dst_base = matrix_a_shared[(stage + kStage - 1) % kStage] +
                           threadIdx.x / kLoadALanesPerRow *
                               (kWmmaM * kBlockRowTiles + kInputSkew) +
                           (threadIdx.x & (kLoadALanesPerRow - 1)) *
                               sizeof(float4) / sizeof(half);

        const half *a_src_base = value +
                                 batched_id * kSeqLength * kHeadSize +
                                 ((k_loop + kStage - 1) * kChunkK * kWmmaK +
                                  threadIdx.x / kLoadALanesPerRow) *
                                     kHeadSize +
                                 (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                     (sizeof(float4) / sizeof(half));

        half *b_dst_base =
            matrix_b_shared[(stage + kStage - 1) % kStage] +
            threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
            (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) /
                sizeof(half);

        const half *b_src_base = query_key_output +
                                 batched_id * kSeqLength * kSeqLength +
                                 (k_loop + kStage - 1) * kChunkK * kWmmaK +
                                 (col_block_id * kBlockColTiles * kWmmaN +
                                  threadIdx.x / kLoadBLanesPerRow) *
                                     kSeqLength +
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
            for (int tile_m = 0; tile_m < kGemmK3WarpRowTiles; ++tile_m) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[tile_m],
                    (matrix_a_shared[stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK3WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
            }
#pragma unroll
            for (int tile_n = 0; tile_n < kGemmK3WarpColTiles; ++tile_n) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_b[tile_n],
                    (matrix_b_shared[stage] +
                     (col_warp_id * kGemmK3WarpColTiles + tile_n) * kWmmaN *
                         (kChunkK * kWmmaK + kInputSkew) +
                     tile_k * kWmmaK),
                    kChunkK * kWmmaK + kInputSkew);
            }
#pragma unroll
            for (int tile_m = 0; tile_m < kGemmK3WarpRowTiles; ++tile_m) {
#pragma unroll
                for (int tile_n = 0; tile_n < kGemmK3WarpColTiles; ++tile_n) {
                    nvcuda::wmma::mma_sync(
                        wmma_accumulator[tile_m + tile_n * kGemmK3WarpRowTiles],
                        wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
                        wmma_accumulator[tile_m +
                                         tile_n * kGemmK3WarpRowTiles]);
                }
            }
        }

        stage = (stage + 1) % kStage;
    }


    // Epilogue
#pragma unroll
    for (int s = kStage - 1; s >= 1; --s) {
        k_loop = (kSeqLength / kChunkK / kWmmaK) - s;
        pipe.consumer_wait();
        __syncthreads();
        pipe.consumer_release();
#pragma unroll
        for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
            for (int tile_m = 0; tile_m < kGemmK3WarpRowTiles; ++tile_m) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[tile_m],
                    (matrix_a_shared[stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK3WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
            }
#pragma unroll
            for (int tile_n = 0; tile_n < kGemmK3WarpColTiles; ++tile_n) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_b[tile_n],
                    (matrix_b_shared[stage] +
                     (col_warp_id * kGemmK3WarpColTiles + tile_n) * kWmmaN *
                         (kChunkK * kWmmaK + kInputSkew) +
                     tile_k * kWmmaK),
                    kChunkK * kWmmaK + kInputSkew);
            }
#pragma unroll
            for (int tile_m = 0; tile_m < kGemmK3WarpRowTiles; ++tile_m) {
#pragma unroll
                for (int tile_n = 0; tile_n < kGemmK3WarpColTiles; ++tile_n) {
                    nvcuda::wmma::mma_sync(
                        wmma_accumulator[tile_m + tile_n * kGemmK3WarpRowTiles],
                        wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
                        wmma_accumulator[tile_m +
                                         tile_n * kGemmK3WarpRowTiles]);
                }
            }
        }
        stage = (stage + 1) % kStage;
    }

#pragma unroll
    for (int tile_n = 0; tile_n < kGemmK3WarpColTiles; ++tile_n) {
#pragma unroll
        for (int tile_m = 0; tile_m < kGemmK3WarpRowTiles; ++tile_m) {
            nvcuda::wmma::store_matrix_sync(
                acc_shared +
                    (col_warp_id * kGemmK3WarpColTiles + tile_n) * kWmmaK *
                        (kBlockRowTiles * kWmmaM + kAccSkew) +
                    (row_warp_id * kGemmK3WarpRowTiles + tile_m) * kWmmaM,
                wmma_accumulator[tile_n * kGemmK3WarpRowTiles + tile_m],
                (kBlockRowTiles * kWmmaM + kAccSkew),
                nvcuda::wmma::mem_col_major);
        }
    }

    __syncthreads();
    const int c_dst_stride = kStoreCColsPerIter * kHiddenDim;
    const int c_src_stride =
        kStoreCColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);

    half *c_dst_base = attn_value_output + batched_id * kHeadSize +
                       (col_block_id * kBlockColTiles * kWmmaN +
                        threadIdx.x / kStoreCLanesPerRow) *
                           kHiddenDim +
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
  }// End of attn_value 
  grid.sync();
  profile_grid_clock[clock_idx * 108 * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
}


__global__ void debug_fused_sqq_bert(const half *__restrict__ qkv_weight, 
                                half *__restrict__ src,
                                const half *__restrict__ qkv_bias,
                                half *__restrict__ qkv_output,
                                half *__restrict__ query_key_output,
                                half *__restrict__ query_key_mask,
                                float * query_key_softmax_sum,
                                half *__restrict__ attn_value_output,
                                half* __restrict__ attn_fc_weight,
                                half* __restrict__ attn_fc_output,
                                float* attn_layer_norm_sum,
                                float* attn_layer_norm_variance,
                                half eps, half h_gama, half h_beta,
                                half* __restrict__ feed_forward_fc1_weight,
                                half* __restrict__ feed_forward_fc1_output,
                                half* __restrict__ feed_forward_fc2_weight,
                                half* __restrict__ feed_forward_fc2_output,
                                float* feed_forward_layernorm_sum,
                                float* feed_forward_layernorm_variance,
                                int64_t* profile_grid_clock,
                                // Pointers from pytorch
                                half* ptr_t_attn_fc_output,
                                half* ptr_t_attn_fc_short_cut_add
                                ){
  using namespace nvcuda;
  extern __shared__ half all_shared_mem[];
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  int clock_idx = 0;
  unsigned int c = 0;
  const int warpIdx = threadIdx.x >> 5;
  
  profile_grid_clock[clock_idx * 108 * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
  
  // Begin of fused QKV matmul
  if(blockIdx.x < 96){
    enum {
        kBlockRowTiles = kBlockRowWarps * kGemmK1WarpRowTiles,
        kBlockColTiles = kBlockColWarps * kGemmK1WarpColTiles,
    };

    half *matrix_a_shared[3][kStage], *matrix_b_shared[kStage];
    half *acc_shared;

    matrix_a_shared[0][0] = all_shared_mem;
    matrix_a_shared[0][1] =
        all_shared_mem +
        kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[0][2] =
        all_shared_mem +
        2 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[1][0] =
        matrix_a_shared[0][0] +
        3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[1][1] =
        matrix_a_shared[0][1] +
        3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[1][2] =
        matrix_a_shared[0][2] +
        3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[2][0] =
        matrix_a_shared[1][0] +
        3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[2][1] =
        matrix_a_shared[1][1] +
        3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[2][2] =
        matrix_a_shared[1][2] +
        3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);

    matrix_b_shared[0] =
        all_shared_mem +
        9 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_b_shared[1] =
        all_shared_mem +
        9 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
        kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);
    matrix_b_shared[2] =
        all_shared_mem +
        9 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
        2 * kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);

    acc_shared = all_shared_mem;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::col_major>
        wmma_matrix_a[3][kGemmK1WarpRowTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::col_major>
        wmma_matrix_b[kGemmK1WarpColTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                           half>
        wmma_accumulator[3][kGemmK1WarpColTiles * kGemmK1WarpRowTiles];

    const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
    const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;
    const int row_block_id =
        blockIdx.x % (kHiddenDim / kBlockRowTiles / kWmmaM);
    const int col_block_id =
        blockIdx.x / (kHiddenDim / kBlockRowTiles / kWmmaM);

#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int col = 0; col < kGemmK1WarpColTiles; ++col) {
#pragma unroll
            for (int row = 0; row < kGemmK1WarpRowTiles; ++row) {
                nvcuda::wmma::fill_fragment(
                    wmma_accumulator[i][col * kGemmK1WarpRowTiles + row], 0.0f);
            }
        }
    }

    enum {
        kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kLoadALanesPerRow =
            kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
        kLoadAColsPerIter = kThreads / kLoadALanesPerRow,

        kLoadBLanesPerRow = kWmmaK * kChunkK / (sizeof(float4) / sizeof(half)),
        kLoadBColsPerIter = kThreads / kLoadBLanesPerRow,

        kAddBiasLanesPerRow =
            kWmmaM * kBlockRowTiles / (sizeof(half2) / sizeof(half)),
        kAddBiasColsPerIter = kThreads / kAddBiasLanesPerRow,

        kStoreCLanesPerRow = kLoadALanesPerRow,
        kStoreCColsPerIter = kLoadAColsPerIter,
    };

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
    int stage = 0;
    int k_loop = 0;

    const int a_dst_stride =
        kLoadAColsPerIter * (kWmmaM * kBlockRowTiles + kInputSkew);
    const int a_src_stride = kLoadAColsPerIter * kHiddenDim;

    const int b_dst_stride =
        kLoadBColsPerIter * (kWmmaK * kChunkK + kInputSkew);
    const int b_src_stride = kLoadBColsPerIter * kHiddenDim;

#pragma unroll
    for (int s = 0; s < kStage - 1; ++s) {
        pipe.producer_acquire();
        half *a_dst_base_0 = matrix_a_shared[0][(stage + s) % kStage] +
                             threadIdx.x / kLoadALanesPerRow *
                                 (kWmmaM * kBlockRowTiles + kInputSkew) +
                             (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                 sizeof(float4) / sizeof(half);
        half *a_dst_base_1 =
            a_dst_base_0 +
            3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
        half *a_dst_base_2 =
            a_dst_base_0 +
            6 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);

        const half *a_src_base_0 = qkv_weight +
                                   row_block_id * kBlockRowTiles * kWmmaM +
                                   ((k_loop + s) * kChunkK * kWmmaK +
                                    threadIdx.x / kLoadALanesPerRow) *
                                       kHiddenDim +
                                   (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                       (sizeof(float4) / sizeof(half));
        const half *a_src_base_1 = a_src_base_0 + kHiddenDim * kHiddenDim;
        const half *a_src_base_2 = a_src_base_1 + kHiddenDim * kHiddenDim;

        half *b_dst_base =
            matrix_b_shared[(stage + s) % kStage] +
            threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
            (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) /
                sizeof(half);

        const half *b_src_base = src + (k_loop + s) * kChunkK * kWmmaK +
                                 (col_block_id * kBlockColTiles * kWmmaN +
                                  threadIdx.x / kLoadBLanesPerRow) *
                                     kHiddenDim +
                                 (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                                     (sizeof(float4) / sizeof(half));

#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
          // printf("a: %f \n", __half2float((a_src_base_0 + i * a_src_stride)[0]));
            cuda::memcpy_async(a_dst_base_0 + i * a_dst_stride,
                               a_src_base_0 + i * a_src_stride, shape, pipe);
        }
#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
            cuda::memcpy_async(a_dst_base_1 + i * a_dst_stride,
                               a_src_base_1 + i * a_src_stride, shape, pipe);
        }
#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
            cuda::memcpy_async(a_dst_base_2 + i * a_dst_stride,
                               a_src_base_2 + i * a_src_stride, shape, pipe);
        }

#pragma unroll
        for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
          // printf("b: %f \n", __half2float((b_src_base + i * b_src_stride)[0]));
            cuda::memcpy_async(b_dst_base + i * b_dst_stride,
                               b_src_base + i * b_src_stride, shape, pipe);
        }
        pipe.producer_commit();
    }

#pragma unroll
    for (; k_loop < (kHiddenDim / kChunkK / kWmmaK) - (kStage - 1); ++k_loop) {
        pipe.producer_acquire();

        half *a_dst_base_0 = matrix_a_shared[0][(stage + kStage - 1) % kStage] +
                             threadIdx.x / kLoadALanesPerRow *
                                 (kWmmaM * kBlockRowTiles + kInputSkew) +
                             (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                 sizeof(float4) / sizeof(half);
        half *a_dst_base_1 =
            a_dst_base_0 +
            3 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
        half *a_dst_base_2 =
            a_dst_base_0 +
            6 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
        const half *a_src_base_0 = qkv_weight +
                                   row_block_id * kBlockRowTiles * kWmmaM +
                                   ((k_loop + kStage - 1) * kChunkK * kWmmaK +
                                    threadIdx.x / kLoadALanesPerRow) *
                                       kHiddenDim +
                                   (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                       (sizeof(float4) / sizeof(half));
        const half *a_src_base_1 = a_src_base_0 + kHiddenDim * kHiddenDim;
        const half *a_src_base_2 = a_src_base_1 + kHiddenDim * kHiddenDim;

        half *b_dst_base =
            matrix_b_shared[(stage + kStage - 1) % kStage] +
            threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
            (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) /
                sizeof(half);

        const half *b_src_base = src +
                                 (k_loop + kStage - 1) * kChunkK * kWmmaK +
                                 (col_block_id * kBlockColTiles * kWmmaN +
                                  threadIdx.x / kLoadBLanesPerRow) *
                                     kHiddenDim +
                                 (threadIdx.x & (kLoadBLanesPerRow - 1)) *
                                     (sizeof(float4) / sizeof(half));

#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
            cuda::memcpy_async(a_dst_base_0 + i * a_dst_stride,
                               a_src_base_0 + i * a_src_stride, shape, pipe);
        }
#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
            cuda::memcpy_async(a_dst_base_1 + i * a_dst_stride,
                               a_src_base_1 + i * a_src_stride, shape, pipe);
        }
#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
            cuda::memcpy_async(a_dst_base_2 + i * a_dst_stride,
                               a_src_base_2 + i * a_src_stride, shape, pipe);
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
            for (int tile_m = 0; tile_m < kGemmK1WarpRowTiles; ++tile_m) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[0][tile_m],
                    (matrix_a_shared[0][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[1][tile_m],
                    (matrix_a_shared[1][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[2][tile_m],
                    (matrix_a_shared[2][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
            }
#pragma unroll
            for (int tile_n = 0; tile_n < kGemmK1WarpColTiles; ++tile_n) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_b[tile_n],
                    (matrix_b_shared[stage] +
                     (col_warp_id * kGemmK1WarpColTiles + tile_n) * kWmmaN *
                         (kChunkK * kWmmaK + kInputSkew) +
                     tile_k * kWmmaK),
                    kChunkK * kWmmaK + kInputSkew);
            }
#pragma unroll
            for (int tile_m = 0; tile_m < kGemmK1WarpRowTiles; ++tile_m) {
#pragma unroll
                for (int i = 0; i < 3; ++i) {
#pragma unroll
                    for (int tile_n = 0; tile_n < kGemmK1WarpColTiles;
                         ++tile_n) {
                        nvcuda::wmma::mma_sync(
                            wmma_accumulator[i][tile_m +
                                                tile_n * kGemmK1WarpRowTiles],
                            wmma_matrix_a[i][tile_m], wmma_matrix_b[tile_n],
                            wmma_accumulator[i][tile_m +
                                                tile_n * kGemmK1WarpRowTiles]);
                    }
                }
            }
        }
        stage = (stage + 1) % kStage;
    }

#pragma unroll
    for (int s = kStage - 1; s >= 1; --s) {
        k_loop = (kHiddenDim / kChunkK / kWmmaK) - s;
        pipe.consumer_wait();
        __syncthreads();
        pipe.consumer_release();

#pragma unroll
        for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
            for (int tile_m = 0; tile_m < kGemmK1WarpRowTiles; ++tile_m) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[0][tile_m],
                    (matrix_a_shared[0][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[1][tile_m],
                    (matrix_a_shared[1][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[2][tile_m],
                    (matrix_a_shared[2][stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
            }
#pragma unroll
            for (int tile_n = 0; tile_n < kGemmK1WarpColTiles; ++tile_n) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_b[tile_n],
                    (matrix_b_shared[stage] +
                     (col_warp_id * kGemmK1WarpColTiles + tile_n) * kWmmaN *
                         (kChunkK * kWmmaK + kInputSkew) +
                     tile_k * kWmmaK),
                    kChunkK * kWmmaK + kInputSkew);
            }
#pragma unroll
            for (int tile_m = 0; tile_m < kGemmK1WarpRowTiles; ++tile_m) {
#pragma unroll
                for (int i = 0; i < 3; ++i) {
#pragma unroll
                    for (int tile_n = 0; tile_n < kGemmK1WarpColTiles;
                         ++tile_n) {
                        nvcuda::wmma::mma_sync(
                            wmma_accumulator[i][tile_m +
                                                tile_n * kGemmK1WarpRowTiles],
                            wmma_matrix_a[i][tile_m], wmma_matrix_b[tile_n],
                            wmma_accumulator[i][tile_m +
                                                tile_n * kGemmK1WarpRowTiles]);
                    }
                }
            }
        }
        stage = (stage + 1) % kStage;
    }

#pragma unroll
    for (int i = 0; i < 3; ++i) {
#pragma unroll
        for (int tile_n = 0; tile_n < kGemmK1WarpColTiles; ++tile_n) {
#pragma unroll
            for (int tile_m = 0; tile_m < kGemmK1WarpRowTiles; ++tile_m) {
                nvcuda::wmma::store_matrix_sync(
                    acc_shared +
                        i * kBlockColTiles * kWmmaN *
                            (kBlockRowTiles * kWmmaM + kAccSkew) +
                        (col_warp_id * kGemmK1WarpColTiles + tile_n) * kWmmaK *
                            (kBlockRowTiles * kWmmaM + kAccSkew) +
                        (row_warp_id * kGemmK1WarpRowTiles + tile_m) * kWmmaM,
                    wmma_accumulator[i][tile_n * kGemmK1WarpRowTiles + tile_m],
                    (kBlockRowTiles * kWmmaM + kAccSkew),
                    nvcuda::wmma::mem_col_major);
            }
        }
    }

    __syncthreads();

    const int bias_stride =
        kAddBiasColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);
    half *bias_dst_base = acc_shared +
                          threadIdx.x / kAddBiasLanesPerRow *
                              (kBlockRowTiles * kWmmaM + kAccSkew) +
                          (threadIdx.x & (kAddBiasLanesPerRow - 1)) *
                              sizeof(half2) / sizeof(half);
    const half *bias_src_base = qkv_bias + row_block_id * kBlockRowTiles * kWmmaM +
                                (threadIdx.x & (kAddBiasLanesPerRow - 1)) *
                                    sizeof(half2) / sizeof(half);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
#pragma unroll
        for (int i = 0; i < kBlockColTiles * kWmmaN / kAddBiasColsPerIter;
             ++i) {
            *reinterpret_cast<half2 *>(
                bias_dst_base +
                j * kBlockColTiles * kWmmaN *
                    (kBlockRowTiles * kWmmaM + kAccSkew) +
                i * bias_stride) +=
                __ldg(reinterpret_cast<const half2 *>(bias_src_base +
                                                      j * kHiddenDim));
        }
    }

    __syncthreads();

    const int c_dst_stride = kStoreCColsPerIter * kHeadSize;
    const int c_src_stride =
        kStoreCColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);

    half *c_dst_base =
        qkv_output +
        (row_block_id / 2) * 2 * kBlockRowTiles * kWmmaM * kSeqLength +
        (row_block_id % 2) * kBlockRowTiles * kWmmaM +
        (col_block_id * kBlockColTiles * kWmmaN +
         threadIdx.x / kStoreCLanesPerRow) *
            kHeadSize +
        (threadIdx.x & (kStoreCLanesPerRow - 1)) * sizeof(float4) /
            sizeof(half);
    half *c_src_base = acc_shared +
                       threadIdx.x / kStoreCLanesPerRow *
                           (kBlockRowTiles * kWmmaM + kAccSkew) +
                       (threadIdx.x & (kStoreCLanesPerRow - 1)) *
                           sizeof(float4) / sizeof(half);

#pragma unroll
    for (int j = 0; j < 3; ++j) {
#pragma unroll
        for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
            *reinterpret_cast<float4 *>(c_dst_base + i * c_dst_stride +
                                        j * kHiddenDim * kSeqLength) =
                *reinterpret_cast<float4 *>(
                    c_src_base + i * c_src_stride +
                    j * kBlockColTiles * kWmmaN *
                        (kBlockRowTiles * kWmmaM + kAccSkew));
        }
    }
  } // End of fused QKV matmul
/* ----------------------------------------------- */
 
  grid.sync();
  
  profile_grid_clock[clock_idx * 108 * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
  // Begin of Query-Key bmm
  if(blockIdx.x < 108){
    const half* __restrict__ query = qkv_output;
    const half* __restrict__ key = query + BertScaleParams::kBatchSize * BertScaleParams::kSeqLength * BertScaleParams::kHiddenDim;
    enum {
        kBlockRowTiles = kBlockRowWarps * kGemmK2WarpRowTiles,
        kBlockColTiles = kBlockColWarps * kGemmK2WarpColTiles,
    };

    half *matrix_a_shared = all_shared_mem;

    half *matrix_b_shared =
        matrix_a_shared + kBlockRowTiles * kWmmaM * (kHeadSize + kInputSkew);

    half *acc_shared = all_shared_mem;

    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::row_major>
        wmma_matrix_a[kGemmK2WarpRowTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::col_major>
        wmma_matrix_b[kGemmK2WarpColTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                           half>
        wmma_accumulator[kGemmK2WarpColTiles * kGemmK2WarpRowTiles];

    const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
    const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;
    const int batch_stride = (kSeqLength / kBlockColTiles / kWmmaN) *
                             (kSeqLength / kBlockRowTiles / kWmmaM);
    const int batched_id = blockIdx.x / batch_stride;
    const int row_block_id =
        blockIdx.x % batch_stride % (kSeqLength / kBlockRowTiles / kWmmaM);
    const int col_block_id =
        blockIdx.x % batch_stride / (kSeqLength / kBlockRowTiles / kWmmaM);

#pragma unroll
    for (int col = 0; col < kGemmK2WarpColTiles; ++col) {
#pragma unroll
        for (int row = 0; row < kGemmK2WarpRowTiles; ++row) {
            nvcuda::wmma::fill_fragment(
                wmma_accumulator[col * kGemmK2WarpRowTiles + row], 0.0f);
        }
    }

    enum {
        kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kLoadLanesPerRow = kHeadSize / (sizeof(float4) / sizeof(half)),
        kLoadColsPerIter = kThreads / kLoadLanesPerRow,

        kStoreLanesPerRow =
            kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
        kStoreColsPerIter = kThreads / kStoreLanesPerRow,
    };

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));

    pipe.producer_acquire();
#pragma unroll
    for (int i = 0; i < kBlockRowTiles * kWmmaM / kLoadColsPerIter; ++i) {
        cuda::memcpy_async(
            reinterpret_cast<float4 *>(
                matrix_a_shared +
                (i * kLoadColsPerIter + threadIdx.x / kLoadLanesPerRow) *
                    (kHeadSize + kInputSkew) +
                (threadIdx.x & (kLoadLanesPerRow - 1)) * sizeof(float4) /
                    sizeof(half)),
            reinterpret_cast<const float4 *>(
                key + batched_id * kSeqLength * kHeadSize +
                (row_block_id * kBlockRowTiles * kWmmaM + i * kLoadColsPerIter +
                 threadIdx.x / kLoadLanesPerRow) *
                    kHeadSize +
                (threadIdx.x & (kLoadLanesPerRow - 1)) *
                    (sizeof(float4) / sizeof(half))),
            shape, pipe);
    }

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadColsPerIter; ++i) {
        cuda::memcpy_async(
            reinterpret_cast<float4 *>(
                matrix_b_shared +
                (i * kLoadColsPerIter + threadIdx.x / kLoadLanesPerRow) *
                    (kHeadSize + kInputSkew) +
                (threadIdx.x & (kLoadLanesPerRow - 1)) * sizeof(float4) /
                    sizeof(half)),
            reinterpret_cast<const float4 *>(
                query + batched_id * kSeqLength * kHeadSize +
                (col_block_id * kBlockColTiles * kWmmaN + i * kLoadColsPerIter +
                 threadIdx.x / kLoadLanesPerRow) *
                    kHeadSize +
                (threadIdx.x & (kLoadLanesPerRow - 1)) *
                    (sizeof(float4) / sizeof(half))),
            shape, pipe);
    }
    pipe.producer_commit();
    pipe.consumer_wait();
    __syncthreads();

#pragma unroll
    for (int tile_k = 0; tile_k < kHeadSize / kWmmaK; ++tile_k) {
#pragma unroll
        for (int tile_m = 0; tile_m < kGemmK2WarpRowTiles; ++tile_m) {
            nvcuda::wmma::load_matrix_sync(
                wmma_matrix_a[tile_m],
                (matrix_a_shared +
                 (row_warp_id * kGemmK2WarpRowTiles + tile_m) * kWmmaM *
                     (kHeadSize + kInputSkew) +
                 tile_k * kWmmaK),
                kHeadSize + kInputSkew);
        }
#pragma unroll
        for (int tile_n = 0; tile_n < kGemmK2WarpColTiles; ++tile_n) {
            nvcuda::wmma::load_matrix_sync(
                wmma_matrix_b[tile_n],
                (matrix_b_shared +
                 (col_warp_id * kGemmK2WarpColTiles + tile_n) * kWmmaN *
                     (kHeadSize + kInputSkew) +
                 tile_k * kWmmaK),
                kHeadSize + kInputSkew);
        }
#pragma unroll
        for (int tile_m = 0; tile_m < kGemmK2WarpRowTiles; ++tile_m) {
#pragma unroll
            for (int tile_n = 0; tile_n < kGemmK2WarpColTiles; ++tile_n) {
                nvcuda::wmma::mma_sync(
                    wmma_accumulator[tile_m + tile_n * kGemmK2WarpRowTiles],
                    wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
                    wmma_accumulator[tile_m + tile_n * kGemmK2WarpRowTiles]);
            }
        }
    }
    pipe.consumer_release();
    __syncthreads();

#pragma unroll
    for (int tile_n = 0; tile_n < kGemmK2WarpColTiles; ++tile_n) {
#pragma unroll
        for (int tile_m = 0; tile_m < kGemmK2WarpRowTiles; ++tile_m) {
            nvcuda::wmma::store_matrix_sync(
                acc_shared +
                    (col_warp_id * kGemmK2WarpColTiles + tile_n) * kWmmaK *
                        (kBlockRowTiles * kWmmaM + kAccSkew) +
                    (row_warp_id * kGemmK2WarpRowTiles + tile_m) * kWmmaM,
                wmma_accumulator[tile_n * kGemmK2WarpRowTiles + tile_m],
                (kBlockRowTiles * kWmmaM + kAccSkew),
                nvcuda::wmma::mem_col_major);
        }
    }

    __syncthreads();
    // Each acc_shared contains (128, 128) elements
    // fused sqrt(hidden_size) + Softmax_reduce_sum
    // Now all values are in shared memory, need to find the layout in shared memory
    // The shared memory uses the same layout as global memory
    const uint64_t attn_mask_base_idx = batched_id * kSeqLength * kSeqLength + 
      (col_block_id * kBlockColTiles * kWmmaN) * kSeqLength + 
      row_block_id * kBlockRowTiles * kWmmaM + threadIdx.x * kSeqLength;
    const int row_shared_size = (kBlockRowTiles * kWmmaM + kAccSkew);
    const int col_shared_size = (kBlockColTiles * kWmmaN);
    float softmax_sum = 0;
    half scale = half(1.0) / hsqrt(__float2half(kHiddenDim));
    half2 scale_h2(scale, scale);
    const int kHalf2Vec = sizeof(half2) / sizeof(half);

    // Now we let one thread to compute half # of elements of the row
    // const int reduce_shared_stride = (threadIdx.x & 63) * row_shared_size + ((threadIdx.x >> 6) << 5);
    const int kSplitNum = 128 / (kBlockColTiles * kWmmaN);
    const int reduce_shared_stride = threadIdx.x * row_shared_size;
    for(int i=0; i<kBlockRowTiles * kWmmaM / kHalf2Vec; ++i){
        int idx = reduce_shared_stride + i * kHalf2Vec;
        auto scaled_acc = ((half2*)(acc_shared + idx))[0] * scale_h2;
        auto mask_h2 = ((half2*)(query_key_mask + attn_mask_base_idx + i * kHalf2Vec))[0];
        auto new_attn_value = h2exp(scaled_acc + mask_h2);
        ((half2*)(acc_shared + idx))[0] = new_attn_value;
        softmax_sum += (__half2float(new_attn_value.x) + __half2float(new_attn_value.y));
    }
    atomicAdd(query_key_softmax_sum + batched_id * kSeqLength + col_block_id *  (kBlockColTiles * kWmmaN) + threadIdx.x, softmax_sum);
    __syncthreads();
  }
  grid.sync();
  
  profile_grid_clock[clock_idx * 108 * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
  if(blockIdx.x<108){
    enum {
        kBlockRowTiles = kBlockRowWarps * kGemmK2WarpRowTiles,
        kBlockColTiles = kBlockColWarps * kGemmK2WarpColTiles,
    };
    const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
    const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;
    const int batch_stride = (kSeqLength / kBlockColTiles / kWmmaN) *
                             (kSeqLength / kBlockRowTiles / kWmmaM);
    const int batched_id = blockIdx.x / batch_stride;
    const int row_block_id =
        blockIdx.x % batch_stride % (kSeqLength / kBlockRowTiles / kWmmaM);
    const int col_block_id =
        blockIdx.x % batch_stride / (kSeqLength / kBlockRowTiles / kWmmaM);
    enum {
        kThreads = kBlockRowWarps * kBlockColWarps * kWarpSize,
        kLoadLanesPerRow = kHeadSize / (sizeof(float4) / sizeof(half)),
        kLoadColsPerIter = kThreads / kLoadLanesPerRow,

        kStoreLanesPerRow =
            kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
        kStoreColsPerIter = kThreads / kStoreLanesPerRow,
    };
    half *acc_shared = all_shared_mem;
    // Do query-key-softmax normalization
    const int kHalf2Vec = 2;
    const int row_shared_size = (kBlockRowTiles * kWmmaM + kAccSkew);
    const int col_shared_size = (kBlockColTiles * kWmmaN);
    float* softmax_sum_base_ptr = query_key_softmax_sum + batched_id * kSeqLength + 
        col_block_id * kBlockColTiles * kWmmaN;
    float* softmax_shared = (float*)(acc_shared + col_shared_size * row_shared_size);
    // Load to softmax to shared
    softmax_shared[threadIdx.x] = 1.0 / softmax_sum_base_ptr[threadIdx.x];
    __syncthreads();
    const int kNormalizePerIter = kThreads * kHalf2Vec / (kBlockRowTiles * kWmmaM);
    for(int i=0; i<kBlockColTiles * kWmmaN / kNormalizePerIter; ++i){
      const int row_shared = i * kNormalizePerIter + (threadIdx.x >> 6);
      int idx = row_shared * row_shared_size + (threadIdx.x & 63) * kHalf2Vec;
      float softmax_sum = (softmax_sum_base_ptr + row_shared)[0];
      auto attn_value = ((half2*)(acc_shared + idx))[0];
      half2 normalized;
      normalized.x = __float2half(__half2float(attn_value.x) * softmax_sum);
      normalized.y = __float2half(__half2float(attn_value.y) * softmax_sum);
      ((half2*)(acc_shared + idx))[0] = normalized;
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreColsPerIter; ++i) {
        *reinterpret_cast<float4 *>(
            query_key_output + batched_id * kSeqLength * kSeqLength +
            row_block_id * kBlockRowTiles * kWmmaM +
            (col_block_id * kBlockColTiles * kWmmaN + i * kStoreColsPerIter +
             threadIdx.x / kStoreLanesPerRow) *
                kSeqLength +
            (threadIdx.x & (kStoreLanesPerRow - 1)) * sizeof(float4) /
                sizeof(half)) =
            *reinterpret_cast<float4 *>(
                acc_shared +
                (i * kStoreColsPerIter + threadIdx.x / kStoreLanesPerRow) *
                    (kBlockRowTiles * kWmmaM + kAccSkew) +
                (threadIdx.x & (kStoreLanesPerRow - 1)) * sizeof(float4) /
                    sizeof(half));
    }
  }
  /* ------------------------------------------------------------- */
  grid.sync();
  
  profile_grid_clock[clock_idx * 108 * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
  
  // Begin of attn_value
  if(blockIdx.x < 72){
    enum {
        kBlockRowTiles = kBlockRowWarps * kGemmK3WarpRowTiles,
        kBlockColTiles = kBlockColWarps * kGemmK3WarpColTiles,
    };
    half* value = qkv_output + 2 * kSeqLength * kHiddenDim;
    
    half *matrix_a_shared[kStage], *matrix_b_shared[kStage];
    half *acc_shared;
    // Three stage for matrix 
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
        wmma_matrix_a[kGemmK3WarpRowTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::col_major>
        wmma_matrix_b[kGemmK3WarpColTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                           half>
        wmma_accumulator[kGemmK3WarpColTiles * kGemmK3WarpRowTiles];

    const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
    const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;
    const int batch_stride = kSeqLength / kBlockColTiles / kWmmaN;
    const int batched_id = blockIdx.x / batch_stride;
    const int col_block_id = blockIdx.x % batch_stride;

#pragma unroll
    for (int col = 0; col < kGemmK3WarpColTiles; ++col) {
#pragma unroll
        for (int row = 0; row < kGemmK3WarpRowTiles; ++row) {
            nvcuda::wmma::fill_fragment(
                wmma_accumulator[col * kGemmK3WarpRowTiles + row], 0.0f);
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
    const int a_src_stride = kLoadAColsPerIter * kHeadSize;

    const int b_dst_stride =
        kLoadBColsPerIter * (kWmmaK * kChunkK + kInputSkew);
    const int b_src_stride = kLoadBColsPerIter * kSeqLength;

    // Prologue
#pragma unroll
    for (int s = 0; s < kStage - 1; ++s) {
        pipe.producer_acquire();
        half *a_dst_base = matrix_a_shared[(stage + s) % kStage] +
                           threadIdx.x / kLoadALanesPerRow *
                               (kWmmaM * kBlockRowTiles + kInputSkew) +
                           (threadIdx.x & (kLoadALanesPerRow - 1)) *
                               sizeof(float4) / sizeof(half);

        const half *a_src_base = value +
                                 batched_id * kSeqLength * kHeadSize +
                                 ((k_loop + s) * kChunkK * kWmmaK +
                                  threadIdx.x / kLoadALanesPerRow) *
                                     kHeadSize +
                                 (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                     (sizeof(float4) / sizeof(half));

        half *b_dst_base =
            matrix_b_shared[(stage + s) % kStage] +
            threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
            (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) /
                sizeof(half);

        const half *b_src_base = query_key_output +
                                 batched_id * kSeqLength * kSeqLength +
                                 (k_loop + s) * kChunkK * kWmmaK +
                                 (col_block_id * kBlockColTiles * kWmmaN +
                                  threadIdx.x / kLoadBLanesPerRow) *
                                     kSeqLength +
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
    for (; k_loop < (kSeqLength / kChunkK / kWmmaK) - (kStage - 1); ++k_loop) {
        pipe.producer_acquire();

        half *a_dst_base = matrix_a_shared[(stage + kStage - 1) % kStage] +
                           threadIdx.x / kLoadALanesPerRow *
                               (kWmmaM * kBlockRowTiles + kInputSkew) +
                           (threadIdx.x & (kLoadALanesPerRow - 1)) *
                               sizeof(float4) / sizeof(half);

        const half *a_src_base = value +
                                 batched_id * kSeqLength * kHeadSize +
                                 ((k_loop + kStage - 1) * kChunkK * kWmmaK +
                                  threadIdx.x / kLoadALanesPerRow) *
                                     kHeadSize +
                                 (threadIdx.x & (kLoadALanesPerRow - 1)) *
                                     (sizeof(float4) / sizeof(half));

        half *b_dst_base =
            matrix_b_shared[(stage + kStage - 1) % kStage] +
            threadIdx.x / kLoadBLanesPerRow * (kWmmaK * kChunkK + kInputSkew) +
            (threadIdx.x & (kLoadBLanesPerRow - 1)) * sizeof(float4) /
                sizeof(half);

        const half *b_src_base = query_key_output +
                                 batched_id * kSeqLength * kSeqLength +
                                 (k_loop + kStage - 1) * kChunkK * kWmmaK +
                                 (col_block_id * kBlockColTiles * kWmmaN +
                                  threadIdx.x / kLoadBLanesPerRow) *
                                     kSeqLength +
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
            for (int tile_m = 0; tile_m < kGemmK3WarpRowTiles; ++tile_m) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[tile_m],
                    (matrix_a_shared[stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK3WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
            }
#pragma unroll
            for (int tile_n = 0; tile_n < kGemmK3WarpColTiles; ++tile_n) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_b[tile_n],
                    (matrix_b_shared[stage] +
                     (col_warp_id * kGemmK3WarpColTiles + tile_n) * kWmmaN *
                         (kChunkK * kWmmaK + kInputSkew) +
                     tile_k * kWmmaK),
                    kChunkK * kWmmaK + kInputSkew);
            }
#pragma unroll
            for (int tile_m = 0; tile_m < kGemmK3WarpRowTiles; ++tile_m) {
#pragma unroll
                for (int tile_n = 0; tile_n < kGemmK3WarpColTiles; ++tile_n) {
                    nvcuda::wmma::mma_sync(
                        wmma_accumulator[tile_m + tile_n * kGemmK3WarpRowTiles],
                        wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
                        wmma_accumulator[tile_m +
                                         tile_n * kGemmK3WarpRowTiles]);
                }
            }
        }
        stage = (stage + 1) % kStage;
    }

    // Epilogue
#pragma unroll
    for (int s = kStage - 1; s >= 1; --s) {
        k_loop = (kSeqLength / kChunkK / kWmmaK) - s;
        pipe.consumer_wait();
        __syncthreads();
        pipe.consumer_release();

#pragma unroll
        for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
            for (int tile_m = 0; tile_m < kGemmK3WarpRowTiles; ++tile_m) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[tile_m],
                    (matrix_a_shared[stage] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kGemmK3WarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
            }
#pragma unroll
            for (int tile_n = 0; tile_n < kGemmK3WarpColTiles; ++tile_n) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_b[tile_n],
                    (matrix_b_shared[stage] +
                     (col_warp_id * kGemmK3WarpColTiles + tile_n) * kWmmaN *
                         (kChunkK * kWmmaK + kInputSkew) +
                     tile_k * kWmmaK),
                    kChunkK * kWmmaK + kInputSkew);
            }
#pragma unroll
            for (int tile_m = 0; tile_m < kGemmK3WarpRowTiles; ++tile_m) {
#pragma unroll
                for (int tile_n = 0; tile_n < kGemmK3WarpColTiles; ++tile_n) {
                    nvcuda::wmma::mma_sync(
                        wmma_accumulator[tile_m + tile_n * kGemmK3WarpRowTiles],
                        wmma_matrix_a[tile_m], wmma_matrix_b[tile_n],
                        wmma_accumulator[tile_m +
                                         tile_n * kGemmK3WarpRowTiles]);
                }
            }
        }
        stage = (stage + 1) % kStage;
    }

#pragma unroll
    for (int tile_n = 0; tile_n < kGemmK3WarpColTiles; ++tile_n) {
#pragma unroll
        for (int tile_m = 0; tile_m < kGemmK3WarpRowTiles; ++tile_m) {
            nvcuda::wmma::store_matrix_sync(
                acc_shared +
                    (col_warp_id * kGemmK3WarpColTiles + tile_n) * kWmmaK *
                        (kBlockRowTiles * kWmmaM + kAccSkew) +
                    (row_warp_id * kGemmK3WarpRowTiles + tile_m) * kWmmaM,
                wmma_accumulator[tile_n * kGemmK3WarpRowTiles + tile_m],
                (kBlockRowTiles * kWmmaM + kAccSkew),
                nvcuda::wmma::mem_col_major);
        }
    }

    __syncthreads();

    const int c_dst_stride = kStoreCColsPerIter * kHiddenDim;
    const int c_src_stride =
        kStoreCColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);

    half *c_dst_base = attn_value_output + batched_id * kHeadSize +
                       (col_block_id * kBlockColTiles * kWmmaN +
                        threadIdx.x / kStoreCLanesPerRow) *
                           kHiddenDim +
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
  }// End of attn_value 
    grid.sync();
    
  profile_grid_clock[clock_idx * 108 * 4 + blockIdx.x * 4 + warpIdx] = clock64(); clock_idx++;
}