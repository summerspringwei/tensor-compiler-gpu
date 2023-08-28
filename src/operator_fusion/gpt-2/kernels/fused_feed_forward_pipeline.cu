#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <mma.h>

#include "../gpt2-large.h"
#include "../../../cuda_kernel_utils.h"

using namespace souffle::gpt2;

__global__ void fused_feed_forwad_pipeline_kernel(
        half* __restrict__ residual,
        half* __restrict__ input_tensor,
        half eps, half gama, half beta,
        half* __restrict__ feed_forward_fc1_weight,
        half* __restrict__ feed_forward_fc1_output,
        half* __restrict__ feed_forward_fc2_weight,
        half* __restrict__ feed_forward_fc2_output,
        half* feed_forward_layernorm_sum,
        half* feed_forward_layernorm_variance,
        half* __restrict__ next_attn_layer_norm){
    using namespace nvcuda;
    extern __shared__ half all_shared_mem[];
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));

    if(blockIdx.x < FeedForwardFC1LimitedBlocksParams::kGridBlocks){
        using namespace souffle::gpt2::FeedForwardFC1LimitedBlocksParams;
        const int M = FeedForwardFC1LimitedBlocksParams::KGEMMFFM;
        const int N = FeedForwardFC1LimitedBlocksParams::KGEMMFFN;
        const int K = FeedForwardFC1LimitedBlocksParams::KGEMMFFK;
        const int B = FeedForwardFC1LimitedBlocksParams::kGEMMFFB;

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
        // Declare tensor core matrics
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,
                            nvcuda::wmma::col_major>
            wmma_matrix_a[kWarpRowTiles];
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
                            nvcuda::wmma::col_major>
            wmma_matrix_b[kWarpColTiles];
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                            half>
            wmma_accumulator[kWarpColTiles * kWarpRowTiles];
        
        // Tile the matrix to reduce the number of blocks required 
        // to meet the constraint of global synchronization
    for(int limit_tile_m = 0; limit_tile_m < kMTiles; ++ limit_tile_m){

        for(int limit_tile_n=0; limit_tile_n < kNTiles; ++ limit_tile_n){
        // 3. 
        const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
        const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;
        const int batch_stride =
            ((N / kNTiles) / kBlockColTiles / kWmmaN) * ((M / kMTiles) / kBlockRowTiles / kWmmaM);
        const int batched_id = blockIdx.x / batch_stride;
        const int row_block_id =
            blockIdx.x % batch_stride % ((M / kMTiles) / kBlockRowTiles / kWmmaM) + 
            limit_tile_m * ((M / kMTiles) / kBlockRowTiles / kWmmaM);
        const int col_block_id =
            blockIdx.x % batch_stride / ((M / kMTiles) / kBlockRowTiles / kWmmaM) + 
            limit_tile_n * ((N / kNTiles) / kBlockColTiles / kWmmaN);

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

            const half *b_src_base = input_tensor + batched_id * N * K +
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

            const half *b_src_base = input_tensor + batched_id * N * K +
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

        // Pipeline fc2 load weight
        if(limit_tile_m ==1 && blockIdx.x < FeedForwardFC2Params::kGridBlocks){
            using namespace souffle::gpt2::FeedForwardFC2Params;
            
            const int slicek_warp_id = threadIdx.x / kWarpSize;
            const int row_block_id =
                blockIdx.x % (kHiddenDim / kGemmK6BlockRowTiles / kWmmaM);
            const int col_block_id =
                blockIdx.x / (kHiddenDim / kGemmK6BlockRowTiles / kWmmaM);
            
            half *matrix_a_shared[kStage], *matrix_b_shared[kStage];
            matrix_a_shared[0] = all_shared_mem;
            matrix_a_shared[1] =
                all_shared_mem + kGemmK6BlockSliceKTiles * kWmmaK *
                                    (kGemmK6BlockRowTiles * kWmmaM + kInputSkew);
            matrix_a_shared[2] =
                all_shared_mem + 2 * kGemmK6BlockSliceKTiles * kWmmaK *
                                    (kGemmK6BlockRowTiles * kWmmaM + kInputSkew);
            
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

                    #pragma unroll
                for (int i = 0;
                    i < kGemmK6BlockSliceKTiles * kWmmaK / kLoadAColsPerIter; ++i) {
                    cuda::memcpy_async(a_dst_base + i * a_dst_stride,
                                    a_src_base + i * a_src_stride, shape, pipe);
                }
                pipe.producer_commit();
            }
        }// end of fc2 load weight

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
        // Do activation (Relu)
        // const int kLoadHalf2PerIter = sizeof(half2) / sizeof(half) * kBlockThreads;
        // const int kLoadRowsPerIter = kBlockColTiles * kWmmaM / kLoadHalf2PerIter;
        const int numWarp = FeedForwardFC1LimitedBlocksParams::kBlockThreads / warpSize;
        const int warpIdx = threadIdx.x / warpSize;
        const int laneIdx = threadIdx.x % 32;
        const int vecLength = sizeof(half2) / sizeof(half);
        // Each warp process one line
        for(int iter=0; iter < kBlockColTiles * kWmmaN / numWarp; ++ iter){
            const int offset = ((iter*numWarp) + warpIdx) * (kBlockRowTiles * kWmmaM + kAccSkew) + laneIdx * vecLength;
            #pragma unroll
            for(int warp_iter = 0; warp_iter < kBlockRowTiles * kWmmaM / warpSize / vecLength; ++warp_iter){
                const int idx = offset + (warp_iter * warpSize) * vecLength;
                half2 ele = *(half2*)&(acc_shared[idx]);
                if(ele.x < half(0.0)) ele.x = half(0.0);
                if(ele.y < half(0.0)) ele.y = half(0.0);
                ((half2*)&(acc_shared[idx]))[0] = ele;
            }
        }
        __syncthreads();
        const int c_dst_stride = kStoreCColsPerIter * M;
        const int c_src_stride =
            kStoreCColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);

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
        }

        // Sync load fc2's weight
        if(limit_tile_m ==1 && blockIdx.x < FeedForwardFC2Params::kGridBlocks){
            for (int s = 0; s < kStage - 1; ++s) {
                pipe.consumer_wait();
                __syncthreads();
                pipe.consumer_release();
            }
        }// end of sync load fc2's weight

    }// end of tileM
}

grid.sync();

if(blockIdx.x < FeedForwardFC2Params::kGridBlocks){
        using namespace souffle::gpt2::FeedForwardFC2Params;
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
        // Do reduction across K dimension on shared memory
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
        
        const int c_dst_stride = kStoreCColsPerIter * kHiddenDim;
        const int c_src_stride =
            kStoreCColsPerIter * (kGemmK6BlockRowTiles * kWmmaM + kAccSkew);

        half *c_dst_base = feed_forward_fc2_output + row_block_id * kGemmK6BlockRowTiles * kWmmaM +
                        (col_block_id * kGemmK6BlockColTiles * kWmmaN +
                            threadIdx.x / kStoreCLanesPerRow) *
                            kHiddenDim +
                        (threadIdx.x & (kStoreCLanesPerRow - 1)) *
                            sizeof(float4) / sizeof(half);
        half *c_residual_base = residual + row_block_id * kGemmK6BlockRowTiles * kWmmaM +
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
    half8 tmp_fc2;
    half8 tmp_residual;
    half8 out;
    #pragma unroll
        for (int i = 0; i < kGemmK6BlockColTiles * kWmmaN / kStoreCColsPerIter;
            ++i) {
            // Short cut add here
            *(float4*)&tmp_fc2 = *(float4 *)(c_src_base + i * c_src_stride);
            *(float4*)&tmp_residual = *(float4 *)(c_residual_base + i * c_dst_stride);
            *(half2*)&(out.data[0]) = __hadd2(*(half2*)&(tmp_fc2.data[0]), *(half2*)&(tmp_residual.data[0]));
            *(half2*)&(out.data[2]) = __hadd2(*(half2*)&(tmp_fc2.data[2]), *(half2*)&(tmp_residual.data[2]));
            *(half2*)&(out.data[4]) = __hadd2(*(half2*)&(tmp_fc2.data[4]), *(half2*)&(tmp_residual.data[4]));
            *(half2*)&(out.data[6]) = __hadd2(*(half2*)&(tmp_fc2.data[6]), *(half2*)&(tmp_residual.data[6]));
            *reinterpret_cast<float4 *>(c_dst_base + i * c_dst_stride) = *(float4*)&out;
        }
    }
    // Compute Layer Norm and save results to another buffer
    // We reuse the tile size of the fc2 output,
    // shared memory shape:
    grid.sync();
    // 1. We first test naive layer norm without shared memory
    half* input_buf = all_shared_mem;
    if(blockIdx.x < FeedForwardFC1LimitedBlocksParams::kGridBlocks){
        const int warpIdx = threadIdx.x / warpSize;
        const int laneIdx = threadIdx.x % warpSize;
        const int numWarp = blockDim.x * blockDim.y * blockDim.z / warpSize;
        const int vecLength = sizeof(float4) / sizeof(half);
        // 1. Load data from global memory to shared memory
    for(int b=0; b<UPDIV(kBatchSize * kSeqLength, FeedForwardFC1LimitedBlocksParams::kGridBlocks * numWarp); ++b){
        const int row_idx = (b * FeedForwardFC1LimitedBlocksParams::kGridBlocks + blockIdx.x) * numWarp + warpIdx;
        const int shared_row_idx = warpIdx;
        float thread_local_sum = 0;
        half8 tmp;
        if(row_idx < kBatchSize * kSeqLength){
            // Iterate through the reduce dim for a warp
            #pragma unroll
            for(int k = 0; k<UPDIV(kHiddenDim, warpSize * vecLength); ++k){
                const int col_idx = (k * warpSize + laneIdx) * vecLength;
                half2 iter_sum(0.0, 0.0);
                if(col_idx < kHiddenDim){
                    *(float4*)&tmp = *(float4*)&(feed_forward_fc2_output[row_idx * kHiddenDim + col_idx]);
                    *(float4*)&(input_buf[shared_row_idx * kHiddenDim + col_idx]) = *(float4*)&tmp;
                    iter_sum += half2(tmp.data[0], tmp.data[1]);
                    iter_sum += half2(tmp.data[2], tmp.data[3]);
                    iter_sum += half2(tmp.data[4], tmp.data[5]);
                    iter_sum += half2(tmp.data[6], tmp.data[7]);
                    thread_local_sum +=  __half2float(iter_sum.x + iter_sum.y);
                }
            }
            thread_local_sum = warpReduceSum(thread_local_sum);
            if(laneIdx == 0){
                feed_forward_layernorm_sum[row_idx] = __float2half(thread_local_sum / kHiddenDim);
            }
            __syncthreads();
            // 2. Compute the variance
            // Iterate through the reduce dim for a warp
            half average = feed_forward_layernorm_sum[row_idx];
            half2 h2_average(average, average);
            thread_local_sum = 0;
            for(int k = 0; k<UPDIV(kHiddenDim, warpSize * vecLength); ++k){
                const int col_idx = (k * warpSize + laneIdx) * vecLength;
                half2 iter_sum(0.0, 0.0);
                #pragma unroll
                if(col_idx < kHiddenDim){
                    *(float4*)&tmp = *(float4*)&(input_buf[shared_row_idx * kHiddenDim + col_idx]);
                    half2 sub = half2(tmp.data[0], tmp.data[1]) - h2_average;
                    iter_sum += sub * sub;
                    sub = half2(tmp.data[2], tmp.data[3]) - h2_average;
                    iter_sum += sub * sub;
                    sub = half2(tmp.data[4], tmp.data[5]) - h2_average;
                    iter_sum += sub * sub;
                    sub = half2(tmp.data[6], tmp.data[7]) - h2_average;
                    iter_sum += sub * sub;
                    thread_local_sum +=  __half2float(iter_sum.x + iter_sum.y);
                }
            }
            thread_local_sum = warpReduceSum(thread_local_sum);
            if(laneIdx == 0){
                feed_forward_layernorm_variance[row_idx] =  __float2half(sqrtf(thread_local_sum / kHiddenDim + __half2float(eps))) ;
            }
            __syncthreads();
            // 3. Compute the layer norm
            half8 tmp;
            half variance = feed_forward_layernorm_variance[row_idx];
            half2 h2_variance(variance, variance);
            thread_local_sum = 0;
            #pragma unroll
            for(int k = 0; k<UPDIV(kHiddenDim, warpSize * vecLength); ++k){
                const int col_idx = (k * warpSize + laneIdx) * vecLength;
                half2 iter_sum(0.0, 0.0);
                if(col_idx < kHiddenDim){
                    *(float4*)&tmp = *(float4*)&(input_buf[shared_row_idx * kHiddenDim + col_idx]);
                    half8 result;
                    *(half2*)&(result.data[0]) = (half2(tmp.data[0], tmp.data[1]) - h2_average) / h2_variance * half2(gama, gama) + half2(beta, beta);
                    *(half2*)&(result.data[2]) = (half2(tmp.data[2], tmp.data[3]) - h2_average) / h2_variance * half2(gama, gama) + half2(beta, beta);
                    *(half2*)&(result.data[4]) = (half2(tmp.data[4], tmp.data[5]) - h2_average) / h2_variance * half2(gama, gama) + half2(beta, beta);
                    *(half2*)&(result.data[6]) = (half2(tmp.data[6], tmp.data[7]) - h2_average) / h2_variance * half2(gama, gama) + half2(beta, beta);
                    *(float4*)&(next_attn_layer_norm[row_idx * kHiddenDim + col_idx]) = *(float4*)&result;
                }
            }
        }
    }
    }
}
