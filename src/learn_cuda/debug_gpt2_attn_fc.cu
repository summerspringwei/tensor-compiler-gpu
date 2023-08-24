
#include "cuda_fp16.h"
#include "../cuda_kernel_utils.h"
#include <stdio.h>

#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <mma.h>
#include "../cuda_utils.h"
#include "../operator_fusion/gpt-2/gpt2-large.h"

using namespace souffle::gpt2;

__global__ void attn_fc_layer_norm(
                                const half *__restrict__ matrix_a,
                                const half *__restrict__ matrix_b,
                                const half *__restrict__ input_tensor,
                                float *__restrict__ layer_norm_sum,
                                float *__restrict__ layer_norm_variance,
                                half eps, half gama, half beta,
                                half *__restrict__ matrix_c) {
    using namespace nvcuda;
    using namespace souffle::gpt2::AttnFcParams;
    enum {
        kWarpRowTiles = kGemmK4WarpRowTiles,
        kWarpColTiles = kGemmK4WarpColTiles,
        M = kHeadNum * kHeadSize,
        N = kSeqLength,
        K = kHeadNum * kHeadSize,
        B = 1,
        kBlockRowTiles = kBlockRowWarps * kWarpRowTiles,
        kBlockColTiles = kBlockColWarps * kWarpColTiles,
    };

    extern __shared__ half all_shared_mem[];

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

        const half *a_src_base = matrix_a + batched_id * K * M +
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

        const half *b_src_base = matrix_b + batched_id * N * K +
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

        const half *a_src_base = matrix_a + batched_id * K * M +
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

        const half *b_src_base = matrix_b + batched_id * N * K +
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

    half *c_dst_base = matrix_c + batched_id * N * M +
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
    const half *input_src_base = input_tensor + batched_id * N * M +
                       row_block_id * kBlockRowTiles * kWmmaM +
                       (col_block_id * kBlockColTiles * kWmmaN +
                        threadIdx.x / kStoreCLanesPerRow) *
                           M +
                       (threadIdx.x & (kStoreCLanesPerRow - 1)) *
                           sizeof(float4) / sizeof(half);
    // First do short cut add
    #pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
        half8 c_dst_half8 = *reinterpret_cast<half8 *>(c_src_base + i * c_src_stride);
        const half8 input_half8 = *reinterpret_cast<const half8 *>(input_src_base + i * c_dst_stride);
        *(half2*)&(c_dst_half8.data[0]) = *(half2*)&(c_dst_half8.data[0]) + *(half2*)&(input_half8.data[0]);
        *(half2*)&(c_dst_half8.data[2]) = *(half2*)&(c_dst_half8.data[2]) + *(half2*)&(input_half8.data[2]);
        *(half2*)&(c_dst_half8.data[4]) = *(half2*)&(c_dst_half8.data[4]) + *(half2*)&(input_half8.data[4]);
        *(half2*)&(c_dst_half8.data[6]) = *(half2*)&(c_dst_half8.data[6]) + *(half2*)&(input_half8.data[6]);
        *reinterpret_cast<half8 *>(c_src_base + i * c_src_stride) = c_dst_half8;
    }
    __syncthreads();
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    const int laneIdx = threadIdx.x % kWarpSize;
    const int warpIdx = threadIdx.x >> 5; // threadIdx.x / 32
    const int warpNum = blockDim.x >> 5; // blockDim.x / 128
    const int vecLength = sizeof(half2) / sizeof(half);
    // Do layer norm sum
    {
        #pragma unroll
        for(int i=0; i< kBlockColTiles * kWmmaN / warpNum; ++i){
            const int shared_row = (i * warpNum + warpIdx);
            const int global_row = batched_id * N + col_block_id * kBlockColTiles * kWmmaN + shared_row;
            half2 local_sum(0.0, 0.0);
            #pragma unroll
            for(int j=0; j<kBlockRowTiles * kWmmaM; j += (warpSize * vecLength)){
                const int shared_col = j + laneIdx * vecLength;
                local_sum += *(half2*)(acc_shared + shared_row * (kBlockRowTiles * kWmmaM + kAccSkew) + shared_col);
            }
            half sum = local_sum.x + local_sum.y;
            sum = warpReduceSum(sum);
            if(laneIdx == 0){
                atomicAdd(layer_norm_sum + global_row, __half2float(sum));
            }
        }
    }
    grid.sync();
    // Do layer norm variance
    {
        #pragma unroll
        for(int i=0; i< kBlockColTiles * kWmmaN / warpNum; ++i){
            const int shared_row = (i * warpNum + warpIdx);
            const int global_row = batched_id * N + col_block_id * kBlockColTiles * kWmmaN + shared_row;
            const half2 mean = __float2half2_rn(layer_norm_sum[global_row] / kHiddenDim);
            half2 local_sum(0.0, 0.0);
            // Loop along the row
            #pragma unroll
            for(int j=0; j<kBlockRowTiles * kWmmaM; j += (warpSize * vecLength)){
                const int shared_col = j + laneIdx * vecLength;
                half2 variance = *(half2*)(acc_shared + shared_row * (kBlockRowTiles * kWmmaM + kAccSkew) + shared_col) - mean;
                local_sum += (variance * variance);
            }
            half sum = local_sum.x + local_sum.y;
            sum = warpReduceSum(sum);
            if(laneIdx == 0){
                atomicAdd(layer_norm_variance + global_row, __half2float(sum));
            }
        }
    }
    grid.sync();
    // Do normalization
    {
        #pragma unroll
        for(int i=0; i< kBlockColTiles * kWmmaN / warpNum; ++i){
            const int shared_row = (i * warpNum + warpIdx);
            const int global_row = batched_id * N + col_block_id * kBlockColTiles * kWmmaN + shared_row;
            const half2 mean = __float2half2_rn(layer_norm_sum[global_row] / kHiddenDim);
            const half2 variance_mean = __float2half2_rn(sqrtf(layer_norm_variance[global_row] / kHiddenDim + __half2float(eps)));
            // if(blockIdx.x==0){
            //     printf("%f %f\n", __half2float(mean.x), __half2float(variance_mean.x));
            // }
            // Loop along the row
            #pragma unroll
            for(int j=0; j<kBlockRowTiles * kWmmaM; j += (warpSize * vecLength)){
                const int shared_col = j + laneIdx * vecLength;
                half2 tmp = *(half2*)(acc_shared + shared_row * (kBlockRowTiles * kWmmaM + kAccSkew) + shared_col);
                tmp = (tmp - mean) / variance_mean * half2(gama, gama) + half2(beta, beta);
                *(half2*)(acc_shared + shared_row * (kBlockRowTiles * kWmmaM + kAccSkew) + shared_col) = tmp;
            }
        }
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
        *reinterpret_cast<float4 *>(c_dst_base + i * c_dst_stride) =
            *reinterpret_cast<float4 *>(c_src_base + i * c_src_stride);
    }
}


int main(int argc, char* argv[]) {
  const int batch_size=1;
  const int max_seq_length = 384;
  const int num_head = 20;
  const int num_hidden = 64;
  const int d_model = 1280;
  const int input_size = max_seq_length * d_model;
  const int weight_size = d_model *d_model;
  const int layer_norm_size = max_seq_length;
  const int output_size = max_seq_length * d_model;

    half* input = (half*)malloc(input_size * sizeof(half));
    half* short_cut = (half*)malloc(input_size * sizeof(half));
    half* weight = (half*)malloc(weight_size * sizeof(half));
    half* bias = (half*)malloc(output_size * sizeof(half));
    float* layer_norm_sum = (float*)malloc(layer_norm_size * sizeof(half));
    float* layer_norm_variance = (float*)malloc(layer_norm_size * sizeof(half));
    half* output = (half*)malloc(output_size * sizeof(half));

    half* d_input;
    half* d_short_cut;
    half* d_weight;
    half* d_bias;
    float* d_layer_norm_sum;
    float* d_layer_norm_variance;
    half* d_output;

    cudaMalloc(&d_input, input_size * sizeof(half));
    cudaMalloc(&d_short_cut, input_size * sizeof(half));
    cudaMalloc(&d_weight, weight_size * sizeof(half));
    cudaMalloc(&d_bias, output_size * sizeof(half));
    cudaMalloc(&d_layer_norm_sum, layer_norm_size * sizeof(float));
    cudaMalloc(&d_layer_norm_variance, layer_norm_size * sizeof(float));
    cudaMalloc(&d_output, output_size * sizeof(half));

    for(int i = 0; i < input_size; i++){
        int row = i / d_model;
        int col = i % d_model;
        input[i] = __float2half(1.0/32*(row%8));
        short_cut[i] = 0.0;
    }
    for(int i = 0; i < weight_size; i++){
        int b = i / d_model / d_model;
        int row = (i % (d_model * d_model)) / d_model;
        int col = i % d_model;
        weight[i] = __float2half(1.0/32 * (col%8));
    }
    for(int i = 0; i < output_size; i++){
        bias[i] = __float2half(0.0);
        output[i] = __float2half(0.0);
    }
    for(int i=0; i<layer_norm_size; ++i){
        layer_norm_sum[i] = __float2half(0.0);
        layer_norm_variance[i] = __float2half(0.0);
    }

    cudaMemcpy(d_input, input, input_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_short_cut, short_cut, input_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, weight_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, output_size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_norm_sum, layer_norm_sum, layer_norm_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_norm_variance, layer_norm_variance, layer_norm_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, output_size * sizeof(half), cudaMemcpyHostToDevice);
    half eps = 0.00001, gama = 1, beta = 0;
    void* args[] = {
        (void*)&d_weight, (void*)&d_input, (void*)&d_short_cut,
        (void*)&d_layer_norm_sum, (void*)&d_layer_norm_variance, 
        (void*)&eps, (void*)&gama, (void*)&beta, (void**)&d_output
    };
  checkCuda(cudaFuncSetAttribute((void*)attn_fc_layer_norm, 
            cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, AttnFcParams::kSharedMemory));
  checkCuda(cudaLaunchCooperativeKernel((void*)attn_fc_layer_norm,
      dim3(AttnFcParams::kGridBlocks, 1, 1), dim3(AttnFcParams::kBlockThreads, 1, 1), 
      args, AttnFcParams::kSharedMemory), __LINE__);
    cudaDeviceSynchronize();
    cudaMemcpy(output, d_output, output_size * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(layer_norm_sum, d_layer_norm_sum, layer_norm_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(layer_norm_variance, d_layer_norm_variance, layer_norm_size * sizeof(float), cudaMemcpyDeviceToHost);
    // for(int i = 0; i < output_size; i++){
    //     if(__half2float(output[i]) != 80.0){
    //         printf("error: <%d, %d> %f\n", i/kSeqLength, i%kSeqLength, __half2float(output[i]));
    //     }
    // }
    for(int i=0; i<10; ++i){
        printf("%f ", layer_norm_sum[i]);
    }printf("\n");
    for(int i=0; i<10; ++i){
        printf("%f ", layer_norm_variance[i]);
    }printf("\n");
    for(int r = 0; r< 32; ++r){
        for(int c=0; c<32; ++c){
            printf("%.2f ", __half2float(output[r*d_model+c]));
        }printf("\n");
    }

    cudaFree(d_input);
    cudaFree(d_short_cut);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_layer_norm_sum);
    cudaFree(d_layer_norm_variance);
    cudaFree(d_output);

}