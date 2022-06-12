#pragma once
#include "base.h"
#include <cooperative_groups.h>
#include <cuda/pipeline>
#include <mma.h>

template <int kChunkK, int kBlockRowWarps, int kBlockColWarps,
          int kWarpRowTiles, int kWarpColTiles, int kInputSkew, int kAccSkew,
          int M, int N, int K>
__global__ void tvm_gemm(const half *__restrict__ matrix_a,
                         const half *__restrict__ matrix_b,
                         half *__restrict__ matrix_c) {
    using namespace nvcuda;
    enum {
        kBlockRowTiles = kBlockRowWarps * kWarpRowTiles,
        kBlockColTiles = kBlockColWarps * kWarpColTiles,
    };

    extern __shared__ half all_shared_mem[];

    half *matrix_a_shared = all_shared_mem;

    half *matrix_b_shared =
        matrix_a_shared +
        kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);

    half *acc_shared = all_shared_mem;

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
            kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)) >=
                    kWarpSize
                ? kWarpSize
                : kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
        kLoadAColsPerIter = kThreads / kLoadALanesPerRow,
        kLoadAInnerLoop = kWmmaM * kBlockRowTiles /
                          (sizeof(float4) / sizeof(half) * kLoadALanesPerRow),

        kLoadBLanesPerRow =
            kWmmaK * kChunkK / (sizeof(float4) / sizeof(half)) >= kWarpSize
                ? kWarpSize
                : kWmmaK * kChunkK / (sizeof(float4) / sizeof(half)),
        kLoadBColsPerIter = kThreads / kLoadBLanesPerRow,
        kLoadBInnerLoop = kWmmaK * kChunkK /
                          (sizeof(float4) / sizeof(half) * kLoadBLanesPerRow),

        kStoreCLanesPerRow = kLoadALanesPerRow,
        kStoreCColsPerIter = kLoadAColsPerIter,
        kStoreCInnerLoop = kLoadAInnerLoop,
    };

    static_assert(kWmmaK * kChunkK % kLoadAColsPerIter == 0);
    static_assert(kWmmaN * kBlockColTiles % kStoreCColsPerIter == 0);
    static_assert(kWmmaN * kBlockColTiles % kLoadBColsPerIter == 0);
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
#pragma unroll
    for (int k_loop = 0; k_loop < (K / kChunkK / kWmmaK); ++k_loop) {
        pipe.producer_acquire();
#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
#pragma unroll
            for (int j = 0; j < kLoadAInnerLoop; ++j) {
                cuda::memcpy_async(
                    reinterpret_cast<float4 *>(
                        matrix_a_shared +
                        (i * kLoadAColsPerIter +
                         threadIdx.x / kLoadALanesPerRow) *
                            (kWmmaM * kBlockRowTiles + kInputSkew) +
                        ((threadIdx.x & (kLoadALanesPerRow - 1)) +
                         j * kLoadALanesPerRow) *
                            sizeof(float4) / sizeof(half)),
                    reinterpret_cast<const float4 *>(
                        matrix_a + blockIdx.z * K * M +
                        blockIdx.x * kBlockRowTiles * kWmmaM +
                        (k_loop * kChunkK * kWmmaK + i * kLoadAColsPerIter +
                         threadIdx.x / kLoadALanesPerRow) *
                            M +
                        ((threadIdx.x & (kLoadALanesPerRow - 1)) +
                         j * kLoadALanesPerRow) *
                            (sizeof(float4) / sizeof(half))),
                    shape, pipe);
            }
        }

#pragma unroll
        for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
#pragma unroll
            for (int j = 0; j < kLoadBInnerLoop; ++j) {
                cuda::memcpy_async(
                    reinterpret_cast<float4 *>(
                        matrix_b_shared +
                        (i * kLoadBColsPerIter +
                         threadIdx.x / kLoadBLanesPerRow) *
                            (kWmmaK * kChunkK + kInputSkew) +
                        ((threadIdx.x & (kLoadBLanesPerRow - 1)) +
                         j * kLoadBLanesPerRow) *
                            sizeof(float4) / sizeof(half)),
                    reinterpret_cast<const float4 *>(
                        matrix_b + blockIdx.z * N * K +
                        k_loop * kChunkK * kWmmaK +
                        (blockIdx.y * kBlockColTiles * kWmmaN +
                         i * kLoadBColsPerIter +
                         threadIdx.x / kLoadBLanesPerRow) *
                            K +
                        ((threadIdx.x & (kLoadBLanesPerRow - 1)) +
                         j * kLoadBLanesPerRow) *
                            (sizeof(float4) / sizeof(half))),
                    shape, pipe);
            }
        }
        pipe.producer_commit();
        pipe.consumer_wait();
        __syncthreads();

#pragma unroll
        for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
            for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[tile_m],
                    (matrix_a_shared +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
            }
#pragma unroll
            for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_b[tile_n],
                    (matrix_b_shared +
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
        pipe.consumer_release();
        __syncthreads();
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
#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
#pragma unroll
        for (int j = 0; j < kStoreCInnerLoop; ++j) {
            *reinterpret_cast<float4 *>(
                matrix_c + blockIdx.z * N * M +
                blockIdx.x * kBlockRowTiles * kWmmaM +
                (blockIdx.y * kBlockColTiles * kWmmaN + i * kStoreCColsPerIter +
                 threadIdx.x / kStoreCLanesPerRow) *
                    M +
                ((threadIdx.x & (kStoreCLanesPerRow - 1)) +
                 j * kStoreCLanesPerRow) *
                    sizeof(float4) / sizeof(half)) =
                *reinterpret_cast<float4 *>(
                    acc_shared +
                    (i * kStoreCColsPerIter +
                     threadIdx.x / kStoreCLanesPerRow) *
                        (kBlockRowTiles * kWmmaM + kAccSkew) +
                    ((threadIdx.x & (kStoreCLanesPerRow - 1)) +
                     j * kStoreCLanesPerRow) *
                        sizeof(float4) / sizeof(half));
        }
    }
}

template <int kChunkK, int kBlockRowWarps, int kBlockColWarps,
          int kWarpRowTiles, int kWarpColTiles, int kInputSkew, int kAccSkew,
          int M, int N, int K>
__global__ void tvm_gemm_two_stage(const half *__restrict__ matrix_a,
                                   const half *__restrict__ matrix_b,
                                   half *__restrict__ matrix_c) {
    using namespace nvcuda;
    enum {
        kBlockRowTiles = kBlockRowWarps * kWarpRowTiles,
        kBlockColTiles = kBlockColWarps * kWarpColTiles,
    };

    extern __shared__ half all_shared_mem[];

    half *matrix_a_shared[2], *matrix_b_shared[2];
    half *acc_shared;

    matrix_a_shared[0] = all_shared_mem;
    matrix_a_shared[1] =
        all_shared_mem +
        kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);

    matrix_b_shared[0] =
        all_shared_mem +
        2 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);

    matrix_b_shared[1] =
        all_shared_mem +
        2 * kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
        kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew);

    acc_shared =
        all_shared_mem +
        2 * (kChunkK * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
             kBlockColTiles * kWmmaN * (kChunkK * kWmmaK + kInputSkew));

    // The entire execution is serialized as :
    // (1) LG (Load global to shmem),
    // (2) LS (Load shmem to reg),
    // (3) C  (Compute the gemm),
    // Thus, the soft pipeline is performed as :
    // Prologue: LG(0), LS(0), LG(1)
    // Pipeline: C(0), LS(1), LG(2); ... ; C(n), LS(n+1), LG(n+2); ...
    // Epilogue: C(N-1), LS(N), C(N)
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::col_major>
        wmma_matrix_a[kChunkK][kWarpRowTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::col_major>
        wmma_matrix_b[kChunkK][kWarpColTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                           half>
        wmma_accumulator[kWarpColTiles * kWarpRowTiles];

    const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
    const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;

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
            kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)) >=
                    kWarpSize
                ? kWarpSize
                : kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
        kLoadAColsPerIter = kThreads / kLoadALanesPerRow,
        kLoadAInnerLoop = kWmmaM * kBlockRowTiles /
                          (sizeof(float4) / sizeof(half) * kLoadALanesPerRow),

        kLoadBLanesPerRow =
            kWmmaK * kChunkK / (sizeof(float4) / sizeof(half)) >= kWarpSize
                ? kWarpSize
                : kWmmaK * kChunkK / (sizeof(float4) / sizeof(half)),
        kLoadBColsPerIter = kThreads / kLoadBLanesPerRow,
        kLoadBInnerLoop = kWmmaK * kChunkK /
                          (sizeof(float4) / sizeof(half) * kLoadBLanesPerRow),

        kStoreCLanesPerRow = kLoadALanesPerRow,
        kStoreCColsPerIter = kLoadAColsPerIter,
        kStoreCInnerLoop = kLoadAInnerLoop,
    };

    static_assert(K / kWmmaK / kChunkK >= 2);
    static_assert(kWmmaK * kChunkK % kLoadAColsPerIter == 0);
    static_assert(kWmmaN * kBlockColTiles % kStoreCColsPerIter == 0);
    static_assert(kWmmaN * kBlockColTiles % kLoadBColsPerIter == 0);
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
    int stage = 0;
    int k_loop = 0;

    // Prologue
    // LG(0)
    pipe.producer_acquire();
#pragma unroll
    for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
#pragma unroll
        for (int j = 0; j < kLoadAInnerLoop; ++j) {
            cuda::memcpy_async(
                reinterpret_cast<float4 *>(
                    matrix_a_shared[stage] +
                    (i * kLoadAColsPerIter + threadIdx.x / kLoadALanesPerRow) *
                        (kWmmaM * kBlockRowTiles + kInputSkew) +
                    ((threadIdx.x & (kLoadALanesPerRow - 1)) +
                     j * kLoadALanesPerRow) *
                        sizeof(float4) / sizeof(half)),
                reinterpret_cast<const float4 *>(
                    matrix_a + blockIdx.z * K * M +
                    blockIdx.x * kBlockRowTiles * kWmmaM +
                    (k_loop * kChunkK * kWmmaK + i * kLoadAColsPerIter +
                     threadIdx.x / kLoadALanesPerRow) *
                        M +
                    ((threadIdx.x & (kLoadALanesPerRow - 1)) +
                     j * kLoadALanesPerRow) *
                        (sizeof(float4) / sizeof(half))),
                shape, pipe);
        }
    }

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
#pragma unroll
        for (int j = 0; j < kLoadBInnerLoop; ++j) {
            cuda::memcpy_async(
                reinterpret_cast<float4 *>(
                    matrix_b_shared[stage] +
                    (i * kLoadBColsPerIter + threadIdx.x / kLoadBLanesPerRow) *
                        (kWmmaK * kChunkK + kInputSkew) +
                    ((threadIdx.x & (kLoadBLanesPerRow - 1)) +
                     j * kLoadBLanesPerRow) *
                        sizeof(float4) / sizeof(half)),
                reinterpret_cast<const float4 *>(
                    matrix_b + blockIdx.z * N * K + k_loop * kChunkK * kWmmaK +
                    (blockIdx.y * kBlockColTiles * kWmmaN +
                     i * kLoadBColsPerIter + threadIdx.x / kLoadBLanesPerRow) *
                        K +
                    ((threadIdx.x & (kLoadBLanesPerRow - 1)) +
                     j * kLoadBLanesPerRow) *
                        (sizeof(float4) / sizeof(half))),
                shape, pipe);
        }
    }
    pipe.producer_commit();

    // LG(1)
    pipe.producer_acquire();
#pragma unroll
    for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
#pragma unroll
        for (int j = 0; j < kLoadAInnerLoop; ++j) {
            cuda::memcpy_async(
                reinterpret_cast<float4 *>(
                    matrix_a_shared[1] +
                    (i * kLoadAColsPerIter + threadIdx.x / kLoadALanesPerRow) *
                        (kWmmaM * kBlockRowTiles + kInputSkew) +
                    ((threadIdx.x & (kLoadALanesPerRow - 1)) +
                     j * kLoadALanesPerRow) *
                        sizeof(float4) / sizeof(half)),
                reinterpret_cast<const float4 *>(
                    matrix_a + blockIdx.z * K * M +
                    blockIdx.x * kBlockRowTiles * kWmmaM +
                    (1 * kChunkK * kWmmaK + i * kLoadAColsPerIter +
                     threadIdx.x / kLoadALanesPerRow) *
                        M +
                    ((threadIdx.x & (kLoadALanesPerRow - 1)) +
                     j * kLoadALanesPerRow) *
                        (sizeof(float4) / sizeof(half))),
                shape, pipe);
        }
    }

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
#pragma unroll
        for (int j = 0; j < kLoadBInnerLoop; ++j) {
            cuda::memcpy_async(
                reinterpret_cast<float4 *>(
                    matrix_b_shared[1] +
                    (i * kLoadBColsPerIter + threadIdx.x / kLoadBLanesPerRow) *
                        (kWmmaK * kChunkK + kInputSkew) +
                    ((threadIdx.x & (kLoadBLanesPerRow - 1)) +
                     j * kLoadBLanesPerRow) *
                        sizeof(float4) / sizeof(half)),
                reinterpret_cast<const float4 *>(
                    matrix_b + blockIdx.z * N * K + 1 * kChunkK * kWmmaK +
                    (blockIdx.y * kBlockColTiles * kWmmaN +
                     i * kLoadBColsPerIter + threadIdx.x / kLoadBLanesPerRow) *
                        K +
                    ((threadIdx.x & (kLoadBLanesPerRow - 1)) +
                     j * kLoadBLanesPerRow) *
                        (sizeof(float4) / sizeof(half))),
                shape, pipe);
        }
    }
    pipe.producer_commit();
    pipe.consumer_wait();
    __syncthreads();
    pipe.consumer_release();

    // LS(0)
#pragma unroll
    for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
        for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
            nvcuda::wmma::load_matrix_sync(
                wmma_matrix_a[tile_k][tile_m],
                (matrix_a_shared[0] +
                 tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                 (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM),
                kBlockRowTiles * kWmmaM + kInputSkew);
        }
#pragma unroll
        for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
            nvcuda::wmma::load_matrix_sync(
                wmma_matrix_b[tile_k][tile_n],
                (matrix_b_shared[0] +
                 (col_warp_id * kWarpColTiles + tile_n) * kWmmaN *
                     (kChunkK * kWmmaK + kInputSkew) +
                 tile_k * kWmmaK),
                kChunkK * kWmmaK + kInputSkew);
        }
    }

    // Soft pipeline
#pragma unroll
    for (; k_loop < (K / kChunkK / kWmmaK) - 2; ++k_loop) {
        pipe.producer_acquire();

        // LG(k + 2)
#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
#pragma unroll
            for (int j = 0; j < kLoadAInnerLoop; ++j) {
                cuda::memcpy_async(
                    reinterpret_cast<float4 *>(
                        matrix_a_shared[(stage + 2) & 0x1] +
                        (i * kLoadAColsPerIter +
                         threadIdx.x / kLoadALanesPerRow) *
                            (kWmmaM * kBlockRowTiles + kInputSkew) +
                        ((threadIdx.x & (kLoadALanesPerRow - 1)) +
                         j * kLoadALanesPerRow) *
                            sizeof(float4) / sizeof(half)),
                    reinterpret_cast<const float4 *>(
                        matrix_a + blockIdx.z * K * M +
                        blockIdx.x * kBlockRowTiles * kWmmaM +
                        ((k_loop + 2) * kChunkK * kWmmaK +
                         i * kLoadAColsPerIter +
                         threadIdx.x / kLoadALanesPerRow) *
                            M +
                        ((threadIdx.x & (kLoadALanesPerRow - 1)) +
                         j * kLoadALanesPerRow) *
                            (sizeof(float4) / sizeof(half))),
                    shape, pipe);
            }
        }

#pragma unroll
        for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
#pragma unroll
            for (int j = 0; j < kLoadBInnerLoop; ++j) {
                cuda::memcpy_async(
                    reinterpret_cast<float4 *>(
                        matrix_b_shared[(stage + 2) & 0x1] +
                        (i * kLoadBColsPerIter +
                         threadIdx.x / kLoadBLanesPerRow) *
                            (kWmmaK * kChunkK + kInputSkew) +
                        ((threadIdx.x & (kLoadBLanesPerRow - 1)) +
                         j * kLoadBLanesPerRow) *
                            sizeof(float4) / sizeof(half)),
                    reinterpret_cast<const float4 *>(
                        matrix_b + blockIdx.z * N * K +
                        (k_loop + 2) * kChunkK * kWmmaK +
                        (blockIdx.y * kBlockColTiles * kWmmaN +
                         i * kLoadBColsPerIter +
                         threadIdx.x / kLoadBLanesPerRow) *
                            K +
                        ((threadIdx.x & (kLoadBLanesPerRow - 1)) +
                         j * kLoadBLanesPerRow) *
                            (sizeof(float4) / sizeof(half))),
                    shape, pipe);
            }
        }
        pipe.producer_commit();

        // C(k)
#pragma unroll
        for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
            for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
#pragma unroll
                for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
                    nvcuda::wmma::mma_sync(
                        wmma_accumulator[tile_m + tile_n * kWarpRowTiles],
                        wmma_matrix_a[tile_k][tile_m],
                        wmma_matrix_b[tile_k][tile_n],
                        wmma_accumulator[tile_m + tile_n * kWarpRowTiles]);
                }
            }
        }

        pipe.consumer_wait();
        __syncthreads();

        pipe.consumer_release();

        // LS(k + 1)
#pragma unroll
        for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
            for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[tile_k][tile_m],
                    (matrix_a_shared[(stage + 1) & 0x1] +
                     tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
            }
#pragma unroll
            for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_b[tile_k][tile_n],
                    (matrix_b_shared[(stage + 1) & 0x1] +
                     (col_warp_id * kWarpColTiles + tile_n) * kWmmaN *
                         (kChunkK * kWmmaK + kInputSkew) +
                     tile_k * kWmmaK),
                    kChunkK * kWmmaK + kInputSkew);
            }
        }
        stage = (stage + 1) & 0x1;
    }

    // Epilogue
    // C(N - 1)
#pragma unroll
    for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
        for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
#pragma unroll
            for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
                nvcuda::wmma::mma_sync(
                    wmma_accumulator[tile_m + tile_n * kWarpRowTiles],
                    wmma_matrix_a[tile_k][tile_m],
                    wmma_matrix_b[tile_k][tile_n],
                    wmma_accumulator[tile_m + tile_n * kWarpRowTiles]);
            }
        }
    }

    pipe.consumer_wait();
    __syncthreads();

    pipe.consumer_release();

    // LS(N)
#pragma unroll
    for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
        for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
            nvcuda::wmma::load_matrix_sync(
                wmma_matrix_a[tile_k][tile_m],
                (matrix_a_shared[(stage + 1) & 0x1] +
                 tile_k * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                 (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM),
                kBlockRowTiles * kWmmaM + kInputSkew);
        }
#pragma unroll
        for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
            nvcuda::wmma::load_matrix_sync(
                wmma_matrix_b[tile_k][tile_n],
                (matrix_b_shared[(stage + 1) & 0x1] +
                 (col_warp_id * kWarpColTiles + tile_n) * kWmmaN *
                     (kChunkK * kWmmaK + kInputSkew) +
                 tile_k * kWmmaK),
                kChunkK * kWmmaK + kInputSkew);
        }
    }

    // C(N)
#pragma unroll
    for (int tile_k = 0; tile_k < kChunkK; ++tile_k) {
#pragma unroll
        for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
#pragma unroll
            for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
                nvcuda::wmma::mma_sync(
                    wmma_accumulator[tile_m + tile_n * kWarpRowTiles],
                    wmma_matrix_a[tile_k][tile_m],
                    wmma_matrix_b[tile_k][tile_n],
                    wmma_accumulator[tile_m + tile_n * kWarpRowTiles]);
            }
        }
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
#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
#pragma unroll
        for (int j = 0; j < kStoreCInnerLoop; ++j) {
            *reinterpret_cast<float4 *>(
                matrix_c + blockIdx.z * N * M +
                blockIdx.x * kBlockRowTiles * kWmmaM +
                (blockIdx.y * kBlockColTiles * kWmmaN + i * kStoreCColsPerIter +
                 threadIdx.x / kStoreCLanesPerRow) *
                    M +
                ((threadIdx.x & (kStoreCLanesPerRow - 1)) +
                 j * kStoreCLanesPerRow) *
                    sizeof(float4) / sizeof(half)) =
                *reinterpret_cast<float4 *>(
                    acc_shared +
                    (i * kStoreCColsPerIter +
                     threadIdx.x / kStoreCLanesPerRow) *
                        (kBlockRowTiles * kWmmaM + kAccSkew) +
                    ((threadIdx.x & (kStoreCLanesPerRow - 1)) +
                     j * kStoreCLanesPerRow) *
                        sizeof(float4) / sizeof(half));
        }
    }
}

template <int kChunkK, int kBlockRowWarps, int kBlockColWarps,
          int kWarpRowTiles, int kWarpColTiles, int kInputSkew, int kAccSkew,
          int M, int N, int K>
__global__ void tvm_gemm_three_stage(const half *__restrict__ matrix_a,
                                     const half *__restrict__ matrix_b,
                                     half *__restrict__ matrix_c) {
    using namespace nvcuda;
    enum {
        kBlockRowTiles = kBlockRowWarps * kWarpRowTiles,
        kBlockColTiles = kBlockColWarps * kWarpColTiles,
    };

    enum {
        kStage = 3,
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
            kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)) >=
                    kWarpSize
                ? kWarpSize
                : kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
        kLoadAColsPerIter = kThreads / kLoadALanesPerRow,
        kLoadAInnerLoop = kWmmaM * kBlockRowTiles /
                          (sizeof(float4) / sizeof(half) * kLoadALanesPerRow),

        kLoadBLanesPerRow =
            kWmmaK * kChunkK / (sizeof(float4) / sizeof(half)) >= kWarpSize
                ? kWarpSize
                : kWmmaK * kChunkK / (sizeof(float4) / sizeof(half)),
        kLoadBColsPerIter = kThreads / kLoadBLanesPerRow,
        kLoadBInnerLoop = kWmmaK * kChunkK /
                          (sizeof(float4) / sizeof(half) * kLoadBLanesPerRow),

        kStoreCLanesPerRow = kLoadALanesPerRow,
        kStoreCColsPerIter = kLoadAColsPerIter,
        kStoreCInnerLoop = kLoadAInnerLoop,
    };

    static_assert(kWmmaK * kChunkK % kLoadAColsPerIter == 0);
    static_assert(kWmmaN * kBlockColTiles % kStoreCColsPerIter == 0);
    static_assert(kWmmaN * kBlockColTiles % kLoadBColsPerIter == 0);

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
    int stage = 0;
    int k_loop = 0;

    constexpr int a_dst_i_stride =
        kLoadAColsPerIter * (kWmmaM * kBlockRowTiles + kInputSkew);
    constexpr int a_dst_j_stride =
        kLoadALanesPerRow * sizeof(float4) / sizeof(half);

    constexpr int a_src_i_stride = kLoadAColsPerIter * M;
    constexpr int a_src_j_stride =
        (kLoadALanesPerRow * sizeof(float4) / sizeof(half));

    constexpr int b_dst_i_stride =
        kLoadBColsPerIter * (kWmmaK * kChunkK + kInputSkew);
    constexpr int b_dst_j_stride =
        kLoadBLanesPerRow * sizeof(float4) / sizeof(half);
    constexpr int b_src_i_stride = kLoadBColsPerIter * K;
    constexpr int b_src_j_stride =
        kLoadBLanesPerRow * sizeof(float4) / sizeof(half);

    // Prologue
#pragma unroll
    for (int s = 0; s < kStage - 1; ++s) {
        pipe.producer_acquire();
        half *a_dst_base =
            matrix_a_shared[(stage + s) % kStage] +
            (0 * kLoadAColsPerIter + threadIdx.x / kLoadALanesPerRow) *
                (kWmmaM * kBlockRowTiles + kInputSkew) +
            ((threadIdx.x & (kLoadALanesPerRow - 1)) + 0 * kLoadALanesPerRow) *
                sizeof(float4) / sizeof(half);

        const half *a_src_base =
            matrix_a + blockIdx.z * K * M +
            blockIdx.x * kBlockRowTiles * kWmmaM +
            ((k_loop + s) * kChunkK * kWmmaK + 0 * kLoadAColsPerIter +
             threadIdx.x / kLoadALanesPerRow) *
                M +
            ((threadIdx.x & (kLoadALanesPerRow - 1)) + 0 * kLoadALanesPerRow) *
                (sizeof(float4) / sizeof(half));

        half *b_dst_base =
            matrix_b_shared[(stage + s) % kStage] +
            (0 * kLoadBColsPerIter + threadIdx.x / kLoadBLanesPerRow) *
                (kWmmaK * kChunkK + kInputSkew) +
            ((threadIdx.x & (kLoadBLanesPerRow - 1)) + 0 * kLoadBLanesPerRow) *
                sizeof(float4) / sizeof(half);

        const half *b_src_base =
            matrix_b + blockIdx.z * N * K + (k_loop + s) * kChunkK * kWmmaK +
            (blockIdx.y * kBlockColTiles * kWmmaN + 0 * kLoadBColsPerIter +
             threadIdx.x / kLoadBLanesPerRow) *
                K +
            ((threadIdx.x & (kLoadBLanesPerRow - 1)) + 0 * kLoadBLanesPerRow) *
                (sizeof(float4) / sizeof(half));

#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
#pragma unroll
            for (int j = 0; j < kLoadAInnerLoop; ++j) {
                cuda::memcpy_async(
                    a_dst_base + i * a_dst_i_stride + j * a_dst_j_stride,
                    a_src_base + i * a_src_i_stride + j * a_src_j_stride, shape,
                    pipe);
            }
        }

#pragma unroll
        for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
#pragma unroll
            for (int j = 0; j < kLoadBInnerLoop; ++j) {
                cuda::memcpy_async(
                    b_dst_base + i * b_dst_i_stride + j * b_dst_j_stride,
                    b_src_base + i * b_src_i_stride + j * b_src_j_stride, shape,
                    pipe);
            }
        }
        pipe.producer_commit();
    }

    // Soft pipeline
#pragma unroll
    for (; k_loop < (K / kChunkK / kWmmaK) - (kStage - 1); ++k_loop) {
        pipe.producer_acquire();

        half *a_dst_base =
            matrix_a_shared[(stage + kStage - 1) % kStage] +
            (0 * kLoadAColsPerIter + threadIdx.x / kLoadALanesPerRow) *
                (kWmmaM * kBlockRowTiles + kInputSkew) +
            ((threadIdx.x & (kLoadALanesPerRow - 1)) + 0 * kLoadALanesPerRow) *
                sizeof(float4) / sizeof(half);

        const half *a_src_base =
            matrix_a + blockIdx.z * K * M +
            blockIdx.x * kBlockRowTiles * kWmmaM +
            ((k_loop + kStage - 1) * kChunkK * kWmmaK + 0 * kLoadAColsPerIter +
             threadIdx.x / kLoadALanesPerRow) *
                M +
            ((threadIdx.x & (kLoadALanesPerRow - 1)) + 0 * kLoadALanesPerRow) *
                (sizeof(float4) / sizeof(half));

        half *b_dst_base =
            matrix_b_shared[(stage + kStage - 1) % kStage] +
            (0 * kLoadBColsPerIter + threadIdx.x / kLoadBLanesPerRow) *
                (kWmmaK * kChunkK + kInputSkew) +
            ((threadIdx.x & (kLoadBLanesPerRow - 1)) + 0 * kLoadBLanesPerRow) *
                sizeof(float4) / sizeof(half);

        const half *b_src_base =
            matrix_b + blockIdx.z * N * K +
            (k_loop + kStage - 1) * kChunkK * kWmmaK +
            (blockIdx.y * kBlockColTiles * kWmmaN + 0 * kLoadBColsPerIter +
             threadIdx.x / kLoadBLanesPerRow) *
                K +
            ((threadIdx.x & (kLoadBLanesPerRow - 1)) + 0 * kLoadBLanesPerRow) *
                (sizeof(float4) / sizeof(half));

#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
#pragma unroll
            for (int j = 0; j < kLoadAInnerLoop; ++j) {
                cuda::memcpy_async(
                    a_dst_base + i * a_dst_i_stride + j * a_dst_j_stride,
                    a_src_base + i * a_src_i_stride + j * a_src_j_stride, shape,
                    pipe);
            }
        }

#pragma unroll
        for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
#pragma unroll
            for (int j = 0; j < kLoadBInnerLoop; ++j) {
                cuda::memcpy_async(
                    b_dst_base + i * b_dst_i_stride + j * b_dst_j_stride,
                    b_src_base + i * b_src_i_stride + j * b_src_j_stride, shape,
                    pipe);
            }
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
        // The sync is not necessary when compute time is large enough
        // __syncthreads();
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

    constexpr int c_dst_i_stride = kStoreCColsPerIter * M;
    constexpr int c_dst_j_stride =
        kStoreCLanesPerRow * sizeof(float4) / sizeof(half);

    constexpr int c_src_i_stride =
        kStoreCColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);
    constexpr int c_src_j_stride =
        kStoreCLanesPerRow * sizeof(float4) / sizeof(half);

    half *c_dst_base =
        matrix_c + blockIdx.z * N * M + blockIdx.x * kBlockRowTiles * kWmmaM +
        (blockIdx.y * kBlockColTiles * kWmmaN + 0 * kStoreCColsPerIter +
         threadIdx.x / kStoreCLanesPerRow) *
            M +
        ((threadIdx.x & (kStoreCLanesPerRow - 1)) + 0 * kStoreCLanesPerRow) *
            sizeof(float4) / sizeof(half);
    half *c_src_base =
        acc_shared +
        (0 * kStoreCColsPerIter + threadIdx.x / kStoreCLanesPerRow) *
            (kBlockRowTiles * kWmmaM + kAccSkew) +
        ((threadIdx.x & (kStoreCLanesPerRow - 1)) + 0 * kStoreCLanesPerRow) *
            sizeof(float4) / sizeof(half);

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
#pragma unroll
        for (int j = 0; j < kStoreCInnerLoop; ++j) {
            *reinterpret_cast<float4 *>(c_dst_base + i * c_dst_i_stride +
                                        j * c_dst_j_stride) =
                *reinterpret_cast<float4 *>(c_src_base + i * c_src_i_stride +
                                            j * c_src_j_stride);
        }
    }
}

template <int kChunkK, int kBlockRowWarps, int kBlockColWarps,
          int kWarpRowTiles, int kWarpColTiles, int kInputSkew, int kAccSkew,
          int M, int N, int K>
__global__ void tvm_gemm_three_stage_v2(const half *__restrict__ matrix_a,
                                        const half *__restrict__ matrix_b,
                                        half *__restrict__ matrix_c) {
    using namespace nvcuda;
    enum {
        kBlockRowTiles = kBlockRowWarps * kWarpRowTiles,
        kBlockColTiles = kBlockColWarps * kWarpColTiles,
    };

    extern __shared__ half all_shared_mem[];

    half *matrix_a_shared[3], *matrix_b_shared[3];
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

    // The entire execution is serialized as :
    // (1) LG (Load global to shmem),
    // (2) LS (Load shmem to reg),
    // (3) C  (Compute the gemm),
    // Thus, the soft pipeline is performed as :
    // Prologue: LG(0), LS(0), LG(1)
    // Pipeline: C(0), LS(1), LG(2); ... ; C(n), LS(n+1), LG(n+2); ...
    // Epilogue: C(N-1), LS(N), C(N)
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::col_major>
        wmma_matrix_a[2][kWarpRowTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, half,
                           nvcuda::wmma::col_major>
        wmma_matrix_b[2][kWarpColTiles];
    nvcuda::wmma::fragment<nvcuda::wmma::accumulator, kWmmaM, kWmmaN, kWmmaK,
                           half>
        wmma_accumulator[kWarpColTiles * kWarpRowTiles];

    const int row_warp_id = (threadIdx.x / kWarpSize) % kBlockRowWarps;
    const int col_warp_id = (threadIdx.x / kWarpSize) / kBlockRowWarps;

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
            kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)) >=
                    kWarpSize
                ? kWarpSize
                : kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
        kLoadAColsPerIter = kThreads / kLoadALanesPerRow,
        kLoadAInnerLoop = kWmmaM * kBlockRowTiles /
                          (sizeof(float4) / sizeof(half) * kLoadALanesPerRow),

        kLoadBLanesPerRow =
            kWmmaK * kChunkK / (sizeof(float4) / sizeof(half)) >= kWarpSize
                ? kWarpSize
                : kWmmaK * kChunkK / (sizeof(float4) / sizeof(half)),
        kLoadBColsPerIter = kThreads / kLoadBLanesPerRow,
        kLoadBInnerLoop = kWmmaK * kChunkK /
                          (sizeof(float4) / sizeof(half) * kLoadBLanesPerRow),

        kStoreCLanesPerRow = kLoadALanesPerRow,
        kStoreCColsPerIter = kLoadAColsPerIter,
        kStoreCInnerLoop = kLoadAInnerLoop,
    };

    static_assert(K / kWmmaK / kChunkK >= 2);
    static_assert(kWmmaK * kChunkK % kLoadAColsPerIter == 0);
    static_assert(kWmmaN * kBlockColTiles % kStoreCColsPerIter == 0);
    static_assert(kWmmaN * kBlockColTiles % kLoadBColsPerIter == 0);
    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
    const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
    int stage = 0;
    int k_loop = 0;

    // Prologue
    // LG(0)
    pipe.producer_acquire();
#pragma unroll
    for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
#pragma unroll
        for (int j = 0; j < kLoadAInnerLoop; ++j) {
            cuda::memcpy_async(
                reinterpret_cast<float4 *>(
                    matrix_a_shared[stage] +
                    (i * kLoadAColsPerIter + threadIdx.x / kLoadALanesPerRow) *
                        (kWmmaM * kBlockRowTiles + kInputSkew) +
                    ((threadIdx.x & (kLoadALanesPerRow - 1)) +
                     j * kLoadALanesPerRow) *
                        sizeof(float4) / sizeof(half)),
                reinterpret_cast<const float4 *>(
                    matrix_a + blockIdx.z * K * M +
                    blockIdx.x * kBlockRowTiles * kWmmaM +
                    (k_loop * kChunkK * kWmmaK + i * kLoadAColsPerIter +
                     threadIdx.x / kLoadALanesPerRow) *
                        M +
                    ((threadIdx.x & (kLoadALanesPerRow - 1)) +
                     j * kLoadALanesPerRow) *
                        (sizeof(float4) / sizeof(half))),
                shape, pipe);
        }
    }

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
#pragma unroll
        for (int j = 0; j < kLoadBInnerLoop; ++j) {
            cuda::memcpy_async(
                reinterpret_cast<float4 *>(
                    matrix_b_shared[stage] +
                    (i * kLoadBColsPerIter + threadIdx.x / kLoadBLanesPerRow) *
                        (kWmmaK * kChunkK + kInputSkew) +
                    ((threadIdx.x & (kLoadBLanesPerRow - 1)) +
                     j * kLoadBLanesPerRow) *
                        sizeof(float4) / sizeof(half)),
                reinterpret_cast<const float4 *>(
                    matrix_b + blockIdx.z * N * K + k_loop * kChunkK * kWmmaK +
                    (blockIdx.y * kBlockColTiles * kWmmaN +
                     i * kLoadBColsPerIter + threadIdx.x / kLoadBLanesPerRow) *
                        K +
                    ((threadIdx.x & (kLoadBLanesPerRow - 1)) +
                     j * kLoadBLanesPerRow) *
                        (sizeof(float4) / sizeof(half))),
                shape, pipe);
        }
    }
    pipe.producer_commit();

    // LG(1)
    pipe.producer_acquire();
#pragma unroll
    for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
#pragma unroll
        for (int j = 0; j < kLoadAInnerLoop; ++j) {
            cuda::memcpy_async(
                reinterpret_cast<float4 *>(
                    matrix_a_shared[1] +
                    (i * kLoadAColsPerIter + threadIdx.x / kLoadALanesPerRow) *
                        (kWmmaM * kBlockRowTiles + kInputSkew) +
                    ((threadIdx.x & (kLoadALanesPerRow - 1)) +
                     j * kLoadALanesPerRow) *
                        sizeof(float4) / sizeof(half)),
                reinterpret_cast<const float4 *>(
                    matrix_a + blockIdx.z * K * M +
                    blockIdx.x * kBlockRowTiles * kWmmaM +
                    (1 * kChunkK * kWmmaK + i * kLoadAColsPerIter +
                     threadIdx.x / kLoadALanesPerRow) *
                        M +
                    ((threadIdx.x & (kLoadALanesPerRow - 1)) +
                     j * kLoadALanesPerRow) *
                        (sizeof(float4) / sizeof(half))),
                shape, pipe);
        }
    }

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
#pragma unroll
        for (int j = 0; j < kLoadBInnerLoop; ++j) {
            cuda::memcpy_async(
                reinterpret_cast<float4 *>(
                    matrix_b_shared[1] +
                    (i * kLoadBColsPerIter + threadIdx.x / kLoadBLanesPerRow) *
                        (kWmmaK * kChunkK + kInputSkew) +
                    ((threadIdx.x & (kLoadBLanesPerRow - 1)) +
                     j * kLoadBLanesPerRow) *
                        sizeof(float4) / sizeof(half)),
                reinterpret_cast<const float4 *>(
                    matrix_b + blockIdx.z * N * K + 1 * kChunkK * kWmmaK +
                    (blockIdx.y * kBlockColTiles * kWmmaN +
                     i * kLoadBColsPerIter + threadIdx.x / kLoadBLanesPerRow) *
                        K +
                    ((threadIdx.x & (kLoadBLanesPerRow - 1)) +
                     j * kLoadBLanesPerRow) *
                        (sizeof(float4) / sizeof(half))),
                shape, pipe);
        }
    }
    pipe.producer_commit();
    pipe.consumer_wait();
    __syncthreads();
    pipe.consumer_release();

    // LS(0)K0
#pragma unroll
    for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_a[0][tile_m],
            (matrix_a_shared[0] +
             0 * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
             (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM),
            kBlockRowTiles * kWmmaM + kInputSkew);
    }
#pragma unroll
    for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_b[0][tile_n],
            (matrix_b_shared[0] +
             (col_warp_id * kWarpColTiles + tile_n) * kWmmaN *
                 (kChunkK * kWmmaK + kInputSkew) +
             0 * kWmmaK),
            kChunkK * kWmmaK + kInputSkew);
    }

    // Soft pipeline
#pragma unroll
    for (; k_loop < (K / kChunkK / kWmmaK) - 2; ++k_loop) {
        pipe.producer_acquire();

        // LG(k + 2)
#pragma unroll
        for (int i = 0; i < kChunkK * kWmmaK / kLoadAColsPerIter; ++i) {
#pragma unroll
            for (int j = 0; j < kLoadAInnerLoop; ++j) {
                cuda::memcpy_async(
                    reinterpret_cast<float4 *>(
                        matrix_a_shared[(stage + 2) % 3] +
                        (i * kLoadAColsPerIter +
                         threadIdx.x / kLoadALanesPerRow) *
                            (kWmmaM * kBlockRowTiles + kInputSkew) +
                        ((threadIdx.x & (kLoadALanesPerRow - 1)) +
                         j * kLoadALanesPerRow) *
                            sizeof(float4) / sizeof(half)),
                    reinterpret_cast<const float4 *>(
                        matrix_a + blockIdx.z * K * M +
                        blockIdx.x * kBlockRowTiles * kWmmaM +
                        ((k_loop + 2) * kChunkK * kWmmaK +
                         i * kLoadAColsPerIter +
                         threadIdx.x / kLoadALanesPerRow) *
                            M +
                        ((threadIdx.x & (kLoadALanesPerRow - 1)) +
                         j * kLoadALanesPerRow) *
                            (sizeof(float4) / sizeof(half))),
                    shape, pipe);
            }
        }

#pragma unroll
        for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
#pragma unroll
            for (int j = 0; j < kLoadBInnerLoop; ++j) {
                cuda::memcpy_async(
                    reinterpret_cast<float4 *>(
                        matrix_b_shared[(stage + 2) % 3] +
                        (i * kLoadBColsPerIter +
                         threadIdx.x / kLoadBLanesPerRow) *
                            (kWmmaK * kChunkK + kInputSkew) +
                        ((threadIdx.x & (kLoadBLanesPerRow - 1)) +
                         j * kLoadBLanesPerRow) *
                            sizeof(float4) / sizeof(half)),
                    reinterpret_cast<const float4 *>(
                        matrix_b + blockIdx.z * N * K +
                        (k_loop + 2) * kChunkK * kWmmaK +
                        (blockIdx.y * kBlockColTiles * kWmmaN +
                         i * kLoadBColsPerIter +
                         threadIdx.x / kLoadBLanesPerRow) *
                            K +
                        ((threadIdx.x & (kLoadBLanesPerRow - 1)) +
                         j * kLoadBLanesPerRow) *
                            (sizeof(float4) / sizeof(half))),
                    shape, pipe);
            }
        }
        pipe.producer_commit();

        // LS(K)C(k)
#pragma unroll
        for (int tile_k = 0; tile_k < kChunkK - 1; ++tile_k) {
#pragma unroll
            for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[(tile_k + 1) & 0x1][tile_m],
                    (matrix_a_shared[stage] +
                     (tile_k + 1) % kChunkK * kWmmaK *
                         (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
            }
#pragma unroll
            for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_b[(tile_k + 1) & 0x1][tile_n],
                    (matrix_b_shared[stage] +
                     (col_warp_id * kWarpColTiles + tile_n) * kWmmaN *
                         (kChunkK * kWmmaK + kInputSkew) +
                     (tile_k + 1) % kChunkK * kWmmaK),
                    kChunkK * kWmmaK + kInputSkew);
            }
#pragma unroll
            for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
#pragma unroll
                for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
                    nvcuda::wmma::mma_sync(
                        wmma_accumulator[tile_m + tile_n * kWarpRowTiles],
                        wmma_matrix_a[tile_k & 0x1][tile_m],
                        wmma_matrix_b[tile_k & 0x1][tile_n],
                        wmma_accumulator[tile_m + tile_n * kWarpRowTiles]);
                }
            }
        }
        // C(K)K3
#pragma unroll
        for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
#pragma unroll
            for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
                nvcuda::wmma::mma_sync(
                    wmma_accumulator[tile_m + tile_n * kWarpRowTiles],
                    wmma_matrix_a[1][tile_m], wmma_matrix_b[1][tile_n],
                    wmma_accumulator[tile_m + tile_n * kWarpRowTiles]);
            }
        }

        pipe.consumer_wait();
        __syncthreads();

        pipe.consumer_release();

        // LS(k + 1)K0
#pragma unroll
        for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
            nvcuda::wmma::load_matrix_sync(
                wmma_matrix_a[0][tile_m],
                (matrix_a_shared[(stage + 1) % 3] +
                 0 * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
                 (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM),
                kBlockRowTiles * kWmmaM + kInputSkew);
        }
#pragma unroll
        for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
            nvcuda::wmma::load_matrix_sync(
                wmma_matrix_b[0][tile_n],
                (matrix_b_shared[(stage + 1) % 3] +
                 (col_warp_id * kWarpColTiles + tile_n) * kWmmaN *
                     (kChunkK * kWmmaK + kInputSkew) +
                 0 * kWmmaK),
                kChunkK * kWmmaK + kInputSkew);
        }
        stage = (stage + 1) % 3;
    }

    // Epilogue
    // LS(N - 1)C(N - 1)
#pragma unroll
    for (int tile_k = 0; tile_k < kChunkK - 1; ++tile_k) {
#pragma unroll
        for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
            nvcuda::wmma::load_matrix_sync(
                wmma_matrix_a[(tile_k + 1) & 0x1][tile_m],
                (matrix_a_shared[stage] +
                 (tile_k + 1) % kChunkK * kWmmaK *
                     (kBlockRowTiles * kWmmaM + kInputSkew) +
                 (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM),
                kBlockRowTiles * kWmmaM + kInputSkew);
        }
#pragma unroll
        for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
            nvcuda::wmma::load_matrix_sync(
                wmma_matrix_b[(tile_k + 1) & 0x1][tile_n],
                (matrix_b_shared[stage] +
                 (col_warp_id * kWarpColTiles + tile_n) * kWmmaN *
                     (kChunkK * kWmmaK + kInputSkew) +
                 (tile_k + 1) % kChunkK * kWmmaK),
                kChunkK * kWmmaK + kInputSkew);
        }
#pragma unroll
        for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
#pragma unroll
            for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
                nvcuda::wmma::mma_sync(
                    wmma_accumulator[tile_m + tile_n * kWarpRowTiles],
                    wmma_matrix_a[tile_k & 0x1][tile_m],
                    wmma_matrix_b[tile_k & 0x1][tile_n],
                    wmma_accumulator[tile_m + tile_n * kWarpRowTiles]);
            }
        }
    }
    // C(N - 1)K3
#pragma unroll
    for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
#pragma unroll
        for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
            nvcuda::wmma::mma_sync(
                wmma_accumulator[tile_m + tile_n * kWarpRowTiles],
                wmma_matrix_a[1][tile_m], wmma_matrix_b[1][tile_n],
                wmma_accumulator[tile_m + tile_n * kWarpRowTiles]);
        }
    }

    pipe.consumer_wait();
    __syncthreads();

    pipe.consumer_release();

    // LS(N)K0
#pragma unroll
    for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_a[0][tile_m],
            (matrix_a_shared[(stage + 1) % 3] +
             0 * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew) +
             (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM),
            kBlockRowTiles * kWmmaM + kInputSkew);
    }
#pragma unroll
    for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
        nvcuda::wmma::load_matrix_sync(
            wmma_matrix_b[0][tile_n],
            (matrix_b_shared[(stage + 1) % 3] +
             (col_warp_id * kWarpColTiles + tile_n) * kWmmaN *
                 (kChunkK * kWmmaK + kInputSkew) +
             0 * kWmmaK),
            kChunkK * kWmmaK + kInputSkew);
    }
    stage = (stage + 1) % 3;

    // LS(N)C(N)
#pragma unroll
    for (int tile_k = 0; tile_k < kChunkK - 1; ++tile_k) {
#pragma unroll
        for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
            nvcuda::wmma::load_matrix_sync(
                wmma_matrix_a[(tile_k + 1) & 0x1][tile_m],
                (matrix_a_shared[stage] +
                 (tile_k + 1) % kChunkK * kWmmaK *
                     (kBlockRowTiles * kWmmaM + kInputSkew) +
                 (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM),
                kBlockRowTiles * kWmmaM + kInputSkew);
        }
#pragma unroll
        for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
            nvcuda::wmma::load_matrix_sync(
                wmma_matrix_b[(tile_k + 1) & 0x1][tile_n],
                (matrix_b_shared[stage] +
                 (col_warp_id * kWarpColTiles + tile_n) * kWmmaN *
                     (kChunkK * kWmmaK + kInputSkew) +
                 (tile_k + 1) % kChunkK * kWmmaK),
                kChunkK * kWmmaK + kInputSkew);
        }
#pragma unroll
        for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
#pragma unroll
            for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
                nvcuda::wmma::mma_sync(
                    wmma_accumulator[tile_m + tile_n * kWarpRowTiles],
                    wmma_matrix_a[tile_k & 0x1][tile_m],
                    wmma_matrix_b[tile_k & 0x1][tile_n],
                    wmma_accumulator[tile_m + tile_n * kWarpRowTiles]);
            }
        }
    }
    // C(N)K3
#pragma unroll
    for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
#pragma unroll
        for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
            nvcuda::wmma::mma_sync(
                wmma_accumulator[tile_m + tile_n * kWarpRowTiles],
                wmma_matrix_a[1][tile_m], wmma_matrix_b[1][tile_n],
                wmma_accumulator[tile_m + tile_n * kWarpRowTiles]);
        }
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
#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
#pragma unroll
        for (int j = 0; j < kStoreCInnerLoop; ++j) {
            *reinterpret_cast<float4 *>(
                matrix_c + blockIdx.z * N * M +
                blockIdx.x * kBlockRowTiles * kWmmaM +
                (blockIdx.y * kBlockColTiles * kWmmaN + i * kStoreCColsPerIter +
                 threadIdx.x / kStoreCLanesPerRow) *
                    M +
                ((threadIdx.x & (kStoreCLanesPerRow - 1)) +
                 j * kStoreCLanesPerRow) *
                    sizeof(float4) / sizeof(half)) =
                *reinterpret_cast<float4 *>(
                    acc_shared +
                    (i * kStoreCColsPerIter +
                     threadIdx.x / kStoreCLanesPerRow) *
                        (kBlockRowTiles * kWmmaM + kAccSkew) +
                    ((threadIdx.x & (kStoreCLanesPerRow - 1)) +
                     j * kStoreCLanesPerRow) *
                        sizeof(float4) / sizeof(half));
        }
    }
}

template <int kBlockRowWarps, int kBlockColWarps, int kBlockSliceKWarps,
          int kWarpRowTiles, int kWarpColTiles, int kWarpSliceKTiles,
          int kInputSkew, int kAccSkew, int M, int N, int K>
__global__ void tvm_gemm_three_stage_v3(const half *__restrict__ matrix_a,
                                        const half *__restrict__ matrix_b,
                                        half *__restrict__ matrix_c) {
    using namespace nvcuda;
    enum {
        kBlockRowTiles = kBlockRowWarps * kWarpRowTiles,
        kBlockColTiles = kBlockColWarps * kWarpColTiles,
        kBlockSliceKTiles = kBlockSliceKWarps * kWarpSliceKTiles,
    };

    enum {
        kStage = 3,
    };

    extern __shared__ half all_shared_mem[];

    half *matrix_a_shared[kStage], *matrix_b_shared[kStage];
    half *acc_shared;

    matrix_a_shared[0] = all_shared_mem;
    matrix_a_shared[1] =
        all_shared_mem +
        kBlockSliceKTiles * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);
    matrix_a_shared[2] =
        all_shared_mem +
        2 * kBlockSliceKTiles * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);

    matrix_b_shared[0] =
        all_shared_mem +
        3 * kBlockSliceKTiles * kWmmaK * (kBlockRowTiles * kWmmaM + kInputSkew);

    matrix_b_shared[1] =
        all_shared_mem +
        3 * kBlockSliceKTiles * kWmmaK *
            (kBlockRowTiles * kWmmaM + kInputSkew) +
        kBlockColTiles * kWmmaN * (kBlockSliceKTiles * kWmmaK + kInputSkew);

    matrix_b_shared[2] =
        all_shared_mem +
        3 * kBlockSliceKTiles * kWmmaK *
            (kBlockRowTiles * kWmmaM + kInputSkew) +
        2 * kBlockColTiles * kWmmaN * (kBlockSliceKTiles * kWmmaK + kInputSkew);

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

    const int row_warp_id =
        threadIdx.x / kWarpSize / kBlockSliceKWarps % kBlockRowWarps;
    const int col_warp_id =
        threadIdx.x / kWarpSize / kBlockSliceKWarps / kBlockRowWarps;
    const int slicek_warp_id =
        threadIdx.x / kWarpSize / kBlockColWarps / kBlockRowWarps;

#pragma unroll
    for (int col = 0; col < kWarpColTiles; ++col) {
#pragma unroll
        for (int row = 0; row < kWarpRowTiles; ++row) {
            nvcuda::wmma::fill_fragment(
                wmma_accumulator[col * kWarpRowTiles + row], 0.0f);
        }
    }

    enum {
        kThreads =
            kBlockRowWarps * kBlockColWarps * kBlockSliceKWarps * kWarpSize,
        kLoadALanesPerRow =
            kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)) >=
                    kWarpSize
                ? kWarpSize
                : kWmmaM * kBlockRowTiles / (sizeof(float4) / sizeof(half)),
        kLoadAColsPerIter = kThreads / kLoadALanesPerRow,
        kLoadAInnerLoop = kWmmaM * kBlockRowTiles /
                          (sizeof(float4) / sizeof(half) * kLoadALanesPerRow),

        kLoadBLanesPerRow =
            kWmmaK * kBlockSliceKTiles / (sizeof(float4) / sizeof(half)) >=
                    kWarpSize
                ? kWarpSize
                : kWmmaK * kBlockSliceKTiles / (sizeof(float4) / sizeof(half)),
        kLoadBColsPerIter = kThreads / kLoadBLanesPerRow,
        kLoadBInnerLoop = kWmmaK * kBlockSliceKTiles /
                          (sizeof(float4) / sizeof(half) * kLoadBLanesPerRow),

        kReduceCLanesPerRow =
            kWmmaM * kBlockRowTiles / (sizeof(half2) / sizeof(half)) >=
                    kWarpSize
                ? kWarpSize
                : kWmmaM * kBlockRowTiles / (sizeof(half2) / sizeof(half)),
        kReduceCColsPerIter = kThreads / kReduceCLanesPerRow,
        kReduceCInnerLoop =
            kWmmaM * kBlockRowTiles /
            (sizeof(half2) / sizeof(half) * kReduceCLanesPerRow),

        kStoreCLanesPerRow = kLoadALanesPerRow,
        kStoreCColsPerIter = kLoadAColsPerIter,
        kStoreCInnerLoop = kLoadAInnerLoop,
    };

    static_assert(kWmmaK * kBlockSliceKTiles % kLoadAColsPerIter == 0);
    static_assert(kWmmaN * kBlockColTiles % kStoreCColsPerIter == 0);
    static_assert(kWmmaN * kBlockColTiles % kLoadBColsPerIter == 0);

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
    int stage = 0;
    int k_loop = 0;

    constexpr int a_dst_i_stride =
        kLoadAColsPerIter * (kWmmaM * kBlockRowTiles + kInputSkew);
    constexpr int a_dst_j_stride =
        kLoadALanesPerRow * sizeof(float4) / sizeof(half);

    constexpr int a_src_i_stride = kLoadAColsPerIter * M;
    constexpr int a_src_j_stride =
        (kLoadALanesPerRow * sizeof(float4) / sizeof(half));

    constexpr int b_dst_i_stride =
        kLoadBColsPerIter * (kWmmaK * kBlockSliceKTiles + kInputSkew);
    constexpr int b_dst_j_stride =
        kLoadBLanesPerRow * sizeof(float4) / sizeof(half);
    constexpr int b_src_i_stride = kLoadBColsPerIter * K;
    constexpr int b_src_j_stride =
        kLoadBLanesPerRow * sizeof(float4) / sizeof(half);

    // Prologue
#pragma unroll
    for (int s = 0; s < kStage - 1; ++s) {
        pipe.producer_acquire();
        half *a_dst_base =
            matrix_a_shared[(stage + s) % kStage] +
            (0 * kLoadAColsPerIter + threadIdx.x / kLoadALanesPerRow) *
                (kWmmaM * kBlockRowTiles + kInputSkew) +
            ((threadIdx.x & (kLoadALanesPerRow - 1)) + 0 * kLoadALanesPerRow) *
                sizeof(float4) / sizeof(half);

        const half *a_src_base =
            matrix_a + blockIdx.z * K * M +
            blockIdx.x * kBlockRowTiles * kWmmaM +
            ((k_loop + s) * kBlockSliceKTiles * kWmmaK + 0 * kLoadAColsPerIter +
             threadIdx.x / kLoadALanesPerRow) *
                M +
            ((threadIdx.x & (kLoadALanesPerRow - 1)) + 0 * kLoadALanesPerRow) *
                (sizeof(float4) / sizeof(half));

        half *b_dst_base =
            matrix_b_shared[(stage + s) % kStage] +
            (0 * kLoadBColsPerIter + threadIdx.x / kLoadBLanesPerRow) *
                (kWmmaK * kBlockSliceKTiles + kInputSkew) +
            ((threadIdx.x & (kLoadBLanesPerRow - 1)) + 0 * kLoadBLanesPerRow) *
                sizeof(float4) / sizeof(half);

        const half *b_src_base =
            matrix_b + blockIdx.z * N * K +
            (k_loop + s) * kBlockSliceKTiles * kWmmaK +
            (blockIdx.y * kBlockColTiles * kWmmaN + 0 * kLoadBColsPerIter +
             threadIdx.x / kLoadBLanesPerRow) *
                K +
            ((threadIdx.x & (kLoadBLanesPerRow - 1)) + 0 * kLoadBLanesPerRow) *
                (sizeof(float4) / sizeof(half));

#pragma unroll
        for (int i = 0; i < kBlockSliceKTiles * kWmmaK / kLoadAColsPerIter;
             ++i) {
#pragma unroll
            for (int j = 0; j < kLoadAInnerLoop; ++j) {
                cuda::memcpy_async(
                    a_dst_base + i * a_dst_i_stride + j * a_dst_j_stride,
                    a_src_base + i * a_src_i_stride + j * a_src_j_stride, shape,
                    pipe);
            }
        }

#pragma unroll
        for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
#pragma unroll
            for (int j = 0; j < kLoadBInnerLoop; ++j) {
                cuda::memcpy_async(
                    b_dst_base + i * b_dst_i_stride + j * b_dst_j_stride,
                    b_src_base + i * b_src_i_stride + j * b_src_j_stride, shape,
                    pipe);
            }
        }
        pipe.producer_commit();
    }

    // Soft pipeline
#pragma unroll
    for (; k_loop < (K / kBlockSliceKTiles / kWmmaK) - (kStage - 1); ++k_loop) {
        pipe.producer_acquire();

        half *a_dst_base =
            matrix_a_shared[(stage + kStage - 1) % kStage] +
            (0 * kLoadAColsPerIter + threadIdx.x / kLoadALanesPerRow) *
                (kWmmaM * kBlockRowTiles + kInputSkew) +
            ((threadIdx.x & (kLoadALanesPerRow - 1)) + 0 * kLoadALanesPerRow) *
                sizeof(float4) / sizeof(half);

        const half *a_src_base =
            matrix_a + blockIdx.z * K * M +
            blockIdx.x * kBlockRowTiles * kWmmaM +
            ((k_loop + kStage - 1) * kBlockSliceKTiles * kWmmaK +
             0 * kLoadAColsPerIter + threadIdx.x / kLoadALanesPerRow) *
                M +
            ((threadIdx.x & (kLoadALanesPerRow - 1)) + 0 * kLoadALanesPerRow) *
                (sizeof(float4) / sizeof(half));

        half *b_dst_base =
            matrix_b_shared[(stage + kStage - 1) % kStage] +
            (0 * kLoadBColsPerIter + threadIdx.x / kLoadBLanesPerRow) *
                (kWmmaK * kBlockSliceKTiles + kInputSkew) +
            ((threadIdx.x & (kLoadBLanesPerRow - 1)) + 0 * kLoadBLanesPerRow) *
                sizeof(float4) / sizeof(half);

        const half *b_src_base =
            matrix_b + blockIdx.z * N * K +
            (k_loop + kStage - 1) * kBlockSliceKTiles * kWmmaK +
            (blockIdx.y * kBlockColTiles * kWmmaN + 0 * kLoadBColsPerIter +
             threadIdx.x / kLoadBLanesPerRow) *
                K +
            ((threadIdx.x & (kLoadBLanesPerRow - 1)) + 0 * kLoadBLanesPerRow) *
                (sizeof(float4) / sizeof(half));

#pragma unroll
        for (int i = 0; i < kBlockSliceKTiles * kWmmaK / kLoadAColsPerIter;
             ++i) {
#pragma unroll
            for (int j = 0; j < kLoadAInnerLoop; ++j) {
                cuda::memcpy_async(
                    a_dst_base + i * a_dst_i_stride + j * a_dst_j_stride,
                    a_src_base + i * a_src_i_stride + j * a_src_j_stride, shape,
                    pipe);
            }
        }

#pragma unroll
        for (int i = 0; i < kBlockColTiles * kWmmaN / kLoadBColsPerIter; ++i) {
#pragma unroll
            for (int j = 0; j < kLoadBInnerLoop; ++j) {
                cuda::memcpy_async(
                    b_dst_base + i * b_dst_i_stride + j * b_dst_j_stride,
                    b_src_base + i * b_src_i_stride + j * b_src_j_stride, shape,
                    pipe);
            }
        }
        pipe.producer_commit();

        pipe.consumer_wait();
        __syncthreads();
        pipe.consumer_release();

#pragma unroll
        for (int tile_k = 0; tile_k < kWarpSliceKTiles; ++tile_k) {
#pragma unroll
            for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[tile_m],
                    (matrix_a_shared[stage] +
                     (tile_k * kBlockSliceKWarps + slicek_warp_id) * kWmmaK *
                         (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
            }
#pragma unroll
            for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_b[tile_n],
                    (matrix_b_shared[stage] +
                     (col_warp_id * kWarpColTiles + tile_n) * kWmmaN *
                         (kBlockSliceKTiles * kWmmaK + kInputSkew) +
                     (tile_k * kBlockSliceKWarps + slicek_warp_id) * kWmmaK),
                    kBlockSliceKTiles * kWmmaK + kInputSkew);
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
        // The sync is not necessary when compute time is large enough
        // __syncthreads();
    }

    // Epilogue
#pragma unroll
    for (int s = kStage - 1; s >= 1; --s) {
        k_loop = (K / kBlockSliceKTiles / kWmmaK) - s;
        pipe.consumer_wait();
        __syncthreads();
        pipe.consumer_release();

#pragma unroll
        for (int tile_k = 0; tile_k < kWarpSliceKTiles; ++tile_k) {
#pragma unroll
            for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_a[tile_m],
                    (matrix_a_shared[stage] +
                     (tile_k * kBlockSliceKWarps + slicek_warp_id) * kWmmaK *
                         (kBlockRowTiles * kWmmaM + kInputSkew) +
                     (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM),
                    kBlockRowTiles * kWmmaM + kInputSkew);
            }
#pragma unroll
            for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
                nvcuda::wmma::load_matrix_sync(
                    wmma_matrix_b[tile_n],
                    (matrix_b_shared[stage] +
                     (col_warp_id * kWarpColTiles + tile_n) * kWmmaN *
                         (kBlockSliceKTiles * kWmmaK + kInputSkew) +
                     (tile_k * kBlockSliceKWarps + slicek_warp_id) * kWmmaK),
                    kBlockSliceKTiles * kWmmaK + kInputSkew);
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
    for (int tile_m = 0; tile_m < kWarpRowTiles; ++tile_m) {
#pragma unroll
        for (int tile_n = 0; tile_n < kWarpColTiles; ++tile_n) {
            nvcuda::wmma::store_matrix_sync(
                acc_shared +
                    (slicek_warp_id * kBlockColTiles +
                     col_warp_id * kWarpColTiles + tile_n) *
                        kWmmaN * (kBlockRowTiles * kWmmaM + kAccSkew) +
                    (row_warp_id * kWarpRowTiles + tile_m) * kWmmaM,
                wmma_accumulator[tile_n * kWarpRowTiles + tile_m],
                (kBlockRowTiles * kWmmaM + kAccSkew),
                nvcuda::wmma::mem_col_major);
        }
    }

    __syncthreads();

    constexpr int c_reduce_i_stride =
        kReduceCColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);
    constexpr int c_reduce_j_stride =
        kReduceCLanesPerRow * sizeof(half2) / sizeof(half);
    constexpr int c_reduce_k_stride = kBlockColTiles * kWmmaN *
                                      (kBlockRowTiles * kWmmaM + kAccSkew) *
                                      sizeof(half) / sizeof(half2);
    half *c_reduce_base = acc_shared +
                          threadIdx.x / kReduceCLanesPerRow *
                              (kBlockRowTiles * kWmmaM + kAccSkew) +
                          (threadIdx.x & (kReduceCLanesPerRow - 1)) *
                              sizeof(half2) / sizeof(half);
#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kReduceCColsPerIter; ++i) {
#pragma unroll
        for (int j = 0; j < kReduceCInnerLoop; ++j) {
            half2 *c_reduce_src = reinterpret_cast<half2 *>(
                c_reduce_base + i * c_reduce_i_stride + j * c_reduce_j_stride);
#pragma unroll
            for (int k = 1; k < kBlockSliceKWarps; ++k) {
                *c_reduce_src += *(c_reduce_src + k * c_reduce_k_stride);
            }
        }
    }
    __syncthreads();

    constexpr int c_dst_i_stride = kStoreCColsPerIter * M;
    constexpr int c_dst_j_stride =
        kStoreCLanesPerRow * sizeof(float4) / sizeof(half);

    constexpr int c_src_i_stride =
        kStoreCColsPerIter * (kBlockRowTiles * kWmmaM + kAccSkew);
    constexpr int c_src_j_stride =
        kStoreCLanesPerRow * sizeof(float4) / sizeof(half);

    half *c_dst_base =
        matrix_c + blockIdx.z * N * M + blockIdx.x * kBlockRowTiles * kWmmaM +
        (blockIdx.y * kBlockColTiles * kWmmaN + 0 * kStoreCColsPerIter +
         threadIdx.x / kStoreCLanesPerRow) *
            M +
        ((threadIdx.x & (kStoreCLanesPerRow - 1)) + 0 * kStoreCLanesPerRow) *
            sizeof(float4) / sizeof(half);
    half *c_src_base =
        acc_shared +
        (0 * kStoreCColsPerIter + threadIdx.x / kStoreCLanesPerRow) *
            (kBlockRowTiles * kWmmaM + kAccSkew) +
        ((threadIdx.x & (kStoreCLanesPerRow - 1)) + 0 * kStoreCLanesPerRow) *
            sizeof(float4) / sizeof(half);

#pragma unroll
    for (int i = 0; i < kBlockColTiles * kWmmaN / kStoreCColsPerIter; ++i) {
#pragma unroll
        for (int j = 0; j < kStoreCInnerLoop; ++j) {
            *reinterpret_cast<float4 *>(c_dst_base + i * c_dst_i_stride +
                                        j * c_dst_j_stride) =
                *reinterpret_cast<float4 *>(c_src_base + i * c_src_i_stride +
                                            j * c_src_j_stride);
        }
    }
}