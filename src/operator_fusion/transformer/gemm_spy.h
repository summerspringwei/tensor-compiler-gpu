#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <mma.h>
#include <cuda/barrier>
#include <cuda/pipeline>

#include "cooperative_groups/memcpy_async.h"

// M, N, K = 64, 16, 16
// dim3(1, 1, 1), dim3(1, 1, 1)
const int M=16, N=16, K=64;
const int TCM=16, TCN=16, TCK=16;

template<typename T>
__global__ void __launch_bounds__(32) gemm_spy_baseline(T* A, T* B, T* C){
  __shared__ T A_shared[TCM*TCK];
  __shared__ T B_shared[TCM*TCK];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> wmma_accumulator_c[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> wmma_matrix_b[1];
  (void)nvcuda::wmma::fill_fragment(wmma_accumulator_c[0], 0.000000e+00f);

  // Load to shared memory
  // ((float4*)A_shared + threadIdx.x)[0] = ((float4*)A + 0 * K * sizeof(T) / sizeof(float4) + threadIdx.x)[0];
  // ((float4*)B_shared + threadIdx.x)[0] = ((float4*)B + 0 * K * sizeof(T) / sizeof(float4) + threadIdx.x)[0];
  const int vec_length = 8;

  for(int ki=0; ki<4; ki++){
    ((float4*)A_shared + threadIdx.x)[0] = ((float4*)A + ki * K * sizeof(T) / sizeof(float4) + threadIdx.x)[0];
    ((float4*)B_shared + threadIdx.x)[0] = ((float4*)B + ki * K * sizeof(T) / sizeof(float4) + threadIdx.x)[0];
    // __pipeline_memcpy_async(
    //   (A_shared + vec_length * threadIdx.x),
    //   (A + ki * K + vec_length * threadIdx.x), vec_length*sizeof(half));
    // __pipeline_memcpy_async(
    //   (B_shared + vec_length * threadIdx.x),
    //   (B + ki * K + vec_length * threadIdx.x), vec_length*sizeof(half));
    // __pipeline_commit();
    // __pipeline_wait_prior(0);
    (void)nvcuda::wmma::load_matrix_sync(wmma_matrix_a[0], ((half *)A_shared+0), TCK);
    (void)nvcuda::wmma::load_matrix_sync(wmma_matrix_b[0], ((half *)B_shared+0), TCK);
    (void)nvcuda::wmma::mma_sync(wmma_accumulator_c[0], wmma_matrix_a[0], wmma_matrix_b[0], wmma_accumulator_c[0]);
  }
  (void)nvcuda::wmma::store_matrix_sync((half*)C, wmma_accumulator_c[0], N, nvcuda::wmma::mem_row_major);
}


template<typename T>
__global__ void __launch_bounds__(32) gemm_spy_pipline(T* A, T* B, T* C){
  __shared__ T A_shared_0[TCM*TCK];
  __shared__ T A_shared_1[TCM*TCK];
  __shared__ T B_shared_0[TCN*TCK];
  __shared__ T B_shared_1[TCN*TCK];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> wmma_accumulator_c[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> wmma_matrix_b[1];
  (void)nvcuda::wmma::fill_fragment(wmma_accumulator_c[0], 0.000000e+00f);

  // Load to shared memory
  T* load_A_shared=&(A_shared_1[0]), *gemm_A_shared=&(A_shared_0[0]);
  T* load_B_shared=&(B_shared_1[0]), *gemm_B_shared=&(B_shared_0[0]); 
  // ((float2*)load_A_shared + threadIdx.x)[0] = ((float2*)A + 0 * K * sizeof(T) / sizeof(float2) + threadIdx.x)[0];
  // ((float2*)load_B_shared + threadIdx.x)[0] = ((float2*)B + 0 * K * sizeof(T) / sizeof(float2) + threadIdx.x)[0];
  const int vec_length = 8;
  __pipeline_memcpy_async(
    (gemm_A_shared + vec_length * threadIdx.x),
    (A + 0 * K + vec_length * threadIdx.x), vec_length*sizeof(half));
  __pipeline_memcpy_async(
    (gemm_B_shared + vec_length * threadIdx.x),
    (B + 0 * K + vec_length * threadIdx.x), vec_length*sizeof(half));
  __pipeline_commit();
  __pipeline_wait_prior(0);

  for(int ki=1; ki<4; ki++){
    __pipeline_memcpy_async(
      (load_A_shared + vec_length * threadIdx.x),
      (A + ki * K + vec_length * threadIdx.x), vec_length*sizeof(half));
    __pipeline_memcpy_async(
      (load_B_shared + vec_length * threadIdx.x),
      (B + ki * K + vec_length * threadIdx.x), vec_length*sizeof(half));
    (void)nvcuda::wmma::load_matrix_sync(wmma_matrix_a[0], ((half *)gemm_A_shared+0), TCK);
    (void)nvcuda::wmma::load_matrix_sync(wmma_matrix_b[0], ((half *)gemm_B_shared+0), TCK);
    (void)nvcuda::wmma::mma_sync(wmma_accumulator_c[0], wmma_matrix_a[0], wmma_matrix_b[0], wmma_accumulator_c[0]);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    T* tmp_a = load_A_shared; load_A_shared=gemm_A_shared; gemm_A_shared=tmp_a;
    T* tmp_b = load_B_shared; load_B_shared=gemm_B_shared; gemm_B_shared=tmp_b;
    __syncthreads();
  }
  (void)nvcuda::wmma::load_matrix_sync(wmma_matrix_a[0], ((half *)gemm_A_shared+0), TCK);
  (void)nvcuda::wmma::load_matrix_sync(wmma_matrix_b[0], ((half *)gemm_B_shared+0), TCK);
  (void)nvcuda::wmma::mma_sync(wmma_accumulator_c[0], wmma_matrix_a[0], wmma_matrix_b[0], wmma_accumulator_c[0]);

  (void)nvcuda::wmma::store_matrix_sync((half*)C, wmma_accumulator_c[0], N, nvcuda::wmma::mem_row_major);
}





template<typename T>
__global__ void __launch_bounds__(32) gemm_spy_pipline_v2(T* A, T* B, T* C){
  __shared__ T A_shared_0[TCM*TCK];
  __shared__ T A_shared_1[TCM*TCK];
  __shared__ T B_shared_0[TCN*TCK];
  __shared__ T B_shared_1[TCN*TCK];
  nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, half> wmma_accumulator_c[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, half, nvcuda::wmma::row_major> wmma_matrix_a[1];
  nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, half, nvcuda::wmma::row_major> wmma_matrix_b[1];
  (void)nvcuda::wmma::fill_fragment(wmma_accumulator_c[0], 0.000000e+00f);

  // Load to shared memory
  T* load_A_shared=&(A_shared_1[0]), *gemm_A_shared=&(A_shared_0[0]);
  T* load_B_shared=&(B_shared_1[0]), *gemm_B_shared=&(B_shared_0[0]); 
  // ((float2*)load_A_shared + threadIdx.x)[0] = ((float2*)A + 0 * K * sizeof(T) / sizeof(float2) + threadIdx.x)[0];
  // ((float2*)load_B_shared + threadIdx.x)[0] = ((float2*)B + 0 * K * sizeof(T) / sizeof(float2) + threadIdx.x)[0];
  const int vec_length = 8;
  cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();
  const auto shape = cuda::aligned_size_t<alignof(float4)>(sizeof(float4));
  pipe.producer_acquire();

  cuda::memcpy_async(
    (gemm_A_shared + vec_length * threadIdx.x),
    (A + 0 * K + vec_length * threadIdx.x), shape, pipe);
  cuda::memcpy_async(
    (gemm_B_shared + vec_length * threadIdx.x),
    (B + 0 * K + vec_length * threadIdx.x), shape, pipe);
  pipe.producer_commit();
  pipe.consumer_wait();

  for(int ki=1; ki<4; ki++){
    cuda::memcpy_async(
      (load_A_shared + vec_length * threadIdx.x),
      (A + ki * K + vec_length * threadIdx.x), shape, pipe);
    cuda::memcpy_async(
      (load_B_shared + vec_length * threadIdx.x),
      (B + ki * K + vec_length * threadIdx.x), shape, pipe);
    (void)nvcuda::wmma::load_matrix_sync(wmma_matrix_a[0], ((half *)gemm_A_shared+0), TCK);
    (void)nvcuda::wmma::load_matrix_sync(wmma_matrix_b[0], ((half *)gemm_B_shared+0), TCK);
    (void)nvcuda::wmma::mma_sync(wmma_accumulator_c[0], wmma_matrix_a[0], wmma_matrix_b[0], wmma_accumulator_c[0]);
    pipe.producer_commit();
    pipe.consumer_wait();
    T* tmp_a = load_A_shared; load_A_shared=gemm_A_shared; gemm_A_shared=tmp_a;
    T* tmp_b = load_B_shared; load_B_shared=gemm_B_shared; gemm_B_shared=tmp_b;
    __syncthreads();
  }
  pipe.consumer_release();
  (void)nvcuda::wmma::load_matrix_sync(wmma_matrix_a[0], ((half *)gemm_A_shared+0), TCK);
  (void)nvcuda::wmma::load_matrix_sync(wmma_matrix_b[0], ((half *)gemm_B_shared+0), TCK);
  (void)nvcuda::wmma::mma_sync(wmma_accumulator_c[0], wmma_matrix_a[0], wmma_matrix_b[0], wmma_accumulator_c[0]);

  (void)nvcuda::wmma::store_matrix_sync((half*)C, wmma_accumulator_c[0], N, nvcuda::wmma::mem_row_major);
}
