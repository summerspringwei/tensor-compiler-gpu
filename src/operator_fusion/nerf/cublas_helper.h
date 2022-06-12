#include <cudnn.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <cublas.h>
#include <cuda.h>
#include <cuda_runtime.h>

template<typename T, cublasOperation_t A_LAYOUT, cublasOperation_t B_LAYOUT>
void cudnn_matmul_wrapper(T* A, T* B, T* C, int m, int n, int k){
  T alpha = 1.0, beta=0.0;
  cublasHandle_t handle;
  cublasCreate_v2(&handle);
  cudaDataType_t cuda_dtype=cudaDataType_t::CUDA_R_32F;
  if(std::is_same<T, half>::value){
    cuda_dtype = cudaDataType_t::CUDA_R_16F;
  }else if(std::is_same<T, float>::value){
    cuda_dtype = cudaDataType_t::CUDA_R_32F;
  }
  auto cublasStat = cublasGemmEx(handle, 
        A_LAYOUT, B_LAYOUT, 
        m, n, k, 
        (const void *)&alpha, 
        A, cuda_dtype, m, 
        B, cuda_dtype, n, 
        (const void *)&beta, 
        C, cuda_dtype, m, 
        cuda_dtype, 
        cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP);
  cudaDeviceSynchronize();
}