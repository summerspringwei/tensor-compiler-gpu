#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <chrono>

#include <cudnn.h>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include <cublas.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "../../cuda_utils.h"

#define CUDNN_CALL(f) { \
  cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    std::cout \
        << "    Error occurred: " << err << " at " << __LINE__ << std::endl; \
    std::exit(1); \
  } \
}

void bench_softmax(){
  half *input = new half[614656];
  half *output = new half[614656];

  cudaError_t err = cudaSuccess;
  half *d_input=NULL;
  half *d_output=NULL;
  half *d_alpha=NULL;
  half *d_beta=NULL;
  err=cudaMalloc((void **)&d_input, sizeof(half)*614656);
  err=cudaMalloc((void **)&d_output, sizeof(half)*614656);

  cudaMemcpy(d_input, input, sizeof(half)*614656, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  cudaMemcpy(output, d_output, sizeof(half)*614656, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  cudnnHandle_t handle;
  CUDNN_CALL(cudnnCreate(&handle));
  cudnnTensorDescriptor_t x_desc, y_desc;
  cudnnCreateTensorDescriptor(&x_desc);
  cudnnCreateTensorDescriptor(&y_desc);
  cudnnSetTensor4dDescriptor(
        x_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF,
        64, 4, 49, 49);
  cudnnSetTensor4dDescriptor(
        y_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_HALF,
        64, 4, 49, 49);
  int loop=10000;
  half alpha=1.0, beta=0.0;
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  
  for(int a=0; a<3; ++a){
    float sum = 0, ms=0;
    auto t1 = std::chrono::steady_clock::now();
    for(int i=0; i<loop; ++i){

      // cudaEventRecord(startEvent,0);
      auto result = cudnnSoftmaxForward(handle, 
        cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE, 
        cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_CHANNEL, 
        &alpha, x_desc, d_input, &beta, y_desc, d_output);
      // checkCuda( cudaEventRecord(stopEvent,0) );
      // checkCuda( cudaEventSynchronize(stopEvent) );
      // checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
      // sum += ms;
    }
    auto t2 = std::chrono::steady_clock::now();
    double latency = std::chrono::duration<double, std::micro>(t2-t1).count();
    printf("chrono: %f\n", latency / loop);
    // printf("%f\n", sum/loop);
  }

  cudnnDestroy(handle);

  delete[] input;
  delete[] output;
  cudaFree(d_input);
  cudaFree(d_output);
}


void bench_query_key(){
  cublasStatus_t cublasHgemm(cublasHandle_t handle,
                           cublasOperation_t transa, cublasOperation_t transb,
                           int m, int n, int k,
                           const __half *alpha,
                           const __half *A, int lda,
                           const __half *B, int ldb,
                           const __half *beta,
                           __half *C, int ldc);
  const int input_size=64*4*64*32;     half *input = new half[input_size];
  const int weight_size=64*4*64*32;    half *weight = new half[weight_size];
  const int output_size=64*4*64*64;    half *output = new half[output_size];

  cudaError_t err = cudaSuccess;
  half *d_input=NULL;
  half *d_weight=NULL;
  half *d_output=NULL;
  err=cudaMalloc((void **)&d_input, sizeof(half)*input_size);
  err=cudaMalloc((void **)&d_weight, sizeof(half)*weight_size);
  err=cudaMalloc((void **)&d_output, sizeof(half)*output_size);

  cudaMemcpy(d_input, input, sizeof(half)*input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, weight, sizeof(half)*weight_size, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  cudaMemcpy(output, d_output, sizeof(half)*output_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  half alpha = 1.0, beta=0.0;

  cublasHandle_t handle;
  cublasCreate_v2(&handle);
  // int m = 49, n = 49, k = 32;
  int m = 64, n = 64, k = 32;
  int batch_size = 64 * 4;
  const void * Aarray[batch_size];
  const void * Barray[batch_size];
  void * Carray[batch_size];
  for(int i=0; i<batch_size; ++i){
    Aarray[i] = d_input + i * m * k;
    Barray[i] = d_weight + i * n * k;
    Carray[i] = d_output + i * m * n;
  }
  
  int loop=10000;
  for(int a=0; a<3; ++a){
    auto t1 = std::chrono::steady_clock::now();
    for(int i=0; i<loop; ++i){
      // cublasHgemmBatched(handle, 
      //   cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_T, 
      //   m, n, k,
      //   &alpha,
      //   Aarray, m,
      //   Barray, n,
      //   &beta,
      //   Carray, m,
      //   batch_size
      // );
      cublasGemmBatchedEx(handle, 
        cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_T, 
        m, n, k,
        (const void *)&alpha,
        Aarray, cudaDataType::CUDA_R_16F, m,
        Barray, cudaDataType::CUDA_R_16F, n,
        &beta,
        Carray, cudaDataType_t::CUDA_R_16F, m,
        batch_size,
        cublasComputeType_t::CUBLAS_COMPUTE_16F,
        cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP
      );
    }
    auto t2 = std::chrono::steady_clock::now();
    double latency = std::chrono::duration<double, std::micro>(t2-t1).count();
    printf("avg chrono: %f\n", latency / loop);
  }
  delete[] input;
  delete[] weight;
  delete[] output;
  cudaFree(d_input);
  cudaFree(d_weight);
  cudaFree(d_output);
}

template<typename T>
void bench_matmul(int m, int n, int k){

  int input_size = m * k;
  int weight_size = n * k;
  int output_size = m * n;

  half *input = new half[input_size];
  half *weight = new half[weight_size];
  half *output = new half[output_size];

  cudaError_t err = cudaSuccess;
  void *d_input=NULL;
  void *d_weight=NULL;
  void *d_output=NULL;
  err=cudaMalloc((void **)&d_input, sizeof(T)*input_size);
  err=cudaMalloc((void **)&d_weight, sizeof(T)*weight_size);
  err=cudaMalloc((void **)&d_output, sizeof(T)*output_size);

  cudaMemcpy((void*)d_input, input, sizeof(T)*input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight, weight, sizeof(T)*weight_size, cudaMemcpyHostToDevice);

  T alpha = 1.0, beta=0.0;

  cublasHandle_t handle;
  cublasCreate_v2(&handle);
  cudaDataType_t cuda_dtype=cudaDataType_t::CUDA_R_32F;
  if(std::is_same<T, half>::value){
    cuda_dtype = cudaDataType_t::CUDA_R_16F;
  }else if(std::is_same<T, float>::value){
    cuda_dtype = cudaDataType_t::CUDA_R_32F;
  }
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  int loop=10000;
  double min_latency = 1e9;
  for(int a=0; a<3; ++a){
    auto t1 = std::chrono::steady_clock::now();
    float sum = 0;
    for(int i=0; i<loop; ++i){
      float ms = 0;
      cudaEventRecord(startEvent,0);
      auto cublasStat = cublasGemmEx(handle, 
        cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_T, 
        m, n, k, 
        (const void *)&alpha, 
        d_input, cuda_dtype, m, 
        d_weight, cuda_dtype, n, 
        (const void *)&beta, 
        d_output, cuda_dtype, m, 
        cuda_dtype, 
        cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT_TENSOR_OP);

        checkCuda( cudaEventRecord(stopEvent,0) );
        checkCuda( cudaEventSynchronize(stopEvent) );
        checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
        sum += ms;
    }
    printf("sum: %f\n", sum/loop);
    auto t2 = std::chrono::steady_clock::now();
    double latency = std::chrono::duration<double, std::micro>(t2-t1).count() / loop;
    if(latency < min_latency){
      min_latency = latency;
    }
  }
  printf("avg chrono: %f\n", min_latency);
  cudaDeviceSynchronize();
  cudaMemcpy(output, d_output, sizeof(half)*output_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  delete[] input;
  delete[] weight;
  delete[] output;
  cudaFree(d_input);
  cudaFree(d_weight);
  cudaFree(d_output);
}


void bench_qvk_matmul(int batch_size, int height, int width, int channel){
  bench_matmul<half>(batch_size * height * width, 3*channel, channel);
}


void bench_FFN_fc1(int batch_size, int height, int width, int channel){
  bench_matmul<half>(batch_size * height * width, 4*channel, channel);
}

void bench_FFN_fc2(int batch_size, int height, int width, int channel){
  bench_matmul<half>(batch_size * height * width, channel, 4*channel);
}


int main(){
  // bench_softmax();
  // bench_query_key();
  // bench_qvk_matmul(1, 64, 64, 128);
  // bench_qvk_matmul(1, 32, 32, 256);
  // bench_qvk_matmul(1, 16, 16, 512);
  // bench_qvk_matmul(1, 8, 8, 1024);
  bench_FFN_fc1(1, 64, 64, 1280);
  bench_FFN_fc1(1, 32, 32, 256);
  bench_FFN_fc1(1, 16, 16, 512);
  bench_FFN_fc1(1, 8, 8, 1024);
  // bench_FFN_fc2(1, 64, 64, 128);
  // bench_FFN_fc2(1, 32, 32, 256);
  // bench_FFN_fc2(1, 16, 16, 512);
  // bench_FFN_fc2(1, 8, 8, 1024);
  return 0;
}
