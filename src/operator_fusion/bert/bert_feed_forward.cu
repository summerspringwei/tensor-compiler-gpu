
#include <assert.h>
#include <cuda.h>
#include <cuda/pipeline>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <stdio.h>

#include "../../cuda_utils.h"
#include "../../utils.h"
#include "../transformer/half.hpp"

#include "kernels/bert_feed_forward_fc1.h"
#include "kernels/bert_feed_forward_fc2.h"
#include "kernels/bert_fused_fc_fc.h"

// Two fc configuration
int m1 = 128, n1 = 3072, k1 = 768;
int m2 = m1, n2 = 768, k2 = n1;

void init_values(half *input, half *weight1, half *weight2, half *output) {
  hf_init_values<half>(input, {m1 * k1}, half(1.0/16), 0);
  hf_init_values<half>(weight1, {n1 * k1}, half(1.0/16), 0);
  hf_init_values<half>(weight2, {n2 * k2}, half(1.0/64), 0);
  hf_init_values<half>(output, {m2 * n2}, 1, 0);
}

#define FUNC1                                                                  \
  checkCuda(cudaLaunchCooperativeKernel((void *)fc1_128_768_3072,              \
                                        dim3(2, 96, 1), dim3(32, 2, 2),        \
                                        kernel_args1, 13056 * sizeof(half)));
#define FUNC2                                                                  \
  checkCuda(cudaLaunchCooperativeKernel((void *)fc2, dim3(4, 24, 1),           \
                                        dim3(32, 4, 1), kernel_args2,          \
                                        8704 * sizeof(half)));
#define FUSED_FUNC                                                             \
  checkCuda(cudaLaunchCooperativeKernel(                                       \
      (void *)fused_fc_fc_v2, dim3(192, 1, 1), dim3(128, 1, 1),                \
      fused_kernel_args, 13056 * sizeof(half)));



void cpu_half_gemm(half* input, half* weight, half* output, int m1, int n1, int k1){
  for(int i=0; i<m1; ++i){
    for(int j=0; j<n1; ++j){
      half_float::half sum = half_float::half_cast<half_float::half, int>(0);
      for(int k=0; k<k1; ++k){
        sum += half_float::half(__half2float(input[i * k1 + k])) * half_float::half(__half2float(weight[j*k1 + k]));
      }
      ((half*)(output + i*n1+j))[0] = ((half*)(&sum))[0];
    }
  }
}

void cpu_fused_bert_feed_forward(half* input, half* weight1, half* output_tmp, half* weight2, float* sum, float* variance, half* output_cmp){
  cpu_half_gemm(input, weight1, output_tmp, m1, n1, k1);
  cpu_half_gemm(output_tmp, weight2, output_cmp, m2, n2, k2);
  // Add
  for(int i=0; i<m1; ++i){
    for(int j=0; j<k1; ++j){
      auto add = half_float::half(__half2float(input[i*k1+j])) + half_float::half(__half2float(output_cmp[i*k1+j]));
      ((half*)output_tmp+i*k1+j)[0] = ((half*)&add)[0];
    }
  }
  // LayerNorm 1. sum
  for(int i=0; i<m1; ++i){
    half_float::half reduce_sum = half_float::half_cast<half_float::half, int>(0);
    for(int j=0; j<k1; ++j){
      reduce_sum += half_float::half(__half2float(output_tmp[i*k1+j]));
    }
    reduce_sum = reduce_sum / half_float::half_cast<half_float::half, int>(k1);
    ((half*)sum+i)[0] = ((half*)&reduce_sum)[0];
  }
  // LayerNorm 2. variance
  for(int i=0; i<m1; ++i){
    half_float::half reduce_sum = half_float::half_cast<half_float::half, int>(0);
    for(int j=0; j<k1; ++j){
      reduce_sum  += half_float::half((__half2float(output_tmp[i*k1+j]) -sum[i])) * half_float::half((__half2float(output_tmp[i*k1+j]) -sum[i]));
    }
    reduce_sum = reduce_sum / half_float::half_cast<half_float::half, int>(k1);
    ((half*)variance+i)[0] = ((half*)&reduce_sum)[0];
  }
  for(int i=0; i<m1; ++i){
    for(int j=0; j<k1; ++j){
      auto tmp = ((half_float::half)(output_tmp[i*k1+j]) - (half_float::half)sum[i]) / half_float::sqrt(half_float::half(variance[i]) + half_float::half(0.00001));
      ((half*)(output_cmp+i*k1+j))[0] = ((half*)&tmp)[0];
    }
  }
}


int main() {
  int error_cnt_threshold = 100000;

  const int input_size = m1 * k1;
  half *input = new half[input_size];
  const int weight1_size = n1 * k1;
  half *weight1 = new half[weight1_size];
  const int weight2_size = n2 * k2;
  half *weight2 = new half[weight2_size];
  const int output1_size = m1 * n1;
  half *output1 = new half[output1_size];
  const int output2_size = m2 * n2;
  half *output2 = new half[output2_size];
  float *sum = new float[m1];
  float *variance = new float[m1];
  float *sum_cmp = new float[m1];
  float *variance_cmp = new float[m1];
  half *output_tmp = new half[output1_size];
  half *output_cmp = new half[output2_size];

  init_values(input, weight1, weight2, output1);

  // cpu_fused_bert_feed_forward(input, weight1, output_tmp, weight2, sum_cmp, variance_cmp, output_cmp);
  int dev = 0;
  int supportsCoopLaunch = 0;
  cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch,
                         dev);
  if (supportsCoopLaunch) {
    printf("Device support CoopLaunch\n");
  }

  cudaError_t err = cudaSuccess;
  half *d_input = NULL;
  half *d_weight1 = NULL;
  half *d_weight2 = NULL;
  half *d_output1 = NULL;
  half *d_output2 = NULL;
  half *d_sum = NULL;
  half *d_variance = NULL;
  half *d_output_tmp = NULL;
  half *d_output_cmp = NULL;
  err = cudaMalloc((void **)&d_input, sizeof(half) * input_size);
  err = cudaMalloc((void **)&d_weight1, sizeof(half) * weight1_size);
  err = cudaMalloc((void **)&d_weight2, sizeof(half) * weight2_size);
  err = cudaMalloc((void **)&d_output1, sizeof(half) * output1_size);
  err = cudaMalloc((void **)&d_output2, sizeof(half) * output2_size);
  err = cudaMalloc((void **)&d_sum, sizeof(float) * m1);
  err = cudaMalloc((void **)&d_variance, sizeof(float) * m1);
  err = cudaMalloc((void **)&d_output2, sizeof(half) * output2_size);
  err = cudaMalloc((void **)&d_output_tmp, sizeof(half) * output1_size);
  err = cudaMalloc((void **)&d_output_cmp, sizeof(half) * output2_size);

  cudaMemcpy(d_input, input, sizeof(half) * input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight1, weight1, sizeof(half) * weight1_size,
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_weight2, weight2, sizeof(half) * weight2_size,
             cudaMemcpyHostToDevice);

  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);
  int numBlocksPerSm;
  int numThreads = 128;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm, (void *)fc1_128_768_3072, numThreads, 0);
  printf("fc1: OccupancyMaxActiveBlocksPerMultiprocessor: %d, "
         "multiProcessorCount: %d\n",
         numBlocksPerSm, deviceProp.multiProcessorCount);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, (void *)fc2,
                                                numThreads, 0);
  printf("fc2: OccupancyMaxActiveBlocksPerMultiprocessor: %d, "
         "multiProcessorCount: %d\n",
         numBlocksPerSm, deviceProp.multiProcessorCount);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm, (void *)fused_fc_fc_v2, numThreads, 0);
  printf("fused_fc_fc: OccupancyMaxActiveBlocksPerMultiprocessor: %d, "
         "multiProcessorCount: %d\n",
         numBlocksPerSm, deviceProp.multiProcessorCount);

  void *kernel_args1[] = {(void *)&(d_input), (void *)&(d_weight1),
                          (void *)&(d_output_tmp)};
  void *kernel_args2[] = {(void *)&(d_output_tmp), (void *)&(d_weight2),
                          (void *)&(d_output_cmp)};
  half eps = 0.00001, beta = 0, gama = 1;
  void *fused_kernel_args[] = {(void *)&(d_input), (void *)&(d_weight1),
                               (void *)&(d_output1), (void *)&(d_weight2),
                               (void *)&(d_output2), (void *)&(d_sum), 
                               (void *)&(d_variance), (void *)&(eps), 
                               (void *)&(gama), (void *)&(beta)};
  // FUNC1
  // FUNC2
  FUSED_FUNC
  cudaDeviceSynchronize();
  cudaMemcpy(output1, d_output1, sizeof(half) * output1_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(output2, d_output2, sizeof(half) * output2_size,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(sum, d_sum, sizeof(float) * m1,
             cudaMemcpyDeviceToHost);
  cudaMemcpy(variance, d_variance, sizeof(float) * m1,
             cudaMemcpyDeviceToHost);
  // cudaMemcpy(output_tmp, d_output_tmp, sizeof(half) * output1_size,
  //            cudaMemcpyDeviceToHost);
  // cudaMemcpy(output_cmp, d_output_cmp, sizeof(half) * output2_size,
  //            cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();
  
  // Compare results
  int error_cnt = 0;

  // for(int i=0; i<m1*n1; ++i){
  // 	if( abs(__half2float(output1[i]) - __half2float(output_tmp[i])) >
  // ((__half2float(output1[i]) + __half2float(output_tmp[i])) / 1000)){
  // 		// printf("(%d, %d): %f\n", i/256, i%256,
  // __half2float(output[i])); 		printf("(%d, %d): ours: %f, tvm: %f\n", i/n1,
  // i%n1, __half2float(output1[i]), __half2float(output_tmp[i]));
  // 	}
  // }printf("\n");
  // for (int i = 0; i < m1; ++i) {
  //   if (abs(__half2float(sum[i]) - half_float::half(sum_cmp[i])) >
  //           ((__half2float(sum[i]) + half_float::half(sum_cmp[i])) / 1000)){  
  //     printf("sum: (%d, %d): ours: %f, tvm: %f\n", i / n2, i % n2,
  //            __half2float(sum[i]), half_float::half(sum_cmp[i]));
  //     error_cnt++;
  //     if (error_cnt > error_cnt_threshold) {
  //       break;
  //     }
  //   }
  // }

  // for (int i = 0; i < m1; ++i) {
  //   if (abs(__half2float(variance[i]) - __half2float(variance_cmp[i])) >
  //           ((__half2float(variance[i]) + __half2float(variance_cmp[i])) / 1000)){  
  //     printf("variance: (%d, %d): ours: %f, tvm: %f\n", i / n2, i % n2,
  //            __half2float(variance[i]), __half2float(variance_cmp[i]));
  //     error_cnt++;
  //     if (error_cnt > error_cnt_threshold) {
  //       break;
  //     }
  //   }
  // }
  // for (int i = 0; i < m2 * n2; ++i) {
  //   if (abs(__half2float(output2[i]) - __half2float(output_cmp[i])) >
  //           ((__half2float(output2[i]) + __half2float(output_cmp[i])) / 1000)) {
  //     printf("output2: (%d, %d): ours: %f, tvm: %f\n", i / n2, i % n2,
  //            __half2float(output2[i]), __half2float(output_cmp[i]));
  //     error_cnt++;
  //     if (error_cnt > error_cnt_threshold) {
  //       break;
  //     }
  //   }
  // }
  // printf("\n");
  // printf("error_cnd: %d\n", error_cnt);

  if (err != cudaSuccess) {
    fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n",
            cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  const int round_cout = 3, loop = 10000;
  float ms = 0, latency_sum = 0;
  // 1. For original pointwise conv
  for (int round = 0; round < round_cout; ++round) {
    ms = 0, latency_sum = 0;
    for (int i = 0; i < loop; ++i) {
      checkCuda(cudaEventRecord(startEvent, 0));
      FUNC1
      FUNC2
      checkCuda(cudaEventRecord(stopEvent, 0));
      checkCuda(cudaEventSynchronize(stopEvent));
      checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
      latency_sum += ms;
    }
    printf("Before fuse avg time %f\n", latency_sum / loop);
  }

  for (int round = 0; round < round_cout; ++round) {
    ms = 0, latency_sum = 0;
    for (int i = 0; i < loop; ++i) {
      checkCuda(cudaEventRecord(startEvent, 0));
      FUSED_FUNC
      checkCuda(cudaEventRecord(stopEvent, 0));
      checkCuda(cudaEventSynchronize(stopEvent));
      checkCuda(cudaEventElapsedTime(&ms, startEvent, stopEvent));
      latency_sum += ms;
    }
    printf("After fuse avg time %f\n", latency_sum / loop);
  }

  delete[] input;
  delete[] weight1;
  delete[] weight2;
  delete[] output1;
  delete[] output2;
  delete[] output_cmp;
  delete[] output_tmp;
  cudaFree(d_input);
  cudaFree(d_weight1);
  cudaFree(d_weight2);
  cudaFree(d_output1);
  cudaFree(d_output2);
  cudaFree(d_output_tmp);
  cudaFree(d_output_cmp);
  return 0;
}
