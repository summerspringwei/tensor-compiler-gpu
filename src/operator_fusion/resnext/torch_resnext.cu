#include <iostream>
#include <vector>
#include <math.h>
#include <sstream>

#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "torch/all.h"

#include "kernels/resnext_fused_conv1x1_14_14_1024_16x64.cu"
#include "kernels/resnext_fused_conv3x3_14_14_16x64.cu"
#include "kernels/fused_conv1x1_conv3x3_14_14_16x64.cu"

#include "npy.hpp"

#include "../../utils.h"
#include "../../cuda_utils.h"
#include "../torch_utils.h"


template<int64_t batch, int64_t height, int64_t width, int64_t in_channel, int64_t out_channel, int64_t groups>
void resnext_module(int round_cout=1, int loop=1, int func_id=0){
  auto input = torch::ones({batch, height, width, in_channel}, options_fp32);
  auto conv1x1_weight = torch::ones({groups, in_channel, out_channel}, options_fp32);
  auto conv1x1_output = torch::zeros({batch, groups, height, width, out_channel}, options_fp32);
  auto bnw1 = torch::ones({1, groups, out_channel}, options_fp32);
  auto bnw2 = torch::ones({1, groups, out_channel}, options_fp32);
  auto conv3x3_weight = torch::ones({groups, 3, 3, out_channel, out_channel}, options_fp32);
  auto conv3x3_output = torch::zeros({batch, groups, height, width, out_channel}, options_fp32);
  auto conv1x1_profile_clock = torch::zeros({18, 224, 4}, options_int64);

  float* ptr_input = input.data<float>();
  float* ptr_conv1x1_weight = conv1x1_weight.data<float>();
  float* ptr_conv1x1_output = conv1x1_output.data<float>();
  float* ptr_bnw1 = bnw1.data<float>();
  float* ptr_bnw2 = bnw2.data<float>();
  float* ptr_conv3x3_weight = conv3x3_weight.data<float>();
  float* ptr_conv3x3_output = conv3x3_output.data<float>();
  int64_t* ptr_conv1x1_profile_clock = conv1x1_profile_clock.data<int64_t>();

  void* conv1x1_kernel_args[] = {
    (void *)&(ptr_input), 
    (void *)&(ptr_conv1x1_weight), 
    (void *)&(ptr_conv1x1_output), 
    (void *)&(ptr_bnw1), 
    (void *)&(ptr_bnw2),
    (void *)&(ptr_conv1x1_profile_clock)
  };
  void* conv3x3_kernel_args [] = {
    (void *)&(ptr_conv1x1_output),
    (void *)&(ptr_conv3x3_weight),
    (void *)&(ptr_conv3x3_output)
  };
  void* fused_conv1x1_conv3x3_kernel_args[] = {
    (void *)&(ptr_input), 
    (void *)&(ptr_conv1x1_weight), 
    (void *)&(ptr_conv1x1_output), 
    (void *)&(ptr_bnw1), 
    (void *)&(ptr_bnw2),
    (void *)&(ptr_conv1x1_profile_clock),
    (void *)&(ptr_conv3x3_weight),
    (void *)&(ptr_conv3x3_output)
  };
  int conv1x1NumBlocksPerSm = 0, conv1x1NumThreads=112;
  int conv3x3NumBlocksPerSm = 0, conv3x3NumThreads=128;
  int fusedConvNumBlocksPerSm = 0, fuseConvNumThreads=128;
  const size_t shared_memory_size = 48*1024;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&conv1x1NumBlocksPerSm, resnext_fused_conv1x1_14_14_1024_16x64, conv1x1NumThreads, shared_memory_size); 
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&conv3x3NumBlocksPerSm, resnext_fused_conv3x3_14_14_16x64, conv3x3NumThreads, shared_memory_size); 
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fusedConvNumBlocksPerSm, fused_conv1x1_conv3x3_14_14_16x64, fuseConvNumThreads, shared_memory_size); 
  printf("Conv1x1 OccupancyMaxActiveBlocksPerMultiprocessor: %d\n", conv1x1NumBlocksPerSm);
  printf("Conv3x3 OccupancyMaxActiveBlocksPerMultiprocessor: %d\n", conv3x3NumBlocksPerSm);
  printf("fusedConv OccupancyMaxActiveBlocksPerMultiprocessor: %d\n", fusedConvNumBlocksPerSm);
  checkCuda(cudaFuncSetAttribute((void*)resnext_fused_conv1x1_14_14_1024_16x64, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size), __LINE__);
  checkCuda(cudaFuncSetAttribute((void*)resnext_fused_conv3x3_14_14_16x64, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size), __LINE__);
  checkCuda(cudaFuncSetAttribute((void*)fused_conv1x1_conv3x3_14_14_16x64, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size), __LINE__);
  
  auto device_func = [&](int func_id){
    switch (func_id)
    {
    case 0:
      checkCuda(cudaLaunchCooperativeKernel((const void*)resnext_fused_conv1x1_14_14_1024_16x64, 
        dim3(224, 1, 1), dim3(112, 1, 1), conv1x1_kernel_args, shared_memory_size), __LINE__);
      break;
    case 1:
      resnext_fused_conv3x3_14_14_16x64_v0<<<dim3(224, 1, 1), dim3(128, 1, 1), shared_memory_size>>>(ptr_conv1x1_output, ptr_conv3x3_weight, ptr_conv3x3_output);
      break;
    case 2:
      checkCuda(cudaLaunchCooperativeKernel((const void*)resnext_fused_conv3x3_14_14_16x64, 
        dim3(224, 1, 1), dim3(128, 1, 1), conv3x3_kernel_args, shared_memory_size), __LINE__);
      break;
    case 3:
      checkCuda(cudaLaunchCooperativeKernel((const void*)fused_conv1x1_conv3x3_14_14_16x64, 
        dim3(224, 1, 1), dim3(128, 1, 1), fused_conv1x1_conv3x3_kernel_args, shared_memory_size), __LINE__);
      break;
    default:
      break;
    }
    
  };

  // Benchmark
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));

  // Warm up
  for(int i=0; i<1; ++i){
    device_func(func_id);
  }
  
  // 1. For original pointwise conv
  float min_avg = 1e10;
  for(int round =0; round<round_cout; ++round){
    float ms = 0, latency_sum = 0;
    for(int i=0; i<loop; ++i){
      checkCuda( cudaEventRecord(startEvent,0) );
      device_func(func_id);
      checkCuda( cudaEventRecord(stopEvent,0) );
      checkCuda( cudaEventSynchronize(stopEvent) );
      checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
      latency_sum += ms;
    }
    auto avg = latency_sum/loop;
    if(avg<min_avg){
      min_avg = avg;
    }
    printf("Run iter %d loops %d finished, avg %f us\n", round, loop, min_avg * 1000);
  }

  checkCuda(cudaEventDestroy(startEvent));
  checkCuda(cudaEventDestroy(stopEvent));

  torch::print(conv1x1_output);
  torch::print(conv3x3_output);
  // torch::save(conv1x1_profile_clock, "conv1x1_profile_clock.pt");
}


int main(int argc, char** argv){
  int round = 1, loop = 1, type=0;
  if(argc>2){
    round = atoi(argv[1]);
    loop = atoi(argv[2]);
  }if(argc>3){
    type = atoi(argv[3]);
  }
  resnext_module<1, 14, 14, 1024, 16, 64>(round, loop, type);
  return 0;
}
