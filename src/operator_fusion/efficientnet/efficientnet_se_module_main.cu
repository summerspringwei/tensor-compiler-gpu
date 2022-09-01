#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>

#include "torch/all.h"

#include "../../utils.h"
#include "../../cuda_utils.h"
#include "../torch_utils.h"

#include "se_module.cu"


template<int64_t batch, int64_t height, int64_t width, int64_t in_channel, int64_t reduce_channel, int64_t tile_size_in_channel>
void efficient_se_module(int round_cout=1, int loop=1, int func_id=0, size_t shared_memory_size=48*1024){
  auto input = torch::ones({batch, height, width, in_channel}, options_fp32);
  auto reduce_output = torch::ones({batch, in_channel}, options_fp32);
  auto se_reduce_weight = torch::ones({in_channel, reduce_channel}, options_fp32);
  auto se_reduce_output = torch::zeros({batch, reduce_channel}, options_fp32);
  auto se_expand_weight = torch::ones({reduce_channel, in_channel}, options_fp32);
  auto se_expand_output = torch::zeros({batch, in_channel}, options_fp32);
  auto se_mul_output = torch::zeros({batch, height, width, in_channel}, options_fp32);

  float* ptr_input = input.data<float>();
  float* ptr_reduce_output = reduce_output.data<float>();
  float* ptr_se_reduce_weight = se_reduce_weight.data<float>();
  float* ptr_se_reduce_output = se_reduce_output.data<float>();
  float* ptr_se_expand_weight = se_expand_weight.data<float>();
  float* ptr_se_expand_output = se_expand_output.data<float>();
  float* ptr_se_mul_output = se_mul_output.data<float>();

  void* se_kernel_args[] = {
    (void *)&(ptr_input), 
    (void *)&(ptr_reduce_output), 
    (void *)&(ptr_se_reduce_weight), 
    (void *)&(ptr_se_reduce_output), 
    (void *)&(ptr_se_expand_weight), 
    (void *)&(ptr_se_expand_output)
  };

  checkCuda(cudaFuncSetAttribute((void*)efficientnet_se_module<batch, height, width, in_channel, reduce_channel, tile_size_in_channel>, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
  checkCuda(cudaFuncSetAttribute((void*)efficientnet_se_module_pipeline<batch, height, width, in_channel, reduce_channel, tile_size_in_channel>, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, shared_memory_size));
  auto device_func = [&](int func_id){
    switch (func_id)
    {
    case 0:
      checkCuda(cudaLaunchCooperativeKernel((const void*)efficientnet_se_module<batch, height, width, in_channel, reduce_channel, tile_size_in_channel>, 
        dim3(in_channel/tile_size_in_channel, 1, 1), dim3(128, 1, 1), se_kernel_args, shared_memory_size), __LINE__);
      break;
    case 1:
      checkCuda(cudaLaunchCooperativeKernel((const void*)efficientnet_se_module_pipeline<batch, height, width, in_channel, reduce_channel, tile_size_in_channel>, 
        dim3(in_channel/tile_size_in_channel, 1, 1), dim3(128, 1, 1), se_kernel_args, shared_memory_size), __LINE__);
      break;
    default:
      break;
    }
  };

  device_func(func_id);
  torch::print(reduce_output);
  torch::print(se_reduce_output);
  torch::print(se_expand_output);

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

}

int main(int argc, char** argv){
  int round = 1, loop = 1, type=0, func=0;
  if(argc>2){
    round = atoi(argv[1]);
    loop = atoi(argv[2]);
  }if(argc>3){
    type = atoi(argv[3]);
  }
  if(argc>4){
    func = atoi(argv[4]);
  }
  // switch (func)
  // {
  // case 0:
  //   efficient_se_module<1, 112, 112, 32, 8, 2>(round, loop, type, 132*1024);
  //   break;
  // case 1:
  //   efficient_se_module<1, 56, 56, 96, 4, 4>(round, loop, type, 132*1024);
  //   break;
  // case 2:
  //   efficient_se_module<1, 56, 56, 144, 6, 4>(round, loop, type, 132*1024);
  //   break;
  // case 3:
  //   efficient_se_module<1, 28, 28, 144, 6, 4>(round, loop, type, 48*1024);
  //   break;
  // case 4:
  //   efficient_se_module<1, 28, 28, 240, 10, 4>(round, loop, type, 48*1024);
  //   break;
  // case 5:
  //   efficient_se_module<1, 14, 14, 240, 10, 4>(round, loop, type, 48*1024);
  //   break;
  // case 6:
  //   efficient_se_module<1, 14, 14, 480, 20, 4>(round, loop, type, 48*1024);
  //   break;
  // case 7:
  //   efficient_se_module<1, 14, 14, 672, 28, 4>(round, loop, type, 32*1024);
  //   break;
  // case 8:
  //   efficient_se_module<1, 7, 7, 1152, 48, 8>(round, loop, type, 16*1024);
  //   break;
  // default:
  //   break;
  // }

  switch (func)
  {
  case 0:
    efficient_se_module<1, 112, 112, 32, 8, 1>(round, loop, type, 132*1024);
    break;
  case 1:
    efficient_se_module<1, 56, 56, 96, 4, 1>(round, loop, type, 132*1024);
    break;
  case 2:
    efficient_se_module<1, 56, 56, 144, 6, 2>(round, loop, type, 132*1024);
    break;
  case 3:
    efficient_se_module<1, 28, 28, 144, 6, 2>(round, loop, type, 48*1024);
    break;
  case 4:
    efficient_se_module<1, 28, 28, 240, 10, 2>(round, loop, type, 48*1024);
    break;
  case 5:
    efficient_se_module<1, 14, 14, 240, 10, 2>(round, loop, type, 48*1024);
    break;
  case 6:
    efficient_se_module<1, 14, 14, 480, 20, 4>(round, loop, type, 48*1024);
    break;
  case 7:
    efficient_se_module<1, 14, 14, 672, 28, 4>(round, loop, type, 32*1024);
    break;
  case 8:
    efficient_se_module<1, 7, 7, 1152, 48, 4>(round, loop, type, 16*1024);
    break;
  default:
    break;
  }

  // efficient_se_module<1, 112, 112, 32, 8, 4>(round, loop, type);
  // efficient_se_module<1, 14, 14, 672, 28, 8>(round, loop, type);
  // efficient_se_module<1, 14, 14, 672, 28, 16>(round, loop, type);
  return 0;
}
