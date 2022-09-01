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

#include "../../utils.h"
#include "../../cuda_utils.h"
#include "../torch_utils.h"

#include "kernels/feed_forward_fc1_m4096_n512_k128.cu"
#include "kernels/feed_forward_fc2_m4096_n128_k512.cu"

template<int64_t B, int64_t M, int64_t N, int64_t K>
float test_fused_feed_forward(int round_cout=1, int loop=1, int func_id=0){
  auto x = torch::nn::init::uniform_(
    torch::randn({B*M, N}, options_fp16), 0, 1);
  // auto x_mean = torch::
  auto x_mean = torch::mean(x, {1,}, true);
  auto x_variance = torch::sub(x, x_mean);
  auto x_variance_sum = torch::sum(x_variance * x_variance, {1});
  auto fc1_weight = torch::ones({K, N}, options_fp16);
  auto fc1_output = torch::zeros({B*M, K}, options_fp16);
  auto fc2_weight = torch::ones({N, K}, options_fp16);
  auto fc2_output = torch::zeros({B*M, N}, options_fp16);

  at::Half* ptr_x = x.data<at::Half>();
  at::Half* ptr_x_mean = x_mean.data<at::Half>();
  at::Half* ptr_x_variance_sum = x_variance_sum.data<at::Half>();
  at::Half* ptr_fc1_weight = fc1_weight.data<at::Half>();
  at::Half* ptr_fc1_output = fc1_output.data<at::Half>();
  at::Half* ptr_fc2_weight = fc2_weight.data<at::Half>();
  at::Half* ptr_fc2_output = fc2_output.data<at::Half>();


  void* feedforward_fc1_kernel_args[] = {
    (void *)&(ptr_x),
    (void *)&(ptr_x_mean),
    (void *)&(ptr_x_variance_sum),
    (void *)&(ptr_fc1_weight),
    (void *)&(ptr_fc1_output)
  };
  void* feedforward_fc2_kernel_args[] = {
    (void *)&(ptr_fc1_output),
    (void *)&(ptr_fc2_weight),
    (void *)&(ptr_fc2_output),
    (void *)&(ptr_x)
  };

  void* fused_feed_forward_fc1_fc2_kernel_args [] = {
    (void *)&(ptr_x),
    (void *)&(ptr_x_mean),
    (void *)&(ptr_x_variance_sum),
    (void *)&(ptr_fc1_weight),
    (void *)&(ptr_fc1_output),
    (void *)&(ptr_fc2_weight),
    (void *)&(ptr_fc2_output),
    (void *)&(ptr_x)
  };
  int fc1_num_blocks, fc2_num_blocks;
  
  const size_t fused_feed_forward_shared_memory = 23040;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fc1_num_blocks, feed_forward_fc1_m4096_n512_k128, 256, fused_feed_forward_shared_memory); 
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fc2_num_blocks, feed_forward_fc1_m4096_n512_k128, 256, fused_feed_forward_shared_memory); 
  printf("fc1 OccupancyMaxActiveBlocksPerMultiprocessor: %d\n", fc1_num_blocks);
  printf("fc2 OccupancyMaxActiveBlocksPerMultiprocessor: %d\n", fc2_num_blocks);

  auto device_func = [&](int func_id){
    switch (func_id)
    {
    case 0:
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(fc2_output.type(), "feef_forward_fc1", [&]{
        // checkCuda(cudaLaunchCooperativeKernel((void*)feed_forward_fc1_m4096_n512_k128, 
        // dim3(128, 4, 1), dim3(32, 2, 4), feedforward_fc1_kernel_args, fused_feed_forward_shared_memory));
        feed_forward_fc1_m4096_n512_k128<<<dim3(128, 4, 1), dim3(32, 2, 4)>>>((half*)ptr_x, (half*)ptr_x_mean, (half*)ptr_x_variance_sum, (half*)ptr_fc1_weight, (half*)ptr_fc1_output);
      });
      break;
    case 1:
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(fc2_output.type(), "bert_attn_qkv", [&]{
        // checkCuda(cudaLaunchCooperativeKernel((void*)feed_forward_fc2_m4096_n128_k512, 
        // dim3(64, 2, 1), dim3(32, 2, 2), feedforward_fc2_kernel_args, fused_feed_forward_shared_memory));
        feed_forward_fc2_m4096_n128_k512<<<dim3(64, 2, 1), dim3(32, 2, 2)>>>((half*)ptr_fc1_output, (half*)ptr_fc2_weight, (half*)ptr_fc2_output, (half*)ptr_x);
      });
      break;
    default:
      break;
    }
  };

  // Run device function
  device_func(func_id);
  cudaDeviceSynchronize();

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
    printf("Run iter %d loops %d finished, avg %f us\n", round, loop, min_avg);
  }

  checkCuda(cudaEventDestroy(startEvent));
  checkCuda(cudaEventDestroy(stopEvent));
}


int main(int argc, char** argv){
  int round = 1, loop = 1, type=0;
  if(argc>2){
    round = atoi(argv[1]);
    loop = atoi(argv[2]);
  }if(argc>3){
    type = atoi(argv[3]);
  }
  test_fused_feed_forward<1, 4096, 128, 512>(round, loop, type);
  return 0;
}
