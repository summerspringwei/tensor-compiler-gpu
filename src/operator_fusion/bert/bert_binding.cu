// #include  <pybind11/pybind11.h>

// int add(int i, int j) {
// return i + j;
// }
// PYBIND11_MODULE(bert, m) {
// m.doc() = "pybind11 example plugin"; // optional module docstring
// m.def("add", &add, "A function that adds two numbers");
// }

#include <iostream>

#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>

#include <torch/extension.h>

#include "kernels/bert_fused_fc_fc.h"
#include "kernels/bert_query_key_matmul_softmax.h"
#include "kernels/bert_qkv.h"
#include "kernels/bert_qkv_matmul_transpose.h"


torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")


void check_compatability(int numThreads, void* cuda_kernel){
  int dev = 0;
  int supportsCoopLaunch = 0;
  cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
  if(supportsCoopLaunch){
    printf("Device support CoopLaunch\n");
  }
  cudaDeviceProp deviceProp; \
  cudaGetDeviceProperties(&deviceProp, dev); \
  int numBlocksPerSm;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, cuda_kernel, numThreads, 0); 
  printf("fused_fc_fc: OccupancyMaxActiveBlocksPerMultiprocessor: %d, multiProcessorCount: %d\n", numBlocksPerSm, deviceProp.multiProcessorCount);
}


template<int64_t batch_size, int64_t num_heads, int64_t max_seq_length, int64_t hidden_size>
torch::Tensor fused_query_key_matmul_softmax(torch::Tensor query, torch::Tensor key) {
  // Check input
  CHECK_CUDA(query);
  CHECK_CUDA(key);
  // assert(query.dense_dim()==3 && key.dense_dim()==3);
  assert(query.size(0)==batch_size*num_heads && query.size(1)==max_seq_length && query.size(2)==hidden_size);
  assert(key.size(0)==batch_size*num_heads && key.size(1)==max_seq_length && key.size(2)==hidden_size);
  
  // Check compatability
  check_compatability(32, (void*)fused_query_key_matmul_softmax_v3);

  auto options = torch::TensorOptions()
    .dtype(torch::kFloat16)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
  auto sum = torch::zeros({batch_size*num_heads, max_seq_length}, options);
  auto output = torch::zeros({batch_size*num_heads, max_seq_length, max_seq_length}, options);
  // void *kernel_args[] = { (void *)(query.data_ptr()), (void *)(key.data_ptr()), (void *)(output.data_ptr()), (void *)(sum.data_ptr()) };
  at::Half* ptr_query = query.data<at::Half>();
  at::Half* ptr_key = key.data<at::Half>();
  at::Half* ptr_output = output.data<at::Half>();
  at::Half* ptr_sum = sum.data<at::Half>();
  void *kernel_args[] = { (void *)(&ptr_query), (void *)(&ptr_key), (void *)(&ptr_output), (void *)(&ptr_sum) };
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.type(), "fused_query_key_matmul_softmax", [&]{
    checkCuda(cudaLaunchCooperativeKernel((void*)fused_query_key_matmul_softmax_v3, dim3(4, 4,12), dim3(32,1,1), kernel_args, 8704*sizeof(half)));
  });
  cudaDeviceSynchronize();
  return output;
}

template<int64_t batch_size, int64_t num_heads, int64_t max_seq_length, int64_t hidden_size, int64_t dim_feedforward>
std::vector<torch::Tensor> fused_feed_forward(torch::Tensor src, torch::Tensor weight1, torch::Tensor weight2){
  // Check input
  CHECK_CUDA(src);
  CHECK_CUDA(weight1);
  CHECK_CUDA(weight2);
  assert(src.size(0)==batch_size*max_seq_length && src.size(1)==num_heads*hidden_size);
  assert(weight1.size(0)==dim_feedforward && weight1.size(1)==num_heads*hidden_size);
  assert(weight2.size(0)==num_heads*hidden_size && weight2.size(1)==dim_feedforward);

  check_compatability(128, (void*)fused_fc_fc_v2);
  // Create outputs
  auto options_fp16 = torch::TensorOptions()
    .dtype(torch::kFloat16)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
  auto output1 = torch::zeros({batch_size*max_seq_length, dim_feedforward}, options_fp16);
  auto output2 = torch::zeros({batch_size*max_seq_length, num_heads*hidden_size}, options_fp16);
  auto options_fp32 = torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
  auto sum = torch::zeros({batch_size*max_seq_length,}, options_fp32);
  auto variance = torch::zeros({batch_size*max_seq_length,}, options_fp32);

  at::Half* ptr_src = src.data<at::Half>();
  at::Half* ptr_weight1 = weight1.data<at::Half>();
  at::Half* ptr_output1 = output1.data<at::Half>();
  at::Half* ptr_weight2 = weight2.data<at::Half>();
  at::Half* ptr_output2 = output2.data<at::Half>();
  float* ptr_sum = sum.data<float>();
  float* ptr_variance = variance.data<float>();
  half eps = 0.00001, gama=1, beta=0;
  // fused_fc_fc_v2(half *__restrict__ x, half *__restrict__ placeholder,
  //               half *__restrict__ T_dense, half *__restrict__ placeholder2,
  //               half *__restrict__ T_dense2, half* sum, half* variance, half eps, half gama, half beta) 
  

  void *fused_kernel_args[] = { (void *)&(ptr_src), (void *)&(ptr_weight1), 
    (void *)&(ptr_output1), (void *)&(ptr_weight2), (void *)&(ptr_output2), 
    (void *)&(ptr_sum), (void *)&(ptr_variance), (void*)&eps, (void*)&gama, (void*)&beta};
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(output2.type(), "fused_feed_forward", [&]{
    checkCuda(cudaLaunchCooperativeKernel((void*)fused_fc_fc_v2, dim3(192, 1, 1), dim3(128, 1, 1), fused_kernel_args, 13056 * sizeof(half)));
  });
  
  return {output2, sum, variance};
}


template<int64_t batch_size, int64_t num_heads, int64_t max_seq_length, int64_t hidden_size>
std::vector<torch::Tensor> fused_attn_qkv_matmul_transpose(torch::Tensor src, torch::Tensor weight_qkv){
  // Check input
  CHECK_CUDA(src);
  CHECK_CUDA(weight_qkv);
  assert(src.size(0)==batch_size*max_seq_length && src.size(1)==num_heads*hidden_size);
  assert(weight_qkv.size(0)==num_heads*hidden_size*3 && weight_qkv.size(1)==num_heads*hidden_size);
  check_compatability(64, (void*)fused_attn_qkv_matmul_transpose_kernel_v2);

  auto options_fp16 = torch::TensorOptions()
    .dtype(torch::kFloat16)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
  
  auto output_qkv = torch::zeros({batch_size*max_seq_length, num_heads*hidden_size*3}, options_fp16);
  auto query = torch::zeros({batch_size*num_heads, max_seq_length, hidden_size}, options_fp16);
  auto key = torch::zeros({batch_size*num_heads, max_seq_length, hidden_size}, options_fp16);
  auto value = torch::zeros({batch_size*num_heads, hidden_size, max_seq_length}, options_fp16);

  at::Half* ptr_src = src.data<at::Half>();
  at::Half* ptr_weight_qkv = weight_qkv.data<at::Half>();
  at::Half* ptr_output_qkv = output_qkv.data<at::Half>();
  at::Half* ptr_query = query.data<at::Half>();
  at::Half* ptr_key = key.data<at::Half>();
  at::Half* ptr_value = value.data<at::Half>();

  void *fused_kernel_args[] = { (void *)&(ptr_src), (void *)&(ptr_weight_qkv), 
    (void *)&(ptr_output_qkv), (void *)&(ptr_query), (void *)&(ptr_key), (void *)&(ptr_value)};
  
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "fused_attn_qkv_matmul_transpose", [&]{
    checkCuda(cudaLaunchCooperativeKernel((void*)fused_attn_qkv_matmul_transpose_kernel_v2, dim3(4, 36,1), dim3(32,2,1), fused_kernel_args, 13056 * sizeof(half)));
  });
  cudaDeviceSynchronize();
  return {output_qkv, query, key, value};
}



template<int64_t batch_size, int64_t num_heads, int64_t max_seq_length, int64_t hidden_size>
float benchmark_fused_attn_qkv_matmul_transpose(torch::Tensor src, torch::Tensor weight_qkv, int round_cout, int loop){
  // Check input
  CHECK_CUDA(src);
  CHECK_CUDA(weight_qkv);
  assert(src.size(0)==batch_size*max_seq_length && src.size(1)==num_heads*hidden_size);
  assert(weight_qkv.size(0)==num_heads*hidden_size*3 && weight_qkv.size(1)==num_heads*hidden_size);
  check_compatability(64, (void*)fused_attn_qkv_matmul_transpose_kernel);

  auto options_fp16 = torch::TensorOptions()
    .dtype(torch::kFloat16)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
  
  auto output_qkv = torch::zeros({batch_size*max_seq_length, num_heads*hidden_size*3}, options_fp16);
  auto query = torch::zeros({batch_size*num_heads, max_seq_length, hidden_size}, options_fp16);
  auto key = torch::zeros({batch_size*num_heads, max_seq_length, hidden_size}, options_fp16);
  auto value = torch::zeros({batch_size*num_heads, hidden_size, max_seq_length}, options_fp16);

  at::Half* ptr_src = src.data<at::Half>();
  at::Half* ptr_weight_qkv = weight_qkv.data<at::Half>();
  at::Half* ptr_output_qkv = output_qkv.data<at::Half>();
  at::Half* ptr_query = query.data<at::Half>();
  at::Half* ptr_key = key.data<at::Half>();
  at::Half* ptr_value = value.data<at::Half>();

  void *fused_kernel_args[] = { (void *)&(ptr_src), (void *)&(ptr_weight_qkv), 
    (void *)&(ptr_output_qkv), (void *)&(ptr_query), (void *)&(ptr_key), (void *)&(ptr_value)};
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  // Warm up
  for(int i=0; i<100; ++i){
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "fused_attn_qkv_matmul_transpose", [&]{
        checkCuda(cudaLaunchCooperativeKernel((void*)fused_attn_qkv_matmul_transpose_kernel, dim3(4, 36,1), dim3(32,2,1), fused_kernel_args, 13056 * sizeof(half)));
      });
  }
  float ms = 0, sum = 0;
  // 1. For original pointwise conv
  for(int round =0; round<round_cout; ++round){
    for(int i=0; i<loop; ++i){
      checkCuda( cudaEventRecord(startEvent,0) );
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "fused_attn_qkv_matmul_transpose", [&]{
        checkCuda(cudaLaunchCooperativeKernel((void*)fused_attn_qkv_matmul_transpose_kernel, dim3(4, 36,1), dim3(32,2,1), fused_kernel_args, 13056 * sizeof(half)));
      });
      checkCuda( cudaEventRecord(stopEvent,0) );
      checkCuda( cudaEventSynchronize(stopEvent) );
      checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
      sum += ms;
    }
    printf("Run iter %d  loops %d finished\n", round, loop);
  }
  cudaDeviceSynchronize();
  checkCuda(cudaEventDestroy(startEvent));
  checkCuda(cudaEventDestroy(stopEvent));
  return (sum / (round_cout * loop));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("d_sigmoid", &d_sigmoid, "d_sigmoid function");
  m.def("fused_query_key_matmul_softmax", &fused_query_key_matmul_softmax<1, 12, 128, 64>, 
    "bert fused_query_key_matmul_softmax with num_heads=12, max_seq_length=128, hidden_size=64");
  m.def("fused_feed_forward", &fused_feed_forward<1, 12, 128, 64, 3072>, 
    "bert fused_feed_forward with num_heads=12, max_seq_length=128, hidden_size=64, dim_feedforward=3072");
  m.def("fused_attn_qkv_matmul_transpose", &fused_attn_qkv_matmul_transpose<1, 12, 128, 64>, 
    "bert fused_attn_qkv_matmul_transpose with num_heads=12, max_seq_length=128, hidden_size=64");
  m.def("benchmark_fused_attn_qkv_matmul_transpose", &benchmark_fused_attn_qkv_matmul_transpose<1, 12, 128, 64>, 
    "bert benchmark_fused_attn_qkv_matmul_transpose with num_heads=12, max_seq_length=128, hidden_size=64");
}
