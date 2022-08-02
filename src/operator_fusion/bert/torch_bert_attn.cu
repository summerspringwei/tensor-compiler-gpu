#include <iostream>
#include <math.h>

#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>

#include "torch/all.h"

#include "../../utils.h"


#include "kernels/bert_main_kernel.h"
#include "kernels/bert_main_kernel_v2.h"
#include "kernels/bert_attn_fc.h"
#include "kernels/bert_fused_fc_fc_v2.h"
#include "kernels/bert_fused_fc_fc.h"


template<int64_t batch_size, int64_t num_heads, int64_t max_seq_length, int64_t hidden_size>
float test_bert_attn(int round_cout=1, int loop=1){
  auto options_fp16 = torch::TensorOptions()
    .dtype(torch::kFloat16)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
  auto options_fp32 = torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
  
  float scale = 32;
  // All same test pass
  // auto src = torch::div(
  //   torch::ones({batch_size*max_seq_length, num_heads*hidden_size}, options_fp16), 
  //   torch::tensor({scale,}, options_fp16));
  // auto weight_qkv = torch::div(
  //   torch::ones({num_heads*hidden_size*3, num_heads*hidden_size}, options_fp16), 
  //   torch::tensor({scale,}, options_fp16));
  // auto attn_value_output = torch::div(
  //   torch::ones({batch_size*num_heads, max_seq_length, hidden_size}, options_fp16),
  //   torch::tensor({scale,}, options_fp16));
  auto attn_fc_weight = torch::div(
    torch::ones({num_heads*hidden_size, num_heads*hidden_size}, options_fp16), 
    torch::tensor({1,}, options_fp16));
  auto src = torch::nn::init::uniform_(
    torch::randn({batch_size*max_seq_length, num_heads*hidden_size}, options_fp16),
    -1/scale, 1/scale);
  auto weight_qkv = torch::nn::init::uniform_(
    torch::randn({num_heads*hidden_size*3, num_heads*hidden_size}, options_fp16),
    -1.0/scale, 1.0/scale);
  auto weight_feed_forward_fc1 = torch::nn::init::uniform_(
    torch::randn({num_heads*hidden_size*4, num_heads*hidden_size}, options_fp16),
    -1.0/scale, 1.0/scale);
  auto weight_feed_forward_fc2 = torch::nn::init::uniform_(
    torch::randn({num_heads*hidden_size, num_heads*hidden_size*4}, options_fp16),
    -1.0/scale, 1.0/scale);
  // auto attn_fc_weight = torch::nn::init::uniform_(
  //   torch::randn({num_heads*hidden_size, num_heads*hidden_size}, options_fp16),
  //   -1.0/scale, 1.0/scale);
  auto t_attn_fc_weight = torch::clone(attn_fc_weight);

  // Torch implementation
  namespace F = torch::nn::functional;
  // fused QKV matmul
  auto t_output_qkv = torch::matmul(src, torch::transpose(weight_qkv, -2, -1));
  auto qkv = torch::split(t_output_qkv, 768, 1);
  auto t_query = torch::permute(
    torch::reshape(qkv[0], {max_seq_length, num_heads, hidden_size}), {1, 0, 2}); // Now (12, 128, 64)
  auto t_key = torch::permute(
    torch::reshape(qkv[1], {max_seq_length, num_heads, hidden_size}), {1, 0, 2});  // Now (12, 128, 64)
  auto t_value = torch::permute(
    torch::reshape(qkv[2], {{max_seq_length, num_heads, hidden_size}}), {1, 2, 0}); // Now (12, 64, 128)
  // softmax(query_key_matmul / sqrt(hidden_size))
  float factor = 64;
  auto t_factor = torch::sqrt(torch::tensor({factor,}, options_fp16));
  auto t_query_key_output = torch::softmax(
    torch::divide(
      torch::bmm(t_query, torch::permute(t_key, {0, 2, 1})), t_factor), 2); // Now (12, 128, 128)
  // qk * value
  auto t_attn_value_output = torch::bmm(t_query_key_output, torch::permute(t_value, {0, 2, 1})); // Now (12, 128, 64)
  // reshape
  auto t_attn_value_output_permuted = torch::reshape(
    torch::permute(t_attn_value_output, {1, 0, 2}), {batch_size*max_seq_length, num_heads * hidden_size});// Now (128, 768)
  // attn fc
  auto t_attn_fc_output_tmp = torch::matmul(t_attn_value_output_permuted, torch::permute(t_attn_fc_weight, {1, 0}));
  // Short cut
  auto t_attn_fc_output = torch::add(src, t_attn_fc_output_tmp);
  auto t_reduce_sum = torch::sum(t_attn_fc_output, 1, false, torch::kFloat32);
  auto t_attn_layer_norm_output = torch::layer_norm(t_attn_fc_output, {768,});
  // Feed forward
  auto t_feed_forward_fc1_output = torch::matmul(t_attn_layer_norm_output, torch::permute(weight_feed_forward_fc1, {1, 0}));
  auto t_feed_forward_fc2_output = torch::matmul(t_feed_forward_fc1_output, torch::permute(weight_feed_forward_fc2, {1, 0}));
  auto t_feed_forward_short_cut_output = torch::add(t_feed_forward_fc2_output, t_attn_layer_norm_output);
  auto t_feed_forward_layer_norm_output = torch::layer_norm(t_feed_forward_short_cut_output, {768,});


  // Our implementation
  auto output_qkv = torch::zeros({batch_size*max_seq_length, num_heads*hidden_size*3}, options_fp16);
  auto query = torch::zeros({batch_size*num_heads, max_seq_length, hidden_size}, options_fp16);
  auto key = torch::zeros({batch_size*num_heads, max_seq_length, hidden_size}, options_fp16);
  auto value = torch::zeros({batch_size*num_heads, hidden_size, max_seq_length}, options_fp16);
  auto query_key_output = torch::zeros({batch_size*num_heads, max_seq_length, max_seq_length}, options_fp16);
  auto sum = torch::zeros({batch_size*num_heads, max_seq_length}, options_fp32); // Reduce_sum of query_key matmul
  auto attn_value_output = torch::zeros({batch_size*num_heads, max_seq_length, hidden_size}, options_fp16);
  auto attn_fc_output = torch::zeros({batch_size * max_seq_length, num_heads * hidden_size}, options_fp16);
  auto single_attn_fc_output = torch::zeros({batch_size * max_seq_length, num_heads * hidden_size}, options_fp16);
  auto inter_attn_fc_output = torch::zeros({batch_size * max_seq_length, num_heads * hidden_size}, options_fp16);
  auto variance = torch::zeros({batch_size * max_seq_length,}, options_fp32);
  auto feed_forward_fc1_output = torch::zeros({batch_size * max_seq_length, num_heads * hidden_size * 4}, options_fp16);
  auto feed_forward_fc2_output = torch::zeros({batch_size * max_seq_length, num_heads * hidden_size}, options_fp16);


  at::Half* ptr_src = src.data<at::Half>();
  at::Half* ptr_weight_qkv = weight_qkv.data<at::Half>();
  at::Half* ptr_output_qkv = output_qkv.data<at::Half>();
  at::Half* ptr_query = query.data<at::Half>();
  at::Half* ptr_key = key.data<at::Half>();
  at::Half* ptr_value = value.data<at::Half>();
  at::Half* ptr_query_key_output = query_key_output.data<at::Half>();
  float* ptr_sum = sum.data<float>();
  at::Half* ptr_attn_value_output = attn_value_output.data<at::Half>();
  at::Half* ptr_inter_attn_fc_output = inter_attn_fc_output.data<at::Half>();
  // at::Half* ptr_attn_value_output = t_attn_value_output_permuted.data<at::Half>();
  at::Half* ptr_single_attn_fc_output = single_attn_fc_output.data<at::Half>();
  at::Half* ptr_attn_fc_weight = attn_fc_weight.data<at::Half>();
  at::Half* ptr_attn_fc_output = attn_fc_output.data<at::Half>();
  float* ptr_variance = variance.data<float>();
  half eps = 0.00001, gama=1, beta = 0;
  at::Half* ptr_weight_feed_forward_fc1 = weight_feed_forward_fc1.data<at::Half>();
  at::Half* ptr_feed_forward_fc1_output = feed_forward_fc1_output.data<at::Half>();
  at::Half* ptr_weight_feed_forward_fc2 = weight_feed_forward_fc2.data<at::Half>();
  at::Half* ptr_feed_forward_fc2_output = feed_forward_fc2_output.data<at::Half>();

  // pointers from pytorch
  at::Half* t_ptr_attn_value_output = t_attn_value_output_permuted.data<at::Half>();
  at::Half* t_ptr_attn_fc_weight = t_attn_fc_weight.data<at::Half>();
  at::Half* t_ptr_fc_output_tmp = t_attn_fc_output_tmp.data<at::Half>();
  at::Half* t_ptr_attn_fc_output = t_attn_fc_output.data<at::Half>();

  void *fused_kernel_args[] = {(void *)&(ptr_src), (void *)&(ptr_weight_qkv), 
    (void *)&(ptr_output_qkv), (void *)&(ptr_query), (void *)&(ptr_key), 
    (void *)&(ptr_value), (void*)&(ptr_query_key_output), (void*)&(ptr_sum),
    (void*)&(ptr_attn_value_output), (void*)&(ptr_attn_fc_weight), (void*)&(ptr_attn_fc_output),
    (void*)&(ptr_variance), (void *)&(eps), (void *)&(gama), (void *)&(beta)};
  
  // (void *)&(t_ptr_attn_value_output), (void *)&(t_ptr_attn_fc_weight), 
  //   (void *)&(t_ptr_fc_output_tmp), (void *)&(t_ptr_attn_fc_output), (void *)&(ptr_inter_attn_fc_output)

  void * single_attn_fc_kernel_args[] = {
    (void *)&(ptr_attn_value_output), (void *)&(ptr_attn_fc_weight), (void *)&(ptr_single_attn_fc_output)
  };

  void * fused_feed_forward_kernel_args[] = {
    (void *)&(ptr_attn_fc_output), (void *)&(ptr_weight_feed_forward_fc1), 
    (void *)&(ptr_feed_forward_fc1_output), (void *)&(ptr_weight_feed_forward_fc2), 
    (void *)&(ptr_feed_forward_fc2_output),
    (void *)&(ptr_sum), (void *)&(ptr_variance), 
    (void *)&(eps), (void *)&(gama), (void *)&(beta)
  };

  cudaFuncSetAttribute((void*)bert_attn_kernel_v2, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, 66*1024);
  cudaFuncSetAttribute((void*)fused_fc_fc_v4, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, 66*1024);
  cudaFuncSetAttribute((void*)fused_fc_fc_v5, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, 66*1024);
  check_compatability(128, 26112 * sizeof(half), (void*)bert_attn_kernel_v2);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "bert_attn", [&]{
    // checkCuda(cudaLaunchCooperativeKernel((void*)bert_attn_kernel, dim3(192, 1,1), dim3(32*4,1,1), fused_kernel_args, 13056 * sizeof(half)));
    // checkCuda(cudaLaunchCooperativeKernel((void*)bert_attn_kernel_v2, dim3(192, 1,1), dim3(32*4,1,1), fused_kernel_args, 26112 * sizeof(half))); 

    // checkCuda(cudaLaunchCooperativeKernel((void*)fused_fc_fc_v3, dim3(192, 1, 1), dim3(128, 1, 1), fused_feed_forward_kernel_args, 13056 * sizeof(half)));
    // checkCuda(cudaLaunchCooperativeKernel((void*)fused_fc_fc_v4, dim3(192, 1, 1), dim3(128, 1, 1), fused_feed_forward_kernel_args, 26112 * sizeof(half)));
    // checkCuda(cudaLaunchCooperativeKernel((void*)fused_fc_fc_v5, dim3(192, 1, 1), dim3(128, 1, 1), fused_feed_forward_kernel_args, 26112 * sizeof(half)));
    checkCuda(cudaLaunchCooperativeKernel((void*)fused_fc_fc_v6, dim3(192, 1, 1), dim3(128, 1, 1), fused_feed_forward_kernel_args, (8704+8704+4352) * sizeof(half)));
  });
  // AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "bert_attn_fc", [&]{
  //   checkCuda(cudaLaunchCooperativeKernel((void*)attn_fc, dim3(96, 1,1), dim3(32*4,1,1), single_attn_fc_kernel_args, 13056 * sizeof(half)));
  // });
  cudaDeviceSynchronize();
  
  // Print tensor for debug
  // torch::save(query_key_output, "query_key_output.pt");
  // torch::save(t_query_key_output, "t_query_key_output.pt");
  std::vector<int64_t> shape_output_qkv = {batch_size*max_seq_length, 3*num_heads*hidden_size,};
  printf("output_qkv\n");
  torch::print(torch::reshape(output_qkv, shape_output_qkv), 768*100);
  printf("t_output_qkv\n");
  torch::print(torch::reshape(t_output_qkv, shape_output_qkv), 768*100);
  printf("query_key_output\n");
  torch::print(query_key_output, 768*100);
  printf("t_query_key_output\n");
  torch::print(t_query_key_output, 768*100);
  printf("attn_value_output\n");
  torch::print(torch::reshape(attn_value_output, {max_seq_length, num_heads * hidden_size}), 768*100);
  printf("t_attn_value_output_permuted\n");
  torch::print(t_attn_value_output_permuted, 768*100);
  printf("attn_fc_output\n");
  torch::print(attn_fc_output, 768*100);
  printf("t_attn_fc_output\n");
  torch::print(t_attn_fc_output, 768*100);
  printf("t_attn_layer_norm_output\n");
  torch::print(t_attn_layer_norm_output, 768*100);
  printf("t_attn_fc_output_tmp\n");
  torch::print(t_attn_fc_output_tmp, 768*100);
  printf("attn_fc_weight\n");
  torch::print(attn_fc_weight, 768*100);
  // printf("single_attn_fc_output\n");
  // torch::print(single_attn_fc_output, 768*100);
  // printf("inter_attn_fc_output\n");
  // torch::print(inter_attn_fc_output, 768*100);
  
  // my_compare(attn_fc_output.cpu().data<at::Half>(), t_attn_layer_norm_output.cpu().data<at::Half>(), 128, 768, 1.0/16, 1.0/1024);
  // Check result
  // assert(torch::allclose(
  //   torch::reshape(output_qkv, shape_output_qkv), 
  //   torch::reshape(t_output_qkv, shape_output_qkv), 1.0/16, 1.0/1024));
  // assert(torch::allclose(query, t_query, 1.0/16, 1.0/1024));
  // assert(torch::allclose(key, t_key, 1.0/16, 1.0/1024));
  // assert(torch::allclose(value, t_value, 1.0/16, 1.0/1024));
  // assert(torch::allclose(query_key_output, t_query_key_output, 1.0/16, 1.0/1024));
  // assert(torch::allclose(
  //   torch::reshape(attn_value_output, {max_seq_length, num_heads * hidden_size}), 
  //   t_attn_value_output_permuted, 1.0/16, 1.0/1024));
  // assert(torch::allclose(attn_fc_weight, t_attn_fc_weight, 1.0/16, 1.0/1024));
  // // assert(torch::allclose(t_attn_fc_output_tmp, single_attn_fc_output, 1.0/16, 1.0/1024));
  // // assert(torch::allclose(inter_attn_fc_output, t_attn_fc_output, 1.0/16, 1.0/1024));
  // assert(torch::allclose(attn_fc_output, t_attn_layer_norm_output, 1.0/16, 1.0/1024));

  // Benchmark
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  // Warm up
  for(int i=0; i<100; ++i){
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "bert_attn", [&]{
      // checkCuda(cudaLaunchCooperativeKernel((void*)bert_attn_kernel, dim3(192, 1,1), dim3(32*4,1,1), fused_kernel_args, 13056 * sizeof(half)));
      // User larger shared memory for double buffer
      // checkCuda(cudaLaunchCooperativeKernel((void*)bert_attn_kernel_v2, dim3(192, 1,1), dim3(32*4,1,1), fused_kernel_args, 17408 * sizeof(half))); 
  });
  }
  
  // 1. For original pointwise conv
  float min_avg = 1e10;
  for(int round =0; round<round_cout; ++round){
    float ms = 0, latency_sum = 0;
    for(int i=0; i<loop; ++i){
      checkCuda( cudaEventRecord(startEvent,0) );
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "bert_attn", [&]{
        // checkCuda(cudaLaunchCooperativeKernel((void*)bert_attn_kernel, dim3(192, 1,1), dim3(32*4,1,1), fused_kernel_args, 13056 * sizeof(half)));
        // checkCuda(cudaLaunchCooperativeKernel((void*)bert_attn_kernel_v2, dim3(192, 1,1), dim3(32*4,1,1), fused_kernel_args, 26112 * sizeof(half))); 

        // checkCuda(cudaLaunchCooperativeKernel((void*)fused_fc_fc_v3, dim3(192, 1, 1), dim3(128, 1, 1), fused_feed_forward_kernel_args, 13056 * sizeof(half)));
        // checkCuda(cudaLaunchCooperativeKernel((void*)fused_fc_fc_v4, dim3(192, 1, 1), dim3(128, 1, 1), fused_feed_forward_kernel_args, 26112 * sizeof(half)));
        // checkCuda(cudaLaunchCooperativeKernel((void*)fused_fc_fc_v5, dim3(192, 1, 1), dim3(128, 1, 1), fused_feed_forward_kernel_args, 26112 * sizeof(half)));
        checkCuda(cudaLaunchCooperativeKernel((void*)fused_fc_fc_v6, dim3(192, 1, 1), dim3(128, 1, 1), fused_feed_forward_kernel_args,  (8704+8704+4352) * sizeof(half)));
      });
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
  return min_avg;
}

int main(){
  test_bert_attn<1, 12, 128, 64>(3, 10000);
  return 0;
}