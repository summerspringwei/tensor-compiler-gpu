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


// #include "kernels/gemm_three_stages.h"

#include "npy.hpp"

#include "../../utils.h"
#include "../../cuda_utils.h"
#include "../torch_utils.h"

#include "kernels/fused_bert_global_sync.cu"
// using namespace bert;
/* This bert is based on the implementation of Qianqi Sun*/

// template<T>
// void print_arr(){

// }

template<int64_t batch_size, int64_t num_heads, int64_t max_seq_length, int64_t hidden_size, int64_t d_intermedia>
float test_bert_attn(int round_cout=1, int loop=1, int type=0){
  int compare_level = 1;
  const int d_model = num_heads * hidden_size;
  // Load weight from model file
  auto load_file_query = std::vector<float>(d_model*d_model);
  std::vector<unsigned long> shape = {d_model, d_model};
  bool fortran_order;
  std::string dir_path("/home/xiachunwei/models/bert_weights/");
  npy::LoadArrayFromNumpy<float>(dir_path+std::string("bert_encoder_layer_0_attention_self_query_kernel_0.npy"), 
    shape, fortran_order,  load_file_query);
  auto file_query = torch::from_blob(load_file_query.data(), {d_model, d_model}).clone().to(torch::kCUDA);
  npy::LoadArrayFromNumpy<float>(dir_path+std::string("bert_encoder_layer_0_attention_self_key_kernel_0.npy"), 
    shape, fortran_order,  load_file_query);
  auto file_key = torch::from_blob(load_file_query.data(), {d_model, d_model}).clone().to(torch::kCUDA);
  npy::LoadArrayFromNumpy<float>(dir_path+std::string("bert_encoder_layer_0_attention_self_value_kernel_0.npy"), 
    shape, fortran_order,  load_file_query);
  auto file_value = torch::from_blob(load_file_query.data(), {d_model, d_model}).clone().to(torch::kCUDA);
  auto qkv_weight = torch::cat({file_query, file_key, file_value}, 0).reshape({3, d_model, d_model}).toType(torch::kHalf);
  auto weight_query = file_query.toType(torch::kHalf);
  auto weight_key = file_key.toType(torch::kHalf);
  auto weight_value = file_value.toType(torch::kHalf);
  npy::LoadArrayFromNumpy<float>(dir_path+std::string("bert_encoder_layer_0_attention_output_dense_kernel_0.npy"), 
    shape, fortran_order,  load_file_query);
  auto attn_fc_weight = torch::from_blob(load_file_query.data(), {d_model, d_model}).clone().toType(torch::kHalf).to(torch::kCUDA);
  auto load_file_feed_forward = std::vector<float>(d_model*d_intermedia);
  npy::LoadArrayFromNumpy<float>(dir_path+std::string("bert_encoder_layer_0_intermediate_dense_kernel_0.npy"), 
    shape, fortran_order,  load_file_feed_forward);
  auto feed_forward_fc1_weight = torch::from_blob(load_file_feed_forward.data(), {d_model, d_intermedia}).clone().toType(torch::kHalf).to(torch::kCUDA);
  npy::LoadArrayFromNumpy<float>(dir_path+std::string("bert_encoder_layer_0_output_dense_kernel_0.npy"), 
    shape, fortran_order,  load_file_feed_forward);
  auto feed_forward_fc2_weight = torch::from_blob(load_file_feed_forward.data(), {d_intermedia, d_model}).clone().toType(torch::kHalf).to(torch::kCUDA);

  auto bias_qkv = torch::zeros({3, d_model}, options_fp16);

  float scale = 16;
  auto src = torch::nn::init::uniform_(
    torch::randn({batch_size*max_seq_length, d_model}, options_fp16), 0, 1);

  // Torch implementation
  auto t_query = torch::matmul(src, weight_query);
  auto t_key = torch::matmul(src, weight_key);
  auto t_value = torch::matmul(src, weight_value);
  auto t_reshaped_query = torch::permute(
    torch::reshape(t_query, {batch_size, max_seq_length, num_heads, hidden_size}),
    {0, 2, 1, 3});
  auto t_reshaped_key = torch::permute(
    torch::reshape(t_key, {batch_size, max_seq_length, num_heads, hidden_size}),
    {0, 2, 1, 3});
  auto t_reshaped_value = torch::permute(
    torch::reshape(t_value, {batch_size, max_seq_length, num_heads, hidden_size}),
    {0, 2, 1, 3});
  // fused QKV matmul
  // auto batched_src = torch::reshape(src.repeat({3, 1, 1}), {3, max_seq_length, d_model});
  // auto t_output_qkv = torch::permute(
  //   torch::reshape(
  //     torch::bmm(batched_src, qkv_weight), 
  //       {3, max_seq_length, num_heads, hidden_size}), {0, 2, 1, 3});
  // auto qkv = torch::split(t_output_qkv, 1, 0);
  // auto t_value = torch::reshape(qkv[2], {num_heads, max_seq_length, hidden_size});
  // auto t_query = torch::reshape(qkv[0], {batch_size*num_heads, max_seq_length, hidden_size});
  // auto t_key = torch::reshape(qkv[1], {batch_size*num_heads, max_seq_length, hidden_size});
  auto t_query_key_output = t_reshaped_query.matmul(torch::permute(t_reshaped_key, {0, 1, 3, 2}));
  // auto t_query_key_output = t_query.bmm(torch::permute(t_key, {0, 2, 1}));
  auto t_attn_mask = torch::zeros({batch_size*num_heads, max_seq_length, max_seq_length}, options_fp16);
  float v_d_model[] = {d_model,};
  auto t_d_model = torch::from_blob(v_d_model, {1,}).to(torch::kCUDA);
  auto t_query_key_output_sum = torch::sum(torch::exp(t_query_key_output / torch::sqrt(t_d_model)) + t_attn_mask, -1);
  auto t_query_key_softmax = torch::softmax(
    (t_query_key_output / torch::sqrt(t_d_model)) + 
      t_attn_mask, -1, torch::kHalf);
  // auto t_attn_value_output = torch::bmm(t_query_key_softmax, t_value); // Now (12, 384, 64)
  // auto t_attn_value_output_permuted = torch::reshape(
  //   torch::permute(t_attn_value_output, {1, 0, 2}), {max_seq_length, d_model});
  // auto t_attn_fc_output = torch::matmul(t_attn_value_output_permuted, attn_fc_weight);
  // auto t_attn_fc_short_cut_add = t_attn_fc_output + src;
  // auto t_attn_fc_layer_norm_x_2 = torch::sum(t_attn_fc_short_cut_add * t_attn_fc_short_cut_add, 1);
  // auto t_attn_fc_layer_norm_x = torch::sum(t_attn_fc_short_cut_add, 1);
  // auto t_attn_fc_layer_norm_output = torch::layer_norm(t_attn_fc_short_cut_add, {d_model,});
  // auto t_feed_forward_fc1_output = torch::matmul(t_attn_fc_layer_norm_output, feed_forward_fc1_weight);
  // auto t_feed_forward_fc1_activation_output = torch::relu(t_feed_forward_fc1_output);
  // auto t_feed_forward_fc2_output = torch::matmul(t_feed_forward_fc1_activation_output, feed_forward_fc2_weight);
  // auto t_feed_forward_fc2_short_cut_output = t_feed_forward_fc2_output + t_attn_fc_layer_norm_output;
  // auto t_feed_forward_fc2_layer_norm_sum_x_2 = torch::sum(t_feed_forward_fc2_short_cut_output * t_feed_forward_fc2_short_cut_output, 1);
  // auto t_feed_forward_fc2_layer_norm_sum_x = torch::sum(t_feed_forward_fc2_output, 1);
  // auto t_feed_forward_fc2_layer_norm = torch::layer_norm(t_feed_forward_fc2_short_cut_output, {d_model,});
  printf("Torch finshed\n");

  // Our implementation
  auto output_query = torch::zeros({batch_size * max_seq_length, num_heads * hidden_size}, options_fp16);
  auto output_key = torch::zeros({batch_size * max_seq_length, num_heads * hidden_size}, options_fp16);
  auto output_value = torch::zeros({batch_size * max_seq_length, num_heads * hidden_size}, options_fp16);
  auto reshaped_query = torch::zeros({batch_size, num_heads, max_seq_length, hidden_size}, options_fp16);
  auto reshaped_key = torch::zeros({batch_size, num_heads, max_seq_length, hidden_size}, options_fp16);
  auto reshaped_value = torch::zeros({batch_size, num_heads, max_seq_length, hidden_size}, options_fp16);

  auto query_key_output = torch::zeros({batch_size*num_heads, max_seq_length, max_seq_length}, options_fp16);
  auto query_key_softmax_sum = torch::zeros({batch_size*num_heads, max_seq_length}, options_fp32);
  auto tvm_query_key_output = torch::zeros({batch_size*num_heads, max_seq_length, max_seq_length}, options_fp16);
  auto attn_value_output = torch::zeros({batch_size*max_seq_length, d_model}, options_fp16);
  auto attn_fc_output = torch::zeros({batch_size*max_seq_length, d_model}, options_fp16);
  auto feed_forward_fc1_output = torch::zeros({batch_size*max_seq_length, d_intermedia}, options_fp16);
  auto feed_forward_fc2_output = torch::zeros({batch_size*max_seq_length, d_model}, options_fp16);
  auto attn_layer_norm_sum = torch::zeros({batch_size*max_seq_length, }, options_fp32);
  auto attn_layer_norm_variance = torch::zeros({batch_size*max_seq_length, }, options_fp32);
  auto feed_forward_layer_norm_sum = torch::zeros({batch_size*max_seq_length, }, options_fp32);
  auto feed_forward_layer_norm_variance = torch::zeros({batch_size*max_seq_length, }, options_fp32);
  const int kProfileStages = 13, max_blocks=108, max_num_warp=4;
  auto profile_clock = torch::zeros({kProfileStages, max_blocks, max_num_warp}, options_int64);
  const int kAttnProfileStages = 9, kAttnBlocks=108, kFeedForwardProfileStages = 5, kFeedForwardBlocks = 96;
  auto attn_profile_clock = torch::zeros({kAttnProfileStages, kAttnBlocks, max_num_warp}, options_int64);
  auto feed_forward_profile_clock = torch::zeros({kFeedForwardProfileStages, kFeedForwardBlocks, max_num_warp}, options_int64);
  

  at::Half* ptr_src = src.data_ptr<at::Half>();
  at::Half* ptr_weight_query = weight_query.data_ptr<at::Half>();
  at::Half* ptr_weight_key = weight_key.data_ptr<at::Half>();
  at::Half* ptr_weight_value = weight_value.data_ptr<at::Half>();
  at::Half* ptr_output_query = output_query.data_ptr<at::Half>();
  at::Half* ptr_output_key = output_key.data_ptr<at::Half>();
  at::Half* ptr_output_value = output_value.data_ptr<at::Half>();
  at::Half* ptr_bias_query = bias_qkv.data_ptr<at::Half>();
  at::Half* ptr_bias_key = bias_qkv.data_ptr<at::Half>() + hidden_size;
  at::Half* ptr_bias_value = bias_qkv.data_ptr<at::Half>() + 2* hidden_size;
  at::Half* ptr_reshaped_query = reshaped_query.data_ptr<at::Half>();
  at::Half* ptr_reshaped_key = reshaped_key.data_ptr<at::Half>();
  at::Half* ptr_reshaped_value = reshaped_value.data_ptr<at::Half>();
  at::Half* ptr_query_key_output = query_key_output.data_ptr<at::Half>();
  // at::Half* ptr_t_attn_mask = t_attn_mask.data_ptr<at::Half>();
  // float* ptr_query_key_softmax_sum = query_key_softmax_sum.data_ptr<float>();
  // at::Half* ptr_attn_value_output = attn_value_output.data_ptr<at::Half>();
  // at::Half* ptr_attn_fc_weight = attn_fc_weight.data_ptr<at::Half>();
  // at::Half* ptr_attn_fc_output = attn_fc_output.data_ptr<at::Half>();
  // at::Half* ptr_feed_forward_fc1_weight = feed_forward_fc1_weight.data_ptr<at::Half>();
  // at::Half* ptr_feed_forward_fc1_output = feed_forward_fc1_output.data_ptr<at::Half>();
  // at::Half* ptr_feed_forward_fc2_weight = feed_forward_fc2_weight.data_ptr<at::Half>();
  // at::Half* ptr_feed_forward_fc2_output = feed_forward_fc2_output.data_ptr<at::Half>();
  // float* ptr_attn_layer_norm_sum = attn_layer_norm_sum.data_ptr<float>();
  // float* ptr_attn_layer_norm_variance = attn_layer_norm_variance.data_ptr<float>();
  // float* ptr_feed_forward_layer_norm_sum = feed_forward_layer_norm_sum.data_ptr<float>();
  // float* ptr_feed_forward_layer_norm_variance = feed_forward_layer_norm_variance.data_ptr<float>();
  // int64_t* ptr_profile_clock = profile_clock.data_ptr<int64_t>();
  // int64_t* ptr_attn_profile_clock = attn_profile_clock.data_ptr<int64_t>();
  // int64_t* ptr_feed_forward_profile_clock = feed_forward_profile_clock.data_ptr<int64_t>();


  // Pointers from torch
  // at::Half* ptr_t_attn_fc_layer_norm_output = t_attn_fc_layer_norm_output.data_ptr<at::Half>();
  // at::Half* ptr_t_feed_forward_fc2_output = t_feed_forward_fc2_output.data_ptr<at::Half>();
  // at::Half* ptr_t_attn_fc_output = t_attn_fc_output.data_ptr<at::Half>();
  // at::Half* ptr_t_attn_fc_short_cut_add = t_attn_fc_short_cut_add.data_ptr<at::Half>();
  // at::Half* ptr_t_feed_forward_fc2_short_cut_output =t_feed_forward_fc2_short_cut_output.data_ptr<at::Half>();

  half eps = 0.00001, gama=1, beta = 0;

  const size_t fused_bert_shared_mem = 108*1024;
  checkCuda(cudaFuncSetAttribute((void*)bert_only_global_sync, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, fused_bert_shared_mem));
  
  void* fused_bert_kernel_args[] = {
    (void*)(&ptr_weight_query),
    (void*)(&ptr_src),
    (void*)(&ptr_output_query),
    (void*)(&ptr_weight_key),
    (void*)(&ptr_src),
    (void*)(&ptr_output_key),
    (void*)(&ptr_weight_value),
    (void*)(&ptr_src),
    (void*)(&ptr_output_value),
    (void*)(&ptr_bias_query),
    (void*)(&ptr_output_query),
    (void*)(&ptr_bias_key),
    (void*)(&ptr_output_key),
    (void*)(&ptr_bias_value),
    (void*)(&ptr_output_value),
    (void*)(&ptr_reshaped_query),
    (void*)(&ptr_reshaped_key),
    (void*)(&ptr_reshaped_value),
    (void*)(&ptr_query_key_output)
  };
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_query.scalar_type(), "fused_bert", [&]{
      checkCuda(cudaLaunchCooperativeKernel((void*)bert_only_global_sync, 
      dim3(108, 1, 1), dim3(128, 1, 1), fused_bert_kernel_args, fused_bert_shared_mem));
  });

  cudaDeviceSynchronize();
  
  // Check result
  if(type>0){
    torch::print(t_query);
    torch::print(output_query);
    torch::print(torch::max(torch::abs(t_query-output_query)));
    torch::print(t_query_key_output);
    torch::print(query_key_output);
    torch::print(torch::max(torch::abs(t_query_key_output-query_key_output)));
    for(auto s: t_query_key_output.sizes()){
      printf("%ld ", s);
    }printf("\n");
  }
  float rtol = 1e-1, atol = 1e-1;
  assert(torch::allclose(t_query, output_query, rtol, atol));
  assert(torch::allclose(t_key, output_key, rtol, atol));
  assert(torch::allclose(t_value, output_value, rtol, atol));
  assert(torch::allclose(t_reshaped_query, reshaped_query, rtol, atol));
  assert(torch::allclose(t_query_key_output, query_key_output, rtol, atol));
  
  // assert(torch::allclose(t_query_key_output, query_key_output, rtol, atol));
  // my_compare(cloned_query, output_query, 1.0/16, 1.0/1024, compare_level);
  // my_compare(t_query_key_output_sum, query_key_softmax_sum, 1.0/16, 1.0/1024, compare_level);
  // my_compare(t_query_key_softmax, query_key_output, 1.0/16, 1.0/1024, compare_level);
  // my_compare(t_attn_value_output_permuted, attn_value_output, 1.0/16, 1.0/1024, compare_level);
  // my_compare(t_attn_fc_short_cut_add, attn_fc_output, 1.0/16, 1.0/1024, 2);
  // auto attn_fc_layer_norm_x_2 = torch::slice(query_key_softmax_sum, 0, 0, 1);
  // my_compare(t_attn_fc_layer_norm_x, attn_fc_layer_norm_x_2, 1.0/16, 1.0/1024, 2);
  // my_compare(t_attn_fc_layer_norm_x_2, layer_norm_variance, 1.0/16, 1.0/1024, 2);
  // my_compare(t_attn_fc_layer_norm_output, attn_fc_output, 1.0/16, 1.0/1024, compare_level);
  // my_compare(t_feed_forward_fc1_activation_output, feed_forward_fc1_output, 1.0/16, 1.0/1024, compare_level);
  // my_compare(t_feed_forward_fc2_output, feed_forward_fc2_output, 1.0/16, 1.0/1024, 2);
  // my_compare(t_feed_forward_fc2_short_cut_output, feed_forward_fc2_output, 1.0/16, 1.0/1024, 2);
  // my_compare(t_feed_forward_fc2_layer_norm_sum_x_2, feed_forward_layer_norm_variance, 1.0/16, 1.0/1024, compare_level);
  // my_compare(t_feed_forward_fc2_layer_norm_sum_x, feed_forward_layer_norm_sum, 1.0/16, 1.0/1024, compare_level);
  // my_compare(t_feed_forward_fc2_layer_norm, feed_forward_fc2_output, 1.0/16, 1.0/1024, compare_level);
  printf("Comparing results finshed\n");

  return 0;
}

int main(int argc, char** argv){
  int round = 1, loop = 1, type=0;
  if(argc>2){
    round = atoi(argv[1]);
    loop = atoi(argv[2]);
  }if(argc>3){
    type = atoi(argv[3]);
  }
  test_bert_attn<1, 12, 384, 64, 3072>(round, loop, type);
  return 0;
}
