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

#include "kernels/gemm.cu"

#include "kernels/bert.h"
// #include "kernels/gemm_three_stages.h"

#include "npy.hpp"

#include "../../utils.h"
#include "../../cuda_utils.h"
#include "../torch_utils.h"

using namespace fuselage::experiments::networks::bert;
/* This bert is based on the implementation of Qianqi Sun*/


template<int64_t batch_size, int64_t num_heads, int64_t max_seq_length, int64_t hidden_size, int64_t d_intermedia>
float test_bert_attn(int round_cout=1, int loop=1, int func_id=0){
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

  // Create dummy data
  // auto src = torch::div(
  //   torch::ones({batch_size*max_seq_length, d_model}, options_fp16), 
  //   torch::tensor({scale,}, options_fp16));
  // auto qkv_weight = torch::div(
  //   torch::ones({3, d_model, d_model}, options_fp16), 
  //   torch::tensor({scale,}, options_fp16));
  // auto attn_value_output = torch::div(
  //   torch::ones({batch_size*num_heads, max_seq_length, hidden_size}, options_fp16),
  //   torch::tensor({scale,}, options_fp16));
  // auto attn_fc_weight = torch::div(
  //   torch::ones({d_model, d_model}, options_fp16), 
  //   torch::tensor({1,}, options_fp16));
  auto bias_qkv = torch::zeros({3, d_model}, options_fp16);
  float scale = 16;
  auto src = torch::nn::init::uniform_(
    torch::randn({batch_size*max_seq_length, d_model}, options_fp16), 0, 1);


  // Torch implementation
  // fused QKV matmul
  auto batched_src = torch::reshape(src.repeat({3, 1, 1}), {3, max_seq_length, d_model});
  auto t_output_qkv = torch::permute(
    torch::reshape(
      torch::bmm(batched_src, qkv_weight), 
        {3, max_seq_length, num_heads, hidden_size}), {0, 2, 1, 3}); //(3, num_heads, max_seq_length, hidden_size)
  auto qkv = torch::split(t_output_qkv, 1, 0);
  // auto t_query = torch::nn::init::xavier_normal_(
  //   torch::randn({num_heads, max_seq_length, hidden_size}, options_fp16));
  // auto t_query = torch::ones({num_heads, max_seq_length, hidden_size}, options_fp16);
  // auto t_key = torch::nn::init::uniform_(
  //   torch::randn({num_heads, max_seq_length, hidden_size}, options_fp16));
  // auto t_value = torch::nn::init::uniform_(
  //   torch::randn({num_heads, max_seq_length, hidden_size}, options_fp16));
  // auto t_query = torch::reshape(qkv[0], {num_heads, max_seq_length, hidden_size});
  // auto t_key = torch::reshape(qkv[1], {num_heads, max_seq_length, hidden_size});
  auto t_value = torch::reshape(qkv[2], {num_heads, max_seq_length, hidden_size});
  auto cloned_query = torch::reshape(qkv[0], {batch_size*num_heads, max_seq_length, hidden_size});
  auto cloned_key = torch::reshape(qkv[1], {batch_size*num_heads, max_seq_length, hidden_size});
  // auto t_query_key_output = t_query.bmm(torch::permute(t_key, {0, 2, 1}));
  auto t_query_key_output = cloned_query.bmm(torch::permute(cloned_key, {0, 2, 1}));
  auto t_attn_mask = torch::zeros({batch_size*num_heads, max_seq_length, max_seq_length}, options_fp16);
  float v_d_model[] = {d_model,};
  auto t_d_model = torch::from_blob(v_d_model, {1,}).to(torch::kCUDA);
  auto t_query_key_output_sum = torch::sum(torch::exp(t_query_key_output / torch::sqrt(t_d_model)) + t_attn_mask, -1);
  auto t_query_key_softmax = torch::softmax(
    (t_query_key_output / torch::sqrt(t_d_model)) + 
      t_attn_mask, -1, torch::kHalf);
  auto t_attn_value_output = torch::bmm(t_query_key_softmax, t_value); // Now (12, 384, 64)
  auto t_attn_value_output_permuted = torch::reshape(
    torch::permute(t_attn_value_output, {1, 0, 2}), {max_seq_length, d_model});
  auto t_attn_fc_output = torch::matmul(t_attn_value_output_permuted, attn_fc_weight);
  auto t_attn_fc_short_cut_add = t_attn_fc_output + src;
  auto t_attn_fc_layer_norm_x_2 = torch::sum(t_attn_fc_short_cut_add * t_attn_fc_short_cut_add, 1);
  auto t_attn_fc_layer_norm_x = torch::sum(t_attn_fc_short_cut_add, 1);
  auto t_attn_fc_layer_norm_output = torch::layer_norm(t_attn_fc_short_cut_add, {d_model,});
  auto t_feed_forward_fc1_output = torch::matmul(t_attn_fc_layer_norm_output, feed_forward_fc1_weight);
  auto t_feed_forward_fc1_activation_output = torch::relu(t_feed_forward_fc1_output);
  auto t_feed_forward_fc2_output = torch::matmul(t_feed_forward_fc1_activation_output, feed_forward_fc2_weight);
  auto t_feed_forward_fc2_short_cut_output = t_feed_forward_fc2_output + t_attn_fc_layer_norm_output;
  auto t_feed_forward_fc2_layer_norm_sum_x_2 = torch::sum(t_feed_forward_fc2_short_cut_output * t_feed_forward_fc2_short_cut_output, 1);
  auto t_feed_forward_fc2_layer_norm_sum_x = torch::sum(t_feed_forward_fc2_output, 1);
  // auto t_feed_forward_fc2_short_cut_output = t_feed_forward_fc2_output + torch::ones({max_seq_length, d_model}, options_fp16);
  auto t_feed_forward_fc2_layer_norm = torch::layer_norm(t_feed_forward_fc2_short_cut_output, {d_model,});
  printf("Torch finshed\n");

  // Our implementation
  auto output_qkv = torch::zeros({batch_size*3, num_heads, max_seq_length, hidden_size}, options_fp16);
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
  

  at::Half* ptr_src = src.data<at::Half>();
  at::Half* ptr_weight_qkv = qkv_weight.data<at::Half>();
  at::Half* ptr_bias_qkv = bias_qkv.data<at::Half>();
  at::Half* ptr_output_qkv = output_qkv.data<at::Half>();  
  at::Half* ptr_query = ptr_output_qkv + (max_seq_length * d_model);
  at::Half* ptr_key = ptr_query + (max_seq_length * d_model);
  at::Half* ptr_query_key_output = query_key_output.data<at::Half>();
  at::Half* ptr_t_attn_mask = t_attn_mask.data<at::Half>();
  float* ptr_query_key_softmax_sum = query_key_softmax_sum.data<float>();
  at::Half* ptr_attn_value_output = attn_value_output.data<at::Half>();
  at::Half* ptr_attn_fc_weight = attn_fc_weight.data<at::Half>();
  at::Half* ptr_attn_fc_output = attn_fc_output.data<at::Half>();
  at::Half* ptr_feed_forward_fc1_weight = feed_forward_fc1_weight.data<at::Half>();
  at::Half* ptr_feed_forward_fc1_output = feed_forward_fc1_output.data<at::Half>();
  at::Half* ptr_feed_forward_fc2_weight = feed_forward_fc2_weight.data<at::Half>();
  at::Half* ptr_feed_forward_fc2_output = feed_forward_fc2_output.data<at::Half>();
  float* ptr_attn_layer_norm_sum = attn_layer_norm_sum.data<float>();
  float* ptr_attn_layer_norm_variance = attn_layer_norm_variance.data<float>();
  float* ptr_feed_forward_layer_norm_sum = feed_forward_layer_norm_sum.data<float>();
  float* ptr_feed_forward_layer_norm_variance = feed_forward_layer_norm_variance.data<float>();
  int64_t* ptr_profile_clock = profile_clock.data<int64_t>();
  int64_t* ptr_attn_profile_clock = attn_profile_clock.data<int64_t>();
  int64_t* ptr_feed_forward_profile_clock = feed_forward_profile_clock.data<int64_t>();


  // Pointers from torch
  at::Half* ptr_t_attn_fc_layer_norm_output = t_attn_fc_layer_norm_output.data<at::Half>();
  at::Half* ptr_t_feed_forward_fc2_output = t_feed_forward_fc2_output.data<at::Half>();
  at::Half* ptr_t_attn_fc_output = t_attn_fc_output.data<at::Half>();
  at::Half* ptr_t_attn_fc_short_cut_add = t_attn_fc_short_cut_add.data<at::Half>();
  at::Half* ptr_t_feed_forward_fc2_short_cut_output =t_feed_forward_fc2_short_cut_output.data<at::Half>();

  half eps = 0.00001, gama=1, beta = 0;
  // 0. My Fused bert
  auto tmp_qkv_output = torch::ones({3, max_seq_length, d_model}, options_fp16);
  auto tmp_query_key_output = torch::ones({num_heads, max_seq_length, max_seq_length}, options_fp16);
  at::Half* ptr_tmp_qkv_output = tmp_qkv_output.data<at::Half>();
  at::Half* ptr_tmp_query_key_output = tmp_query_key_output.data<at::Half>();
  
  void * fused_bert_kernel_args[] = {
    (void *)&(ptr_weight_qkv), 
    (void *)&(ptr_src), 
    (void *)&(ptr_bias_qkv), 
    (void *)&(ptr_output_qkv), 
    // (void *)&(ptr_tmp_qkv_output), 
    (void *)&(ptr_query_key_output),
    // (void *)&(ptr_tmp_query_key_output),
    (void *)&(ptr_t_attn_mask),
    (void *)&(ptr_query_key_softmax_sum),
    (void *)&(ptr_attn_value_output),
    (void *)&(ptr_attn_fc_weight),
    (void *)&(ptr_attn_fc_output),
    (void *)&(ptr_attn_layer_norm_sum),
    (void *)&(ptr_attn_layer_norm_variance),
    (void *)&(eps), (void *)&(gama), (void *)&(beta),
    (void *)&(ptr_feed_forward_fc1_weight),
    (void *)&(ptr_feed_forward_fc1_output),
    (void *)&(ptr_feed_forward_fc2_weight),
    (void *)&(ptr_feed_forward_fc2_output),
    (void *)&(ptr_feed_forward_layer_norm_sum),
    (void *)&(ptr_feed_forward_layer_norm_variance),
    (void *)&(ptr_profile_clock),
    (void *)&(ptr_t_feed_forward_fc2_output),
    (void *)&(ptr_t_feed_forward_fc2_short_cut_output),
  };
  void * fused_bert_attn_kernel_args[] = {
    (void *)&(ptr_weight_qkv), 
    (void *)&(ptr_src), 
    (void *)&(ptr_bias_qkv), 
    (void *)&(ptr_output_qkv), 
    (void *)&(ptr_query_key_output),
    (void *)&(ptr_t_attn_mask),
    (void *)&(ptr_query_key_softmax_sum),
    (void *)&(ptr_attn_value_output),
    (void *)&(ptr_attn_fc_weight),
    (void *)&(ptr_attn_fc_output),
    (void *)&(ptr_attn_layer_norm_sum),
    (void *)&(ptr_attn_layer_norm_variance),
    (void *)&(eps), (void *)&(gama), (void *)&(beta),
    (void *)&(ptr_attn_profile_clock),
  };
  const size_t fused_bert_shared_mem = 108*1024;
  checkCuda(cudaFuncSetAttribute((void*)fused_sqq_bert, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, fused_bert_shared_mem));
  checkCuda(cudaFuncSetAttribute((void*)fused_sqq_bert_pipelined, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, fused_bert_shared_mem));
  checkCuda(cudaFuncSetAttribute((void*)fused_sqq_bert_pipelined_v2, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, fused_bert_shared_mem));
  checkCuda(cudaFuncSetAttribute((void*)fused_sqq_feedforward_pipelined, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, fused_bert_shared_mem));
  checkCuda(cudaFuncSetAttribute((void*)fused_sqq_bert_attn, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, fused_bert_shared_mem));
  checkCuda(cudaFuncSetAttribute((void*)fused_sqq_feedforward_pipelined_v2, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, fused_bert_shared_mem));
  checkCuda(cudaFuncSetAttribute((void*)fused_sqq_bert_query_key_softmax, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, fused_bert_shared_mem));
  
  // 1. fused qkv matmul
  void* fused_attn_kernel_args[] = {(void *)&(ptr_weight_qkv), (void *)&(ptr_src), 
    (void *)&(ptr_bias_qkv), (void *)&(ptr_output_qkv)
  };
  // (K, M) * (N, K) -> (N, M); (768, 768*3), (384, 768)-> (384, 768*3)
  // (384)/(2*2*16) * (768)/(2*16) = 6*12 = 72
  const size_t gemm_k1_shared_mem =
    (kStage * /* 3x (3x 4x16 x (2x1x16+8) +  2x3x16 x (4x16+8))*/
      (3 * kChunkK * kWmmaK *
          (kBlockRowWarps * kGemmK1WarpRowTiles * kWmmaM + kInputSkew) +
      kBlockColWarps * kGemmK1WarpColTiles * kWmmaN *
          (kChunkK * kWmmaK + kInputSkew))) *
    sizeof(half);
  // each block compute(2*16, 4*16)->(32, 64), need
  check_compatability(128, gemm_k1_shared_mem, (void*)gemm_add_qkv_bias);
  printf("qkv matmul shared memory: %ld, blocks %d\n", gemm_k1_shared_mem, 24*4);
  checkCuda(cudaFuncSetAttribute((void*)gemm_add_qkv_bias, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, gemm_k1_shared_mem));

  // 2. query key matmul
  auto t_ptr_key = cloned_key.data<at::Half>();
  auto t_ptr_query = cloned_query.data<at::Half>();
  void* fused_attn_query_key_kernel_args[] = {
    (void*)&(t_ptr_key), (void*)&(t_ptr_query),
    (void *)&(ptr_query_key_output)
  };
  const int gemm_k2_blocks =
        (max_seq_length / (kBlockRowWarps * kGemmK2WarpRowTiles * kWmmaM)) * /*3*/
        (max_seq_length / (kBlockColWarps * kGemmK2WarpColTiles * kWmmaN)) * /*3*/
        kGemmK2BatchedNum; /*12*/
  const int gemm_k2_shared_mem = ((kBlockRowWarps * kGemmK2WarpRowTiles * kWmmaM) * (kChunkK * kWmmaK + kInputSkew) +
    (kBlockColWarps * kGemmK2WarpColTiles * kWmmaN) * (kChunkK * kWmmaK + kInputSkew)) * sizeof(half);
  checkCuda(cudaFuncSetAttribute((void*)gemm_k2, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, gemm_k2_shared_mem));
  printf("gemm_k2 matmul shared memory: %ld, blocks %d\n", gemm_k2_shared_mem, 3*3*12);

  // tvm_query_key_matmul_cuda<<<dim3(768, 1, 1), dim3(192, 1, 1)>>>((half*)t_ptr_query, (half*)t_ptr_key, (half*)ptr_tvm_query_key_output);
  // checkCuda(cudaDeviceSynchronize());

  // 3. inputA:(12,384,64) [value], inputB:(12,384,384) [softmax_qk], output:(384,768)
  // (12, 384, 384) x (12, 384, 64) -> (12, 384, 64) ->reshape-> (384, 768)
  auto ptr_value = t_value.data<at::Half>();
  auto ptr_qk = t_query_key_output.data<at::Half>();
  // (64/(2*2*16)) * (384/(2*2*16)) * 12 = 6*12 = 72
  const int gemm_k3_blocks =
        (kHeadSize / (kBlockRowWarps * kGemmK3WarpRowTiles * kWmmaM)) *
        (batch_size*max_seq_length / (kBlockColWarps * kGemmK3WarpColTiles * kWmmaN)) *
        kGemmK3BatchedNum;
  const int gemm_k3_shared_mem =
        (kStage *
         (kChunkK * kWmmaK *
              (kBlockRowWarps * kGemmK3WarpRowTiles * kWmmaM + kInputSkew) +
          kBlockColWarps * kGemmK3WarpColTiles * kWmmaN *
              (kChunkK * kWmmaK + kInputSkew))) *
        sizeof(half);
  
  printf("gemm_k3 shared memory %d, blocks %d\n", gemm_k3_shared_mem, gemm_k3_blocks);
  void* fused_attn_value_kernel_args[] = {
    (void*)&(ptr_value), (void*)&(ptr_qk), (void*)&(ptr_attn_value_output)
  };

  // 4. inputA: (768, 768), inputB: (384, 768), C(384, 768)
  const int gemm_k4_blocks =
            (d_model / (kBlockRowWarps * kGemmK4WarpRowTiles * kWmmaM)) *
            (batch_size*max_seq_length / (kBlockColWarps * kGemmK4WarpColTiles * kWmmaN));
  const int gemm_k4_shared_mem =
      (kStage *
        (kChunkK * kWmmaK *
            (kBlockRowWarps * kGemmK4WarpRowTiles * kWmmaM + kInputSkew) +
        kBlockColWarps * kGemmK4WarpColTiles * kWmmaN *
            (kChunkK * kWmmaK + kInputSkew))) *
      sizeof(half);
  printf("gemm_k4 shared memory %d, blocks %d\n", gemm_k4_shared_mem, gemm_k4_blocks);
  void* fused_attn_fc_kernel_args[] = {
    (void*)&(ptr_attn_fc_weight), (void*)&(ptr_attn_value_output), (void*)&(ptr_attn_fc_output)
  };
  checkCuda(cudaFuncSetAttribute((const void *)gemm_three_stage<kGemmK4WarpRowTiles, kGemmK4WarpColTiles,
                        d_model, max_seq_length, d_model, 1>, 
                        cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, gemm_k4_shared_mem));

  // 5. inputA:(768,3072), inputB: (384,768), output:(384,3072)
  void* fused_feed_forward_fc1_kernel_args[] = {
    (void*)&(ptr_feed_forward_fc1_weight), (void*)&(ptr_attn_fc_output), (void*)&(ptr_feed_forward_fc1_output)
  };
  const int gemm_k5_blocks =
            (d_intermedia / (kBlockRowWarps * kGemmK5WarpRowTiles * kWmmaM)) *
            (batch_size*max_seq_length / (kBlockColWarps * kGemmK5WarpColTiles * kWmmaN));
  const int gemm_k5_shared_mem =
      (kStage *
        (kChunkK * kWmmaK *
            (kBlockRowWarps * kGemmK5WarpRowTiles * kWmmaM + kInputSkew) +
        kBlockColWarps * kGemmK5WarpColTiles * kWmmaN *
            (kChunkK * kWmmaK + kInputSkew))) *
      sizeof(half);
  printf("gemm_k5 shared memory %d, blocks %d\n", gemm_k5_shared_mem, gemm_k5_blocks);
  checkCuda(cudaFuncSetAttribute((const void *)gemm_three_stage<kGemmK5WarpRowTiles, kGemmK5WarpColTiles,
                                       kHiddenSize * kHiddenDim, kSeqLength,
                                       kHiddenDim, 1>, 
                                  cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, fused_bert_shared_mem));
  // 6.
  void* fused_feed_forward_fc2_kernel_args[] = {
    (void*)&(ptr_feed_forward_fc2_weight), (void*)&(ptr_feed_forward_fc1_output), (void*)&(ptr_feed_forward_fc2_output)
  };
  const int gemm_k6_blocks = (d_model / (kGemmK6BlockRowTiles * kWmmaM)) *
                                   (batch_size*max_seq_length / (kGemmK6BlockColTiles * kWmmaN));
  const int gemm_k6_shared_mem =
            (kStage * (kGemmK6BlockSliceKTiles * kWmmaK *
                           (kGemmK6BlockRowTiles * kWmmaM + kInputSkew) +
                       kGemmK6BlockColTiles * kWmmaN *
                           (kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew))) *
            sizeof(half);
  printf("gemm_k6 shared memory %d, blocks %d\n", gemm_k6_shared_mem, gemm_k6_blocks);
  checkCuda(cudaFuncSetAttribute((const void *)gemm_k6, 
                                  cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, gemm_k6_shared_mem));

  // 7. we use the result from torch
  auto tmp_feed_forward_fc1_output = torch::ones({batch_size*max_seq_length, d_intermedia}, options_fp16);
  at::Half* ptr_tmp_feed_forward_fc1_output = tmp_feed_forward_fc1_output.data<at::Half>();
  auto tmp_feed_forward_fc2_weight = torch::ones({d_intermedia, d_model}, options_fp16);
  at::Half* ptr_tmp_feed_forward_fc2_weight = tmp_feed_forward_fc2_weight.data<at::Half>();
  
  void* fused_feedforward_kernel_args[] = {
    // (void *)&(ptr_attn_fc_output),
    (void *)&(ptr_t_attn_fc_layer_norm_output),
    (void *)&(eps), (void *)&(gama), (void *)&(beta),
    (void *)&(ptr_feed_forward_fc1_weight),
    // (void *)&(ptr_tmp_feed_forward_fc1_output),
    // (void *)&(ptr_tmp_feed_forward_fc2_weight),
    (void *)&(ptr_feed_forward_fc1_output),
    (void *)&(ptr_feed_forward_fc2_weight),
    (void *)&(ptr_feed_forward_fc2_output),
    (void *)&(ptr_feed_forward_layer_norm_sum),
    (void *)&(ptr_feed_forward_layer_norm_variance),
    (void *)&(ptr_feed_forward_profile_clock),
    (void *)&(ptr_t_feed_forward_fc2_output)
  };
  const size_t fused_feed_forward_shared_mem_size = gemm_k5_shared_mem + kStage * (kGemmK6BlockSliceKTiles * kWmmaK *
                           (kGemmK6BlockRowTiles * kWmmaM + kInputSkew)) * sizeof(half2);
  checkCuda(cudaFuncSetAttribute((const void *)fused_sqq_feedforward, 
                                  cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, fused_feed_forward_shared_mem_size));
  
  // 8. For debug single kernel performance
  checkCuda(cudaFuncSetAttribute((const void *)debug_feed_forward_fc1, 
                                  cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, 120*1024));
  checkCuda(cudaFuncSetAttribute((const void *)debug_fused_sqq_bert_pipelined, 
                                  cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, fused_bert_shared_mem));
                                  

  auto device_func = [&](int func_id){
    switch (func_id)
    {
    case 0:
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "fused_bert", [&]{
        checkCuda(cudaLaunchCooperativeKernel((void*)fused_sqq_bert, 
        dim3(108, 1, 1), dim3(128, 1, 1), fused_bert_kernel_args, fused_bert_shared_mem));
      });
      break;
    case 1:
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "bert_attn_qkv", [&]{
        checkCuda(cudaLaunchCooperativeKernel((void*)gemm_add_qkv_bias, 
        dim3(24*4, 1, 1), dim3(128, 1, 1), fused_attn_kernel_args, gemm_k1_shared_mem));
      });
      break;
    case 2:
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(query_key_output.type(), "bert_attn_query_key", [&]{
        checkCuda(cudaLaunchCooperativeKernel((void*)gemm_k2, 
        dim3(3*3*12, 1, 1), dim3(128, 1, 1), fused_attn_query_key_kernel_args, gemm_k2_shared_mem));
      });
      break;
    case 3:
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "bert_attn_value", [&]{
        checkCuda(cudaLaunchCooperativeKernel((void*)gemm_reshape, 
        dim3(72, 1, 1), dim3(128, 1, 1), fused_attn_value_kernel_args, gemm_k3_shared_mem));
      });
      break;
    case 4:
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "bert_attn_fc", [&]{
      gemm_three_stage<kGemmK4WarpRowTiles, kGemmK4WarpColTiles, 
        d_model, max_seq_length, d_model, 1>
        <<<dim3(gemm_k4_blocks, 1, 1), dim3(128, 1, 1)>>>(
          (half*)ptr_attn_fc_weight, (half*)ptr_attn_value_output, (half*)ptr_attn_fc_output
        );
    });
      // AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "bert_attn_fc", [&]{
      //   checkCuda(cudaLaunchCooperativeKernel((const void *)gemm_three_stage<kGemmK4WarpRowTiles, kGemmK4WarpColTiles, 
      //                   d_model, max_seq_length, d_model, 1>, 
      //                   dim3(gemm_k4_blocks, 1, 1), dim3(128, 1, 1), fused_attn_fc_kernel_args, gemm_k4_shared_mem));
      // });
      break;
    case 5:
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "bert_feed_forward_fc1", [&]{
        // m=3072, n=384, k=768
        checkCuda(cudaLaunchCooperativeKernel((const void *)gemm_three_stage<kGemmK5WarpRowTiles, kGemmK5WarpColTiles,
                                       kHiddenSize * kHiddenDim, kSeqLength, kHiddenDim, 1>, 
                                      dim3(96,1,1), dim3(128, 1,1), fused_feed_forward_fc1_kernel_args, fused_bert_shared_mem));
      });
      break;
    case 6:
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "bert_feed_forward_fc2", [&]{
        checkCuda(cudaLaunchCooperativeKernel((const void *)gemm_k6, 
                                      dim3(gemm_k6_blocks,1,1), dim3(128, 1,1), fused_feed_forward_fc2_kernel_args, gemm_k6_shared_mem));
      });
      break;
    case 7:
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "bert_fused_feed_forward", [&]{
        checkCuda(cudaLaunchCooperativeKernel((const void *)fused_sqq_feedforward, 
                                      dim3(gemm_k5_blocks,1,1), dim3(128, 1,1), fused_feedforward_kernel_args, fused_bert_shared_mem));
      });
      break;
    case 8:
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "bert_fused_feed_forward_pipelined", [&]{
        checkCuda(cudaLaunchCooperativeKernel((const void *)fused_sqq_feedforward_pipelined, 
                                      dim3(gemm_k5_blocks,1,1), dim3(128, 1,1), fused_feedforward_kernel_args, fused_bert_shared_mem));
      });
      break;
    case 9:
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "fused_sqq_bert_pipelined_v2", [&]{
        checkCuda(cudaLaunchCooperativeKernel((const void *)fused_sqq_bert_pipelined_v2, 
                                      dim3(108,1,1), dim3(128, 1,1), fused_bert_kernel_args, fused_bert_shared_mem));
      });
      break;
    case 10:
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "debug_feed_forward_fc1", [&]{
        checkCuda(cudaLaunchCooperativeKernel((const void *)debug_feed_forward_fc1, 
                                      dim3(96,1,1), dim3(128, 1,1), fused_feed_forward_fc1_kernel_args, fused_bert_shared_mem));
      });
      break;
    case 11:
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "debug_fused_sqq_bert_pipelined", [&]{
        checkCuda(cudaLaunchCooperativeKernel((const void *)debug_fused_sqq_bert_pipelined, 
                                      dim3(108,1,1), dim3(128, 1,1), fused_bert_kernel_args, fused_bert_shared_mem));
      });
      break;
    case 12:
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "fused_sqq_feedforward_pipelined_v2", [&]{
        checkCuda(cudaLaunchCooperativeKernel((const void *)fused_sqq_feedforward_pipelined_v2, 
                                      dim3(gemm_k5_blocks,1,1), dim3(128, 1,1), fused_feedforward_kernel_args, fused_bert_shared_mem));
      });
      break;
    case 13:
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "attn_feed_forward", [&]{
        checkCuda(cudaLaunchCooperativeKernel((const void *)fused_sqq_bert_attn, 
                                      dim3(108,1,1), dim3(128, 1,1), fused_bert_attn_kernel_args, fused_bert_shared_mem));
        checkCuda(cudaLaunchCooperativeKernel((const void *)fused_sqq_feedforward_pipelined_v2, 
                                      dim3(gemm_k5_blocks,1,1), dim3(128, 1,1), fused_feedforward_kernel_args, fused_bert_shared_mem));
      });
      break;
    case 14:
      AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "attn_feed_forward", [&]{
        checkCuda(cudaLaunchCooperativeKernel((const void *)fused_sqq_bert_query_key_softmax, 
                                      dim3(108,1,1), dim3(128, 1,1), fused_bert_attn_kernel_args, fused_bert_shared_mem));
        checkCuda(cudaLaunchCooperativeKernel((const void *)fused_sqq_feedforward_pipelined_v2, 
                                      dim3(gemm_k5_blocks,1,1), dim3(128, 1,1), fused_feedforward_kernel_args, fused_bert_shared_mem));
      });
      break;
    default:
      break;
    }
  };

  // Run device function
  device_func(func_id);
  cudaDeviceSynchronize();
  
  // Check result
  auto value = torch::reshape(torch::split(output_qkv, 1, 0)[2], {num_heads, max_seq_length, hidden_size});
  my_compare(t_value, value, 1.0/16, 1.0/1024, compare_level);
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

  // Dump the clock profile in fused_bert
  // for(int i=0; i<kProfileStages; ++i){
  //   for(int j=0; j<108; ++j){
  //     for(int k=0; k<4; ++k){
  //       if(i>0){
  //         printf("stage: %d, block: %d, warp: %d, cycles: %ld\n", 
  //           i-1, j, k, profile_clock[i][j][k].item().toLong() - profile_clock[i-1][j][k].item().toLong());
  //       }
  //     }
  //   }
  // }
  
  torch::save(profile_clock, "profile_clock.pt");
  torch::save(attn_profile_clock, "attn_profile_clock.pt");
  torch::save(feed_forward_profile_clock, "feed_forward_profile_clock.pt");

  checkCuda(cudaEventDestroy(startEvent));
  checkCuda(cudaEventDestroy(stopEvent));
  return min_avg;
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
