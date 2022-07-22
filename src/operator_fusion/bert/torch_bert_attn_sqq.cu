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

#include "npy.hpp"

#include "kernels/bert.h"
// #include "kernels/gemm_add_qkv_bias.h"
// #include "kernels/gemm_k2.h"
// #include "kernels/gemm_reshape.h"
// #include "kernels/gemm_three_stages.h"

#include "kernels/fused_sqq_bert.h"

#include "../../utils.h"
#include "../../cuda_utils.h"

using namespace fuselage::experiments::networks::bert;
/* This bert is based on the implementation of Qianqi Sun*/

void print_tensor_shape(torch::Tensor tensor){
  for(int i=0; i<tensor.sizes().size(); ++i){
    printf("%ld ", tensor.size(i));
  }printf("\n");
}

std::string idx2cordinate(uint64_t idx, std::vector<uint64_t>& acc_mul){
  std::vector<uint64_t> coordinate;
  const int dim = acc_mul.size();
  for(int i=0; i<dim-1; ++i){
    coordinate.push_back(idx / acc_mul[i+1]);
    idx = idx % acc_mul[i+1];
  }
  coordinate.push_back(idx);
  std::stringstream ss;
  // printf("(");
  ss << "(";
  for(int j=0; j<coordinate.size(); ++j){
    // printf("%u ", coordinate[j]);
    ss<<coordinate[j]<<" ";
  }
  ss <<")";
  // printf(")");
  return ss.str();
}


void my_compare(torch::Tensor& a, torch::Tensor& b, float rotl, float aotl, int print_detail=0){
  auto shape = a.sizes();
  const int dim = shape.size();
  std::vector<uint64_t> acc_mul(dim);
  acc_mul[dim-1] = shape[dim-1]; // acc_mul[2] = shape[2]
  for(int i=0; i<dim-1; ++i){
    acc_mul[dim-2-i] = acc_mul[dim-i-1] * shape[dim-2-i]; 
  }
  std::stringstream ss;
  int error_cnt = 0;
  auto num_elements = a.numel();
  auto reshaped_a = torch::reshape(a, {num_elements, });
  auto reshaped_b = torch::reshape(b, {num_elements, });
  for(uint64_t i=0; i<num_elements; ++i){
    auto x = reshaped_a[i].item().toHalf();
    auto y = reshaped_b[i].item().toHalf();
    if(std::abs(x - y) > rotl * std::abs(x) + aotl){
      error_cnt ++;
      if(print_detail>=1){
        ss << "diff " << idx2cordinate(i, acc_mul);
        ss << __half2float(x) << " " << __half2float(y) << "\n";
        // printf("diff ");
        // printf(" %f %f\n", __half2float(x), __half2float(y));
      }
    }else{
      if(print_detail>=2){
        ss << "same " << idx2cordinate(i, acc_mul);
        ss << __half2float(x) << " " << __half2float(y) << "\n";
        // printf("same ");
        // idx2cordinate(i, acc_mul);
        // printf(" %f %f\n", __half2float(x), __half2float(y));
      }
    }
  }
  printf("%s\n", ss.str().c_str());
  printf("my_compare error_cnt %d, total %d, error ratio %.3f\n", error_cnt, num_elements, ((float)error_cnt) / ((float)num_elements));
}


template<int64_t batch_size, int64_t num_heads, int64_t max_seq_length, int64_t hidden_size, int64_t d_intermedia>
float test_bert_attn(int round_cout=1, int loop=1){
  auto options_fp16 = torch::TensorOptions()
    .dtype(torch::kFloat16)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
  auto options_fp16_cpu = torch::TensorOptions()
    .dtype(torch::kFloat16)
    .layout(torch::kStrided)
    .device(torch::kCPU, 0)
    .requires_grad(false);
  auto options_fp32 = torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCUDA, 0)
    .requires_grad(false);
  auto options_fp32_cpu = torch::TensorOptions()
    .dtype(torch::kFloat32)
    .layout(torch::kStrided)
    .device(torch::kCPU, 0)
    .requires_grad(false);
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
    torch::randn({batch_size*max_seq_length, d_model}, options_fp16),
    0, 1);


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
  auto layer_norm_variance = torch::zeros({batch_size*max_seq_length, }, options_fp32);
  

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
  float* ptr_layer_norm_variance = layer_norm_variance.data<float>();
  at::Half* ptr_t_attn_fc_layer_norm_output = t_attn_fc_layer_norm_output.data<at::Half>();
  at::Half* ptr_t_feed_forward_fc2_output = t_feed_forward_fc2_output.data<at::Half>();
  half eps = 0.00001, gama=1, beta = 0;
  // My Fused bert
  void * fused_bert_kernel_args[] = {
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
    (void *)&(ptr_layer_norm_variance),
    (void *)&(eps), (void *)&(gama), (void *)&(beta),
    (void *)&(ptr_feed_forward_fc1_weight),
    (void *)&(ptr_feed_forward_fc1_output),
    (void *)&(ptr_feed_forward_fc2_weight),
    (void *)&(ptr_feed_forward_fc2_output),
  };
  const size_t fused_bert_shared_mem = 128*1024;
  checkCuda(cudaFuncSetAttribute((void*)fused_sqq_bert, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, fused_bert_shared_mem));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "fused_bert", [&]{
    checkCuda(cudaLaunchCooperativeKernel((void*)fused_sqq_bert, dim3(108, 1, 1), dim3(128, 1, 1), fused_bert_kernel_args, fused_bert_shared_mem));
  });
  checkCuda(cudaDeviceSynchronize());
  printf("Ours finshed\n");
  // Check result
  auto value = torch::reshape(torch::split(output_qkv, 1, 0)[2], {num_heads, max_seq_length, hidden_size});
  // my_compare(t_value, value, 1.0/16, 1.0/1024, 2);
  // my_compare(t_query_key_softmax, query_key_output, 1.0/16, 1.0/1024, 2);
  // my_compare(t_attn_value_output_permuted, attn_value_output, 1.0/16, 1.0/1024, 2);
  // my_compare(t_attn_fc_short_cut_add, attn_fc_output, 1.0/16, 1.0/1024, 2);
  auto attn_fc_layer_norm_x_2 = torch::slice(query_key_softmax_sum, 0, 0, 1);
  // my_compare(t_attn_fc_layer_norm_x_2, attn_fc_layer_norm_x_2, 1.0/16, 1.0/1024, 2);
  // my_compare(t_attn_fc_layer_norm_x, layer_norm_variance, 1.0/16, 1.0/1024, 2);
  my_compare(t_attn_fc_layer_norm_output, attn_fc_output, 1.0/16, 1.0/1024, 2);
  my_compare(t_feed_forward_fc1_activation_output, feed_forward_fc1_output, 1.0/16, 1.0/1024, 2);
  // my_compare(t_feed_forward_fc2_output, feed_forward_fc2_output, 1.0/16, 1.0/1024, 2);
  // my_compare(t_feed_forward_fc2_short_cut_output, feed_forward_fc2_output, 1.0/16, 1.0/1024, 2);
  my_compare(t_feed_forward_fc2_layer_norm_sum_x_2, attn_fc_layer_norm_x_2, 1.0/16, 1.0/1024, 2);
  my_compare(t_feed_forward_fc2_layer_norm_sum_x, layer_norm_variance, 1.0/16, 1.0/1024, 2);
  my_compare(t_feed_forward_fc2_layer_norm, feed_forward_fc2_output, 1.0/16, 1.0/1024, 2);
  printf("Comparing results finshed\n");
  // // 1. fused qkv matmul
  // void* fused_attn_kernel_args[] = {(void *)&(ptr_weight_qkv), (void *)&(ptr_src), 
  //   (void *)&(ptr_bias_qkv), (void *)&(ptr_output_qkv)
  // };
  // const size_t gemm_k1_shared_mem =
  //   (kStage * /* 3x (3x 4x16 x (2x1x16+8) +  2x3x16 x (4x16+8))*/
  //     (3 * kChunkK * kWmmaK *
  //         (kBlockRowWarps * kGemmK1WarpRowTiles * kWmmaM + kInputSkew) +
  //     kBlockColWarps * kGemmK1WarpColTiles * kWmmaN *
  //         (kChunkK * kWmmaK + kInputSkew))) *
  //   sizeof(half);
  // check_compatability(128, gemm_k1_shared_mem, (void*)gemm_add_qkv_bias);
  // printf("qkv matmul shared memory: %ld, blocks %d\n", gemm_k1_shared_mem, 24*4);
  // checkCuda(cudaFuncSetAttribute((void*)gemm_add_qkv_bias, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, gemm_k1_shared_mem));
  // AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "bert_attn_qkv", [&]{
  //   checkCuda(cudaLaunchCooperativeKernel((void*)gemm_add_qkv_bias, dim3(24*4, 1, 1), dim3(128, 1, 1), fused_attn_kernel_args, gemm_k1_shared_mem));
  // });
  // checkCuda(cudaDeviceSynchronize());

  // // 2. query key matmul
  // auto t_ptr_key = cloned_key.data<at::Half>();
  // auto t_ptr_query = cloned_query.data<at::Half>();
  // void* fused_attn_query_key_kernel_args[] = {
  //   (void*)&(t_ptr_key), (void*)&(t_ptr_query),
  //   (void *)&(ptr_query_key_output)
  // };
  // const int gemm_k2_blocks =
  //       (max_seq_length / (kBlockRowWarps * kGemmK2WarpRowTiles * kWmmaM)) * /*3*/
  //       (max_seq_length / (kBlockColWarps * kGemmK2WarpColTiles * kWmmaN)) * /*3*/
  //       kGemmK2BatchedNum; /*12*/
  const int gemm_k2_shared_mem = ((kBlockRowWarps * kGemmK2WarpRowTiles * kWmmaM) * (kChunkK * kWmmaK + kInputSkew) +
    (kBlockColWarps * kGemmK2WarpColTiles * kWmmaN) * (kChunkK * kWmmaK * kInputSkew)) * sizeof(half);
  // checkCuda(cudaFuncSetAttribute((void*)gemm_k2, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, gemm_k2_shared_mem));
  // printf("gemm_k2 matmul shared memory: %ld, blocks %d\n", gemm_k2_shared_mem, 3*3*12);
  // AT_DISPATCH_FLOATING_TYPES_AND_HALF(query_key_output.type(), "bert_attn_query_key", [&]{
  //   checkCuda(cudaLaunchCooperativeKernel((void*)gemm_k2, dim3(3*3*12, 1, 1), dim3(128, 1, 1), fused_attn_query_key_kernel_args, gemm_k2_shared_mem));
  // });
  // checkCuda(cudaDeviceSynchronize());

  // tvm_query_key_matmul_cuda<<<dim3(768, 1, 1), dim3(192, 1, 1)>>>((half*)t_ptr_query, (half*)t_ptr_key, (half*)ptr_tvm_query_key_output);
  // checkCuda(cudaDeviceSynchronize());

  // // 3. inputA:(12,384,64) [value], inputB:(12,384,384) [softmax_qk], output:(384,768)
  // // (12, 384, 384) x (12, 384, 64) -> (12, 384, 64) ->reshape-> (384, 768)
  // auto ptr_value = t_value.data<at::Half>();
  // auto ptr_qk = t_query_key_output.data<at::Half>();
  // // (64/(2*2*16)) * (384/(2*2*16)) * 12 = 6*12 = 72
  // const int gemm_k3_blocks =
  //       (kHeadSize / (kBlockRowWarps * kGemmK3WarpRowTiles * kWmmaM)) *
  //       (batch_size*max_seq_length / (kBlockColWarps * kGemmK3WarpColTiles * kWmmaN)) *
  //       kGemmK3BatchedNum;
  // const int gemm_k3_shared_mem =
  //       (kStage *
  //        (kChunkK * kWmmaK *
  //             (kBlockRowWarps * kGemmK3WarpRowTiles * kWmmaM + kInputSkew) +
  //         kBlockColWarps * kGemmK3WarpColTiles * kWmmaN *
  //             (kChunkK * kWmmaK + kInputSkew))) *
  //       sizeof(half);
  
  // printf("gemm_k3 shared memory %d, blocks %d\n", gemm_k3_shared_mem, gemm_k3_blocks);
  // void* fused_attn_value_kernel_args[] = {
  //   (void*)&(ptr_value), (void*)&(ptr_qk), (void*)&(ptr_attn_value_output)
  // };
  // checkCuda(cudaFuncSetAttribute((void*)gemm_reshape, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, gemm_k3_shared_mem));
  // AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "bert_attn_value", [&]{
  //   checkCuda(cudaLaunchCooperativeKernel((void*)gemm_reshape, dim3(72, 1, 1), dim3(128, 1, 1), fused_attn_value_kernel_args, gemm_k3_shared_mem));
  // });
  // checkCuda(cudaDeviceSynchronize());

  // // 4. inputA: (768, 768), inputB: (384, 768), C(384, 768)
  // auto ptr_attn_fc_weight = attn_fc_weight.data<at::Half>();
  // auto ptr_attn_fc_output = attn_fc_output.data<at::Half>();
  // const int gemm_k4_blocks =
  //           (d_model / (kBlockRowWarps * kGemmK4WarpRowTiles * kWmmaM)) *
  //           (batch_size*max_seq_length / (kBlockColWarps * kGemmK4WarpColTiles * kWmmaN));
  // const int gemm_k4_shared_mem =
  //     (kStage *
  //       (kChunkK * kWmmaK *
  //           (kBlockRowWarps * kGemmK4WarpRowTiles * kWmmaM + kInputSkew) +
  //       kBlockColWarps * kGemmK4WarpColTiles * kWmmaN *
  //           (kChunkK * kWmmaK + kInputSkew))) *
  //     sizeof(half);
  // printf("gemm_k4 shared memory %d, blocks %d\n", gemm_k4_shared_mem, gemm_k4_blocks);
  // void* fused_attn_fc_kernel_args[] = {
  //   (void*)&(ptr_attn_fc_weight), (void*)&(ptr_attn_value_output), (void*)&(ptr_attn_fc_output)
  // };
  // checkCuda(cudaFuncSetAttribute((const void *)gemm_three_stage<kGemmK4WarpRowTiles, kGemmK4WarpColTiles,
  //                       d_model, max_seq_length, d_model, 1>, 
  //                       cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, gemm_k4_shared_mem));
  // AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "bert_attn_fc", [&]{
  //   checkCuda(cudaLaunchCooperativeKernel((const void *)gemm_three_stage<kGemmK4WarpRowTiles, kGemmK4WarpColTiles,
  //                       d_model, max_seq_length, d_model, 1>, 
  //                       dim3(gemm_k4_blocks, 1, 1), dim3(128, 1, 1), fused_attn_fc_kernel_args, gemm_k4_shared_mem));
  // });
  // checkCuda(cudaDeviceSynchronize());

  // // 5. inputA:(768,3072), inputB: (384,768), output:(384,3072)
  // void* fused_feed_forward_fc1_kernel_args[] = {
  //   (void*)&(ptr_feed_forward_fc1_weight), (void*)&(ptr_attn_fc_output), (void*)&(ptr_feed_forward_fc1_output)
  // };
  // const int gemm_k5_blocks =
  //           (d_intermedia / (kBlockRowWarps * kGemmK5WarpRowTiles * kWmmaM)) *
  //           (batch_size*max_seq_length / (kBlockColWarps * kGemmK5WarpColTiles * kWmmaN));
  // const int gemm_k5_shared_mem =
  //     (kStage *
  //       (kChunkK * kWmmaK *
  //           (kBlockRowWarps * kGemmK5WarpRowTiles * kWmmaM + kInputSkew) +
  //       kBlockColWarps * kGemmK5WarpColTiles * kWmmaN *
  //           (kChunkK * kWmmaK + kInputSkew))) *
  //     sizeof(half);
  // printf("gemm_k5 shared memory %d, blocks %d\n", gemm_k5_shared_mem, gemm_k5_blocks);
  // checkCuda(cudaFuncSetAttribute((const void *)gemm_three_stage<kGemmK5WarpRowTiles, kGemmK5WarpColTiles,
  //                                 kHiddenSize * kHiddenDim, kSeqLength, kHiddenDim, 1>, 
  //                                 cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, gemm_k5_shared_mem));
  // AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "bert_feed_forward_fc1", [&]{
  //   checkCuda(cudaLaunchCooperativeKernel((const void *)gemm_three_stage<kGemmK5WarpRowTiles, kGemmK5WarpColTiles,
  //                                 kHiddenSize * kHiddenDim, kSeqLength, kHiddenDim, 1>, 
  //                                 dim3(gemm_k5_blocks,1,1), dim3(128, 1,1), fused_feed_forward_fc1_kernel_args, gemm_k5_shared_mem));
  // });
  // checkCuda(cudaDeviceSynchronize());

  // void* fused_feed_forward_fc2_kernel_args[] = {
  //   (void*)&(ptr_feed_forward_fc2_weight), (void*)&(ptr_feed_forward_fc1_output), (void*)&(ptr_feed_forward_fc2_output)
  // };
  // const int gemm_k6_blocks = (d_model / (kGemmK6BlockRowTiles * kWmmaM)) *
  //                                  (batch_size*max_seq_length / (kGemmK6BlockColTiles * kWmmaN));
  // const int gemm_k6_shared_mem =
  //           (kStage * (kGemmK6BlockSliceKTiles * kWmmaK *
  //                          (kGemmK6BlockRowTiles * kWmmaM + kInputSkew) +
  //                      kGemmK6BlockColTiles * kWmmaN *
  //                          (kGemmK6BlockSliceKTiles * kWmmaK + kInputSkew))) *
  //           sizeof(half);
  // printf("gemm_k6 shared memory %d, blocks %d\n", gemm_k6_shared_mem, gemm_k6_blocks);
  // checkCuda(cudaFuncSetAttribute((const void *)gemm_k6, 
  //                                 cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, gemm_k6_shared_mem));
  // AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "bert_feed_forward_fc2", [&]{
  //   checkCuda(cudaLaunchCooperativeKernel((const void *)gemm_k6, 
  //                                 dim3(gemm_k6_blocks,1,1), dim3(128, 1,1), fused_feed_forward_fc2_kernel_args, gemm_k6_shared_mem));
  // });
  // checkCuda(cudaDeviceSynchronize());

  // std::vector<int64_t> shape_output_qkv = {batch_size*max_seq_length, 3*d_model,};
  // std::vector<int64_t> shape_output_qkv = {3, d_model, max_seq_length};
  // printf("\noutput_qkv\n");
  // torch::print(torch::reshape(output_qkv, shape_output_qkv), 768*100);
  // printf("\nt_output_qkv\n");
  // torch::print(torch::reshape(t_output_qkv, shape_output_qkv), 768*100);
  // printf("\nquery_key_output\n");
  // torch::print(query_key_output, 768*100);
  // printf("\nt_query_key_output\n");
  // torch::print(t_query_key_output, 768*100);
  // printf("\nquery_key_output\n");
  // torch::print(query_key_output, 768*100);
  // printf("\nt_query_key_softmax\n");
  // torch::print(t_query_key_softmax, 768*100);
  // printf("\nquery_key_softmax_sum\n");
  // torch::print(query_key_softmax_sum, 768*100);
  // printf("\nt_query_key_output_sum\n");
  // torch::print(t_query_key_output_sum, 768*100);
  
  
  // printf("attn_fc_output\n");
  // torch::print(attn_fc_output, 768*100);
  // printf("t_attn_fc_output\n");
  // torch::print(t_attn_fc_output, 768*100);
  // printf("t_attn_layer_norm_output\n");
  // torch::print(t_attn_layer_norm_output, 768*100);
  // printf("t_attn_fc_output_tmp\n");
  // torch::print(t_attn_fc_output_tmp, 768*100);
  // printf("attn_fc_weight\n");
  // torch::print(attn_fc_weight, 768*100);
  // printf("single_attn_fc_output\n");
  // torch::print(single_attn_fc_output, 768*100);
  // printf("inter_attn_fc_output\n");
  // torch::print(inter_attn_fc_output, 768*100);
  
  
  
  // assert(torch::allclose(
  //   torch::reshape(output_qkv, {3, max_seq_length, d_model}), 
  //   torch::reshape(t_output_qkv, {3, max_seq_length, d_model}), 1.0/16, 1.0/1024));
  // assert(torch::allclose(query, t_query, 1.0/16, 1.0/1024));
  // assert(torch::allclose(key, t_key, 1.0/16, 1.0/1024));
  // assert(torch::allclose(value, t_value, 1.0/16, 1.0/1024));
  // assert(torch::allclose(
  //   torch::reshape(t_query_key_output, {num_heads, max_seq_length, max_seq_length}),
  //   torch::reshape(query_key_output, {num_heads, max_seq_length, max_seq_length}), 1.0/16, 1.0/1024, true));
  // assert(torch::allclose(
  //   torch::reshape(attn_value_output, {max_seq_length, d_model}), 
  //   t_attn_value_output_permuted, 1.0/16, 1.0/1024));
  // assert(torch::allclose(attn_fc_weight, t_attn_fc_weight, 1.0/16, 1.0/1024));
  // // assert(torch::allclose(t_attn_fc_output_tmp, single_attn_fc_output, 1.0/16, 1.0/1024));
  // printf("max sub:\n");
  // torch::print(torch::max(torch::abs(torch::sub(attn_fc_output, t_attn_fc_output))));
  // auto cmp_attn_fc_output = attn_fc_output.to(torch::kCPU);
  // auto cmp_t_attn_fc_output = t_attn_fc_output.to(torch::kCPU);
  // my_compare(cmp_attn_fc_output, cmp_t_attn_fc_output, 1.0/16, 1.0/128);
  // my_compare(attn_fc_output, t_attn_fc_output, 1.0/16, 1.0/128);
  // my_compare(t_feed_forward_fc1_output, feed_forward_fc1_output, 1.0/16, 1.0/128);
  // my_compare(t_feed_forward_fc2_output, feed_forward_fc2_output, 1.0/16, 1.0/128);
  // assert(torch::allclose(attn_fc_output, t_attn_fc_output, 1.0/16, 1.0/128));

  // assert(torch::allclose(attn_fc_output, t_attn_layer_norm_output, 1.0/16, 1.0/1024));


  // Benchmark
  cudaEvent_t startEvent, stopEvent;
  checkCuda(cudaEventCreate(&startEvent));
  checkCuda(cudaEventCreate(&stopEvent));
  // Warm up
  for(int i=0; i<100; ++i){
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "fused_bert", [&]{
    checkCuda(cudaLaunchCooperativeKernel((void*)fused_sqq_bert, dim3(108, 1, 1), dim3(128, 1, 1), fused_bert_kernel_args, fused_bert_shared_mem));
  });
  }
  
  // 1. For original pointwise conv
  float min_avg = 1e10;
  for(int round =0; round<round_cout; ++round){
    float ms = 0, latency_sum = 0;
    for(int i=0; i<loop; ++i){
      checkCuda( cudaEventRecord(startEvent,0) );
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "fused_bert", [&]{
    checkCuda(cudaLaunchCooperativeKernel((void*)fused_sqq_bert, dim3(108, 1, 1), dim3(128, 1, 1), fused_bert_kernel_args, fused_bert_shared_mem));
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
  test_bert_attn<1, 12, 384, 64, 3072>(3, 10000);
  return 0;
}
