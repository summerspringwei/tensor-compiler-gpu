#include <iostream>
#include <vector>
#include <math.h>

#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "torch/all.h"

#include "kernels/bert.h"
#include "kernels/gemm_add_qkv_bias.h"
#include "kernels/gemm_k2.h"

#include "../../utils.h"
#include "../../cuda_utils.h"

using namespace fuselage::experiments::networks::bert;
/* This bert is based on the implementation of Qianqi Sun*/

// dim3(24*4, 1, 1), dim3(128, 1, 1)
// matrix_a: (3*768, 768), matrix_b: (384, 768), bias: (3, 768), matrix_c: (3, 768, 384)
// __global__ void gemm_add_qkv_bias(const half *__restrict__ matrix_a,
//                                   const half *__restrict__ matrix_b,
//                                   const half *__restrict__ bias,
//                                   half *__restrict__ matrix_c);

void my_compare(torch::Tensor a, torch::Tensor b, float rotl, float aotl){
  at::Half* ptr_a = a.data<at::Half>();
  at::Half* ptr_b = b.data<at::Half>();
  size_t max_seq_length = a.size(1);
  size_t hidden_size = a.size(2);
  for(int i=0; i<a.size(0); ++i){
    for(int j=0; j<a.size(1); ++j){
      for(int k=0; k<a.size(2); ++k){
         int idx = i*max_seq_length*hidden_size + j * hidden_size + k;
         float x = __half2float(ptr_a[idx]);
         float y = __half2float(ptr_b[idx]);
        if(std::abs(x - y) > 
          rotl * std::abs(x) + aotl){
            printf("diff %d %d %d %f %f\n", i, j, k, x, y);
        }else{
          printf("same %d %d %d %f %f\n", i, j, k, x, y);
        }
      }
    }
  }
}


template<int64_t batch_size, int64_t num_heads, int64_t max_seq_length, int64_t hidden_size>
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
  const int d_model = num_heads * hidden_size;
  
  float scale = 32;
  
  // auto src = torch::div(
  //   torch::ones({batch_size*max_seq_length, d_model}, options_fp16), 
  //   torch::tensor({scale,}, options_fp16));
  // auto weight_qkv = torch::div(
  //   torch::ones({3, d_model, d_model}, options_fp16), 
  //   torch::tensor({scale,}, options_fp16));
  // auto attn_value_output = torch::div(
  //   torch::ones({batch_size*num_heads, max_seq_length, hidden_size}, options_fp16),
  //   torch::tensor({scale,}, options_fp16));
  auto attn_fc_weight = torch::div(
    torch::ones({d_model, d_model}, options_fp16), 
    torch::tensor({1,}, options_fp16));
  auto bias_qkv = torch::zeros({3, d_model}, options_fp16);
  
  
  auto src = torch::nn::init::uniform_(
    torch::randn({batch_size*max_seq_length, d_model}, options_fp16),
    -1/scale, 1/scale);
  auto weight_qkv = torch::nn::init::uniform_(
    torch::randn({3, d_model, d_model}, options_fp16),
    -1.0/scale, 1.0/scale);
  // auto bias_qkv = torch::nn::init::uniform_(
  //   torch::randn({3, d_model}, options_fp16),
  //   -1.0/scale, 1.0/scale);
  // auto weight_feed_forward_fc1 = torch::nn::init::uniform_(
  //   torch::randn({d_model*4, d_model}, options_fp16),
  //   -1.0/scale, 1.0/scale);
  // auto weight_feed_forward_fc2 = torch::nn::init::uniform_(
  //   torch::randn({d_model, d_model*4}, options_fp16),
  //   -1.0/scale, 1.0/scale);
  // auto attn_fc_weight = torch::nn::init::uniform_(
  //   torch::randn({d_model, d_model}, options_fp16),
  //   -1.0/scale, 1.0/scale);
  // auto t_attn_fc_weight = torch::clone(attn_fc_weight);

  // Torch implementation
  namespace F = torch::nn::functional;
  // fused QKV matmul
  auto batched_src = torch::reshape(src.repeat({3, 1, 1}), {3, max_seq_length, d_model});
  auto t_output_qkv = torch::permute(torch::reshape(
    torch::bmm(batched_src, weight_qkv/*torch::transpose(weight_qkv, -2, -1)*/), 
    {3, max_seq_length, num_heads, hidden_size}), {0, 2, 1, 3}); //(3, num_heads, max_seq_length, hidden_size)
  auto qkv = torch::split(t_output_qkv, 1, 0);

  // auto v_query_cpu = torch::empty({num_heads, max_seq_length, hidden_size}, options_fp16_cpu);
  // auto v_key_cpu = torch::empty({num_heads, max_seq_length, hidden_size}, options_fp16_cpu);
  float* v_query = (float*)malloc(max_seq_length*d_model*sizeof(float));
  float* v_key = (float*)malloc(max_seq_length*d_model*sizeof(float));
  for(int i=0; i<num_heads; ++i){
    for(int j=0; j<max_seq_length; ++j){
      for(int k=0; k<hidden_size; ++k){
        // v_query_cpu[i][j][k] =__float2half((j%10)/10.0);
        // v_key_cpu[i][j][k] =__float2half((j%10)/10.0); 
        int idx = i*max_seq_length*hidden_size + j * hidden_size + k;
        // v_query[idx] = __float2half((j%10)/10.0);
        // v_key[idx] = __float2half((j%384)/10.0);
        // v_key[idx] = (j%10)/10.0;
        v_key[idx] = 1;
        v_query[idx] = 1;
        // if(j<max_seq_length/2){
        //   v_query[idx] = (j%10)/10.0;
        // }else{
        //   v_query[idx] = 1;
        // }
      }
    }
  }
  // auto v_query = v_query_cpu.to(torch::kCUDA);
  // auto v_key = v_key_cpu.to(torch::kCUDA);
  
  torch::Tensor my_query = torch::from_blob(v_query, {num_heads, max_seq_length, hidden_size})
    .toType(torch::kFloat16).to(torch::kCUDA);
  torch::Tensor my_key = torch::from_blob(v_key, {num_heads, max_seq_length, hidden_size})
    .toType(torch::kFloat16).to(torch::kCUDA);
  auto cloned_query = qkv[0].clone();
  auto cloned_key = qkv[1].clone();
  auto t_query_key_output =  
    torch::bmm(
    torch::reshape(qkv[0], {num_heads, max_seq_length, hidden_size}), 
    torch::permute(torch::reshape(qkv[1], {num_heads, max_seq_length, hidden_size}), {0, 2, 1})
  );
  // 12, 384, 64
  
  // auto t_query_key_output = torch::bmm(
  //   torch::reshape(qkv[1], {num_heads, max_seq_length, hidden_size}), 
  //   torch::permute(torch::reshape(qkv[0], {num_heads, max_seq_length, hidden_size}), {0, 2, 1})
  //   );

  // auto t_output_qkv = torch::add(t_output_qkv_tmp, bias_qkv.index({"...", torch::indexing::None, "..."}));
  // auto t_output_qkv = torch::matmul(src, torch::transpose(weight_qkv, -2, -1)); // Now ()

  // auto qkv = torch::split(t_output_qkv, 768, 1);
  // auto arr_bias_qkv = torch::split(t_output_qkv, 768, 1);

  // auto t_query = torch::permute(
  //   torch::reshape(qkv[0]+, {max_seq_length, num_heads, hidden_size}), {1, 0, 2}); // Now (12, 128, 64)
  // auto t_key = torch::permute(
  //   torch::reshape(qkv[1], {max_seq_length, num_heads, hidden_size}), {1, 0, 2});  // Now (12, 128, 64)
  // auto t_value = torch::permute(
  //   torch::reshape(qkv[2], {{max_seq_length, num_heads, hidden_size}}), {1, 2, 0}); // Now (12, 64, 128)
  // // softmax(query_key_matmul / sqrt(hidden_size))
  // float factor = 64;
  // auto t_factor = torch::sqrt(torch::tensor({factor,}, options_fp16));
  // auto t_query_key_output = torch::softmax(
  //   torch::divide(
  //     torch::bmm(t_query, torch::permute(t_key, {0, 2, 1})), t_factor), 2); // Now (12, 128, 128)
  // // qk * value
  // auto t_attn_value_output = torch::bmm(t_query_key_output, torch::permute(t_value, {0, 2, 1})); // Now (12, 128, 64)
  // // reshape
  // auto t_attn_value_output_permuted = torch::reshape(
  //   torch::permute(t_attn_value_output, {1, 0, 2}), {batch_size*max_seq_length, d_model});// Now (128, 768)
  // // attn fc
  // auto t_attn_fc_output_tmp = torch::matmul(t_attn_value_output_permuted, torch::permute(t_attn_fc_weight, {1, 0}));
  // // Short cut
  // auto t_attn_fc_output = torch::add(src, t_attn_fc_output_tmp);
  // auto t_reduce_sum = torch::sum(t_attn_fc_output, 1, false, torch::kFloat32);
  // auto t_attn_layer_norm_output = torch::layer_norm(t_attn_fc_output, {768,});
  // // Feed forward
  // auto t_feed_forward_fc1_output = torch::matmul(t_attn_layer_norm_output, torch::permute(weight_feed_forward_fc1, {1, 0}));
  // auto t_feed_forward_fc2_output = torch::matmul(t_feed_forward_fc1_output, torch::permute(weight_feed_forward_fc2, {1, 0}));
  // auto t_feed_forward_short_cut_output = torch::add(t_feed_forward_fc2_output, t_attn_layer_norm_output);
  // auto t_feed_forward_layer_norm_output = torch::layer_norm(t_feed_forward_short_cut_output, {768,});


  // Our implementation
  auto output_qkv = torch::zeros({batch_size*max_seq_length, d_model*3}, options_fp16);
  // auto query = torch::zeros({batch_size*num_heads, max_seq_length, hidden_size}, options_fp16);
  // auto key = torch::zeros({batch_size*num_heads, max_seq_length, hidden_size}, options_fp16);
  // auto value = torch::zeros({batch_size*num_heads, hidden_size, max_seq_length}, options_fp16);
  auto query_key_output = torch::zeros({batch_size*num_heads, max_seq_length, max_seq_length}, options_fp16);
  // auto sum = torch::zeros({batch_size*num_heads, max_seq_length}, options_fp32); // Reduce_sum of query_key matmul
  // auto attn_value_output = torch::zeros({batch_size*num_heads, max_seq_length, hidden_size}, options_fp16);
  // auto attn_fc_output = torch::zeros({batch_size * max_seq_length, d_model}, options_fp16);
  // auto single_attn_fc_output = torch::zeros({batch_size * max_seq_length, d_model}, options_fp16);
  // auto inter_attn_fc_output = torch::zeros({batch_size * max_seq_length, d_model}, options_fp16);
  // auto variance = torch::zeros({batch_size * max_seq_length,}, options_fp32);
  // auto feed_forward_fc1_output = torch::zeros({batch_size * max_seq_length, d_model * 4}, options_fp16);
  // auto feed_forward_fc2_output = torch::zeros({batch_size * max_seq_length, d_model}, options_fp16);


  at::Half* ptr_src = src.data<at::Half>();
  at::Half* ptr_weight_qkv = weight_qkv.data<at::Half>();
  at::Half* ptr_bias_qkv = bias_qkv.data<at::Half>();
  at::Half* ptr_output_qkv = output_qkv.data<at::Half>();
  
  at::Half* ptr_query = ptr_output_qkv + (max_seq_length * d_model);
  at::Half* ptr_key = ptr_query + (max_seq_length * d_model);
  // at::Half* ptr_value = value.data<at::Half>();
  at::Half* ptr_query_key_output = query_key_output.data<at::Half>();
  // float* ptr_sum = sum.data<float>();
  // at::Half* ptr_attn_value_output = attn_value_output.data<at::Half>();
  // at::Half* ptr_inter_attn_fc_output = inter_attn_fc_output.data<at::Half>();
  // // at::Half* ptr_attn_value_output = t_attn_value_output_permuted.data<at::Half>();
  // at::Half* ptr_single_attn_fc_output = single_attn_fc_output.data<at::Half>();
  // at::Half* ptr_attn_fc_weight = attn_fc_weight.data<at::Half>();
  // at::Half* ptr_attn_fc_output = attn_fc_output.data<at::Half>();
  // float* ptr_variance = variance.data<float>();
  // half eps = 0.00001, gama=1, beta = 0;
  // at::Half* ptr_weight_feed_forward_fc1 = weight_feed_forward_fc1.data<at::Half>();
  // at::Half* ptr_feed_forward_fc1_output = feed_forward_fc1_output.data<at::Half>();
  // at::Half* ptr_weight_feed_forward_fc2 = weight_feed_forward_fc2.data<at::Half>();
  // at::Half* ptr_feed_forward_fc2_output = feed_forward_fc2_output.data<at::Half>();

  // // pointers from pytorch
  // at::Half* t_ptr_attn_value_output = t_attn_value_output_permuted.data<at::Half>();
  // at::Half* t_ptr_attn_fc_weight = t_attn_fc_weight.data<at::Half>();
  // at::Half* t_ptr_fc_output_tmp = t_attn_fc_output_tmp.data<at::Half>();
  // at::Half* t_ptr_attn_fc_output = t_attn_fc_output.data<at::Half>();

  // void *fused_kernel_args[] = { (void *)&(ptr_src), (void *)&(ptr_weight_qkv), 
  //   (void *)&(ptr_output_qkv), (void *)&(ptr_query), (void *)&(ptr_key), 
  //   (void *)&(ptr_value), (void*)&(ptr_query_key_output), (void*)&(ptr_sum),
  //   (void*)&(ptr_attn_value_output), (void*)&(ptr_attn_fc_weight), (void*)&(ptr_attn_fc_output),
  //   (void*)&(ptr_variance), (void *)&(eps), (void *)&(gama), (void *)&(beta)};
  
  // (void *)&(t_ptr_attn_value_output), (void *)&(t_ptr_attn_fc_weight), 
  //   (void *)&(t_ptr_fc_output_tmp), (void *)&(t_ptr_attn_fc_output), (void *)&(ptr_inter_attn_fc_output)

  // void * single_attn_fc_kernel_args[] = {
  //   (void *)&(ptr_attn_value_output), (void *)&(ptr_attn_fc_weight), (void *)&(ptr_single_attn_fc_output)
  // };

  // void * fused_feed_forward_kernel_args[] = {
  //   (void *)&(ptr_attn_fc_output), (void *)&(ptr_weight_feed_forward_fc1), 
  //   (void *)&(ptr_feed_forward_fc1_output), (void *)&(ptr_weight_feed_forward_fc2), 
  //   (void *)&(ptr_feed_forward_fc2_output),
  //   (void *)&(ptr_sum), (void *)&(ptr_variance), 
  //   (void *)&(eps), (void *)&(gama), (void *)&(beta)
  // };

  void* fused_attn_kernel_args[] = {(void *)&(ptr_weight_qkv), (void *)&(ptr_src), 
    (void *)&(ptr_bias_qkv), (void *)&(ptr_output_qkv)
  };
  const size_t gemm_k1_shared_mem =
    (kStage * /* 3x (3x 4x16 x (2x1x16+8) +  2x3x16 x (4x16+8))*/
      (3 * kChunkK * kWmmaK *
          (kBlockRowWarps * kGemmK1WarpRowTiles * kWmmaM + kInputSkew) +
      kBlockColWarps * kGemmK1WarpColTiles * kWmmaN *
          (kChunkK * kWmmaK + kInputSkew))) *
    sizeof(half);
  check_compatability(128, gemm_k1_shared_mem, (void*)gemm_add_qkv_bias);
  printf("shared memory: %ld\n", gemm_k1_shared_mem);
  checkCuda(cudaFuncSetAttribute((void*)gemm_add_qkv_bias, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, gemm_k1_shared_mem));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "bert_attn_qkv", [&]{
    checkCuda(cudaLaunchCooperativeKernel((void*)gemm_add_qkv_bias, dim3(24*4, 1, 1), dim3(128, 1, 1), fused_attn_kernel_args, gemm_k1_shared_mem));
  });
  auto t_ptr_key = cloned_key.data<at::Half>();
  auto t_ptr_query = cloned_key.data<at::Half>();
  // auto t_ptr_key = torch::permute(cloned_key.data<at::Half>(), {0, 1, 3, 2});
  // auto t_ptr_query = torch::permute(cloned_query.data<at::Half>(), {0, 1, 3, 2});
  // auto t_ptr_key = my_key.data<at::Half>();
  // auto t_ptr_query = my_query.data<at::Half>();
  void* fused_attn_query_key_kernel_args[] = {
    (void*)&(t_ptr_key), (void*)&(t_ptr_query), 
    (void *)&(ptr_query_key_output)
  };
  const int gemm_k2_blocks =
        (max_seq_length / (kBlockRowWarps * kGemmK2WarpRowTiles * kWmmaM)) * /*3*/
        (max_seq_length / (kBlockColWarps * kGemmK2WarpColTiles * kWmmaN)) * /*3*/
        kGemmK2BatchedNum; /*12*/
  const int gemm_k2_shared_mem =
        (kChunkK * kWmmaK *
             (kBlockRowWarps * kGemmK2WarpRowTiles * kWmmaM + kInputSkew) +
         kBlockColWarps * kGemmK2WarpColTiles * kWmmaN *
             (kChunkK * kWmmaK + kInputSkew)) *
        sizeof(half);
  checkCuda(cudaFuncSetAttribute((void*)gemm_k2, cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, gemm_k2_shared_mem));
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(output_qkv.type(), "bert_attn_query_key", [&]{
    checkCuda(cudaLaunchCooperativeKernel((void*)gemm_k2, dim3(3*3*12, 1, 1), dim3(128, 1, 1), fused_attn_query_key_kernel_args, gemm_k2_shared_mem));
  });
  checkCuda(cudaDeviceSynchronize());
  my_compare(query_key_output.to(torch::kCPU), t_query_key_output.to(torch::kCPU), 1/16.0, 1/1024.0);

  // std::vector<int64_t> shape_output_qkv = {batch_size*max_seq_length, 3*d_model,};
  std::vector<int64_t> shape_output_qkv = {3, d_model, max_seq_length};
  printf("output_qkv\n");
  torch::print(torch::reshape(output_qkv, shape_output_qkv), 768*100);
  printf("t_output_qkv\n");
  torch::print(torch::reshape(t_output_qkv, shape_output_qkv), 768*100);
  printf("query_key_output\n");
  torch::print(query_key_output, 768*100);
  printf("t_query_key_output\n");
  torch::print(t_query_key_output, 768*100);
  printf("query\n");
  torch::print(qkv[0], 768*100);
  printf("key\n");
  torch::print(qkv[1], 768*100);
  // printf("attn_value_output\n");
  // torch::print(torch::reshape(attn_value_output, {max_seq_length, d_model}), 768*100);
  // printf("t_attn_value_output_permuted\n");
  // torch::print(t_attn_value_output_permuted, 768*100);
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
  
  // my_compare(attn_fc_output.cpu().data<at::Half>(), t_attn_layer_norm_output.cpu().data<at::Half>(), 128, 768, 1.0/16, 1.0/1024);
  // Check result
  assert(torch::allclose(
    torch::reshape(output_qkv, {3, max_seq_length, d_model}), 
    torch::reshape(t_output_qkv, {3, max_seq_length, d_model}), 1.0/16, 1.0/1024));
  // assert(torch::allclose(query, t_query, 1.0/16, 1.0/1024));
  // assert(torch::allclose(key, t_key, 1.0/16, 1.0/1024));
  // assert(torch::allclose(value, t_value, 1.0/16, 1.0/1024));
  assert(torch::allclose(
    torch::reshape(t_query_key_output, {num_heads, max_seq_length, max_seq_length}),
    torch::reshape(query_key_output, {num_heads, max_seq_length, max_seq_length}), 1.0/16, 1.0/128, true));
  // assert(torch::allclose(
  //   torch::reshape(attn_value_output, {max_seq_length, d_model}), 
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
    
  }
  
  // 1. For original pointwise conv
  float min_avg = 1e10;
  for(int round =0; round<round_cout; ++round){
    float ms = 0, latency_sum = 0;
    for(int i=0; i<loop; ++i){
      checkCuda( cudaEventRecord(startEvent,0) );
      
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
  test_bert_attn<1, 12, 384, 64>(1, 1);
  return 0;
}