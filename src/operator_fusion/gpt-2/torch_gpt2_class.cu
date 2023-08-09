#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <mma.h>

#include <iostream>
#include <sstream>
#include <vector>

#include "torch/all.h"

#include "../../cuda_utils.h"
#include "../../utils.h"
#include "../torch_utils.h"

#include "gpt2-large.h"
#include "kernels/gemm.cu"
#include "kernels/fused_feed_forward_pipeline.cu"

using namespace souffle::gpt2;

template <int64_t batch_size, int64_t num_heads, int64_t max_seq_length,
          int64_t hidden_size, int64_t d_intermedia>
class FeedForward {
 public:
  FeedForward(std::string folder_path, torch::Tensor input_tensor) {
    this->folder_path = folder_path;
    this->input_tensor = input_tensor;
    load_weight();
    init_intermedia_tensor();
    init_tensor_pointers();
  }
  ~FeedForward() {}

  void load_weight() {
    this->feed_forward_fc1_weight =
        torch_load_tensor(folder_path + "gpt2-torch-data/MLP_c_fc.pt")
            .to(torch::kCUDA)
            .to(torch::kHalf);
    this->feed_forward_fc2_weight =
        torch_load_tensor(folder_path + "gpt2-torch-data/MLP_c_proj.pt")
            .to(torch::kCUDA)
            .to(torch::kHalf);
  }

  void init_intermedia_tensor() {
    // auto attn_fc_output =
    //     torch::ones({batch_size * max_seq_length, d_model}, options_fp16);
    // feed_forward_fc1_weight = torch::ones({d_model, d_intermedia},
    // options_fp16);
    feed_forward_fc1_bias = torch::zeros({d_intermedia}, options_fp16);
    feed_forward_fc1_output =
        torch::zeros({batch_size * max_seq_length, d_intermedia}, options_fp16);
    feed_forward_fc2_bias = torch::zeros({d_model}, options_fp16);
    feed_forward_fc2_output =
        torch::zeros({batch_size * max_seq_length, d_model}, options_fp16);
    feed_forward_fc2_layer_norm_sum =
        torch::zeros({batch_size * max_seq_length,}, options_fp16);
    feed_forward_fc2_layer_norm_sum_x_2 = torch::zeros(
        {batch_size * max_seq_length,}, options_fp16);
  }

  void init_tensor_pointers() {
    // Note, change here
    ptr_residual = input_tensor.data_ptr<at::Half>();
    ptr_input_tensor = input_tensor.data_ptr<at::Half>();
    ptr_feed_forward_fc1_weight = feed_forward_fc1_weight.data_ptr<at::Half>();
    ptr_feed_forward_fc1_bias = feed_forward_fc1_bias.data_ptr<at::Half>();
    ptr_feed_forward_fc1_output = feed_forward_fc1_output.data_ptr<at::Half>();
    ptr_feed_forward_fc2_weight = feed_forward_fc2_weight.data_ptr<at::Half>();
    ptr_feed_forward_fc2_bias = feed_forward_fc2_bias.data_ptr<at::Half>();
    ptr_feed_forward_fc2_output = feed_forward_fc2_output.data_ptr<at::Half>();
    ptr_feed_forward_fc2_layer_norm_sum =  feed_forward_fc2_layer_norm_sum.data_ptr<at::Half>();
    ptr_feed_forward_fc2_layer_norm_sum_x_2 = feed_forward_fc2_layer_norm_sum_x_2.data_ptr<at::Half>();
  }

  void torch_forward() {
    // 0. Layer norm
    // t_input_layer_norm = torch::layer_norm(input_tensor, {d_model,});
    // 1. fc1
    t_feed_forward_fc1_output =
        torch::matmul(input_tensor, feed_forward_fc1_weight);
    t_feed_forward_fc1_output += feed_forward_fc1_bias;
    // 2. relu
    t_feed_forward_fc1_output = torch::relu(t_feed_forward_fc1_output);
    // 3. fc2
    t_feed_forward_fc2_output =
        torch::matmul(t_feed_forward_fc1_output, feed_forward_fc2_weight);
    t_feed_forward_fc2_output += feed_forward_fc2_bias;
    // 4. short cut add
    t_feed_forward_fc2_short_cut_output = t_feed_forward_fc2_output + input_tensor;
    // 5. layer norm
    // t_feed_forward_fc2_layer_norm = torch::layer_norm(t_feed_forward_fc2_short_cut_output, {d_model,});
  }

  void souffle_forward() {
    // fc1_limited_blocks();
    // fc1();
    // fc2();
    fused_feed_forward_pipelined();
  }

  void fc1() {
    // 1. fc1
    void *fused_feed_forward_fc1_kernel_args[] = {
        (void *)&(ptr_feed_forward_fc1_weight), (void *)&(ptr_input_tensor),
        (void *)&(ptr_feed_forward_fc1_output)};
    const int feed_forward_fc1_shared_mem =
        (kStage *
         (kChunkK * kWmmaK *
              (kBlockRowWarps * FeedForwardFC1Params::kBlockRowTiles * kWmmaM +
               kInputSkew) +
          kBlockColWarps * FeedForwardFC1Params::kBlockColTiles * kWmmaN *
              (kChunkK * kWmmaK + kInputSkew))) *
        sizeof(half);
    printf("fc1 shared memory %d KB, grid blocks %d\n",
           feed_forward_fc1_shared_mem / 1024,
           FeedForwardFC1Params::kGridBlocks);
    const void *cuda_kernel_func = (const void *)
        gemm_three_stage<FeedForwardFC1Params::kWarpRowTiles,
                         FeedForwardFC1Params::kWarpColTiles,
                         kHiddenSize * kHiddenDim, kSeqLength, kHiddenDim, 1>;
    checkCuda(
        cudaFuncSetAttribute(
            cuda_kernel_func,
            cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
            feed_forward_fc1_shared_mem),
        __LINE__);
    checkCuda(cudaLaunchKernel(cuda_kernel_func,
                               dim3(FeedForwardFC1Params::kGridBlocks, 1, 1),
                               dim3(FeedForwardFC1Params::kBlockThreads, 1, 1),
                               fused_feed_forward_fc1_kernel_args,
                               feed_forward_fc1_shared_mem),
              __LINE__);
    cudaDeviceSynchronize();
  }

  void fc1_limited_blocks() {
    void *fused_feed_forward_fc1_kernel_args[] = {
        (void *)&(ptr_feed_forward_fc1_weight), (void *)&(ptr_input_tensor),
        (void *)&(ptr_feed_forward_fc1_output)};
    const int feed_forward_fc1_shared_mem = FeedForwardFC1LimitedBlocksParams::kSharedMemory;
    printf("fc1 shared memory %d KB, grid blocks %d\n",
           feed_forward_fc1_shared_mem / 1024,
           FeedForwardFC1LimitedBlocksParams::kGridBlocks);
    const void *cuda_kernel_func =
        (const void *)gemm_three_stage_limited_blocks<
            FeedForwardFC1LimitedBlocksParams::kWarpRowTiles,
            FeedForwardFC1LimitedBlocksParams::kWarpColTiles,
            FeedForwardFC1LimitedBlocksParams::kMTiles /*kMTiles*/,
            FeedForwardFC1LimitedBlocksParams::kNTiles /*kNTiles*/,
            kHiddenSize * kHiddenDim, kSeqLength, kHiddenDim, 1>;
    checkCuda(
        cudaFuncSetAttribute(
            cuda_kernel_func,
            cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
            feed_forward_fc1_shared_mem),
        __LINE__);
    checkCuda(
        cudaLaunchKernel(
            cuda_kernel_func,
            dim3(FeedForwardFC1LimitedBlocksParams::kGridBlocks, 1, 1),
            dim3(FeedForwardFC1LimitedBlocksParams::kBlockThreads, 1, 1),
            fused_feed_forward_fc1_kernel_args, feed_forward_fc1_shared_mem),
        __LINE__);
    cudaDeviceSynchronize();
  }

  void fc2() {
    void *fused_feed_forward_fc2_kernel_args[] = {
        (void *)&(ptr_feed_forward_fc2_weight),
        (void *)&(ptr_feed_forward_fc1_output),
        (void *)&(ptr_feed_forward_fc2_output)};
    
    const int gemm_k6_shared_mem = FeedForwardFC2Params::kSharedMemory;
    const int kGemmK6BlockThreads = 128;
    printf("gemm_k6 shared memory %d KB, grid blocks %d\n",
           gemm_k6_shared_mem / 1024, FeedForwardFC2Params::kGridBlocks);
    checkCuda(cudaFuncSetAttribute(
        (const void *)gemm_k6,
        cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
        gemm_k6_shared_mem));
    checkCuda(cudaLaunchKernel(
        (const void *)gemm_k6, dim3(FeedForwardFC2Params::kGridBlocks, 1, 1),
        dim3(FeedForwardFC2Params::kBlockThreads, 1, 1), fused_feed_forward_fc2_kernel_args,
        gemm_k6_shared_mem));
  }

  void fused_fc1_fc2_layernorm_relu() {}

  void fused_feed_forward_pipelined() {
    void* fused_feedforward_kernel_args[] = {
        (void *)&(ptr_input_tensor),
        (void *)&(ptr_input_tensor),
        (void *)&(eps), (void *)&(gama), (void *)&(beta),
        (void *)&(ptr_feed_forward_fc1_weight),
        (void *)&(ptr_feed_forward_fc1_output),
        (void *)&(ptr_feed_forward_fc2_weight),
        (void *)&(ptr_feed_forward_fc2_output),
        (void *)&(ptr_feed_forward_fc2_layer_norm_sum),
        (void *)&(ptr_feed_forward_fc2_layer_norm_sum_x_2)
    };
    const int fused_shared_memory = FeedForwardFC1LimitedBlocksParams::kSharedMemory;
    // std::max(
    //     FeedForwardFC1LimitedBlocksParams::kSharedMemory,
    //     FeedForwardFC2Params::kSharedMemory);
    
    const int fused_grid_blocks = (int)FeedForwardFC1LimitedBlocksParams::kGridBlocks;
    // std::max(
    //     (int)FeedForwardFC1LimitedBlocksParams::kGridBlocks,
    //     (int)FeedForwardFC2Params::kGridBlocks);
    printf("fused_feed_forward shared memory %d KB, grid blocks %d\n",
           fused_shared_memory / 1024, fused_grid_blocks);
    checkCuda(cudaFuncSetAttribute(
        (const void *)fused_feed_forwad_pipeline,
        cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
        fused_shared_memory));
    checkCuda(cudaLaunchCooperativeKernel((const void *)fused_feed_forwad_pipeline,
        dim3(fused_grid_blocks, 1, 1),
        dim3(FeedForwardFC1LimitedBlocksParams::kBlockThreads, 1, 1),
        fused_feedforward_kernel_args, fused_shared_memory));
    cudaDeviceSynchronize();
  }

  void print() {
    printf("feed_forward_fc1_output:");
    torch::print(this->feed_forward_fc1_output);
    printf("\nt_feed_forward_fc1_output:");
    torch::print(this->t_feed_forward_fc1_output);
    printf("\nfeed_forward_fc2_output:");
    torch::print(this->feed_forward_fc2_output);
    printf("\nt_feed_forward_fc2_output:");
    torch::print(this->t_feed_forward_fc2_output);
    printf("\nfeed_forward_fc2_short_cut_output:");
    torch::print(this->t_feed_forward_fc2_short_cut_output);
    my_compare(this->feed_forward_fc1_output, this->t_feed_forward_fc1_output, 1.0/16, 1.0/16, kPrintDiff);
    // my_compare(this->feed_forward_fc2_output, this->t_feed_forward_fc2_output, 1.0/16, 1.0/16, kPrintDiff);
    my_compare(this->feed_forward_fc2_output, this->t_feed_forward_fc2_short_cut_output, 1.0/16, 1.0/16, kPrintDiff);
  }

  std::vector<at::Half *> get_pointers() {
    std::vector<at::Half *> pointers;
    pointers.push_back(ptr_feed_forward_fc1_weight);
    pointers.push_back(ptr_feed_forward_fc1_bias);
    pointers.push_back(ptr_feed_forward_fc1_output);
    pointers.push_back(ptr_feed_forward_fc2_weight);
    pointers.push_back(ptr_feed_forward_fc2_bias);
    pointers.push_back(ptr_feed_forward_fc2_output);

    return pointers;
  }

  const int64_t d_model = num_heads * hidden_size;
  std::string folder_path;
  torch::Tensor input_tensor;
  // Weights
  torch::Tensor input_layer_norm;
  torch::Tensor feed_forward_fc1_weight;
  torch::Tensor feed_forward_fc1_bias;
  torch::Tensor feed_forward_fc1_output;
  torch::Tensor feed_forward_fc2_weight;
  torch::Tensor feed_forward_fc2_bias;
  torch::Tensor feed_forward_fc2_output;
  torch::Tensor feed_forward_fc2_shortcut_output;
  torch::Tensor feed_forward_fc2_layer_norm_sum;
  torch::Tensor feed_forward_fc2_layer_norm_sum_x_2;
  // Torch output tensors
  torch::Tensor t_input_layer_norm;
  torch::Tensor t_feed_forward_fc1_output;
  torch::Tensor t_feed_forward_fc1_activation_output;
  torch::Tensor t_feed_forward_fc2_output;
  torch::Tensor t_feed_forward_fc2_short_cut_output;
  torch::Tensor t_feed_forward_fc2_layer_norm;
  // Pointers
  at::Half *ptr_residual;
  at::Half *ptr_input_tensor;
  at::Half *ptr_feed_forward_fc1_weight;
  at::Half *ptr_feed_forward_fc1_bias;
  at::Half *ptr_feed_forward_fc1_output;
  at::Half *ptr_feed_forward_fc2_weight;
  at::Half *ptr_feed_forward_fc2_bias;
  at::Half *ptr_feed_forward_fc2_output;
  at::Half *ptr_feed_forward_fc2_shortcut_output;
  at::Half *ptr_feed_forward_fc2_layer_norm_sum;
  at::Half *ptr_feed_forward_fc2_layer_norm_sum_x_2;
  half eps = 0.00001, gama = 1, beta = 0;
};

int main(int argc, char *argv[]) {
  std::string folder_path =
      "/home/xiachunwei/Projects/tensor-compiler-gpu/src/"
      "operator_fusion/gpt-2/";
//   torch::Tensor feed_forward_input_tensor =
//       torch::ones({384, 20 * 64}, torch::kCUDA).to(torch::kHalf);
  torch::Tensor feed_forward_input_tensor =
      torch_load_tensor(folder_path + "gpt2-torch-data/MLP_input_hidden_states.pt")
          .to(torch::kCUDA)
          .to(torch::kHalf);
  FeedForward<1, 20, 384, 64, 5120> module_feed_forward(
      folder_path, feed_forward_input_tensor);
  module_feed_forward.torch_forward();
  module_feed_forward.souffle_forward();
  module_feed_forward.print();
}
