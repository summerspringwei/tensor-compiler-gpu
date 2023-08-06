#include <iostream>
#include <math.h>
#include <sstream>
#include <vector>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "torch/all.h"

#include "gpt2-large.h"
#include "kernels/gemm.cu"

#include "../../cuda_utils.h"
#include "../../utils.h"
#include "../torch_utils.h"

using namespace souffle::gpt2;

template <int64_t batch_size, int64_t num_heads, int64_t max_seq_length,
          int64_t hidden_size, int64_t d_intermedia>
class FeedForward {

public:
    FeedForward(std::string folder_path, torch::Tensor input_tensor){
        this->folder_path = folder_path;
        this->input_tensor = input_tensor;
        load_weight();
        init_intermedia_tensor();
        init_tensor_pointers();
    }
    ~FeedForward() {

    }
    
    void load_weight(){
        this->feed_forward_fc1_weight =
            torch_load_tensor(folder_path + "gpt2-torch-data/MLP_c_fc.pt")
                .to(torch::kCUDA)
                .to(torch::kHalf);
        this->feed_forward_fc2_weight =
            torch_load_tensor(folder_path + "gpt2-torch-data/MLP_c_proj.pt")
                .to(torch::kCUDA)
                .to(torch::kHalf);
    }

    void init_intermedia_tensor(){
        // auto attn_fc_output =
        //     torch::ones({batch_size * max_seq_length, d_model}, options_fp16);
        // feed_forward_fc1_weight = torch::ones({d_model, d_intermedia}, options_fp16);
        feed_forward_fc1_bias = torch::zeros({d_intermedia}, options_fp16);
        feed_forward_fc1_output =
            torch::zeros({batch_size * max_seq_length, d_intermedia}, options_fp16);
        feed_forward_fc2_bias = torch::zeros({d_model}, options_fp16);
        feed_forward_fc2_output =
          torch::zeros({batch_size * max_seq_length, d_model}, options_fp16);
    }

    void init_tensor_pointers(){
        ptr_input_tensor = input_tensor.data_ptr<at::Half>();
        ptr_feed_forward_fc1_weight = feed_forward_fc1_weight.data_ptr<at::Half>();
        ptr_feed_forward_fc1_bias = feed_forward_fc1_bias.data_ptr<at::Half>();
        ptr_feed_forward_fc1_output = feed_forward_fc1_output.data_ptr<at::Half>();
        ptr_feed_forward_fc2_weight = feed_forward_fc2_weight.data_ptr<at::Half>();
        ptr_feed_forward_fc2_bias = feed_forward_fc2_bias.data_ptr<at::Half>();
        ptr_feed_forward_fc2_output = feed_forward_fc2_output.data_ptr<at::Half>();
    }

    void torch_forward(){
        t_feed_forward_fc1_output = torch::matmul(input_tensor, feed_forward_fc1_weight);
        t_feed_forward_fc1_output += feed_forward_fc1_bias;
        t_feed_forward_fc2_output = torch::matmul(t_feed_forward_fc1_output, feed_forward_fc2_weight);
        t_feed_forward_fc2_output += feed_forward_fc2_bias;
    }

    void souffle_forward(){
        fc1();
        fc2();
    }

    void fc1() {
        // 1. fc1
        void *fused_feed_forward_fc1_kernel_args[] = {
            (void *)&(ptr_feed_forward_fc1_weight), (void *)&(ptr_input_tensor),
            (void *)&(ptr_feed_forward_fc1_output)
        };
        const int feed_forward_fc1_shared_mem =
        (kStage *
         (kChunkK * kWmmaK *
              (kBlockRowWarps * FeedForwardFC1Params::kBlockRowTiles * kWmmaM +
               kInputSkew) +
          kBlockColWarps * FeedForwardFC1Params::kBlockColTiles * kWmmaN *
              (kChunkK * kWmmaK + kInputSkew))) *
        sizeof(half);
        printf("fc1 shared memory %d KB, grid block %d\n", feed_forward_fc1_shared_mem / 1024, FeedForwardFC1Params::kGridBlocks);
        const void* cuda_kernel_func = (const void *)gemm_three_stage<FeedForwardFC1Params::kWarpRowTiles,
                                        FeedForwardFC1Params::kWarpColTiles,
                                        kHiddenSize * kHiddenDim, kSeqLength,
                                        kHiddenDim, 1>;
        checkCuda(cudaFuncSetAttribute(
            cuda_kernel_func,
            cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
            feed_forward_fc1_shared_mem), __LINE__);
        checkCuda(cudaLaunchKernel(cuda_kernel_func,
                     dim3(FeedForwardFC1Params::kGridBlocks, 1, 1),
                     dim3(FeedForwardFC1Params::kBlockThreads, 1, 1),
                     fused_feed_forward_fc1_kernel_args, 
                     feed_forward_fc1_shared_mem), __LINE__);
        t_feed_forward_fc1_output = torch::matmul(input_tensor, feed_forward_fc1_weight);
        cudaDeviceSynchronize();
    }

    void fc2() {
        
    }

    void print(){
        torch::print(this->feed_forward_fc1_output);
        torch::print(this->t_feed_forward_fc1_output);
    }
    std::vector<at::Half*> get_pointers() {
        std::vector<at::Half*> pointers;
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
    torch::Tensor feed_forward_fc1_weight;
    torch::Tensor feed_forward_fc1_bias;
    torch::Tensor feed_forward_fc1_output;
    torch::Tensor feed_forward_fc2_weight;
    torch::Tensor feed_forward_fc2_bias;
    torch::Tensor feed_forward_fc2_output;
    // Torch output tensors
    torch::Tensor t_feed_forward_fc1_output;
    torch::Tensor t_feed_forward_fc2_output;
    // Pointers
    at::Half *ptr_input_tensor;
    at::Half *ptr_feed_forward_fc1_weight;
    at::Half *ptr_feed_forward_fc1_bias;
    at::Half *ptr_feed_forward_fc1_output;
    at::Half *ptr_feed_forward_fc2_weight;
    at::Half *ptr_feed_forward_fc2_bias;
    at::Half *ptr_feed_forward_fc2_output;
};


int main(int argc, char* argv[]) {
    std::string folder_path = "/home/xiachunwei/Projects/tensor-compiler-gpu/src/operator_fusion/gpt-2/";
    torch::Tensor feed_forward_input_tensor = torch::ones({384, 20*64}, torch::kCUDA).to(torch::kHalf);
    FeedForward<1, 20, 384, 64, 5120> module_feed_forward(folder_path, feed_forward_input_tensor);
    module_feed_forward.torch_forward();
    module_feed_forward.souffle_forward();
    module_feed_forward.print();
}
