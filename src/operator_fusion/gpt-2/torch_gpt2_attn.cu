#include <mma.h>

#include <iostream>
#include <vector>

#include "torch/all.h"

#include "../../cuda_utils.h"
#include "../../utils.h"
#include "../torch_utils.h"

#include "gpt2-large.h"
#include "kernels/gemm.cu"


template <int64_t batch_size, int64_t num_heads, int64_t max_seq_length,
          int64_t hidden_size, int64_t d_intermedia>

class Attn {
    public:
    Attn(std::string folder_path, torch::Tensor input_tensor){
        this->folder_path = folder_path;
        this->input_tensor = torch::permute(input_tensor, {1, 0, 2}).contiguous();
        load_weight();
        init_intermedia_tensor();
        init_tensor_pointers();
    }
    ~Attn(){}

    void load_weight() {
        this->qkv_weight =
        torch_load_tensor(folder_path + "gpt2-torch-data/c_attn.pt")
            .to(torch::kCUDA)
            .to(torch::kHalf);
        this->attn_fc_weight =
            torch_load_tensor(folder_path + "gpt2-torch-data/attn_c_proj.pt")
                .to(torch::kCUDA)
                .to(torch::kHalf);
        this->permuted_qkv_weight = torch::reshape(torch::permute(qkv_weight, {1, 0}), {3, d_model, d_model}).contiguous();// 3840, 1280 -> 3 * 1280, 1280
    }

    void init_intermedia_tensor() {
        this->t_attn_mask = torch::zeros({batch_size*num_heads, max_seq_length, max_seq_length}, options_fp16);
        // this->output_qkv = torch::zeros({batch_size*3, num_heads, max_seq_length, hidden_size}, options_fp16);
        this->output_qkv = torch::zeros({3*max_seq_length, d_model}, options_fp16);
        this->qkv_bias = torch::zeros({3, d_model}, options_fp16);
        this->query_key_output = torch::zeros({batch_size*num_heads, max_seq_length, max_seq_length}, options_fp16);
        this->query_key_softmax_sum = torch::zeros({batch_size*num_heads, max_seq_length}, options_fp32);
        this->attn_value_output = torch::zeros({batch_size*max_seq_length, d_model}, options_fp16);
        this->attn_fc_output = torch::zeros({batch_size*max_seq_length, d_model}, options_fp16);
        float v_d_model[] = {d_model,};
        this->t_d_model = torch::from_blob(v_d_model, {1,}).to(torch::kCUDA);
    }

    void init_tensor_pointers() {
        this->ptr_input_tensor = (at::Half*)this->input_tensor.data_ptr<at::Half>();
        // this->ptr_qkv_weight = (at::Half*)this->qkv_weight.data_ptr<at::Half>();
        this->ptr_qkv_weight = (at::Half*)this->permuted_qkv_weight.data_ptr<at::Half>();
        this->ptr_qkv_bias = (at::Half*)this->qkv_bias.data_ptr<at::Half>();
        this->ptr_output_qkv = (at::Half*)this->output_qkv.data_ptr<at::Half>();
        // this->ptr_query = this->ptr_output_qkv + (max_seq_length * d_model);
        // this->ptr_key = this->ptr_query + (max_seq_length * d_model);
        // this->ptr_value = this->ptr_key + (max_seq_length * d_model);
        
        this->ptr_query_key_output = this->query_key_output.data_ptr<at::Half>();
        this->ptr_t_attn_mask = this->t_attn_mask.data_ptr<at::Half>();
        this->ptr_query_key_softmax_sum = this->query_key_softmax_sum.data_ptr<float>();
        this->ptr_attn_value_output = this->attn_value_output.data_ptr<at::Half>();
        this->ptr_attn_fc_weight = this->attn_fc_weight.data_ptr<at::Half>();
        this->ptr_attn_fc_output = this->attn_fc_output.data_ptr<at::Half>();
    }

    void torch_forward() {
        auto batched_src = torch::reshape(input_tensor.repeat({3, 1, 1}), {3, max_seq_length, d_model});
        // auto torch_permuted_qkv_weight = torch::permute(qkv_weight.reshape({d_model, 3, d_model}), {1, 0, 2});
        auto torch_permuted_qkv_weight = torch::reshape(torch::permute(qkv_weight, {1, 0}), {3, d_model, d_model});// (3*1280r, 1280)
        printf("torch_permuted_qkv_weight shape: %s\n", get_torch_tensor_shape_str(torch_permuted_qkv_weight).c_str());
        this->bmm_output = torch::bmm(batched_src, torch_permuted_qkv_weight);// (3, seq_length, d_model)
        auto t_output_qkv = torch::permute(torch::reshape(bmm_output,
                {3, max_seq_length, num_heads, hidden_size}), {0, 2, 1, 3});// (3, num_heads, seq_length, hidden_size)
        t_qkv = torch::split(t_output_qkv, 1, 0);
        t_query = torch::reshape(t_qkv[0], {batch_size*num_heads, max_seq_length, hidden_size}).contiguous();
        t_key = torch::reshape(t_qkv[1], {batch_size*num_heads, max_seq_length, hidden_size}).contiguous();
        t_value = torch::reshape(t_qkv[2], {batch_size*num_heads, max_seq_length, hidden_size}).contiguous();
        // (20, 384, 64) * （20， 384， 64) -> (20, 384, 384)
        t_query_key_output = t_query.bmm(torch::permute(t_key, {0, 2, 1}));
        t_query_key_softmax = torch::softmax(
            (t_query_key_output / torch::sqrt(t_d_model)) + 
            t_attn_mask, -1, torch::kHalf);
        t_attn_value_output = torch::bmm(t_query_key_softmax, t_value); 
        t_attn_value_output_permuted = torch::reshape(
            torch::permute(t_attn_value_output, {1, 0, 2}), {max_seq_length, d_model});
        t_attn_fc_output = torch::matmul(t_attn_value_output_permuted, attn_fc_weight);
        t_attn_fc_short_cut_add = t_attn_fc_output + input_tensor;
        t_attn_fc_layernorm_output = torch::layer_norm(t_attn_fc_short_cut_add, {d_model,});
    }

    void qkv(){
        printf("permuted_qkv_weight shape: %s\n", get_torch_tensor_shape_str(this->permuted_qkv_weight).c_str());
        // auto weight = torch::ones({3, d_model, d_model}, options_fp16) / 16;
        // at::Half* ptr_weight = weight.data_ptr<at::Half>();
        // auto input = torch::ones({3, max_seq_length, d_model}, options_fp16);
        // at::Half* ptr_input = input.data_ptr<at::Half>();
        void* fused_attn_kernel_args[] = {(void *)&(ptr_qkv_weight), (void *)&(ptr_input_tensor), 
            (void *)&(ptr_qkv_bias), (void *)&(ptr_output_qkv)
        };
        // void* fused_attn_kernel_args[] = {(void *)&(ptr_weight), (void *)&(ptr_input), 
        //     (void *)&(ptr_qkv_bias), (void *)&(ptr_output_qkv)
        // };
        checkCuda(cudaFuncSetAttribute((void*)gemm_add_qkv_bias, 
            cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, AttnQKVParams::kSharedMemory), __LINE__);
        checkCuda(cudaLaunchKernel((void*)gemm_add_qkv_bias,
            dim3(AttnQKVParams::kGridBlocks, 1, 1), dim3(AttnQKVParams::kBlockThreads, 1, 1), 
            fused_attn_kernel_args, AttnQKVParams::kSharedMemory), __LINE__);
    }

    void query_key() {
        this->ptr_query = this->t_query.data_ptr<at::Half>();
        this->ptr_key = this->t_key.data_ptr<at::Half>();
        this->ptr_query_key_output = this->query_key_output.data_ptr<at::Half>();
        void* args[] = {
            (void *)&(ptr_key), (void *)&(ptr_query), (void *)&(ptr_query_key_output)
        };
        
        printf("blocks %d, shared memory: %d\n", AttnQueryKeyParams::kGridBlocks, AttnQueryKeyParams::kSharedMemory);
        checkCuda(cudaFuncSetAttribute((void*)gemm_k2, 
            cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, AttnQueryKeyParams::kSharedMemory), __LINE__);
        checkCuda(cudaLaunchKernel((void*)gemm_k2,
            dim3(AttnQueryKeyParams::kGridBlocks, 1, 1), dim3(AttnQueryKeyParams::kBlockThreads, 1, 1), 
            args, AttnQueryKeyParams::kSharedMemory), __LINE__);
    }



    void query_key_limited_blocks() {
        this->ptr_query = this->t_query.data_ptr<at::Half>();
        this->ptr_key = this->t_key.data_ptr<at::Half>();
        this->ptr_query_key_output = this->query_key_output.data_ptr<at::Half>();
        void* args[] = {
            (void *)&(ptr_key), (void *)&(ptr_query), (void *)&(ptr_query_key_output)
        };
        printf("blocks %d, shared memory: %d\n", AttnQueryKeyParamsLimitedBlocks::kGridBlocks, AttnQueryKeyParamsLimitedBlocks::kSharedMemory);
        checkCuda(cudaFuncSetAttribute((void*)gemm_k2_limited_blocks, 
            cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, AttnQueryKeyParamsLimitedBlocks::kSharedMemory), __LINE__);
        checkCuda(cudaLaunchKernel((void*)gemm_k2_limited_blocks,
            dim3(AttnQueryKeyParamsLimitedBlocks::kGridBlocks, 1, 1), dim3(AttnQueryKeyParamsLimitedBlocks::kBlockThreads, 1, 1), 
            args, AttnQueryKeyParamsLimitedBlocks::kSharedMemory), __LINE__);
    }

    void query_key_limited_blocks_div_softmax(){
        this->ptr_query = this->t_query.data_ptr<at::Half>();
        this->ptr_key = this->t_key.data_ptr<at::Half>();
        this->ptr_query_key_softmax_sum = this->query_key_softmax_sum.data_ptr<float>();
        this->ptr_query_key_output = this->query_key_output.data_ptr<at::Half>();
        void* args[] = {
            (void *)&(ptr_key), (void *)&(ptr_query), (void*)&(ptr_query_key_softmax_sum), (void *)&(ptr_query_key_output)
        };

        printf("blocks %d, shared memory: %d\n", AttnQueryKeyParamsLimitedBlocks::kGridBlocks, AttnQueryKeyParamsLimitedBlocks::kSharedMemory);
        checkCuda(cudaFuncSetAttribute((void*)gemm_k2_limited_blocks_div_softmax, 
            cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, AttnQueryKeyParamsLimitedBlocks::kSharedMemory), __LINE__);
        checkCuda(cudaLaunchCooperativeKernel((void*)gemm_k2_limited_blocks_div_softmax,
            dim3(AttnQueryKeyParamsLimitedBlocks::kGridBlocks, 1, 1), dim3(AttnQueryKeyParamsLimitedBlocks::kBlockThreads, 1, 1), 
            args, AttnQueryKeyParamsLimitedBlocks::kSharedMemory), __LINE__);
    }
    
    void attn_value() {
        this->ptr_value = this->t_value.data_ptr<at::Half>();
        this->ptr_query_key_output = this->t_query_key_softmax.data_ptr<at::Half>();
        void* args[] = {
            (void *)&(ptr_value), (void *)&(ptr_query_key_output), 
            (void *)&(ptr_attn_value_output), 
        };
        printf("blocks %d, shared memory: %d\n", AttnValueParams::kGridBlocks, AttnValueParams::kSharedMemory);
        checkCuda(cudaFuncSetAttribute((void*)gemm_reshape, 
            cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, AttnValueParams::kSharedMemory), __LINE__);
        checkCuda(cudaLaunchKernel((void*)gemm_reshape, 
            dim3(AttnValueParams::kGridBlocks, 1, 1), dim3(AttnValueParams::kBlockThreads, 1, 1), args, AttnValueParams::kSharedMemory));
    }

    void attn_fc() {
        this->ptr_attn_value_output = t_attn_value_output_permuted.data_ptr<at::Half>();
        void* args [] = {
            (void*)&ptr_attn_fc_weight, (void*)&ptr_attn_value_output, (void*)&ptr_attn_fc_output
        };
        printf("blocks %d, shared memory: %d\n", AttnFcParams::kGridBlocks, AttnFcParams::kSharedMemory);
        void* func_kernel = (void*)gemm_three_stage<AttnFcParams::kGemmK4WarpRowTiles, AttnFcParams::kGemmK4WarpColTiles, 
            kHeadNum * kHeadSize, kSeqLength, kHeadNum * kHeadSize, 1>;
        checkCuda(cudaFuncSetAttribute((void*)func_kernel, 
            cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize, AttnFcParams::kSharedMemory), __LINE__);
        checkCuda(cudaLaunchKernel((void*)func_kernel,
            dim3(AttnFcParams::kGridBlocks, 1, 1), dim3(AttnFcParams::kBlockThreads, 1, 1), 
            args, AttnFcParams::kSharedMemory), __LINE__);
    }


    void souffle_forward() {
        qkv();
        // query_key();
        // query_key_limited_blocks();
        query_key_limited_blocks_div_softmax();
        
        attn_value();
        attn_fc();
    }

    void print() {
        printf("\ninput_tensor:\n");
        torch::print(input_tensor);
        printf("\nqkv_weight:\n");
        torch::print(qkv_weight);
        printf("\nt_query:\n");
        // torch::print(t_query);
        torch::print(t_qkv[0]);
        printf("\nt_key:\n");
        // torch::print(t_key);
        torch::print(t_qkv[1]);
        printf("\nt_value:\n");
        // torch::print(t_value);
        torch::print(t_qkv[2]);
        printf("\nbmm_output:\n");
        torch::print(bmm_output);
        printf("\noutput_qkv\n");
        torch::print(output_qkv);
        
        printf("\nt_query_key_output:\n");
        torch::print(t_query_key_output);
        printf("\nt_query_key_softmax\n");
        torch::print(t_query_key_softmax);
        printf("\nquery_key_output:\n");
        torch::print(query_key_output);

        printf("\nt_attn_value_output:\n");
        torch::print(t_attn_value_output_permuted);
        printf("\nattn_value_output:\n");
        torch::print(attn_value_output);
        printf("\nattn_fc_output:\n");
        torch::print(t_attn_fc_output);
        printf("\nattn_fc_output:\n");
        torch::print(attn_fc_output);
        // my_compare();
    }

public:
    const int64_t d_model = num_heads * hidden_size;
    std::string folder_path;
    // Torch tensors
    torch::Tensor input_tensor;
    std::vector<at::Tensor> t_qkv;
    torch::Tensor permuted_qkv_weight;
    torch::Tensor bmm_output;
    torch::Tensor t_query;
    torch::Tensor t_key;
    torch::Tensor t_value;
    torch::Tensor t_query_key_output;
    torch::Tensor t_attn_mask;
    torch::Tensor t_d_model;
    torch::Tensor t_query_key_softmax;
    torch::Tensor t_attn_value_output;
    torch::Tensor t_attn_value_output_permuted;
    torch::Tensor t_attn_fc_output;
    torch::Tensor t_attn_fc_short_cut_add;
    torch::Tensor t_attn_fc_layernorm_output;

    // Our tensors
    torch::Tensor output_qkv;
    torch::Tensor query_key_output;
    torch::Tensor query_key_softmax_sum;
    torch::Tensor attn_value_output;
    torch::Tensor attn_fc_output;
    // Weights
    torch::Tensor qkv_weight;
    torch::Tensor qkv_bias;
    torch::Tensor attn_fc_weight;
    // Our pointers
    at::Half* ptr_input_tensor;
    at::Half* ptr_qkv_weight;
    at::Half* ptr_qkv_bias;
    at::Half* ptr_output_qkv;
    at::Half* ptr_query;
    at::Half* ptr_key;
    at::Half* ptr_value;
    at::Half* ptr_query_key_output;
    at::Half* ptr_t_attn_mask;
    float* ptr_query_key_softmax_sum;
    at::Half* ptr_attn_value_output;
    at::Half* ptr_attn_fc_weight;
    at::Half* ptr_attn_fc_output;
};


int main(int argc, char* argv) {
    std::string folder_path =
      "/home/xiachunwei/Projects/tensor-compiler-gpu/src/operator_fusion/gpt-2/";
    torch::Tensor input_tensor = 
      torch_load_tensor(folder_path + "gpt2-torch-data/attn_input_hidden_states.pt")
          .to(torch::kCUDA)
          .to(torch::kHalf);
    Attn<1, 20, 384, 64, 5120> module_attn(
      folder_path, input_tensor);
    
    module_attn.torch_forward();
    module_attn.souffle_forward();
    module_attn.print();
    return 0;
}
