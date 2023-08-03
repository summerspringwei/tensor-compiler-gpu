// Deal with batch_size < 4
#include "kernels/vector_matrix_mul.h"
#include "torch/all.h"
#include "../torch_utils.h"


int main(int argc, char* argv[]){
    // Load weight
    torch::Tensor attn_fc_weight =
      torch_load_tensor("gpt2-torch-data/attn_c_proj.pt")
          .to(torch::kCUDA)
          .to(torch::kHalf);
    const int out_dim = attn_fc_weight.sizes()[0];
    const int reduce_dim = attn_fc_weight.sizes()[1];
    const int batch_size = 1;
      auto src = torch::nn::init::uniform_(
    torch::randn({batch_size, reduce_dim}, options_fp16), 0, 1);
    auto output = torch::empty({batch_size, out_dim}, options_fp16);
    
    // Declare pointers
    auto d_ptr_input = src.data_ptr<half>();
    auto d_ptr_weight = attn_fc_weight.data_ptr<half>();
    auto d_ptr_output = output.data_ptr<half>();

    // Launch kernel
    vector_matrix_mul_kernel<1, 1280, 5120>
        <<<dim3(kGridSize, 1, 1), dim3(kBlockSize, 1, 1)>>>(d_ptr_input, d_ptr_weight, d_ptr_output);
    cudaDeviceSynchronize();

    // Check correctness
    auto torch_output = torch::matmul(src, attn_fc_weight);
    int compare_level = 1;
    my_compare(torch_output, output, 1.0/16, 1.0/1024, compare_level);
    return 0;
}