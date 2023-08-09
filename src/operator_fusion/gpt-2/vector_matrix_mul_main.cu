// Deal with batch_size < 4
#include "../torch_utils.h"
#include "cuda_fp16.h"
#include "kernels/vector_matrix_mul.h"
#include "torch/all.h"

int main(int argc, char* argv[]) {
  // Load weight
  std::string folder =
      "/home/xiachunwei/Projects/tensor-compiler-gpu/src/operator_fusion/"
      "gpt-2/";
  torch::Tensor attn_fc_weight =
      torch_load_tensor(folder + "gpt2-torch-data/attn_c_proj.pt")
          .to(torch::kCUDA)
          .to(torch::kHalf);
  // torch::Tensor attn_fc_weight = torch::ones({5120, 1280},
  // torch::kHalf).to(torch::kCUDA);
  const int out_dim = attn_fc_weight.sizes()[0];
  const int reduce_dim = attn_fc_weight.sizes()[1];
  const int batch_size = 1;
  auto src =
      torch::ones({batch_size, reduce_dim}, torch::kHalf).to(torch::kCUDA);
  // auto src = torch::nn::init::uniform_(
  //   torch::randn({batch_size, reduce_dim}, options_fp16), 0,
  //   1).to(torch::kCUDA);
  auto output =
      torch::empty({batch_size, out_dim}, options_fp16).to(torch::kCUDA);

  // Declare pointers
//   auto d_ptr_input = src.data_ptr<at::Half>();
//   auto d_ptr_weight = attn_fc_weight.data_ptr<at::Half>();
//   auto d_ptr_output = output.data_ptr<at::Half>();
  half* d_ptr_input;
  cudaMalloc((void**)&d_ptr_input, sizeof(half) * 1 * 1280);
  half* d_ptr_weight;
  cudaMalloc((void**)&d_ptr_weight, sizeof(half) * 5120 * 1280);
  half* d_ptr_output;
  cudaMalloc((void**)&d_ptr_output, sizeof(half) * 1 * 5120);

  // Launch kernel
//   vector_matrix_mul_kernel<1, 1280, 5120>
//       <<<dim3(kGridSize, 1, 1), dim3(kBlockSize, 1, 1)>>>(
//           (half*)(d_ptr_input), (half*)d_ptr_weight, (half*)d_ptr_output);
  // Launch kernel
  vector_matrix_mul_kernel
      <<<dim3(kGridSize, 1, 1), dim3(kBlockSize, 1, 1)>>>(
          (half*)(d_ptr_input), (half*)d_ptr_weight, (half*)d_ptr_output);
  // vector_matrix_mul_kernel_half2<1, 1280, 5120>
  //     <<<dim3(kGridSize, 1, 1), dim3(kBlockSize, 1,
  //     1)>>>((half*)(d_ptr_input), (half*)d_ptr_weight, (half*)d_ptr_output);
  cudaDeviceSynchronize();

  // Check correctness
  auto torch_output = torch::mm(
      src, torch::permute(attn_fc_weight, {1, 0}));  // (m, k) * (k, n) = (m, n)
  cudaDeviceSynchronize();
  torch::print(torch_output);
  torch::print(output);
  printf("%d\n", torch::allclose(output, torch_output, 1e-2, 1e-3));
  // int compare_level = 1;
  // my_compare(torch_output, output, 1.0/16, 1.0/1024, compare_level);
  return 0;
}
