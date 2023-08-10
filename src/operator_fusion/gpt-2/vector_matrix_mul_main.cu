// Deal with batch_size < 4
#include "../torch_utils.h"
#include "cuda_fp16.h"
#include "torch/all.h"
#include "../../cuda_kernel_utils.h"
#define kBlockSize 256
#define kGridSize 84  // The number of SM on RTX3090 is 84
// #include "kernels/vector_matrix_mul.cu"
__global__ void __launch_bounds__(kBlockSize)
    vector_matrix_mul_kernel(half *__restrict__ input,
                             half *__restrict__ weight,
                             half *__restrict__ output) {
  const int warpIdx = threadIdx.x / 32;
  const int laneIdx = threadIdx.x % 32;
  const int numWarp = kBlockSize / 32;
  const int vectorLength = sizeof(float4) / sizeof(half);
  half local_input[8];
  half local_weight[8];
  const int64_t batch_size = 1;
  const int64_t reduce_dim = 1280;
  const int64_t out_dim = 5120;
  // Iterate over batch_size
  for (int64_t b = 0; b < batch_size; ++b) {
    // Iterate over out_dim
    for (int64_t idx = 0; UPDIV(out_dim, kGridSize * numWarp); ++idx) {
      // Each warp reduce one reduce_dim
      float local_sum = 0;
      const int64_t weight_row_idx =
          (idx * kGridSize * numWarp + blockIdx.x * numWarp + warpIdx);
      // Guard against over indexing
      if (weight_row_idx >= out_dim) break;
#pragma unroll
      for (int64_t k = 0; k < reduce_dim; k += (warpSize * vectorLength)) {
        const int64_t col_idx = k + laneIdx * vectorLength;
        // Guard against over indexing
        if (col_idx >= reduce_dim) break;
        *((float4 *)local_input) =
            *((float4 *)&(input[(b * reduce_dim + col_idx)]));
        *((float4 *)local_weight) =
            *((float4 *)&(weight[(weight_row_idx * reduce_dim + col_idx)]));
        // if(blockIdx.x == 0){
        //   printf("weight_row_idx: %ld, col_idx: %ld, k: %ld, local_input: %f, %f, %f, %f, local_weight: %f, %f, %f, %f\n",
        //          weight_row_idx, col_idx, k,
        //          __half2float(local_input[0]), __half2float(local_input[1]),
        //          __half2float(local_input[2]), __half2float(local_input[3]),
        //          __half2float(local_weight[0]), __half2float(local_weight[1]),
        //          __half2float(local_weight[2]), __half2float(local_weight[3]));
        // }
        float2 tmp;
        tmp = __half22float2(__hmul2(half2(local_input[0], local_input[1]), half2(local_weight[0], local_weight[1])));
        local_sum += (tmp.x + tmp.y);
        tmp = __half22float2(__hmul2(half2(local_input[0], local_input[1]), half2(local_weight[0], local_weight[1])));
        local_sum += (tmp.x + tmp.y);
        tmp = __half22float2(__hmul2(half2(local_input[2], local_input[3]), half2(local_weight[2], local_weight[3])));
        local_sum += (tmp.x + tmp.y);
        tmp = __half22float2(__hmul2(half2(local_input[4], local_input[5]), half2(local_weight[4], local_weight[5])));
        local_sum += (tmp.x + tmp.y);
        tmp = __half22float2(__hmul2(half2(local_input[6], local_input[7]), half2(local_weight[6], local_weight[7])));
        local_sum += (tmp.x + tmp.y);
      }
      // Reduce within warp
      local_sum = warpReduceSum(local_sum);
      // Write to output
      if (laneIdx == 0) {
        output[b * out_dim + weight_row_idx] = __float2half(local_sum);
      }
    }
  }
}

int main(int argc, char* argv[]) {
  // Load weight
  std::string folder_path =
      "/home/xiachunwei/Projects/tensor-compiler-gpu/src/operator_fusion/gpt-2/";
  // Shape (1280, 5120)
  torch::Tensor attn_fc_weight = torch_load_tensor(folder_path + "gpt2-torch-data/MLP_c_fc.pt")
            .to(torch::kCUDA)
            .to(torch::kHalf);
  // torch::Tensor attn_fc_weight =
  //     torch_load_tensor(folder + "gpt2-torch-data/attn_c_proj.pt")
  //         .to(torch::kCUDA)
  //         .to(torch::kHalf);
  // torch::Tensor attn_fc_weight = torch::ones({5120, 1280},
  // torch::kHalf).to(torch::kCUDA);
  const int reduce_dim = attn_fc_weight.sizes()[0];
  const int out_dim = attn_fc_weight.sizes()[1];
  const int batch_size = 1;
  auto src =
      torch::ones({batch_size, reduce_dim}, torch::kHalf).to(torch::kCUDA);
  // auto src = torch::nn::init::uniform_(
  //   torch::randn({batch_size, reduce_dim}, options_fp16), 0,
  //   1).to(torch::kCUDA);
  auto output =
      torch::empty({batch_size, out_dim}, options_fp16).to(torch::kCUDA);
  auto permuted_attn_fc_weight = torch::permute(attn_fc_weight, {1, 0});
  // Declare pointers
  auto d_ptr_input = src.data_ptr<at::Half>();
  // Note, need to permute to make the reduction dimension contiguous
  auto d_ptr_weight = permuted_attn_fc_weight.data_ptr<at::Half>();
  auto d_ptr_output = output.data_ptr<at::Half>();

  //  half* h_weight = new half[5120 * 1280];
//  for(int i=0; i< 5120; i++){
//   for(int j=0;j<1280;j++){
//     h_weight[i*1280+j] = j + (i%32) * 0.01;
//   }
//  }
//  cudaMemcpy(d_ptr_weight, h_weight, 5120*1280*sizeof(half), cudaMemcpyHostToDevice);

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
      src, attn_fc_weight);  // (m, k) * (k, n) = (m, n)
  cudaDeviceSynchronize();
  torch::print(src);
  torch::print(permuted_attn_fc_weight);
  torch::print(torch_output);
  torch::print(output);
  printf("%d\n", torch::allclose(output, torch_output, 1e-2, 1e-3));
  // int compare_level = 1;
  // my_compare(torch_output, output, 1.0/16, 1.0/1024, compare_level);
  // delete h_weight;
  return 0;
}
