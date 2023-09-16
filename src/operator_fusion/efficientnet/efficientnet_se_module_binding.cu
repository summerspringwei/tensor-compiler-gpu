#include <iostream>

#include <cuda.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda/pipeline>

#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")

template <int64_t batch, int64_t height, int64_t width, int64_t in_channel,
          int64_t reduce_channel, int64_t tile_size_in_channel>
void torch_efficientnet_se_module_v2_short_cut_fused(torch::Tensor input, torch::Tensor se_reduce_weight, torch::Tensor se_expand_weight) {
  CHECK_CUDA(input);
    CHECK_CUDA(se_reduce_weight);
    CHECK_CUDA(se_expand_weight);
  // Allocate intermedia tensors
  auto reduce_output = torch::zeros({batch, in_channel}, options_fp32);
  auto se_reduce_output = torch::zeros({batch, reduce_channel}, options_fp32);
  auto se_reduce_sigmoid = torch::zeros({batch, reduce_channel}, options_fp32);
  auto se_reduce_mul = torch::zeros({batch, reduce_channel}, options_fp32);
  auto se_expand_output = torch::zeros({batch, in_channel}, options_fp32);
  auto se_expand_sigmoid = torch::zeros({batch, in_channel}, options_fp32);
  auto se_mul_output =
      torch::zeros({batch, height, width, in_channel}, options_fp32);
  auto se_short_cut_add =
      torch::zeros({batch, height, width, in_channel}, options_fp32);
  // stages, number of blocks, number of warps
  auto profile_clock =
      torch::zeros({6, in_channel / tile_size_in_channel, kBlockSize / 32}, options_int64);
  
  // Get tensors' data pointer
  float *ptr_input = (float *)input.data_ptr<float>();
  float *ptr_reduce_output = (float *)reduce_output.data_ptr<float>();
  float *ptr_se_reduce_weight = (float *)se_reduce_weight.data_ptr<float>();
  float *ptr_se_reduce_output = (float *)se_reduce_output.data_ptr<float>();
  float* ptr_se_reduce_sigmoid = se_reduce_sigmoid.data_ptr<float>();
  float* ptr_se_reduce_mul = se_reduce_mul.data_ptr<float>();
  float *ptr_se_expand_weight = (float *)se_expand_weight.data_ptr<float>();
  float *ptr_se_expand_output = (float *)se_expand_output.data_ptr<float>();
  float *ptr_se_mul_output = (float *)se_mul_output.data_ptr<float>();
  float *ptr_se_expand_sigmoid = (float *)se_expand_sigmoid.data_ptr<float>();
  float *ptr_se_short_cut_add = (float *)se_short_cut_add.data_ptr<float>();
  int64_t *ptr_profile_clock = (int64_t *)profile_clock.data_ptr<int64_t>();

  void* se_kernel_args[] = {
      (void *)&(ptr_input),
      (void *)&(ptr_reduce_output),
      (void *)&(ptr_se_reduce_weight),
      (void *)&(ptr_se_reduce_output),
      (void *)&(ptr_se_expand_weight),
      (void *)&(ptr_se_expand_output),
      (void *)&(ptr_se_short_cut_add),
      (void *)&(ptr_profile_clock)
    };
    auto func_ptr = (const void*)efficientnet_se_module_v2_short_cut_fused<batch, height, width, in_channel,
                                          reduce_channel, tile_size_in_channel>;  
    checkCuda(cudaLaunchCooperativeKernel((const void *)func, dim3(in_channel / tile_size_in_channel, 1, 1), dim3(kBlockSize, 1, 1),,
                             se_kernel_args, shared_memory_size));
    cudaDeviceSynchronize();
}

// TODO(Chunwei Xia) Add a dispatcher function to dispatch according to tensor shape

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("torch_efficientnet_se_module_v2_short_cut_fused", &torch_efficientnet_se_module_v2_short_cut_fused<1, 10, 256, 100>, 
    "");
}
