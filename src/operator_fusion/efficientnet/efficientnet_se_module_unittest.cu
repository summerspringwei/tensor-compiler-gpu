#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/script.h>  // One-stop header.

#include <iostream>
#include <memory>

#include "../../cuda_utils.h"
#include "../../utils.h"
#include "../torch_utils.h"
#include "se_module_v2.cu"
#include "torch/all.h"

/**
 * tensor name format
 *
 * _blocks.8._expand_conv.weight: (1,1,.,.) =
_blocks.8._bn0.weight:  1
_blocks.8._depthwise_conv.weight: (1,1,.,.) =
_blocks.8._bn1.weight:  1
_blocks.8._se_reduce.weight: (1,1,.,.) =
_blocks.8._se_expand.weight: (1,1,.,.) =
_blocks.8._project_conv.weight: (1,1,.,.) =
_blocks.8._bn2.weight:  1
*/
std::unordered_map<std::string, at::Tensor> get_model_tensors(
    const char* argv) {
  std::unordered_map<std::string, at::Tensor> name_tensor_map;
  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv);
    for (auto p : module.named_parameters(/*recurse=*/true)) {
      name_tensor_map[p.name] = p.value;
    }
  } catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
  }

  return name_tensor_map;
}

template <int64_t batch, int64_t height, int64_t width, int64_t in_channel,
          int64_t reduce_channel, int64_t tile_size_in_channel>
void efficient_se_module(
    const std::unordered_map<std::string, at::Tensor> name_tensor_map,
    const int block_id, size_t shared_memory_size = 48 * 1024) {
  auto input = torch::nn::init::uniform_(
      torch::randn({batch, in_channel, height, width}, options_fp32), 0, 1);
  // auto input = torch::ones({batch, in_channel, height, width}, options_fp32);

  // Load tensor value from map
  std::string prefix = "_blocks." + std::to_string(block_id);
  std::string se_reduce_weight_name = prefix + "._se_reduce.weight";
  std::string se_expand_weight_name = prefix + "._se_expand.weight";
  auto it = name_tensor_map.find(se_reduce_weight_name);
  at::Tensor se_reduce_weight = ((*it).second).to(torch::kCUDA);
  it = name_tensor_map.find(se_expand_weight_name);
  at::Tensor se_expand_weight = ((*it).second).to(torch::kCUDA);
  auto t_reduce_channel = se_reduce_weight.sizes()[0];
  auto t_in_channel = se_reduce_weight.sizes()[1];
  assert((t_reduce_channel == reduce_channel) && (t_in_channel == in_channel));

  auto reduce_output = torch::zeros({batch, in_channel}, options_fp32);
  auto se_reduce_output = torch::zeros({batch, reduce_channel}, options_fp32);
  auto se_expand_output = torch::zeros({batch, in_channel}, options_fp32);
  auto se_mul_output =
      torch::zeros({batch, height, width, in_channel}, options_fp32);
  auto profile_clock =
      torch::zeros({3, kBlockSize, kBlockSize / 32}, options_int64);

  float* ptr_input = (float*)input.data_ptr<float>();
  float* ptr_reduce_output = (float*)reduce_output.data_ptr<float>();
  float* ptr_se_reduce_weight = (float*)se_reduce_weight.data_ptr<float>();
  float* ptr_se_reduce_output = (float*)se_reduce_output.data_ptr<float>();
  float* ptr_se_expand_weight = (float*)se_expand_weight.data_ptr<float>();
  float* ptr_se_expand_output = (float*)se_expand_output.data_ptr<float>();
  float* ptr_se_mul_output = (float*)se_mul_output.data_ptr<float>();
  int64_t* ptr_profile_clock = (int64_t*)profile_clock.data_ptr<int64_t>();
  checkCuda(cudaFuncSetAttribute(
      (const void*)efficientnet_se_module_v2_avg_pool_v2<batch, height, width, in_channel,
                                               reduce_channel,
                                               tile_size_in_channel>,
      cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_memory_size));
  checkCuda(cudaFuncSetAttribute(
      (const void*)matmul_with_block_reduce_k<batch, reduce_channel, in_channel>,
      cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_memory_size));
  checkCuda(cudaFuncSetAttribute(
      (void*)efficientnet_se_module_v2_matmul2<batch, height, width, in_channel,
                                               reduce_channel,
                                               tile_size_in_channel>,
      cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_memory_size));

  // PyTorch implementation
  auto t_reduce_output = torch::avg_pool2d(input, height);
  auto t_se_reduce_output = torch::conv2d(t_reduce_output, se_reduce_weight);
  auto t_se_reduce_sigmoid = torch::sigmoid(t_se_reduce_output);
  auto t_se_reduce_mul = t_se_reduce_output * t_se_reduce_sigmoid;
  auto t_se_expand_output = torch::conv2d(t_se_reduce_mul, se_expand_weight);
  auto t_se_expand_sigmoid = torch::sigmoid(t_se_expand_output);
  // Test avg pool
  {
    void* se_kernel_args[] = {(void*)&(ptr_input),
                          (void*)&(ptr_reduce_output),
                          (void*)&(ptr_se_reduce_weight),
                          (void*)&(ptr_se_reduce_output),
                          (void*)&(ptr_se_expand_weight),
                          (void*)&(ptr_se_expand_output),
                          (void*)&(ptr_profile_clock)};
    checkCuda(cudaLaunchKernel(
        (const void*)efficientnet_se_module_v2_avg_pool_v2<batch, height, width, in_channel,
                                               reduce_channel,
                                               tile_size_in_channel>,
        dim3(in_channel / tile_size_in_channel, 1, 1), dim3(kBlockSize, 1, 1),
        se_kernel_args, shared_memory_size));
    auto test_t_reduce_output =
        t_reduce_output.to(torch::kCPU).reshape(t_reduce_output.numel());
    auto test_reduce_output =
        reduce_output.to(torch::kCPU).reshape(reduce_output.numel());
    // torch::print(t_reduce_output);
    // torch::print(reduce_output);
    assert(torch::allclose(test_t_reduce_output, test_reduce_output,
                          1.0 / 16, 1.0 / 16));
  }
  // Test matmul1
  {
    float* ptr_t_reduce_output = (float*)t_reduce_output.data_ptr<float>();
    void* se_kernel_args[] = {(void*)&(ptr_t_reduce_output),
                              (void*)&(ptr_se_reduce_weight),
                              (void*)&(ptr_se_reduce_output)};
    checkCuda(cudaLaunchKernel(
        (const void*)matmul_with_block_reduce_k<batch, reduce_channel, in_channel>,
        dim3(batch*reduce_channel, 1, 1), dim3(kBlockSize, 1, 1),
        se_kernel_args, shared_memory_size));
    auto test_t_se_reduce_output =
        t_se_reduce_output.to(torch::kCPU).reshape(t_se_reduce_output.numel());
    auto test_se_reduce_output =
        se_reduce_output.to(torch::kCPU).reshape(se_reduce_output.numel());
    // my_compare(test_t_se_reduce_output, test_se_reduce_output, 1.0 / 64, 1.0 / 1024, 2);
    assert(torch::allclose(test_t_se_reduce_output, test_se_reduce_output,
                          1.0 / 16, 1.0 / 16));
  }
  // Test matmul2
  {
    float* ptr_t_se_reduce_mul = (float*)t_se_reduce_mul.data_ptr<float>();
    void* se_kernel_args[] = {(void*)&(ptr_input),
                              (void*)&(ptr_reduce_output),
                              (void*)&(ptr_se_reduce_weight),
                              (void*)&(ptr_t_se_reduce_mul),
                              (void*)&(ptr_se_expand_weight),
                              (void*)&(ptr_se_expand_output),
                              (void*)&(ptr_profile_clock)};

    checkCuda(cudaLaunchKernel(
        (const void*)efficientnet_se_module_v2_matmul2<batch, height, width,
                                                      in_channel, reduce_channel,
                                                      tile_size_in_channel>,
        dim3(in_channel / tile_size_in_channel, 1, 1), dim3(kBlockSize, 1, 1),
        se_kernel_args, shared_memory_size));

    auto test_t_se_expand_output =
        t_se_expand_output.to(torch::kCPU).reshape(t_se_expand_output.numel());
    auto test_se_expand_output =
        se_expand_output.to(torch::kCPU).reshape(se_expand_output.numel());
    // my_compare(t_se_expand_output.reshape({in_channel,}),
    // se_expand_output.reshape({in_channel,}), 1.0 / 64, 1.0 / 1024, 2);
    // torch::print(test_t_se_expand_output);
    // torch::print(test_se_expand_output);
    assert(torch::allclose(test_t_se_expand_output, test_se_expand_output,
                          1.0 / 16, 1.0 / 16));
  }
}


void test_cu_func(const void* func_ptr, void* kernel_args[], dim3 grid_dim, dim3 block_dim, uint32_t shared_memory_size){
    checkCuda(cudaFuncSetAttribute(
      func_ptr,
      cudaFuncAttribute::cudaFuncAttributeMaxDynamicSharedMemorySize,
      shared_memory_size));
    checkCuda(cudaLaunchKernel(
      func_ptr,
      grid_dim, block_dim,
      kernel_args, shared_memory_size));
}


int main(int argc, char** argv) {
  auto name_tensor_map = get_model_tensors(argv[1]);
  efficient_se_module<1, 112, 112, 32, 8, 1>(name_tensor_map, 0, 96*1024);
  efficient_se_module<1, 56, 56, 96, 4, 1>(name_tensor_map, 1, 96*1024);
  efficient_se_module<1, 56, 56, 144, 6, 2>(name_tensor_map, 2, 96 * 1024);
  efficient_se_module<1, 28, 28, 144, 6, 1>(name_tensor_map, 3, 48 * 1024);
  efficient_se_module<1, 28, 28, 240, 10, 2>(name_tensor_map, 4, 48 * 1024);
  efficient_se_module<1, 14, 14, 240, 10, 1>(name_tensor_map, 5, 32 * 1024);
  efficient_se_module<1, 14, 14, 480, 20, 1>(name_tensor_map, 7, 24 * 1024);
  efficient_se_module<1, 14, 14, 672, 28, 3>(name_tensor_map, 9, 16 * 1024);
  efficient_se_module<1, 7, 7, 1152, 48, 4>(name_tensor_map, 12, 16 * 1024);
  return 0;
}
