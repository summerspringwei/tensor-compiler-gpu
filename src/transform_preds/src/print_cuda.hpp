#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <torch/extension.h>
#pragma once

cudaError_t checkCuda(cudaError_t result, char* msg);

cudaError_t checkCuda(cudaError_t result);

void print_cuda(float* input, long n);

void print_cuda(torch::PackedTensorAccessor64<float, 1> input, long n);