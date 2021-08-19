#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>
#include <torch/extension.h>
#pragma once

inline cudaError_t checkCuda(cudaError_t result, char* msg);

inline cudaError_t checkCuda(cudaError_t result);

void print_cuda(float* input, long n);

void print_cuda(torch::PackedTensorAccessor64<float, 1> input, long n);