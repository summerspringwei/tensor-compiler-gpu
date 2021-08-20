#pragma once

#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

cudaError_t checkCuda(cudaError_t result, char* msg);

cudaError_t checkCuda(cudaError_t result);

void print_cuda(float* input, long n);
