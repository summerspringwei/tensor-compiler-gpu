#include <cstdlib>
#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>

#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>


__global__ void test_atomic_add(int* arr_syn){
    atomicAdd(&arr_sync[blockIdx.x / 4 + threadIdx.x] + );
}