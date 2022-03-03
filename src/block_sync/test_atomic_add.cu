#include <cstdlib>
#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>

#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "../cuda_utils.h"

const int kReduceFactor = 1;

// __global__ void test_atomic_add(unsigned int* b_sync){
//     volatile unsigned int* arr_sync=b_sync;
//     if(threadIdx.x==0){
//         printf("bync blockIdx.x %d, %u\n",blockIdx.x, b_sync[0]);
//         atomicInc(&(b_sync[0]), 10900);
//     }
//     while((*(arr_sync))<=107){
//         if(threadIdx.x==0 && (blockIdx.x < 8)){
//             printf("blockIdx.x %d, %u\n",blockIdx.x, *(arr_sync));
//         }
//     }
//     if(blockIdx.x % kReduceFactor==0 && threadIdx.x==0){
//         *(arr_sync + (blockIdx.x / kReduceFactor)) += 1;
//     }
// }


// __global__ void test_atomic_add(unsigned int* b_sync){
//     volatile unsigned int* arr_sync = b_sync;
//         atomicAdd(b_sync + (blockIdx.x / kReduceFactor * blockDim.x + threadIdx.x), 1);
//         // atomicInc(b_sync + (blockIdx.x / kReduceFactor * blockDim.x + threadIdx.x), 10000);
//     while(arr_sync[(blockIdx.x / kReduceFactor * blockDim.x + threadIdx.x)]!=kReduceFactor){
//     }
// }

__device__ unsigned int c_sync[2] = {0, 0};
__global__ void test_atomic_add(unsigned int* b_sync){
    volatile unsigned int* arr_sync = &(c_sync[0]);
    if(threadIdx.x % 32 ==0){
        atomicAdd(c_sync, 1);
    }
    // atomicInc(b_sync + (blockIdx.x / kReduceFactor * blockDim.x + threadIdx.x), 10000);
    while(arr_sync[0]<gridDim.x*(threadIdx.x / 32)){
    }
    if(threadIdx.x % 32 ==0){
        printf("<%d %d> %u\n", blockIdx.x, threadIdx.x, arr_sync[0]);
    }
}

// __global__ void test_atomic_add(unsigned int* volatile arr_sync){
//     atomicAdd(arr_sync + (blockIdx.x / kReduceFactor * blockDim.x + threadIdx.x), 1);
//     while(arr_sync[(blockIdx.x / kReduceFactor * blockDim.x + threadIdx.x)]!=kReduceFactor){
//     }
//     // if(blockIdx.x % 4==0){
//     //     arr_sync[(blockIdx.x / kReduceFactor * blockDim.x + threadIdx.x)] += 1;
//     // }
// }


#define CUDA_CHECK_RESULT if (result != cudaSuccess) \
    { \
        const char* msg = cudaGetErrorString(result); \
        std::stringstream safe_call_ss; \
        safe_call_ss << "\nerror: " << " failed with error" \
                    << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg; \
        throw std::runtime_error(safe_call_ss.str()); \
    };

void atomic_add_array(){
    const int reduced_num = 108;
    const int num_blocks = kReduceFactor * reduced_num;
    const int threads_pre_block = 32;
    std::vector<unsigned int> input(reduced_num * threads_pre_block);
    for(int i=0;i<input.size();++i){
        input[i]=0;
    }
     unsigned int* d_input = nullptr;
    cudaMalloc((void**)&d_input, sizeof(unsigned int) * input.size());
    checkCuda(cudaMemcpy(d_input, input.data(), sizeof( unsigned int) * input.size(), cudaMemcpyHostToDevice));
    
    test_atomic_add<<<dim3(num_blocks, 1, 1), dim3(threads_pre_block, 1, 1)>>>(d_input);
    cudaDeviceSynchronize();
    auto result = cudaGetLastError();
    CUDA_CHECK_RESULT
    unsigned int output[reduced_num];
    // checkCuda(cudaMemcpy(output, d_input, sizeof(unsigned int) * reduced_num, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(output, c_sync, sizeof(unsigned int) * 2, cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    // for(auto d: input){
    //     printf("%d ", d);
    // }printf("\n");
    for(int i=0;i<2; ++i){
        printf("%u ", output[i]);
    }printf("\n");
}


int main(){
    atomic_add_array();
}
