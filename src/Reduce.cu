#include <iostream>
#include <list>
#include <vector>
#include <chrono>
#include <cstring>
#include <map>
#include <assert.h>

#include <cuda_runtime.h>


// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

const int block_size = 256;

__global__ void reduce_v1(float* a, float* b, const size_t n){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= n){
        return;
    }
    for(int s = 1; s<blockDim.x; s*=2){
        if(threadIdx.x % (2*s) == 0){
            // printf("<%d-> %d, %d>\n",s, tid, tid+s);
            a[tid] += a[tid+s];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        // printf("fuse: <%d, %d, %f>", blockIdx.x, threadIdx.x, a[blockIdx.x * blockDim.x]);
        b[blockIdx.x]=a[blockIdx.x * blockDim.x];
    }
}

__global__ void reduce_v2(float* a, float* b, const size_t n){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= n){
        return;
    }
    // extern __shared__ float s_data[];
    __shared__ float s_data[block_size];
    s_data[threadIdx.x] = a[tid];
    __syncthreads();

    for(int s = 1; s<blockDim.x; s*=2){
        if(threadIdx.x % (2*s) == 0){
            // printf("<%d-> %d, %d>\n",s, tid, tid+s);
            s_data[threadIdx.x] += s_data[threadIdx.x+s];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        // printf("fuse: <%d, %d, %f>", blockIdx.x, threadIdx.x, a[blockIdx.x * blockDim.x]);
        b[blockIdx.x]=s_data[0];
    }
}


__global__ void reduce_v3(float* a, float* b, const size_t n){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= n){
        return;
    }
    // extern __shared__ float s_data[];
    __shared__ float s_data[block_size];
    s_data[threadIdx.x] = a[tid];
    __syncthreads();

    for(int s = 1; s<blockDim.x; s*=2){
        int reduced_idx = 2 * s * threadIdx.x;
        if(reduced_idx < blockDim.x){
            // printf("<%d-> %d, %d>\n",s, tid, tid+s);
            s_data[reduced_idx] += s_data[reduced_idx+s];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        // printf("fuse: <%d, %d, %f>", blockIdx.x, threadIdx.x, a[blockIdx.x * blockDim.x]);
        b[blockIdx.x]=s_data[0];
    }
}

__global__ void reduce_v4(float* a, float* b, const size_t n){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= n){
        return;
    }
    // extern __shared__ float s_data[];
    __shared__ float s_data[block_size];
    s_data[threadIdx.x] = a[tid];
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s/=2){
        if(threadIdx.x < s){
            // printf("<%d-> %d, %d>\n",s, tid, tid+s);
            s_data[threadIdx.x] += s_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        // printf("fuse: <%d, %d, %f>", blockIdx.x, threadIdx.x, a[blockIdx.x * blockDim.x]);
        b[blockIdx.x]=s_data[0];
    }
}

// Halve number of blocks
__global__ void reduce_v5(float* a, float* b, const size_t n){
    unsigned int tid = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    if(tid >= n){
        return;
    }
    // extern __shared__ float s_data[];
    __shared__ float s_data[block_size];
    s_data[threadIdx.x] = a[tid] + a[tid + blockDim.x];
    __syncthreads();

    for(int s = blockDim.x / 2; s > 0; s/=2) {
        if(threadIdx.x < s){
            // printf("<%d-> %d, %d>\n",s, tid, tid+s);
            s_data[threadIdx.x] += s_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        // printf("fuse: <%d, %d, %f>", blockIdx.x, threadIdx.x, a[blockIdx.x * blockDim.x]);
        b[blockIdx.x]=s_data[0];
    }
}

__device__ void warpReduce(volatile float* sdata, int tid) {
    sdata[tid] += sdata[tid + 32];
    sdata[tid] += sdata[tid + 16];
    sdata[tid] += sdata[tid + 8];
    sdata[tid] += sdata[tid + 4];
    sdata[tid] += sdata[tid + 2];
    sdata[tid] += sdata[tid + 1];
}
// Unroll loops
__global__ void reduce_v6(float* a, float* b, const size_t n){
    unsigned int tid = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    if(tid >= n){
        return;
    }
    // extern __shared__ float s_data[];
    __shared__ float s_data[block_size];
    s_data[threadIdx.x] = a[tid] + a[tid + blockDim.x];
    __syncthreads();

    for(int s = blockDim.x / 2; s > 32; s/=2){
        if(threadIdx.x < s){
            // printf("<%d-> %d, %d>\n",s, tid, tid+s);
            s_data[threadIdx.x] += s_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    if(threadIdx.x < 32){
        warpReduce(s_data, threadIdx.x);
        // s_data[threadIdx.x] += s_data[threadIdx.x + 32];
        // __syncthreads();
        // s_data[threadIdx.x] += s_data[threadIdx.x + 16];
        // __syncthreads();
        // s_data[threadIdx.x] += s_data[threadIdx.x + 8];
        // __syncthreads();
        // s_data[threadIdx.x] += s_data[threadIdx.x + 4];
        // __syncthreads();
        // s_data[threadIdx.x] += s_data[threadIdx.x + 2];
        // __syncthreads();
        // s_data[threadIdx.x] += s_data[threadIdx.x + 1];
        // __syncthreads();
    }
    
    if(threadIdx.x == 0){
        // printf("fuse: <%d, %d, %f>", blockIdx.x, threadIdx.x, a[blockIdx.x * blockDim.x]);
        b[blockIdx.x]=s_data[0];
    }
}

// const int k = 4;

// Unroll loops
template <unsigned int K>
__global__ void reduce_v7(float* a, float* b, const size_t n){
    unsigned int tid = blockIdx.x * blockDim.x * K + threadIdx.x;
    
    __shared__ float s_data[block_size];
    s_data[threadIdx.x] = 0;
    #pragma unroll 2
    for(int i=0; i<K*blockDim.x; i+=2*blockDim.x){
        s_data[threadIdx.x] += (a[tid + i] + a[tid + i+blockDim.x]);
    }
    __syncthreads();
    
    for(int s = blockDim.x / 2; s > 32; s/=2){
        if(threadIdx.x < s){
            // printf("<%d-> %d, %d>\n",s, tid, tid+s);
            s_data[threadIdx.x] += s_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    if(threadIdx.x < 32){
        warpReduce(s_data, threadIdx.x);
        // s_data[threadIdx.x] += s_data[threadIdx.x + 32];
        // __syncthreads();
        // s_data[threadIdx.x] += s_data[threadIdx.x + 16];
        // __syncthreads();
        // s_data[threadIdx.x] += s_data[threadIdx.x + 8];
        // __syncthreads();
        // s_data[threadIdx.x] += s_data[threadIdx.x + 4];
        // __syncthreads();
        // s_data[threadIdx.x] += s_data[threadIdx.x + 2];
        // __syncthreads();
        // s_data[threadIdx.x] += s_data[threadIdx.x + 1];
        // __syncthreads();
    }
    
    if(threadIdx.x == 0){
        // printf("fuse: <%d, %d, %f>", blockIdx.x, threadIdx.x, a[blockIdx.x * blockDim.x]);
        b[blockIdx.x]=s_data[0];
    }
}

template <unsigned int K>
__global__ void reduce_v8(float* a, float* b, const size_t n){
    unsigned int tid = blockIdx.x * blockDim.x * K + threadIdx.x;
    
    __shared__ float s_data[block_size];
    s_data[threadIdx.x] = 0;
    #pragma unroll 2
    for(int i=0; i<K*blockDim.x; i+=2*blockDim.x){
        s_data[threadIdx.x] += (a[tid + i] + a[tid + i+blockDim.x]);
    }
    __syncthreads();
    
    for(int s = blockDim.x / 2; s > 32; s/=2){
        if(threadIdx.x < s){
            // printf("<%d-> %d, %d>\n",s, tid, tid+s);
            s_data[threadIdx.x] += s_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    // Reduce using registers
    if(threadIdx.x < 32){
        float val = s_data[threadIdx.x];
        #define FULL_MASK 0xffffffff
        for (int offset = 32; offset > 0; offset /= 2){
            val += __shfl_down_sync(FULL_MASK, val, offset);
        }
        if(threadIdx.x == 0){
            // printf("fuse: <%d, %d, %f>", blockIdx.x, threadIdx.x, a[blockIdx.x * blockDim.x]);
            b[blockIdx.x]=val;
        }
    }
}


template <unsigned int BLOCK_SIZE, unsigned int K>
__global__ void reduce_v9(float* a, float* b, const size_t n){
    unsigned int tid = blockIdx.x * blockDim.x * K + threadIdx.x;
    
    __shared__ float s_data[block_size];
    s_data[threadIdx.x] = 0;
    #pragma unroll 2
    for(int i=0; i<K*blockDim.x; i+=2*blockDim.x){
        s_data[threadIdx.x] += (a[tid + i] + a[tid + i+blockDim.x]);
    }
    __syncthreads();
    
    if(BLOCK_SIZE >= 256){
        if(threadIdx.x < 128){
            s_data[threadIdx.x] += s_data[threadIdx.x + 128];
            __syncthreads();
        }
    }
    if(BLOCK_SIZE >= 128){
        if(threadIdx.x < 64){
            s_data[threadIdx.x] += s_data[threadIdx.x + 64];
            __syncthreads();
        }
    }
    if(BLOCK_SIZE >= 64){
        if(threadIdx.x < 32){
            s_data[threadIdx.x] += s_data[threadIdx.x + 32];
            __syncthreads();
        }
    }
    // Reduce using registers
    if(threadIdx.x < 32){
        float val = s_data[threadIdx.x];
        #define FULL_MASK 0xffffffff
        for (int offset = 16; offset > 0; offset /= 2){
            val += __shfl_down_sync(FULL_MASK, val, offset);
        }
        if(threadIdx.x == 0){
            // printf("fuse: <%d, %d, %f>", blockIdx.x, threadIdx.x, a[blockIdx.x * blockDim.x]);
            b[blockIdx.x]=val;
        }
    }
}

// __global__ void foo(int a, int b, int *c){
//     int k = a+b;
//     d = c[threadIdx.x];
//     e = c[d];
// }

void reduce_cuda(float* a, float* b, const size_t n){
    // Error code to check return values for CUDA calls
    
    float* d_a = NULL;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&d_a, sizeof(float)*n);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float *d_b = NULL;
    err = cudaMalloc((void **)&d_b, sizeof(float)*n / block_size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_b (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_a, a, sizeof(float)*n, cudaMemcpyHostToDevice);
    cudaEvent_t startEvent, stopEvent;
    float ms = 0.0;
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );
    
    auto t1 = std::chrono::steady_clock::now();
    // Launch the arnold CUDA Kernel
    checkCuda( cudaEventRecord(startEvent,0) );
    dim3 threadsPerBlock(block_size);
    dim3 numBlocks(n / threadsPerBlock.x / 4);
    // reduce_v4<<<numBlocks, threadsPerBlock>>>(d_a, d_b, n);
    reduce_v9<256, 4><<<numBlocks, threadsPerBlock>>>(d_a, d_b, n);
    checkCuda( cudaEventRecord(stopEvent,0) );
    checkCuda( cudaEventSynchronize(stopEvent) );
    checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
    
    cudaDeviceSynchronize();

    auto t2 = std::chrono::steady_clock::now();
    double latency = std::chrono::duration<double, std::micro>(t2-t1).count();
    printf("[%f, %f] bandwidth %f\n", latency, ms,  n*sizeof(float) * 1e3 /1024/1024/ms);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    // printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(b, d_b, sizeof(float) * n / block_size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
}


int main(int argc, char** argv)
{
    if(argc < 2){
        printf("Usage: EncryptionTime img_size loop_count\n");
        return 0;
    }
    
    int n = atoi(argv[1]);
    int loop_count = atoi(argv[2]);
    assert((loop_count>0) && (n>0));
    float* a = (float*)malloc(sizeof(float) * n*n);
    float* b = (float*)malloc(sizeof(float) * n*n / block_size);
    for(int i=0; i<n*n;++i){
        a[i] = (float)1;
    }
    auto t1 = std::chrono::steady_clock::now();
    for(int i=0; i<loop_count; ++i){
        reduce_cuda(a, b, n*n);
    }
    printf("b:\n");
    auto num_out = n*n/block_size;
    for(int i=0; i<(num_out < 16 ? num_out: 16);++i){
        printf("%f ", b[i]);
    }printf("\n");
    // for(int i=num_out-1; i>=(num_out > 16 ? num_out-16: 0);--i){
    //     printf("%f ", b[i]);
    // }printf("\n");
    auto t2 = std::chrono::steady_clock::now();
    printf("%d %d %f\n", n, loop_count, std::chrono::duration<double, std::micro>(t2-t1).count() / loop_count);
   
    return 0;
}

