#include <iostream>
#include <list>
#include <vector>
#include <chrono>
#include <cstring>
#include <map>
#include <assert.h>

#include <cuda_runtime.h>

#include "../utils.h"

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


template <unsigned int BLOCK_SIZE>
__device__ float reduce_on_shared_mem(float* s_data){
    if(BLOCK_SIZE >= 512){
        if(threadIdx.x < 256){
            s_data[threadIdx.x] += s_data[threadIdx.x + 128];
            __syncthreads();
        }
    }
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
        return val;
    }
    return 0;
}


template <unsigned int BLOCK_SIZE, unsigned int K>
__global__ void reduce_sum_to_block(float* dst, float* src, const uint64_t batch, const uint64_t c, const uint64_t h, const uint64_t w){
    unsigned int tid = blockIdx.x * blockDim.x * K + threadIdx.x;
    
    __shared__ float s_data[block_size];

    for(uint64_t b_idx=0; b_idx<batch; ++b_idx){
        uint64_t image_offset = b_idx * c * h * w;
        for(uint64_t c_idx=0; c_idx<c; ++c_idx){
            uint64_t plane_offset = c_idx * h * w;
            uint64_t total_offset = image_offset + plane_offset;
            s_data[threadIdx.x] = 0;
            #pragma unroll 2
            for(int i=0; i<K*blockDim.x; i+=2*blockDim.x){
                s_data[threadIdx.x] += (src[total_offset + tid + i] + src[total_offset + tid + i + blockDim.x]);
            }
            __syncthreads();
            
            float val = reduce_on_shared_mem<BLOCK_SIZE>(s_data);
            
            if(threadIdx.x == 0){
                int num_reduced_to_elements = h * w / BLOCK_SIZE / K;
                dst[(b_idx * c + c_idx) * num_reduced_to_elements + blockIdx.x] = val;
            }
        }
    }
}


// Only optimize for single image inference
template <unsigned int BLOCK_SIZE>
__global__ void reduce_sum_to_thread(float* dst, float* src, const uint64_t batch, const uint64_t c, int num_to_reduce){
    __shared__ float sums[BLOCK_SIZE];
    if(blockIdx.x == 0){
        for(uint64_t b_idx=0; b_idx<batch; ++b_idx){
            uint64_t image_offset = b_idx * c * num_to_reduce;
            for(uint64_t c_idx=0; c_idx<c; ++c_idx){
                uint64_t plane_offset = c_idx * num_to_reduce;
                uint64_t total_offset = image_offset + plane_offset;
                // Sum up all elements to one thread block
                sums[threadIdx.x] = src[total_offset + threadIdx.x];
                int k = num_to_reduce / BLOCK_SIZE;
                for(int i=1; i<k; ++i){
                    sums[threadIdx.x] += src[total_offset + i*blockDim.x + threadIdx.x];
                }
                __syncthreads();
                // Sum up all elements in one blocks' shared memory to one float value
                float val = reduce_on_shared_mem<BLOCK_SIZE>(sums);
                // Sum up the left values
                if(threadIdx.x == 0){
                    for(int j=0; j<(num_to_reduce % blockDim.x); ++j){
                        val += src[num_to_reduce-1-j];
                    }
                    dst[b_idx * c + c_idx] = val;
                }
            }
        }
    }
}


void reduce_cuda(float* a, float* b, std::vector<uint64_t> shape){
    // Error code to check return values for CUDA calls
    uint64_t batch = shape[0], channel = shape[1], height = shape[2], width = shape[3];
    uint64_t n = get_shape_size(shape);
    const int K = 4;
    float* d_a = NULL;
    cudaError_t err = cudaSuccess;
    err = cudaMalloc((void **)&d_a, sizeof(float)*n);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float *d_b = NULL;
    int num_reduced_to_elements = n / block_size / K;
    err = cudaMalloc((void **)&d_b, sizeof(float) * num_reduced_to_elements);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_b (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    float *d_c = NULL;
    err = cudaMalloc((void **)&d_c, sizeof(float) * batch * channel);
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
    dim3 numBlocks(n / block_size / K);
    // reduce_v4<<<numBlocks, threadsPerBlock>>>(d_a, d_b, n);
    reduce_sum_to_block<block_size, K><<<numBlocks, threadsPerBlock>>>(d_b, d_a, batch, channel, height, width);
    // reduce_sum_to_thread<block_size><<<numBlocks, threadsPerBlock>>>(d_c, d_b, batch, channel, num_reduced_to_elements);
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
    err = cudaMemcpy(b, d_b, sizeof(float) * num_reduced_to_elements, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
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
    uint64_t batch = 1, channel = 3, height = n, width = n;
    std::vector<uint64_t> shape = {batch, channel, height, width};
    float* a = (float*)malloc(sizeof(float) * get_shape_size(shape));
    float* b = (float*)malloc(sizeof(float) * n*n / block_size);
    for(int i=0; i<n*n;++i){
        a[i] = (float)1;
    }
    auto t1 = std::chrono::steady_clock::now();
    for(int i=0; i<loop_count; ++i){
        reduce_cuda(a, b, shape);
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

