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

template<typename T, typename T2>
__global__ void normalize_v1(T2* dst, T* src, T2* mean, T2* std, const uint64_t b, const uint64_t c, const uint64_t h, const uint64_t w){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(uint64_t i=0; i<b; ++i) {
	for(uint64_t j=0; j<c; ++j){
            if(idx > h*w) {
                 return;
	    }
	    uint64_t img_idx = i*c*h*w + j*h*w + idx;
	    dst[img_idx] = (src[img_idx] - mean[j]) / std[j];
	}
    }
}

// We expand the channel dim for small images
template<typename T, typename T2>
__global__ void normalize_v2(T2* dst, T* src, T2* mean, T2* std, const uint64_t b, const uint64_t c, const uint64_t h, const uint64_t w){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(uint64_t i=0; i<b; ++i) {
	float channel_mean=0, channel_std=0;
	/*
	for(uint64_t j = 0; j<c;++j){
	    if(idx < j*h*w){
                channel_mean = mean[j];
		channel_std = std[j];
            }
	}
	*/
	
        if(idx <= h*w){
            channel_mean = mean[0];
	    channel_std = std[0];
	}else if(idx <= 2*h*w){
            channel_mean = mean[1];
	    channel_std = std[2];
	}else if(idx <= 3*h*w){
            channel_mean = mean[2];
	    channel_std = std[2];
	}else{
	    return;
	}
	
	uint64_t img_idx = i*c*h*w + idx;
	dst[img_idx] = (src[img_idx] - channel_mean) / channel_std;
    }
}


// For each thread compute K elements
template<typename T, typename T2>
__global__ void normalize_v3(T2* dst, T* src, T2* mean, T2* std, const uint64_t b, const uint64_t c, const uint64_t h, const uint64_t w){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for(uint64_t i=0; i<b; ++i) {
	float channel_mean=0, channel_std=0;

    if(idx <= h*w){
            channel_mean = mean[0];
	    channel_std = std[0];
	}else if(idx <= 2*h*w){
            channel_mean = mean[1];
	    channel_std = std[2];
	}else if(idx <= 3*h*w){
            channel_mean = mean[2];
	    channel_std = std[2];
	}else{
	    return;
	}
	
	uint64_t img_idx = i*c*h*w + idx;
    uchar4 src4 = reinterpret_cast<uchar4*>(src)[img_idx];
    float4 dst4;
    dst4.x = (src4.x - channel_mean) / channel_std;
    dst4.y = (src4.x - channel_mean) / channel_std;
    dst4.z = (src4.z - channel_mean) / channel_std;
    dst4.w = (src4.w - channel_mean) / channel_std;
    reinterpret_cast<float4*>(dst)[img_idx] = dst4;
    }
}


uint64_t get_shape_size(std::vector<uint64_t> shape){
    uint64_t shape_size = 1;
    for(auto s: shape){
        shape_size *= s;
    }
    return shape_size;
}

int block_size = 128;

template<typename T, typename T2>
void normalize_cuda(T2* dst, T* src, T2* mean, T2* std, std::vector<uint64_t> shape, int loop_count){
    // In NCHW format
    // Error code to check return values for CUDA calls
    
    uint64_t n = shape[0], c = shape[1], h = shape[2], w = shape[3];
    T2* d_dst = NULL;
    cudaError_t err = cudaSuccess;
    const uint64_t elements_num = get_shape_size(shape);
    err = cudaMalloc((void **)&d_dst, elements_num * sizeof(T2));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    T *d_src = NULL;
    err = cudaMalloc((void **)&d_src, elements_num * sizeof(T));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector d_b (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_src, src, elements_num * sizeof(T), cudaMemcpyHostToDevice);
    T2* d_mean = NULL;
    err = cudaMalloc((void **)&d_mean, c*sizeof(T2));
    err = cudaMemcpy(d_mean, mean, c * sizeof(T2), cudaMemcpyHostToDevice);
    T2* d_std = NULL;
    err = cudaMalloc((void **)&d_std, c*sizeof(T2));
    err = cudaMemcpy(d_std, std, c*sizeof(T2), cudaMemcpyHostToDevice);
    cudaEvent_t startEvent, stopEvent;
    float total_time = 0.0;
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );

    dim3 threadsPerBlock(block_size);
    dim3 numBlocks(elements_num / threadsPerBlock.x / 4);
    // Warm up
    normalize_v1<T, T2><<<numBlocks, threadsPerBlock>>>(d_dst, d_src, d_mean, d_std, n, c, h, w);
    auto t1 = std::chrono::steady_clock::now();
    for(int i=0; i<loop_count; ++i){
        float ms = 0.0;
        checkCuda( cudaEventRecord(startEvent,0) );
        normalize_v1<T, T2><<<numBlocks, threadsPerBlock>>>(d_dst, d_src, d_mean, d_std, n, c, h, w);
        checkCuda( cudaEventRecord(stopEvent,0) );
        checkCuda( cudaEventSynchronize(stopEvent) );
        checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
        total_time += ms;
    }
    cudaDeviceSynchronize();

    auto t2 = std::chrono::steady_clock::now();
    double latency = std::chrono::duration<double, std::micro>(t2-t1).count();
    printf("[%f, %f] bandwidth %f\n", latency / loop_count, total_time / loop_count,  elements_num*(sizeof(float) + sizeof(uint8_t)) * 1e3 /1024/1024/ (total_time/loop_count));
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    // printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(dst, d_dst, elements_num * sizeof(T), cudaMemcpyDeviceToHost);

    cudaFree(d_src);
    cudaFree(d_dst);
}


int main(int argc, char** argv)
{
    if(argc < 2){
        printf("Usage: normalize img_size loop_count\n");
        return 0;
    }
    
    uint64_t n = atoi(argv[1]);
    int loop_count = atoi(argv[2]);
    assert((loop_count>0) && (n>0));
    const uint32_t img_size = 3*n*n;
    uint8_t* src = (uint8_t*)malloc(sizeof(uint8_t) * img_size);
    float* dst = (float*)malloc(sizeof(float) * img_size);
    float* mean = (float*)malloc(sizeof(float) * 3);
    float* std = (float*)malloc(sizeof(float) * 3);

    for(uint32_t i=0; i<img_size; ++i){
        src[i] = 0;
    }
    for(int i=0; i<3; ++i){
        mean[i] = (float) 127;
	std[i] = (float) 127;
    }

    std::vector<uint64_t> shape = {1, 3, n, n};
    auto t1 = std::chrono::steady_clock::now();
    normalize_cuda<uint8_t, float>(dst, src, mean, std, shape, loop_count);

    printf("dst:\n");
    auto num_out = n*n*3;
    for(int i=0; i<(num_out < 16 ? num_out: 16);++i){
        printf("%f ", dst[i]);
    }printf("\n");
    auto t2 = std::chrono::steady_clock::now();
    printf("%ld %d %f\n", n, loop_count, std::chrono::duration<double, std::micro>(t2-t1).count() / loop_count);
    
    free(src);
    free(dst);
    free(mean);
    free(std);
    return 0;
}

