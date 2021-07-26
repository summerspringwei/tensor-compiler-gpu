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
__global__ void sobel1x3_row_v1(T2* dst, T* src, T2* kernel, const uint64_t b, const uint64_t c, const uint64_t h, const uint64_t w){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx > h * w){
        return;
    }
    int col = idx % w;
    int row = idx / w;
    const uint64_t img_size = c * h * w;
    const uint64_t plane_size = h * w;
    
    for(uint64_t i=0; i<b; ++i) {
        for(uint64_t j=0; j<c; ++j){
            int total_offset = i * img_size + j * plane_size + row * w + col;
            // For mirror pad
            if(col==0){
                dst[total_offset + idx] = src[total_offset + 1] + (src[total_offset] * 2) + src[total_offset + 1];
            }else if(col==w-1){
                dst[total_offset + idx] = src[total_offset - 1] + (src[total_offset] * 2) + src[total_offset - 1];
            }else{
                dst[total_offset + idx] = src[total_offset - 1] + (src[total_offset] * 2) + src[total_offset + 1];
            }
        }
    }
}

// 先做内部的，再做边缘的
// 每个block做image的一行
template<typename T, typename T2>
__global__ void sobel1x3_row_v2(T2* dst, T* src, T2* kernel, const uint64_t b, const uint64_t c, const uint64_t h, const uint64_t w){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row >= h || col >= w){
        return;
    }
    
    const uint64_t img_size = c * h * w;
    const uint64_t plane_size = h * w;
    __shared__ T buf[4096];//
    for(uint64_t i=0; i<b; ++i) {
        for(uint64_t j=0; j<c; ++j){
            uint64_t total_offset = i * img_size + j * plane_size + row * w + col;
            buf[col] = src[total_offset];
            
            __syncthreads();
            if(col == 0){
                dst[total_offset] = src[total_offset + 1] + (src[total_offset] *2 ) + src[total_offset + 1];
            }else if(col == w-1){
                dst[total_offset] = src[total_offset - 1] + (src[total_offset] * 2) + src[total_offset - 1];
            }else{
                dst[total_offset] = buf[col - 1] + (buf[col] * 2) + buf[col + 1];
            }
        }
    }
}


// 先做内部的，再做边缘的
// 每个block做image的tile_size行
template<typename T, typename T2>
__global__ void sobel1x3_row_v4(T2* dst, T* src, T2* kernel, const uint64_t b, const uint64_t c, const uint64_t h, const uint64_t w){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row >= h || col >= w){
        return;
    }
    const int tile_size = 16;
    const uint64_t img_size = c * h * w;
    const uint64_t plane_size = h * w;
    __shared__ T buf[4096];//
    for(uint64_t i=0; i<b; ++i) {
        for(uint64_t j=0; j<c; ++j){
            for(int k=0; k<tile_size; ++k){
                uint64_t total_offset = i * img_size + j * plane_size + (row * tile_size + k) * w + col;
                buf[col] = src[total_offset];
                __syncthreads();
                if(col == 0){
                    dst[total_offset] = src[total_offset + 1] + (src[total_offset] * 2) + src[total_offset + 1];
                }else if(col == w-1){
                    dst[total_offset] = src[total_offset - 1] + (src[total_offset] * 2) + src[total_offset - 1];
                }else{
                    dst[total_offset] = buf[col - 1] + (buf[col] * 2) + buf[col + 1];
                }
            }
        }
    }
}


// Reduce the blockDim to reduce the number of blocks
// Performance is not good
template<typename T, typename T2>
__global__ void sobel1x3_row_v3(T2* dst, T* src, T2* kernel, const uint64_t b, const uint64_t c, const uint64_t h, const uint64_t w){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if(row >= h || col >= w / 4){
        return;
    }
    
    const uint64_t img_size = c * h * w;
    const uint64_t plane_size = h * w;
    const int tile_size = 4;
    // const int stride = w / tile_size;

    __shared__ T buf[4096];//
    for(uint64_t i=0; i<b; ++i) {
        for(uint64_t j=0; j<c; ++j){
            int total_offset = i * img_size + j * plane_size + row * w + col * tile_size;
            buf[col * tile_size] = src[total_offset];
            buf[col * tile_size + 1] = src[total_offset + 1];
            buf[col * tile_size + 2] = src[total_offset + 2];
            buf[col * tile_size + 3] = src[total_offset + 3];
            __syncthreads();
            // if(row==0){
            //     printf("%d\n", total_offset);
            // }
            
            if(col == 0){
                dst[total_offset] = src[total_offset + 1] + (src[total_offset] * 2) + src[total_offset + 1];
                dst[total_offset + 1] = src[total_offset] + (src[total_offset + 1] * 2) + src[total_offset + 2];
                dst[total_offset + 2] = src[total_offset + 1] + (src[total_offset + 2] * 2) + src[total_offset + 3];
                dst[total_offset + 3] = src[total_offset + 2] + (src[total_offset + 3] * 2) + src[total_offset + 4];
            }else if(col*tile_size == w - 4){
                dst[total_offset] = src[total_offset - 1] + (src[total_offset] * 2) + src[total_offset + 1];
                dst[total_offset + 1] = src[total_offset] + (src[total_offset + 1] * 2) + src[total_offset + 2];
                dst[total_offset + 2] = src[total_offset + 1] + (src[total_offset + 2] * 2) + src[total_offset + 3];
                dst[total_offset + 3] = src[total_offset + 2] + (src[total_offset + 3] * 2) + src[total_offset + 2];
            }else{
                dst[total_offset] = buf[col * tile_size - 1] + (buf[col * tile_size] * 2) + buf[col * tile_size + 1];
                dst[total_offset + 1] = buf[col * tile_size] + (buf[col * tile_size + 1] * 2) + buf[col * tile_size + 2];
                dst[total_offset + 2] = buf[col * tile_size + 1] + (buf[col * tile_size + 2] * 2) + buf[col * tile_size + 3];
                dst[total_offset + 3] = buf[col * tile_size + 2] + (buf[col * tile_size + 3] * 2) + buf[col * tile_size + 4];
            }
        }
    }
}


template<typename T, typename T2>
__global__ void sobel3x1_col_v1(T2* dst, T* src, T2* kernel, const uint64_t b, const uint64_t c, const uint64_t h, const uint64_t w){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx > h * w){
        return;
    }
    int col = idx % w;
    int row = idx / w;
    // const uint32_t tile_size = 256;
    const uint64_t img_size = c * h * w;
    const uint64_t plane_size = h * w;
    
    for(uint64_t i=0; i<b; ++i) {
        for(uint64_t j=0; j<c; ++j){
            int total_offset = i * img_size + j * plane_size;
            // For mirror pad
            if(row==0){
                dst[total_offset + idx] = 0;
            }else if(row==h-1){
                dst[total_offset + idx] = 0;
            }else{
                // dst[total_offset + idx] = -src[total_offset + (row-1) * w + col] + src[total_offset + (row+1) * w + col];
                dst[total_offset + idx] = -src[total_offset + idx - w] + src[total_offset + idx + w];
            }
        }
    }
}

template<typename T, typename T2>
__global__ void sobel3x1_col_v2(T2* dst, T* src, T2* kernel, const uint64_t b, const uint64_t c, const uint64_t h, const uint64_t w){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    // const uint32_t tile_size = 256;
    const uint64_t img_size = c * h * w;
    const uint64_t plane_size = h * w;
    // Put several rows to a tile
    const int tile_size = 16;
    __shared__ T buf[tile_size + 2][128];// blockDim.x equal to 128

    if((row+1) * tile_size > h || col > w){
        return;
    }

    for(uint64_t i=0; i<b; ++i) {
        for(uint64_t j=0; j<c; ++j){
            int total_offset = i * img_size + j * plane_size;
            // For mirror pad
            if(row==0){
                dst[total_offset + row * tile_size * w + col] = 0;
            }else if(row==h-1){
                dst[total_offset + row * tile_size * w + col] = 0;
            }else{
                // Pre load (tile_size + 2) rows into shared memory
                for(int k=0; k<tile_size+2; ++k){
                    buf[k][threadIdx.y] = src[total_offset + (row * tile_size - 1 + k) * w + col];
                }
                for(int k=2; k<tile_size+2; ++k){
                    dst[total_offset + (row * tile_size + (k-2)) * w + col] = -buf[k-2][threadIdx.y] + buf[k][threadIdx.y];
                }
            }
        }
    }
}

template<typename T, typename T2>
__global__ void sobel1x3(T2* dst, T* src, T2* kernel, const uint64_t b, const uint64_t c, const uint64_t h, const uint64_t w){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    // const uint32_t tile_size = 256;
    const uint64_t img_size = c * h * w;
    const uint64_t plane_size = h * w;
    // Put several rows to a tile
    const int tile_size = 16;
    const int block_size = 128;
    __shared__ T buf[tile_size + 2][block_size];// blockDim.x equal to 128

    if((row+1) * tile_size > h || col > w){
        return;
    }

    for(uint64_t i=0; i<b; ++i) {
        for(uint64_t j=0; j<c; ++j){
            int total_offset = i * img_size + j * plane_size;
            for(int k=0; k<tile_size; ++k){
                if(col == 0 || col == (w-1)){
                    continue;
                }
                int idx =  total_offset + (row * tile_size + k) * w + col;
                dst[idx] = src[idx - 1] + (src[idx] * 2) + src[idx + 1];
            }

            // For mirror pad
            if(row==0){
                dst[total_offset + row * tile_size * w + col] = 0;
            }else if(row==h-1){
                dst[total_offset + row * tile_size * w + col] = 0;
            }else{
                // Pre load (tile_size + 2) rows into shared memory
                for(int k=0; k<tile_size+2; ++k){
                    buf[k][threadIdx.y] = src[total_offset + (row * tile_size - 1 + k) * w + col];
                }
                for(int k=2; k<tile_size+2; ++k){
                    dst[total_offset + (row * tile_size + (k-2)) * w + col] = -buf[k-2][threadIdx.y] + buf[k][threadIdx.y];
                }
            }
        }
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
void sobel_cuda(T2* dst, T* src, T2* kernel, std::vector<uint64_t> shape, int loop_count){
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
    T2* d_kernel = NULL;
    err = cudaMalloc((void **)&d_kernel, c*sizeof(T2));
    err = cudaMemcpy(d_kernel, kernel, c * sizeof(T2), cudaMemcpyHostToDevice);

    // cudaThreadSetCacheConfig(cudaFuncCachePreferL1);
    cudaEvent_t startEvent, stopEvent;
    float total_time = 0.0;
    checkCuda( cudaEventCreate(&startEvent) );
    checkCuda( cudaEventCreate(&stopEvent) );

    dim3 threadsPerBlockK1(1, w);
    dim3 numBlocksK1(h, 1);
    // Warm up
    checkCuda(cudaEventRecord(startEvent,0) );
    sobel1x3_row_v2<T, T2><<<numBlocksK1, threadsPerBlockK1>>>(d_dst, d_src, d_kernel, n, c, h, w);
    dim3 threadsPerBlockK2(1, block_size);
    dim3 numBlocksK2(h/16, w/block_size);
    // sobel3x1_col_v2<T, T2><<<numBlocksK2, threadsPerBlockK2>>>(d_dst, d_src, d_kernel, n, c, h, w);
    checkCuda(cudaEventRecord(stopEvent, 0) );
    checkCuda(cudaEventSynchronize(stopEvent) );

    // CPU record latency
    auto t1 = std::chrono::steady_clock::now();
    for(int i=0; i<loop_count; ++i) {
        // sobel1x3_row_v1<T, T2><<<numBlocks, threadsPerBlock>>>(d_dst, d_src, d_kernel, n, c, h, w);
        sobel1x3_row_v2<T, T2><<<numBlocksK1, threadsPerBlockK1>>>(d_dst, d_src, d_kernel, n, c, h, w);
        // sobel3x1_col_v1<T, T2><<<numBlocks, threadsPerBlock>>>(d_dst, d_src, d_kernel, n, c, h, w);
        // sobel3x1_col_v2<T, T2><<<numBlocksK2, threadsPerBlockK2>>>(d_dst, d_src, d_kernel, n, c, h, w);
    }
    cudaDeviceSynchronize();
    auto t2 = std::chrono::steady_clock::now();
    double latency = std::chrono::duration<double, std::micro>(t2-t1).count();
    
    // GPU record latency
    for(int i=0; i<loop_count; ++i) {
        float ms = 0.0;
        checkCuda( cudaEventRecord(startEvent,0) );
        // sobel1x3_row_v1<T, T2><<<numBlocks, threadsPerBlock>>>(d_dst, d_src, d_kernel, n, c, h, w);
        sobel1x3_row_v2<T, T2><<<numBlocksK1, threadsPerBlockK1>>>(d_dst, d_src, d_kernel, n, c, h, w);
        // sobel3x1_col_v1<T, T2><<<numBlocks, threadsPerBlock>>>(d_dst, d_src, d_kernel, n, c, h, w);
        // sobel3x1_col_v2<T, T2><<<numBlocksK2, threadsPerBlockK2>>>(d_dst, d_src, d_kernel, n, c, h, w);
        checkCuda( cudaEventRecord(stopEvent,0) );
        checkCuda( cudaEventSynchronize(stopEvent) );
        checkCuda( cudaEventElapsedTime(&ms, startEvent, stopEvent) );
        total_time += ms;
    }
    
    printf("[%f, %f] bandwidth %f\n", latency / loop_count, total_time / loop_count,  elements_num*(sizeof(float) + sizeof(uint32_t)) * 1e3 /1024/1024/ (total_time/loop_count));
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
    cudaFree(d_kernel);
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
    float* src = (float*)malloc(sizeof(float) * img_size);
    float* dst = (float*)malloc(sizeof(float) * img_size);
    float* kernel = (float*)malloc(sizeof(float) * 3);

    for(uint32_t i=0; i<img_size; ++i){
        src[i] = i % n;
    }

    std::vector<uint64_t> shape = {1, 1, n, n};
    auto t1 = std::chrono::steady_clock::now();
    sobel_cuda<float, float>(dst, src, kernel, shape, loop_count);

    printf("dst:\n");
    auto num_out = n*n*3;
    for(int i=0; i<(num_out < 16 ? num_out: 16);++i){
        printf("<%d %f> ", i, dst[i]);
    }printf("\n");
    auto t2 = std::chrono::steady_clock::now();
    printf("%ld %d %f\n", n, loop_count, std::chrono::duration<double, std::micro>(t2-t1).count() / loop_count);
    
    free(src);
    free(dst);
    free(kernel);
    return 0;
}

