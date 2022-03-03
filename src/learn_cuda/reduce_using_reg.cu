#include <stdio.h>
#include <cuda_runtime.h>
#define FULL_MASK 0xffffffff
__global__ void warpReduce(int *d_a, int* d_b) {
    // int laneId = threadIdx.x & 0x1f;
    // Seed starting value as inverse lane ID
    // int value = d_a[threadIdx.x];
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    int value = d_a[idx];
    // __shared__ local_db[blockIdx.x];
    for(int i=16; i>=1; i/=2){
        unsigned mask = __ballot_sync(FULL_MASK, threadIdx.x < i);
        value += __shfl_down_sync(mask, value, i);
        // printf("<%d %d %d>", i, threadIdx.x, value);
        // __syncwarp();
        // value += __reduce_add_sync(mask, value);
    }
    __syncwarp();
    if(threadIdx.x % 32 ==0)
        atomicAdd(&d_b[blockIdx.x + threadIdx.x / 32], value);
    
    // Use XOR mode to perform butterfly reduction
    // for (int i=16; i>=1; i/=2)
    //     value += __shfl_down_sync(0xffffffff, value, i);
    // printf("<%d %d>", threadIdx.x, value);
    // d_a[threadIdx.x] = value;
    // "value" now contains the sum across all threads
}

int main() {
    int n = 1024*10240;
    int num_block = 10240;
    int* a = new int[n];
    int* b = new int[num_block];
    for(int i =0; i<n;++i){
        a[i] = 1;
    }
    int* da = NULL;
    cudaMalloc((void **)&da, sizeof(int) * n);
    cudaMemcpy(da, a, sizeof(int) * n, cudaMemcpyHostToDevice);
    int* db = NULL;
    cudaMalloc((void **)&db, sizeof(int) * num_block);

    warpReduce<<< num_block, n / num_block >>>(da, db);
    // cudaMemcpy(a, da, sizeof(int) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(b, db, sizeof(int) * num_block, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    for(int i =0; i<num_block;++i){
        printf("%d ", b[i]);
    }printf("\n");
    
    cudaFree(da);
    cudaFree(db);
    return 0;
}