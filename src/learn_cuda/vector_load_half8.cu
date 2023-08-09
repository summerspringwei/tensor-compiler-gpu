#include <cuda_fp16.h>
#include <iostream>

const int N = 1024; // Number of elements in the vectors

__global__ void vectorAdd(const half* A, const half* B, half* C, int numElements) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    half localA[8];
    half localB[8];
    if (i * 8 < numElements) {
        *(float4*)&localA = *(float4*)&(A[i*8]);
        *(float4*)&localB = *(float4*)&(B[i*8]);
        // C[i] = __hadd(A[i], B[i]); // Half-precision addition
        #pragma unroll
        for(int j =0;j<8; ++j){
            // C[i*8+j] = __hadd(localA[j], localB[j]);
            C[i*8+j] = localA[j] + localB[j];
        }
    }
}

int main() {
    // Allocate memory on the host for input and output vectors
    half *h_A, *h_B, *h_C;
    h_A = new half[N];
    h_B = new half[N];
    h_C = new half[N];

    // Initialize input vectors
    for (int i = 0; i < N; ++i) {
        h_A[i] = __float2half(float(i) * 0.1); // Convert float to half
        h_B[i] = __float2half(float(i) * 0.2);
    }

    // Allocate memory on the device
    half *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * sizeof(half));
    cudaMalloc((void**)&d_B, N * sizeof(half));
    cudaMalloc((void**)&d_C, N * sizeof(half));

    // Copy input vectors from host to device
    cudaMemcpy(d_A, h_A, N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(half), cudaMemcpyHostToDevice);

    // Launch the vector addition kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy the result back from the device to the host
    cudaMemcpy(h_C, d_C, N * sizeof(half), cudaMemcpyDeviceToHost);
    for(int i =0;i<10;++i){
        std::cout<<__half2float(h_C[i])<<std::endl;
    }
    // Clean up
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
