
#include <cuda.h>
#include <cooperative_groups.h>


__global__ void learn_grid_sync(float* input, float* output, const int totalNumber, const int blockSize){
  cooperative_groups::grid_group grid = cooperative_groups::this_grid();
  input[blockIdx.x * blockSize + threadIdx.x] += 1;
  grid.sync();
  output[blockIdx.x * blockSize + threadIdx.x] = input[totalNumber - 1 - (blockIdx.x * blockSize + threadIdx.x)];
}


int main(){
  const int totalNumber = 128 * 32;
  const int blockSize = 128;
  float* input = new float[totalNumber];
  float* output = new float[totalNumber];

  cudaError_t err = cudaSuccess;
  float* d_input = NULL, *d_output=NULL;
  cudaMalloc((void **)&d_input, sizeof(float)*totalNumber);
  cudaMalloc((void **)&d_output, sizeof(float)*totalNumber);
  for(int i=0; i<totalNumber; ++i){
    input[i] = i;
  }
  
  cudaMemcpy(d_input, input, sizeof(float)*totalNumber, cudaMemcpyHostToDevice);
  void * kernel_args[] = {
    (void*)&(d_input), (void*)&(d_output), (void*)&(totalNumber),(void*)&(blockSize)
  };

  cudaLaunchCooperativeKernel((void*)learn_grid_sync, dim3(32, 1,1), dim3(128,1,1), kernel_args, 1024);
  cudaDeviceSynchronize();
  cudaMemcpy(output, d_output, sizeof(float)*totalNumber, cudaMemcpyDeviceToHost);
  for(int i=0; i<totalNumber; ++i){
    printf("%f ", output[i]);
  }
  return 0;
}
