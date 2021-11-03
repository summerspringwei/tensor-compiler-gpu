// From https://gist.githubusercontent.com/odashi/1c20ba90388cf02330e1b95963d78039/raw/7cb69e7cadddbb07501aa69a6384ffb9ea2fca9a/cudnn_convolution_forward.cu

#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <limits>

#include <cstdlib>

#include <cuda.h>
#include <cudnn.h>

#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
        << "    Error occurred: " << err << " at " << __LINE__ << std::endl; \
    std::exit(1); \
  } \
}

#define CUDNN_CALL(f) { \
  cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    std::cout \
        << "    Error occurred: " << err << " at " << __LINE__ << std::endl; \
    std::exit(1); \
  } \
}

__global__ void dev_const(float *px, float k) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = k;
}

__global__ void dev_iota(float *px) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = tid;
}

void print(const float *data, int n, int c, int h, int w) {
  std::vector<float> buffer(1 << 20);
  CUDA_CALL(cudaMemcpy(
        buffer.data(), data,
        n * c * h * w * sizeof(float),
        cudaMemcpyDeviceToHost));
  int a = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      std::cout << "n=" << i << ", c=" << j << ":" << std::endl;
      for (int k = 0; k < h; ++k) {
        for (int l = 0; l < w; ++l) {
          std::cout << std::setw(4) << std::right << buffer[a];
          ++a;
        }
        std::cout << std::endl;
      }
    }
  }
  std::cout << std::endl;
}

int main() {
  cudnnHandle_t cudnn;
  CUDNN_CALL(cudnnCreate(&cudnn));

  // input
  const int in_n = 1;
  const int in_c = 1024;
  const int in_h = 7;
  const int in_w = 7;
  // const int in_n = 1;
  // const int in_c = 160;
  // const int in_h = 73;
  // const int in_w = 73;
  // filter
  const int filt_k = 1024;
  const int filt_c = in_c;
  const int filt_h = 1;
  const int filt_w = 1;

  std::cout << "in_n: " << in_n << std::endl;
  std::cout << "in_c: " << in_c << std::endl;
  std::cout << "in_h: " << in_h << std::endl;
  std::cout << "in_w: " << in_w << std::endl;
  std::cout << std::endl;

  cudnnTensorDescriptor_t in_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        in_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        in_n, in_c, in_h, in_w));

  float *in_data;
  CUDA_CALL(cudaMalloc(
        &in_data, in_n * in_c * in_h * in_w * sizeof(float)));


  std::cout << "filt_k: " << filt_k << std::endl;
  std::cout << "filt_c: " << filt_c << std::endl;
  std::cout << "filt_h: " << filt_h << std::endl;
  std::cout << "filt_w: " << filt_w << std::endl;
  std::cout << std::endl;

  cudnnFilterDescriptor_t filt_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(
        filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
        filt_k, filt_c, filt_h, filt_w));

  float *filt_data;
  CUDA_CALL(cudaMalloc(
      &filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float)));

  // convolution
  const int pad_h = 1;
  const int pad_w = 1;
  const int str_h = 1;
  const int str_w = 1;
  const int dil_h = 1;
  const int dil_w = 1;
  std::cout << "pad_h: " << pad_h << std::endl;
  std::cout << "pad_w: " << pad_w << std::endl;
  std::cout << "str_h: " << str_h << std::endl;
  std::cout << "str_w: " << str_w << std::endl;
  std::cout << "dil_h: " << dil_h << std::endl;
  std::cout << "dil_w: " << dil_w << std::endl;
  std::cout << std::endl;

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w, str_h, str_w, dil_h, dil_w,
        CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

  // output
  int out_n;
  int out_c;
  int out_h;
  int out_w;
  
  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc,
        &out_n, &out_c, &out_h, &out_w));

  std::cout << "out_n: " << out_n << std::endl;
  std::cout << "out_c: " << out_c << std::endl;
  std::cout << "out_h: " << out_h << std::endl;
  std::cout << "out_w: " << out_w << std::endl;
  std::cout << std::endl;

  cudnnTensorDescriptor_t out_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w));

  float *out_data;
  CUDA_CALL(cudaMalloc(
        &out_data, out_n * out_c * out_h * out_w * sizeof(float)));

  // algorithm
  int max_count=0;
  cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn, &max_count);
  // cudnnConvolutionFwdAlgoPerf_t algo_perf[max_count];
  cudnnConvolutionFwdAlgoPerf_t* algo_perf = (cudnnConvolutionFwdAlgoPerf_t*)malloc(sizeof(cudnnConvolutionFwdAlgoPerf_t) * max_count);
  // cudnnConvolutionFwdAlgo_t algo;
  int returnedAlgoCount=0;
  int req = 10;
  CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm_v7(
        cudnn,
        in_desc, filt_desc, conv_desc, out_desc,
        req, &returnedAlgoCount, algo_perf));
  for(int i=0; i<returnedAlgoCount; ++i){
    std::cout<<algo_perf[i].algo << " " << algo_perf[i].status << " " << algo_perf[i].memory << std::endl;
  }
  int best_algo = 0;
  for(; best_algo < returnedAlgoCount; ++best_algo){
    if(algo_perf[best_algo].status!=CUDNN_STATUS_SUCCESS){
      continue;
    }
  std::cout << "Returned : " << returnedAlgoCount << std::endl;
  std::cout << "Convolution algorithm: " << algo_perf[best_algo].algo << std::endl;
  std::cout << std::endl;

  // workspace
  size_t ws_size;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, in_desc, filt_desc, conv_desc, out_desc, algo_perf[best_algo].algo, &ws_size));

  float *ws_data;
  CUDA_CALL(cudaMalloc(&ws_data, ws_size));

  std::cout << "Workspace size: " << ws_size << std::endl;
  std::cout << std::endl;

  // perform
  float alpha = 1.f;
  float beta = 0.f;
  dev_iota<<<in_w * in_h, in_n * in_c>>>(in_data);
  dev_const<<<filt_w * filt_h, filt_k * filt_c>>>(filt_data, 1.f);
  // Warm up
  for(int i=0;i<10;++i){
    CUDNN_CALL(cudnnConvolutionForward(
        cudnn,
        &alpha, in_desc, in_data, filt_desc, filt_data,
        conv_desc, algo_perf[best_algo].algo, ws_data, ws_size,
        &beta, out_desc, out_data));
  }
  cudaEvent_t start_i, stop_i;
  cudaEventCreate(&start_i);
  cudaEventCreate(&stop_i);
  float ms_max = std::numeric_limits<float>::min();
  float ms_min = std::numeric_limits<float>::max();
  float ms_total=0, ms_i;
  int steps = 100000;
  for(int i=0;i<steps;++i){
cudaEventRecord(start_i, 0);
    CUDNN_CALL(cudnnConvolutionForward(
        cudnn,
        &alpha, in_desc, in_data, filt_desc, filt_data,
        conv_desc, algo_perf[best_algo].algo, ws_data, ws_size,
        &beta, out_desc, out_data));
cudaEventRecord(stop_i, 0);
cudaEventSynchronize(stop_i);
cudaEventElapsedTime(&ms_i, start_i, stop_i);
cudaDeviceSynchronize();
printf("(1,256,56,56)x(256,256,1,1) %f\n", ms_i);
    cudaError_t result = cudaGetLastError();                                                   
    if (result != cudaSuccess)                                                                 
    {                                                                                          
        const char* msg = cudaGetErrorString(result);                                          
        std::stringstream safe_call_ss;                                                        
        safe_call_ss << "\n----------\nerror: " << " failed with error"                                    
                      << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg
                      <<"--------\n";  
        // throw std::runtime_error(safe_call_ss.str());                                          
    }
    // printf("Iteration time %f ms\n", ms_i);
    ms_total += ms_i;
    if (ms_i > ms_max)  ms_max = ms_i;
    if (ms_i < ms_min) ms_min = ms_i;
  }

  
  printf("Summary: [min, max, mean] = [%f, %f, %f] ms\n",  ms_min, ms_max, ms_total / steps);
  // results
  // std::cout << "in_data:" << std::endl;
  // print(in_data, in_n, in_c, in_h, in_w);
  
  // // std::cout << "filt_data:" << std::endl;
  // print(filt_data, filt_k, filt_c, filt_h, filt_w);
  
  // // std::cout << "out_data:" << std::endl;
  // print(out_data, out_n, out_c, out_h, out_w);

  // finalizing
  
}
  // CUDA_CALL(cudaFree(ws_data));
  // CUDA_CALL(cudaFree(out_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDA_CALL(cudaFree(filt_data));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
  CUDA_CALL(cudaFree(in_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
  CUDNN_CALL(cudnnDestroy(cudnn));
  return 0;
}