
#include <cstdlib>
#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>

#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "../cuda_utils.h"
#include "lstm_reuse_shared_memory.h"
#include "lstm_utils.h"


// kNumGatePart=4
// #define LSTM_DEV_FUNC  { \
//     cudaLaunchCooperativeKernel((void*)lstm_reuse_shared_memory_v6<1, 10, 256, 100>, dim3(320, 1, 1), dim3(256, 1, 1), encoder_kernelArgs, 48*1024);};

#define LSTM_DEV_FUNC  { \
    cudaLaunchCooperativeKernel((void*)lstm_reuse_shared_memory_v8<1, 10, 256, 100>, dim3(320, 1, 1), dim3(32, 8, 1), encoder_kernelArgs, 32*1024);};

#define CUDA_CHECK_RESULT if (result != cudaSuccess) \
    { \
        const char* msg = cudaGetErrorString(result); \
        std::stringstream safe_call_ss; \
        safe_call_ss << "\nerror: " << " failed with error" \
                    << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg; \
        throw std::runtime_error(safe_call_ss.str()); \
    };


void benchmark_lstm(int argc, char** argv){
    const int batch = 1;
    const int num_layer = 10, num_timestep = 100, num_hidden = 256;
    
    int steps = 10000;
    if(argc > 1){
        steps = atoi(argv[1]);
    }
    
    int dev = 0;
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, dev);
    if(supportsCoopLaunch){
        printf("Device support CoopLaunch\n");
    }
    auto lstm_data = create_lstm_data(batch, num_layer, num_hidden, num_timestep);

    // int* d_arr_sync=nullptr;
    // cudaMalloc((void**)&d_arr_sync, 8*num_layer*num_layer*sizeof(int));
    // cudaMemset(d_arr_sync, 0, 8*num_layer*num_layer*sizeof(int));
    // Set shared memory for SM
    // int maxbytes = 1024*164;
    // cudaFuncSetAttribute(lstm_reuse_shared_memory_v4<1, 10, 256, 100>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    // int carveout = 50; // prefer shared memory capacity 50% of maximum
    // Named Carveout Values:
    // carveout = cudaSharedmemCarveoutDefault;   //  (-1)
    // carveout = cudaSharedmemCarveoutMaxL1;     //   (0)
    // auto carveout = cudaSharedmemCarveoutMaxShared; // (100)
    // cudaFuncSetAttribute(lstm_wavefront_magic, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    int numThreads = 64*4, numBlocksPerSm=0; \
    cudaDeviceProp deviceProp; \
    cudaGetDeviceProperties(&deviceProp, dev); \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, lstm_reuse_shared_memory_v4<1, 10, 256, 100>, numThreads, 0); \
    printf("OccupancyMaxActiveBlocksPerMultiprocessor: %d, multiProcessorCount: %d\n", numBlocksPerSm, deviceProp.multiProcessorCount);\
    void *encoder_kernelArgs[] = { (void *)&(lstm_data.d_inputs_timestep), (void *)&(lstm_data.d_outputs_timestep), \
        (void *)&(lstm_data.d_c_wavefront), (void *)&(lstm_data.d_h_wavefront), (void *)&(lstm_data.d_input_wavefront), \
        (void *)&(lstm_data.d_weight_input_wavefront), (void *)&(lstm_data.d_weight_state_wavefront), (void *)&(lstm_data.d_bias), \
        (void *)&(lstm_data.d_output_buffer)
        };
    
    LSTM_DEV_FUNC
    cudaDeviceSynchronize();
    
    std::vector<float> lstm_output_timestep(batch * num_hidden * num_timestep);
    checkCuda(cudaMemcpy(lstm_output_timestep.data(), lstm_data.d_outputs_timestep, sizeof(float) * lstm_output_timestep.size() , cudaMemcpyDeviceToHost));
    // printf("%ld\n", encoder_output_timestep.size());
    // printf("Outputs\n");
    // for(int i=0;i<num_layer; ++i){
    //     for(int j=0;j<num_layer;++j){
    //         printf("%f ", encoder_output_timestep[i*num_layer + j]);
    //     }printf("\n");
    // }
    auto result = cudaGetLastError();
    CUDA_CHECK_RESULT
    // return;
    // Warm up
    for (int i=0; i<steps; i++) {
        LSTM_DEV_FUNC
        cudaDeviceSynchronize();
    }
    result = cudaGetLastError();                                                   
    CUDA_CHECK_RESULT
     
    // GPU time measurement
    float ms_max = std::numeric_limits<float>::min();
    float ms_min = std::numeric_limits<float>::max();
    float ms_total, ms_i;
    cudaEvent_t start_i, stop_i;
    cudaEventCreate(&start_i);
    cudaEventCreate(&stop_i);
    ms_total = 0;

    cudaProfilerStart();
    for (int i_=0; i_<steps; i_++)
    {
        cudaEventRecord(start_i, 0);
        LSTM_DEV_FUNC
        cudaEventRecord(stop_i, 0);
        cudaEventSynchronize(stop_i);
        cudaEventElapsedTime(&ms_i, start_i, stop_i);
        cudaDeviceSynchronize();
        // printf("Iteration time %f ms\n", ms_i);
        ms_total += ms_i;
        if (ms_i > ms_max)  ms_max = ms_i;
        if (ms_i < ms_min) ms_min = ms_i;
    }
    cudaProfilerStop();
    cudaDeviceSynchronize();
    printf("Summary: [min, max, mean] = [%f, %f, %f] ms\n",  ms_min, ms_max, ms_total / steps);
    result = cudaGetLastError();
    CUDA_CHECK_RESULT

    lstm_data.free();
}

int main(int argc, char** argv) {
    benchmark_lstm(argc, argv);
    return 0;
}
