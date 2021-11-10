
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


// #define SEQ2SEQ_ENCODER lstm_reuse_shared_memory<1, 8, 128, 100><<<dim3(128, 1, 1), dim3(64, 1, 1)>>>(\
//     encoder_data.d_inputs_timestep, encoder_data.d_outputs_timestep, \
//     encoder_data.d_c_wavefront, encoder_data.d_h_wavefront, encoder_data.d_input_wavefront, \
//     encoder_data.d_weight_input_wavefront, encoder_data.d_weight_state_wavefront, encoder_data.d_bias, \
//     encoder_data.d_output_buffer);

// #define SEQ2SEQ_DECODER lstm_reuse_shared_memory<1, 4, 128, 30><<<dim3(64, 1, 1), dim3(64, 1, 1)>>>(\
//     decoder_data.d_inputs_timestep, decoder_data.d_outputs_timestep, \
//     decoder_data.d_c_wavefront, decoder_data.d_h_wavefront, decoder_data.d_input_wavefront, \
//     decoder_data.d_weight_input_wavefront, decoder_data.d_weight_state_wavefront, decoder_data.d_bias, \
//     decoder_data.d_output_buffer);
// numBlocksPerSm*deviceProp.multiProcessorCount
#define SEQ2SEQ_ENCODER  { \
    cudaLaunchCooperativeKernel((void*)lstm_reuse_shared_memory<1, 8, 128, 100>, dim3(128, 1, 1), dim3(64, 1, 1), kernelArgs, 32*1024);}; 

#define SEQ2SEQ_DECODER  {int dev = 0, numThreads = 64, numBlocksPerSm=0; \
    cudaDeviceProp deviceProp; \
    cudaGetDeviceProperties(&deviceProp, dev); \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, lstm_reuse_shared_memory<1, 4, 128, 30>, numThreads, 0); \
    printf("OccupancyMaxActiveBlocksPerMultiprocessor: %d, multiProcessorCount: %d\n", numBlocksPerSm, deviceProp.multiProcessorCount);\
    void *kernelArgs[] = { decoder_data.d_inputs_timestep, decoder_data.d_outputs_timestep, \
        decoder_data.d_c_wavefront, decoder_data.d_h_wavefront, decoder_data.d_input_wavefront, \
        decoder_data.d_weight_input_wavefront, decoder_data.d_weight_state_wavefront, decoder_data.d_bias, \
        decoder_data.d_output_buffer }; \
    cudaLaunchCooperativeKernel((void*)lstm_reuse_shared_memory<1, 4, 128, 30>, dim3(numBlocksPerSm*deviceProp.multiProcessorCount, 1, 1), dim3(64, 1, 1), kernelArgs, 32*1024);};


#define CUDA_CHECK_RESULT if (result != cudaSuccess) \
    { \
        const char* msg = cudaGetErrorString(result); \
        std::stringstream safe_call_ss; \
        safe_call_ss << "\nerror: " << " failed with error" \
                    << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg; \
        throw std::runtime_error(safe_call_ss.str()); \
    };


void benchmark_seq2seq(int argc, char** argv){
    const int batch = 1;
    const int encoder_num_layer = 8, encoder_num_timestep = 100, encoder_num_hidden = 128;
    const int decoder_num_layer = 4, decoder_num_timestep = 30, decoder_num_hidden = 128; 
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
    auto encoder_data = create_lstm_data(batch, encoder_num_layer, encoder_num_hidden, encoder_num_timestep);
    auto decoder_data = create_lstm_data(batch, decoder_num_layer, decoder_num_hidden, decoder_num_timestep);

    // Set shared memory for SM
    // int maxbytes = 1024*64;
    // cudaFuncSetAttribute(lstm_reuse_shared_memory<1, 8, 128, 100>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    // int carveout = 50; // prefer shared memory capacity 50% of maximum
    // Named Carveout Values:
    // carveout = cudaSharedmemCarveoutDefault;   //  (-1)
    // carveout = cudaSharedmemCarveoutMaxL1;     //   (0)
    // auto carveout = cudaSharedmemCarveoutMaxShared; // (100)
    // cudaFuncSetAttribute(lstm_wavefront_magic, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    int numThreads = 64, numBlocksPerSm=0; \
    cudaDeviceProp deviceProp; \
    cudaGetDeviceProperties(&deviceProp, dev); \
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, lstm_reuse_shared_memory<1, 8, 128, 100>, numThreads, 0); \
    printf("OccupancyMaxActiveBlocksPerMultiprocessor: %d, multiProcessorCount: %d\n", numBlocksPerSm, deviceProp.multiProcessorCount);\
    void *encoder_kernelArgs[] = { (void *)&(encoder_data.d_inputs_timestep), (void *)&(encoder_data.d_outputs_timestep), \
        (void *)&(encoder_data.d_c_wavefront), (void *)&(encoder_data.d_h_wavefront), (void *)&(encoder_data.d_input_wavefront), \
        (void *)&(encoder_data.d_weight_input_wavefront), (void *)&(encoder_data.d_weight_state_wavefront), (void *)&(encoder_data.d_bias), \
        (void *)&(encoder_data.d_output_buffer) };
    void *decoder_kernelArgs[] = { (void *)&(decoder_data.d_inputs_timestep), (void *)&(decoder_data.d_outputs_timestep), \
        (void *)&(decoder_data.d_c_wavefront), (void *)&(decoder_data.d_h_wavefront), (void *)&(decoder_data.d_input_wavefront), \
        (void *)&(decoder_data.d_weight_input_wavefront), (void *)&(decoder_data.d_weight_state_wavefront), (void *)&(decoder_data.d_bias), \
        (void *)&(decoder_data.d_output_buffer) }; 
    
    cudaLaunchCooperativeKernel((void*)lstm_reuse_shared_memory<1, 8, 128, 100>, dim3(128, 1, 1), dim3(64, 1, 1), encoder_kernelArgs, 32*1024);
    cudaLaunchCooperativeKernel((void*)lstm_reuse_shared_memory<1, 4, 128, 30>, dim3(64, 1, 1), dim3(64, 1, 1), decoder_kernelArgs, 32*1024);
    // SEQ2SEQ_DECODER
    cudaDeviceSynchronize();
    
    std::vector<float> encoder_output_timestep(batch * encoder_num_timestep * encoder_num_hidden);
    checkCuda(cudaMemcpy(encoder_output_timestep.data(), encoder_data.d_outputs_timestep, sizeof(float) * encoder_output_timestep.size() , cudaMemcpyDeviceToHost));
    std::vector<float> decoder_output_timestep(batch * decoder_num_timestep * decoder_num_hidden);
    checkCuda(cudaMemcpy(decoder_output_timestep.data(), decoder_data.d_outputs_timestep, sizeof(float) * decoder_output_timestep.size() , cudaMemcpyDeviceToHost));
    // printf("%ld\n", encoder_output_timestep.size());
    printf("Outputs\n");
    for(int i=0;i<encoder_num_timestep; ++i){
        for(int j=0;j<encoder_num_hidden;++j){
            printf("%f ", encoder_output_timestep[i*encoder_num_hidden + j]);
        }printf("\n");
    }
    auto result = cudaGetLastError();
    CUDA_CHECK_RESULT
    
    // Warm up
    for (int i=0; i<steps; i++) {
        // SEQ2SEQ_ENCODER
        cudaLaunchCooperativeKernel((void*)lstm_reuse_shared_memory<1, 8, 128, 100>, dim3(128, 1, 1), dim3(64, 1, 1), encoder_kernelArgs, 32*1024);
        cudaLaunchCooperativeKernel((void*)lstm_reuse_shared_memory<1, 4, 128, 30>, dim3(64, 1, 1), dim3(64, 1, 1), decoder_kernelArgs, 32*1024);
        // SEQ2SEQ_DECODER
        // printf("Iter %d\n", i);
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
        // SEQ2SEQ_DECODER
        cudaLaunchCooperativeKernel((void*)lstm_reuse_shared_memory<1, 8, 128, 100>, dim3(128, 1, 1), dim3(64, 1, 1), encoder_kernelArgs, 32*1024);
        cudaLaunchCooperativeKernel((void*)lstm_reuse_shared_memory<1, 4, 128, 30>, dim3(64, 1, 1), dim3(64, 1, 1), decoder_kernelArgs, 32*1024);
        // SEQ2SEQ_ENCODER
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

    encoder_data.free();
    decoder_data.free();
}

int main(int argc, char** argv) {
    benchmark_seq2seq(argc, argv);
    return 0;
}
