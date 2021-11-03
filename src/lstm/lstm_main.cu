
#include <cstdlib>
#include <iostream>
#include <vector>
#include <chrono>
#include <sstream>

#include <cstdlib>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#include "../cuda_utils.h"

extern "C" __global__ void lstm_wavefront_magic(float* inputs_timestep, float* outputs_timestep, 
    float* c_wavefront, float* h_wavefront, float* input_wavefront,
    float* weight_input_wavefront, float* weight_state_wavefront, float* bias,
    float* output_buffer);

extern "C" __global__ void seq2seq_encoder(float* inputs_timestep, float* outputs_timestep, 
    float* c_wavefront, float* h_wavefront, float* input_wavefront,
    float* weight_input_wavefront, float* weight_state_wavefront, float* bias,
    float* output_buffer);

// #define FUNC_CALL lstm_wavefront_magic<<<numBlocks, threadsPerBlock>>>(\
//     d_inputs_timestep, d_outputs_timestep, \
//     d_c_wavefront, d_h_wavefront, d_input_wavefront, \
//     d_weight_input_wavefront, d_weight_state_wavefront, d_bias, \
//     d_output_buffer);

// #define FUNC_CALL seq2seq_encoder<batch, num_layer, num_timestep, num_hidden><<<numBlocks, threadsPerBlock>>>(\
//     d_inputs_timestep, d_outputs_timestep, \
//     d_c_wavefront, d_h_wavefront, d_input_wavefront, \
//     d_weight_input_wavefront, d_weight_state_wavefront, d_bias, \
//     d_output_buffer);

#define FUNC_CALL seq2seq_encoder<<<numBlocks, threadsPerBlock>>>(\
    d_inputs_timestep, d_outputs_timestep, \
    d_c_wavefront, d_h_wavefront, d_input_wavefront, \
    d_weight_input_wavefront, d_weight_state_wavefront, d_bias, \
    d_output_buffer);

#define CUDA_CHECK_RESULT if (result != cudaSuccess) \
    { \
        const char* msg = cudaGetErrorString(result); \
        std::stringstream safe_call_ss; \
        safe_call_ss << "\nerror: " << " failed with error" \
                    << "\nfile: " << __FILE__ << "\nline: " << __LINE__ << "\nmsg: " << msg; \
        throw std::runtime_error(safe_call_ss.str()); \
    }

void benchmark_lstm_wavefront_magic(int argc, char** argv){
    if(argc < 5){
        printf("argc is less than 4, require 3: batch, num_layer, num_timestep, num_hidden");
        return;
    }
    const int batch = atoi(argv[1]);
    const int num_layer = atoi(argv[2]);
    const int num_timestep = atoi(argv[3]);
    const int num_hidden = atoi(argv[4]);
    int steps = 1000;
    if(argc > 5){
        steps = atoi(argv[5]);
    }
    // Allocate host data
    std::vector<float> input_timestep(num_timestep * num_hidden);
    std::vector<float> output_timestep(num_timestep * num_hidden);
    std::vector<float> input_wavefront(batch*num_layer*num_hidden);
    std::vector<float> c_wavefront(batch*num_layer*num_hidden);
    std::vector<float> h_wavefront(batch*num_layer*num_hidden);
    std::vector<float> weight_input_wavefront(4*num_layer*num_hidden*num_hidden);
    std::vector<float> weight_state_wavefront(4*num_layer*num_hidden*num_hidden);
    std::vector<float> output_buffer(8*num_layer*num_hidden);
    std::vector<float> bias(num_hidden);
    // Set host data
    for(int i=0; i<input_timestep.size(); ++i){
        input_timestep[i] = 1.0;
    }
    for(int i=0;i<weight_input_wavefront.size(); ++i){
        weight_input_wavefront[i] = 1.0;
        weight_state_wavefront[i] = 1.0;
    }
    for(int i=0; i<c_wavefront.size(); ++i){
        c_wavefront[i]=1.0;
        h_wavefront[i]=1.0;
        input_wavefront[i]=1.0;
    }
    // Allocate GPU
    float* d_inputs_timestep=nullptr, *d_outputs_timestep=nullptr;
    float* d_input_wavefront=nullptr, *d_c_wavefront=nullptr, *d_h_wavefront=nullptr;
    float* d_weight_input_wavefront=nullptr, *d_weight_state_wavefront=nullptr, *d_output_buffer=nullptr, *d_bias=nullptr;
    cudaMalloc((void**)&d_inputs_timestep, sizeof(float) * input_timestep.size());
    cudaMalloc((void**)&d_outputs_timestep, sizeof(float) * output_timestep.size());
    cudaMalloc((void**)&d_input_wavefront, sizeof(float) * input_wavefront.size());
    cudaMalloc((void**)&d_c_wavefront, sizeof(float) * c_wavefront.size());
    cudaMalloc((void**)&d_h_wavefront, sizeof(float) * h_wavefront.size());
    cudaMalloc((void**)&d_weight_input_wavefront, sizeof(float) * weight_input_wavefront.size());
    cudaMalloc((void**)&d_weight_state_wavefront, sizeof(float) * weight_state_wavefront.size());
    cudaMalloc((void**)&d_output_buffer, sizeof(float) * output_buffer.size());
    cudaMalloc((void**)&d_bias, sizeof(float) * bias.size());

    //GPU time measurement
    float ms_max = std::numeric_limits<float>::min();
    float ms_min = std::numeric_limits<float>::max();
    float ms_total, ms_i;
    cudaEvent_t start_i, stop_i;
    cudaEventCreate(&start_i);
    cudaEventCreate(&stop_i);

    checkCuda(cudaMemcpy(d_inputs_timestep, input_timestep.data(), sizeof(float) * input_timestep.size(), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_input_wavefront, input_wavefront.data(), sizeof(float) * input_wavefront.size() , cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_wavefront, c_wavefront.data(), sizeof(float) * c_wavefront.size() , cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_h_wavefront, h_wavefront.data(), sizeof(float) * h_wavefront.size() , cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_weight_input_wavefront, weight_input_wavefront.data(), sizeof(float) * weight_input_wavefront.size() , cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_weight_state_wavefront, weight_state_wavefront.data(), sizeof(float) * weight_state_wavefront.size() , cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_bias, bias.data(), sizeof(float) * bias.size() , cudaMemcpyHostToDevice));

    
    // Set shared memory for SM
    // int maxbytes = 1024*64; // 96 KB
    // cudaFuncSetAttribute(lstm_wavefront_magic, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);
    // int carveout = 50; // prefer shared memory capacity 50% of maximum
    // Named Carveout Values:
    // carveout = cudaSharedmemCarveoutDefault;   //  (-1)
    // carveout = cudaSharedmemCarveoutMaxL1;     //   (0)
    // auto carveout = cudaSharedmemCarveoutMaxShared; // (100)
    // cudaFuncSetAttribute(lstm_wavefront_magic, cudaFuncAttributePreferredSharedMemoryCarveout, carveout);
    dim3 threadsPerBlock(num_hidden/2, 1, 1);
    dim3 numBlocks(8*num_layer*2, 1, 1);

    FUNC_CALL
    cudaDeviceSynchronize();
    checkCuda(cudaMemcpy(output_timestep.data(), d_outputs_timestep, sizeof(float) * output_timestep.size() , cudaMemcpyDeviceToHost));
    printf("%ld\n", output_timestep.size());
    for(int i=0;i<num_timestep; ++i){
        for(int j=0;j<num_hidden;++j){
            printf("%f ", output_timestep[i*num_hidden + j]);
        }printf("\n");
    }
    auto result = cudaGetLastError();                                                   
    CUDA_CHECK_RESULT

    // Warm up
    for (int i=0; i<steps; i++) {
        // Run in serial default
        FUNC_CALL
        // printf("warm up %d\n", i);
        cudaDeviceSynchronize();
    }
    result = cudaGetLastError();                                                   
    CUDA_CHECK_RESULT

    //time measurement
    ms_total = 0;

    cudaProfilerStart();
    for (int i_=0; i_<steps; i_++)
    {
        cudaEventRecord(start_i, 0);
        FUNC_CALL
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

    cudaFree(d_inputs_timestep);
    cudaFree(d_outputs_timestep);
    cudaFree(d_c_wavefront);
    cudaFree(d_h_wavefront);
    cudaFree(d_input_wavefront);
    cudaFree(d_weight_input_wavefront);
    cudaFree(d_weight_state_wavefront);
    cudaFree(d_bias);
    cudaFree(d_output_buffer);
}

int main(int argc, char** argv) {
    benchmark_lstm_wavefront_magic(argc, argv);
    return 0;
}