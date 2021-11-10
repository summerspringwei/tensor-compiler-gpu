#ifndef LSTM_UTILS_H
#define LSTM_UTILS_H

#include <vector>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

class CuLstmData
{
    public:
    /* data */
    float* d_inputs_timestep=nullptr, *d_outputs_timestep=nullptr;
    float* d_input_wavefront=nullptr, *d_c_wavefront=nullptr, *d_h_wavefront=nullptr;
    float* d_weight_input_wavefront=nullptr, *d_weight_state_wavefront=nullptr, *d_output_buffer=nullptr, *d_bias=nullptr;
    CuLstmData(float* inputs_timestep, float* outputs_timestep, 
        float* input_wavefront, float* c_wavefront, float* h_wavefront, 
        float* weight_input_wavefront, float* weight_state_wavefront, float* output_buffer, float* bias):
        d_inputs_timestep(inputs_timestep), d_outputs_timestep(outputs_timestep),
        d_input_wavefront(input_wavefront), d_c_wavefront(c_wavefront), d_h_wavefront(h_wavefront),
        d_weight_input_wavefront(weight_state_wavefront), d_weight_state_wavefront(weight_state_wavefront), 
        d_output_buffer(output_buffer), d_bias(bias){
    
    }

    void free(){
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
};


CuLstmData create_lstm_data(int batch, int num_layer, int num_hidden, int num_timestep){
   // Allocate host data
    std::vector<float> input_timestep(batch*num_timestep * num_hidden);
    std::vector<float> output_timestep(batch*num_timestep * num_hidden);
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

    checkCuda(cudaMemcpy(d_inputs_timestep, input_timestep.data(), sizeof(float) * input_timestep.size(), cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_input_wavefront, input_wavefront.data(), sizeof(float) * input_wavefront.size() , cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_c_wavefront, c_wavefront.data(), sizeof(float) * c_wavefront.size() , cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_h_wavefront, h_wavefront.data(), sizeof(float) * h_wavefront.size() , cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_weight_input_wavefront, weight_input_wavefront.data(), sizeof(float) * weight_input_wavefront.size() , cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_weight_state_wavefront, weight_state_wavefront.data(), sizeof(float) * weight_state_wavefront.size() , cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_bias, bias.data(), sizeof(float) * bias.size() , cudaMemcpyHostToDevice));

    return CuLstmData(d_inputs_timestep, d_outputs_timestep, 
        d_input_wavefront, d_c_wavefront, d_h_wavefront, 
        d_weight_input_wavefront, d_weight_state_wavefront, d_output_buffer, d_bias);
}


#endif