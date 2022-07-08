#include <cassert>

#include <stdio.h>

#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cooperative_groups.h>

__device__ __forceinline__ float sigmoid(float x){
    return (1.0f / (1+exp(-x)));
}

// TODO(xiachunwei) using more threads per block to do GEMV
// num_layers: 8, timesteps: 100
// inputs_timestep: [1, 100, 128], outputs_timestep[1, 100, 128]
// input_wavefront: [1, 8, 128], state_wavefront: [1, 8, 128], weight_*_wavefront [32, 128, 128]
// c: [1, 8, 128], output_buffer:[1, 8, 128]
// two block computes one gate, therefore each block compute [1, 1, 128] * [1, 128, 64]
// gridDim(64*2, ), blockDim(64, ) 
template<int batch, int num_layer, int num_hidden, int num_timestep>
    __global__ void lstm_reuse_shared_memory(float* inputs_timestep, float* outputs_timestep, 
    float* c_wavefront, float* h_wavefront, float* input_wavefront,
    float* weight_input_wavefront, float* weight_state_wavefront, float* bias,
    float* output_buffer){
    const int kNumGatePart = 2;
    // const int kNumInputGatesInLstmCell = 4;
    const int kNumGatesInLstmCell = 8;
    const int kPartHidden = num_hidden / kNumGatePart;
    assert(num_hidden % kNumGatePart == 0);
    assert(num_hidden * kPartHidden * sizeof(float) < 48 * 1024);
    if(blockIdx.x >= num_layer*8*kNumGatePart){
        return;
    }
    // __shared__ float shared_weight[num_hidden * kPartHidden];
    extern float __shared__ shared_weight[];
    float * weight_ptr = NULL;
    float* is_ptr = NULL;

    // Feed inputs_timestep[0] to input_wavefront
    if(blockIdx.x < 2){
        input_wavefront[blockIdx.x * kPartHidden + threadIdx.x]=inputs_timestep[0 * num_hidden + blockIdx.x * kPartHidden + threadIdx.x];
    }
    // Load weight to shared memory
    // Weight layout: [hidden_to_reduce(128), hidden_to_out(64)]
    if(blockIdx.x < gridDim.x / 2){
        weight_ptr=weight_input_wavefront;
        is_ptr = input_wavefront;
        #pragma unroll
        for(int i=0; i<num_hidden; ++i){
            shared_weight[i * kPartHidden + threadIdx.x] = weight_ptr[blockIdx.x / 2 * num_hidden * num_hidden + i * num_hidden + (blockIdx.x % 2) * kPartHidden + threadIdx.x];
        }
    }else{
        weight_ptr = weight_state_wavefront;
        is_ptr = h_wavefront;
        #pragma unroll
        for(int i=0; i<num_hidden; ++i){
            shared_weight[i * kPartHidden + threadIdx.x] = weight_ptr[(blockIdx.x-gridDim.x/2) / 2 * num_hidden * num_hidden + i * num_hidden + (blockIdx.x % 2) * kPartHidden + threadIdx.x];
        }
    }
    
    __syncthreads();
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    for(int step=0; step<num_timestep+num_layer-1; ++step){
        // GEMV gate, for even blocks compute first half gate, for odd blocks compute second half gate
        float thread_output = 0;
        // Blocks from 0 to kNumGatePart * kNumInputGatesInLstmCell * num_layer compute the input gate
        bool cond_input = (blockIdx.x < min(step+1, (int)num_layer) * kNumGatesInLstmCell) 
            || (step>=num_timestep && blockIdx.x > (step+1-num_timestep)* kNumGatesInLstmCell);
        // Blocks from kNumGatePart * kNumInputGatesInLstmCell * num_layer to gridDim.x compute the state gate
        bool cond_state = ((blockIdx.x >= gridDim.x / 2) && (blockIdx.x < gridDim.x / 2 + min(step+1, (int)num_layer) * kNumGatesInLstmCell))
            || (step>=num_timestep && (blockIdx.x >= (gridDim.x / 2 + (step+1-num_timestep)* kNumGatesInLstmCell)));
        // GEMV
        if(cond_input){
            #pragma unroll
            for(int i=0; i<num_hidden; ++i){
                thread_output = thread_output + \
                    is_ptr[blockIdx.x / kNumGatePart * num_hidden + i] * \
                        shared_weight[i * kPartHidden + threadIdx.x];
            }
        }else if(cond_state){
            #pragma unroll
            for(int i=0; i<num_hidden; ++i){
                thread_output = thread_output + \
                    is_ptr[(blockIdx.x-gridDim.x/2) / kNumGatePart * num_hidden + i] * \
                        shared_weight[i * kPartHidden + threadIdx.x];
            }
        }
        // TODO(xiachunwei) Can be replaced by atomicAdd?
        // Store thread_output to global memory
        output_buffer[blockIdx.x * kPartHidden + threadIdx.x] = thread_output;
        __threadfence();
        grid.sync();
        
        // Each block compute a cell thus we need num_layer blocks
        if(cond_input && blockIdx.x % kNumGatesInLstmCell==0){// TODO(xiachunwei) Processing last part for wavefront
            // Let one block compute input_gate+state_gate+bias of the cell to reduce sync
            #pragma unroll
            for(int i=0; i<kNumGatesInLstmCell; ++i){
                output_buffer[(blockIdx.x + i) * kPartHidden + threadIdx.x] += \
                    output_buffer[((blockIdx.x+i) + gridDim.x / 2) * kPartHidden + threadIdx.x] + bias[((blockIdx.x+i) % 2) * kPartHidden + threadIdx.x];
            }

            float* i = output_buffer + (blockIdx.x / kNumGatesInLstmCell * 4 + 0) * num_hidden;
            float* j = output_buffer + (blockIdx.x / kNumGatesInLstmCell * 4 + 1) * num_hidden;
            float* f = output_buffer + (blockIdx.x / kNumGatesInLstmCell * 4 + 2) * num_hidden;
            float* o = output_buffer + (blockIdx.x / kNumGatesInLstmCell * 4 + 3) * num_hidden;
            c_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + threadIdx.x] = 
                c_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + threadIdx.x] * sigmoid(f[threadIdx.x] + 1.0) +
                sigmoid(i[threadIdx.x]) * tanh(j[threadIdx.x]);
            c_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + kPartHidden + threadIdx.x] = 
                c_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + kPartHidden + threadIdx.x] * sigmoid(f[kPartHidden + threadIdx.x] + 1.0) +
                sigmoid(i[kPartHidden + threadIdx.x]) * tanh(j[kPartHidden + threadIdx.x]);
            h_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + threadIdx.x] = 
                tanh(c_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + threadIdx.x]) * sigmoid(o[threadIdx.x]);
            h_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + kPartHidden + threadIdx.x] = 
                tanh(c_wavefront[blockIdx.x / kNumGatesInLstmCell * num_hidden + kPartHidden + threadIdx.x]) * sigmoid(o[kPartHidden + threadIdx.x]);
            
            if(blockIdx.x / kNumGatesInLstmCell < num_layer - 1){// Shift h to next layer
                input_wavefront[(blockIdx.x / kNumGatesInLstmCell + 1) * num_hidden + threadIdx.x] = h_wavefront[(blockIdx.x / kNumGatesInLstmCell) * num_hidden + threadIdx.x];
                input_wavefront[(blockIdx.x / kNumGatesInLstmCell + 1) * num_hidden + kPartHidden + threadIdx.x] = h_wavefront[(blockIdx.x / kNumGatesInLstmCell) * num_hidden + kPartHidden + threadIdx.x];
            }else if((step < num_timestep) && (blockIdx.x==0)){// Feed inputs[step] to input_wavefront
                input_wavefront[0*num_hidden + threadIdx.x] = inputs_timestep[step*num_hidden + threadIdx.x];
                input_wavefront[0*num_hidden + kPartHidden + threadIdx.x] = inputs_timestep[step*num_hidden + kPartHidden + threadIdx.x];
            }else if((step >= num_layer-1) && blockIdx.x / kNumGatesInLstmCell == num_layer - 1){// Feed last layer's h to output
                outputs_timestep[(step+1-num_layer)*num_hidden + threadIdx.x]=h_wavefront[(blockIdx.x / kNumGatesInLstmCell) * num_hidden + threadIdx.x];
                outputs_timestep[(step+1-num_layer)*num_hidden + kPartHidden + threadIdx.x]=h_wavefront[(blockIdx.x / kNumGatesInLstmCell) * num_hidden + kPartHidden + threadIdx.x];
            }
        }
    }
    // __threadfence();
    grid.sync();
}


// Each block compute num_hidden/kNumGatePart fma then all blocks sync
template<int batch, int num_layer, int num_hidden, int num_timestep>
    __global__ void __launch_bounds__(128) lstm_reuse_shared_memory_v2(float* inputs_timestep, float* outputs_timestep, 
    float* c_wavefront, float* h_wavefront, float* input_wavefront,
    float* weight_input_wavefront, float* weight_state_wavefront, float* bias,
    float* output_buffer){
    
    const int kNumInputGate = 4;
    const int kNumGatesInLstmCell = 8;
    const int kNumGatePart = gridDim.x / (num_layer*kNumGatesInLstmCell);
    const int kPartHidden = num_hidden / kNumGatePart;

    assert(gridDim.x % (num_layer*kNumGatePart)==0);
    assert(num_hidden % kNumGatePart == 0);
    assert(num_hidden * kPartHidden * sizeof(float) < 48 * 1024);
    if(blockIdx.x >= num_layer*kNumGatesInLstmCell*kNumGatePart){
        return;
    }
    
    extern float __shared__ shared_weight[];
    float * weight_ptr = NULL;
    float* is_ptr = NULL;

    // Feed inputs_timestep[0] to input_wavefront
    if(blockIdx.x < kNumGatePart){
        input_wavefront[blockIdx.x * kPartHidden + threadIdx.x]=inputs_timestep[0 * num_hidden + blockIdx.x * kPartHidden + threadIdx.x];
    }
    // Load weight to shared memory
    // Weight layout: [hidden_to_reduce(128), hidden_to_out(64)]
    if(blockIdx.x < gridDim.x / 2){
        weight_ptr=weight_input_wavefront;
        is_ptr = input_wavefront;
        #pragma unroll
        for(int i=0; i<kPartHidden; ++i){
            shared_weight[i*num_hidden + threadIdx.x] = weight_ptr[blockIdx.x / kNumGatePart * num_hidden * num_hidden + (blockIdx.x % kNumGatePart) * kPartHidden * num_hidden + i*num_hidden + threadIdx.x];
        }
    }else{
        weight_ptr = weight_state_wavefront;
        is_ptr = h_wavefront;
        #pragma unroll
        for(int i=0; i<kPartHidden; ++i){
            shared_weight[i*num_hidden + threadIdx.x] = weight_ptr[(blockIdx.x - gridDim.x / 2) / kNumGatePart * num_hidden * num_hidden + ((blockIdx.x - gridDim.x / 2) %  kNumGatePart) * kPartHidden * num_hidden + i*num_hidden + threadIdx.x];
        }
    }
    output_buffer[(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x] = 0;
    __syncthreads();
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    for(int step=0; step<num_timestep+num_layer-1; ++step){
        float thread_output = 0;
        // Blocks of first half gridDim.x compute the input gate
        bool cond_input = ((step<num_timestep && blockIdx.x < min(step+1, (int)num_layer) * kNumInputGate * kNumGatePart) 
            || (step>=num_timestep && blockIdx.x >= (step+1-num_timestep) * kNumInputGate * kNumGatePart)) && (blockIdx.x < gridDim.x / 2);
        // Blocks of second half gridDim.x compute the input gate
        bool cond_state = (((blockIdx.x >= gridDim.x / 2) && (blockIdx.x < gridDim.x / 2 + min(step+1, (int)num_layer) * kNumInputGate * kNumGatePart))
            || (step>=num_timestep && (blockIdx.x >= (gridDim.x / 2 + (step+1-num_timestep) * kNumInputGate * kNumGatePart)))) && (blockIdx.x >= gridDim.x / 2);
        
        // GEMV
        if(cond_input){
            #pragma unroll
            for(int i=0; i<kPartHidden; ++i){
                thread_output += is_ptr[blockIdx.x / (kNumInputGate * kNumGatePart) * num_hidden + (blockIdx.x % (kNumInputGate * kNumGatePart)) * kPartHidden + i] * shared_weight[i*num_hidden + threadIdx.x];
            }
        }else if(cond_state){
            #pragma unroll
            for(int i=0; i<kPartHidden; ++i){
                thread_output += is_ptr[((blockIdx.x - gridDim.x / 2) / (kNumInputGate * kNumGatePart)) * num_hidden + ((blockIdx.x - gridDim.x/2) % (kNumInputGate * kNumGatePart)) * kPartHidden + i] * shared_weight[i*num_hidden + threadIdx.x];
            }
        }
        
        if(cond_input || cond_state){
            // Store thread_output to global memory
            atomicAdd(output_buffer + ((blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x), thread_output);
            // __syncthreads();
        }
        
        grid.sync();
        
        // Each block compute a cell thus we need num_layer blocks
        //(step>=num_timestep && blockIdx.x==gridDim.x-(num_layer + num_timestep - step) * kNumInputGate))
        if(cond_input && (blockIdx.x % (kNumInputGate * kNumGatePart)==0 || false)){// TODO(xiachunwei) Processing last part for wavefront
            // Let one block compute input_gate+state_gate+bias of the cell to reduce sync
            #pragma unroll
            for(int i=0; i<kNumInputGate; ++i){
                output_buffer[(blockIdx.x / kNumGatePart + i) * num_hidden + threadIdx.x] += \
                    output_buffer[((blockIdx.x + gridDim.x / 2) / kNumGatePart + i) * num_hidden + threadIdx.x] + bias[(blockIdx.x / (kNumInputGate * kNumGatePart)) + threadIdx.x];
            }

            float* i = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 0) * num_hidden;
            float* j = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 1) * num_hidden;
            float* f = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 2) * num_hidden;
            float* o = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 3) * num_hidden;
            c_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x] = 
                c_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x] * sigmoid(f[threadIdx.x] + 1.0) +
                sigmoid(i[threadIdx.x]) * tanh(j[threadIdx.x]);
            
            h_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x] = 
                tanh(c_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x]) * sigmoid(o[threadIdx.x]);
            
            if(blockIdx.x / (kNumInputGate * kNumGatePart) < num_layer - 1){// Shift h to next layer
                input_wavefront[(blockIdx.x / (kNumInputGate * kNumGatePart) + 1) * num_hidden + threadIdx.x] = h_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x];
            }else if((step < num_timestep) && (blockIdx.x==0)){// Feed inputs[step] to input_wavefront
                input_wavefront[0*num_hidden + threadIdx.x] = inputs_timestep[step*num_hidden + threadIdx.x];
            }else if((step >= num_layer-1) && blockIdx.x / (kNumInputGate * kNumGatePart) == num_layer - 1){// Feed last layer's h to output
                outputs_timestep[(step+1-num_layer)*num_hidden + threadIdx.x]=h_wavefront[(blockIdx.x / (kNumInputGate * kNumGatePart)) * num_hidden + threadIdx.x];
            }
            // if(step>= num_timestep && threadIdx.x==0){
            //     printf("step: %d\n", step);
            //     printf("blockIdx.x %d\n", blockIdx.x);
            // }
        }
        output_buffer[(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x] = 0;
        // __threadfence();
        grid.sync();
    }
}


// Each block compute num_hidden/kNumGatePart fma then all blocks sync
// Use atomicAdd to build spin lock to sync between blocks
// Use memory space starts from `output_buffer + gridDim.x * num_hidden` to do sync
// in this way we can save a parameter
template<int batch, int num_layer, int num_hidden, int num_timestep>
    __global__ void lstm_reuse_shared_memory_v3(float* inputs_timestep, float* outputs_timestep, 
    float* c_wavefront, float* h_wavefront, float* input_wavefront,
    float* weight_input_wavefront, float* weight_state_wavefront, float* bias,
    float* output_buffer, int* arr_sync){
    
    const int kNumInputGate = 4;
    const int kNumGatesInLstmCell = 8;
    const int kNumGatePart = gridDim.x / (num_layer*kNumGatesInLstmCell);
    const int kPartHidden = num_hidden / kNumGatePart;

    assert(gridDim.x % (num_layer*kNumGatePart)==0);
    assert(num_hidden % kNumGatePart == 0);
    assert(num_hidden * kPartHidden * sizeof(float) < 48 * 1024);
    if(blockIdx.x >= num_layer*kNumGatesInLstmCell*kNumGatePart){
        return;
    }
    
    extern float __shared__ shared_weight[];
    float * weight_ptr = NULL;
    float* is_ptr = NULL;

    // Feed inputs_timestep[0] to input_wavefront
    if(blockIdx.x < kNumGatePart){
        input_wavefront[blockIdx.x * kPartHidden + threadIdx.x]=inputs_timestep[0 * num_hidden + blockIdx.x * kPartHidden + threadIdx.x];
    }
    // Load weight to shared memory
    // Weight layout: [hidden_to_reduce(128), hidden_to_out(64)]
    if(blockIdx.x < gridDim.x / 2){
        weight_ptr=weight_input_wavefront;
        is_ptr = input_wavefront;
        #pragma unroll
        for(int i=0; i<kPartHidden; ++i){
            shared_weight[i*num_hidden + threadIdx.x] = weight_ptr[blockIdx.x / kNumGatePart * num_hidden * num_hidden + (blockIdx.x % kNumGatePart) * kPartHidden * num_hidden + i*num_hidden + threadIdx.x];
        }
    }else{
        weight_ptr = weight_state_wavefront;
        is_ptr = h_wavefront;
        #pragma unroll
        for(int i=0; i<kPartHidden; ++i){
            shared_weight[i*num_hidden + threadIdx.x] = weight_ptr[(blockIdx.x - gridDim.x / 2) / kNumGatePart * num_hidden * num_hidden + ((blockIdx.x - gridDim.x / 2) %  kNumGatePart) * kPartHidden * num_hidden + i*num_hidden + threadIdx.x];
        }
    }
    output_buffer[(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x] = 0;
    __syncthreads();
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    for(int step=0; step<num_timestep+num_layer-1; ++step){
        float thread_output = 0;
        // Blocks of first half gridDim.x compute the input gate
        bool cond_input = ((step<num_timestep && blockIdx.x < min(step+1, (int)num_layer) * kNumInputGate * kNumGatePart) 
            || (step>=num_timestep && blockIdx.x >= (step+1-num_timestep) * kNumInputGate * kNumGatePart)) && (blockIdx.x < gridDim.x / 2);
        // Blocks of second half gridDim.x compute the input gate
        bool cond_state = (((blockIdx.x >= gridDim.x / 2) && (blockIdx.x < gridDim.x / 2 + min(step+1, (int)num_layer) * kNumInputGate * kNumGatePart))
            || (step>=num_timestep && (blockIdx.x >= (gridDim.x / 2 + (step+1-num_timestep) * kNumInputGate * kNumGatePart)))) && (blockIdx.x >= gridDim.x / 2);
        
        // GEMV
        if(cond_input){
            #pragma unroll
            for(int i=0; i<kPartHidden; ++i){
                thread_output += is_ptr[blockIdx.x / (kNumInputGate * kNumGatePart) * num_hidden + threadIdx.x] * shared_weight[i*num_hidden + threadIdx.x];
            }
        }else if(cond_state){
            #pragma unroll
            for(int i=0; i<kPartHidden; ++i){
                thread_output += is_ptr[((blockIdx.x - gridDim.x / 2) / (kNumInputGate * kNumGatePart)) * num_hidden + threadIdx.x] * shared_weight[i*num_hidden + threadIdx.x];
            }
        }
        
        
        // Store thread_output to global memory
        atomicAdd(output_buffer + ((blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x), thread_output);
        //printf("before step %d blockIdx.x %d threadIdx.x %d spin, value %d\n", step, blockIdx.x, threadIdx.x, arr_sync[(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x]);
         atomicAdd(&arr_sync[(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x], 1);
    //
       // arr_sync[(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x] = 100;
        // printf("blockIdx.x %d, threadIdx.x %d add 1, now is %d, kNumGatePart %d\n", blockIdx.x, threadIdx.x, arr_sync[(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x], kNumGatePart);
        // __syncthreads();
        __threadfence();
        //grid.sync();
        int count = 0;
        while(arr_sync[(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x] != kNumGatePart){
            // Spin here
            if(arr_sync[(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x]==0)
            printf("after step %d blockIdx.x %d threadIdx.x %d spin, value %d\n", step, blockIdx.x, threadIdx.x, arr_sync[(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x]);
            if(count++>10){
            //    break;
            }
        }
        
        
        
        grid.sync();
        
        // Each block compute a cell thus we need num_layer blocks
        if(cond_input && (blockIdx.x % (kNumInputGate * kNumGatePart)==0)){
            // Let one block compute input_gate+state_gate+bias of the cell to reduce sync
            #pragma unroll
            for(int i=0; i<kNumInputGate; ++i){
                output_buffer[(blockIdx.x / kNumGatePart + i) * num_hidden + threadIdx.x] += \
                    output_buffer[((blockIdx.x + gridDim.x / 2) / kNumGatePart + i) * num_hidden + threadIdx.x] + bias[(blockIdx.x / (kNumInputGate * kNumGatePart)) + threadIdx.x];
            }

            float* i = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 0) * num_hidden;
            float* j = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 1) * num_hidden;
            float* f = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 2) * num_hidden;
            float* o = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 3) * num_hidden;
            c_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x] = 
                c_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x] * sigmoid(f[threadIdx.x] + 1.0) +
                sigmoid(i[threadIdx.x]) * tanh(j[threadIdx.x]);
            
            h_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x] = 
                tanh(c_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x]) * sigmoid(o[threadIdx.x]);
            
            if(blockIdx.x / (kNumInputGate * kNumGatePart) < num_layer - 1){// Shift h to next layer
                input_wavefront[(blockIdx.x / (kNumInputGate * kNumGatePart) + 1) * num_hidden + threadIdx.x] = h_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x];
            }else if((step < num_timestep) && (blockIdx.x==0)){// Feed inputs[step] to input_wavefront
                input_wavefront[0*num_hidden + threadIdx.x] = inputs_timestep[step*num_hidden + threadIdx.x];
            }else if((step >= num_layer-1) && blockIdx.x / (kNumInputGate * kNumGatePart) == num_layer - 1){// Feed last layer's h to output
                outputs_timestep[(step+1-num_layer)*num_hidden + threadIdx.x]=h_wavefront[(blockIdx.x / (kNumInputGate * kNumGatePart)) * num_hidden + threadIdx.x];
            }
            // if(step>= num_timestep && threadIdx.x==0){
            //     printf("step: %d\n", step);
            //     printf("blockIdx.x %d\n", blockIdx.x);
            // }
        }
        output_buffer[(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x] = 0;
        // arr_sync[(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x]=0;
        __threadfence();
        grid.sync();
    }
}


// Optimize for hidden size 256, gridDim(256, 1, 1), blockDim(256, 1, 1)
// Using shared memory to store part of the weights
// We set kPartHidden=4, each block somputes [64*256], 
// in which 32*256 elements are in shared memory and 32*256 elements are in global memory
template<int batch, int num_layer, int num_hidden, int num_timestep>
    __global__ void lstm_reuse_shared_memory_v4(float* inputs_timestep, float* outputs_timestep, 
    float* c_wavefront, float* h_wavefront, float* input_wavefront,
    float* weight_input_wavefront, float* weight_state_wavefront, float* bias,
    float* output_buffer){
    
    const int kNumInputGate = 4;
    const int kNumGatesInLstmCell = 8;
    const int kNumGatePart = gridDim.x / (num_layer*kNumGatesInLstmCell);
    const int kPartHidden = num_hidden / kNumGatePart;
    const int kShared = 48; // Note, only for hidden size 256
    const int kGlobal = num_hidden/kNumGatePart - kShared; // Note, only for hidden size 256

    assert(gridDim.x % (num_layer*kNumGatePart)==0);
    assert(num_hidden % kNumGatePart == 0);
    assert(num_hidden * kShared * sizeof(float) <= 48 * 1024);
    if(blockIdx.x >= num_layer*kNumGatesInLstmCell*kNumGatePart){
        return;
    }
    
    extern float __shared__ shared_weight[];
    float thread_output[2];
    float * weight_ptr = NULL;

    // Feed inputs_timestep[0] to input_wavefront
    if(blockIdx.x < kNumGatePart){
        // for(int b=0; b<1; ++b){
            input_wavefront[blockIdx.x * kPartHidden + threadIdx.x]=inputs_timestep[0 * num_hidden + blockIdx.x * kPartHidden + threadIdx.x];
        // }
    }
    // Load weight to shared memory
    // Weight layout: [hidden_to_reduce(128), hidden_to_out(64)]
    if(blockIdx.x < gridDim.x / 2){
        weight_ptr=weight_input_wavefront;
        // is_ptr = input_wavefront;
        #pragma unroll
        for(int i=0; i<kShared; ++i){
            #pragma unroll
            for(int b=0; b<1; ++b){
                shared_weight[i*num_hidden + b*blockDim.x + threadIdx.x] = weight_ptr[blockIdx.x / kNumGatePart * num_hidden * num_hidden + (blockIdx.x % kNumGatePart) * kPartHidden * num_hidden + i*num_hidden + b*blockDim.x + threadIdx.x];
            }
        }
    }else{
        weight_ptr = weight_state_wavefront;
        // is_ptr = h_wavefront;
        #pragma unroll
        for(int i=0; i<kShared; ++i){
            #pragma unroll
            for(int b=0; b<1; ++b){
                shared_weight[i*num_hidden + b*blockDim.x + threadIdx.x] = weight_ptr[(blockIdx.x - gridDim.x / 2) / kNumGatePart * num_hidden * num_hidden + ((blockIdx.x - gridDim.x / 2) %  kNumGatePart) * kPartHidden * num_hidden + i*num_hidden + b*blockDim.x + threadIdx.x];
            }
        }
    }
    for(int b=0; b<1; ++b){
        output_buffer[(blockIdx.x / kNumGatePart) * num_hidden + b*blockDim.x + threadIdx.x] = 0;
    }
    __syncthreads();
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    for(int step=0; step<num_timestep+num_layer-1; ++step){
        for(int b=0; b<1; ++b){
            thread_output[b] = 0;
        }
        // Blocks of first half gridDim.x compute the input gate
        bool cond_input = ((step<num_timestep && blockIdx.x < min(step+1, (int)num_layer) * kNumInputGate * kNumGatePart) 
            || (step>=num_timestep && blockIdx.x >= (step+1-num_timestep) * kNumInputGate * kNumGatePart)) && (blockIdx.x < gridDim.x / 2);
        // Blocks of second half gridDim.x compute the input gate
        bool cond_state = (((blockIdx.x >= gridDim.x / 2) && (blockIdx.x < gridDim.x / 2 + min(step+1, (int)num_layer) * kNumInputGate * kNumGatePart))
            || (step>=num_timestep && (blockIdx.x >= (gridDim.x / 2 + (step+1-num_timestep) * kNumInputGate * kNumGatePart)))) && (blockIdx.x >= gridDim.x / 2);
        
        // GEMV
        if(cond_input){
            // Reduce shared memory part
            #pragma unroll
            for(int i=0; i<kShared; ++i){
                #pragma unroll
                for(int b=0; b<1; ++b){
                    thread_output[b] += input_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) * num_hidden + (blockIdx.x % (kNumInputGate * kNumGatePart)) * kPartHidden + i] * shared_weight[i*num_hidden + b*blockDim.x + threadIdx.x];
                }
            }
            // Reduce global memory part
            #pragma unroll
            for(int i=0; i<kGlobal; ++i){
                #pragma unroll
                for(int b=0; b<1; ++b){
                    thread_output[b] += input_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) * num_hidden + (blockIdx.x % (kNumInputGate * kNumGatePart)) * kPartHidden + kShared + i] * 
                        weight_input_wavefront[blockIdx.x / kNumGatePart * num_hidden * num_hidden + ((blockIdx.x % kNumGatePart) * kPartHidden + kShared + i) * num_hidden + b*blockDim.x + threadIdx.x];
                }
            }
        }else if(cond_state){
            // Reduce shared memory part
            #pragma unroll
            for(int i=0; i<kShared; ++i){
                #pragma unroll
                for(int b=0; b<1; ++b){
                    thread_output[b] += h_wavefront[((blockIdx.x - gridDim.x / 2) / (kNumInputGate * kNumGatePart)) * num_hidden + ((blockIdx.x - gridDim.x / 2) % (kNumInputGate * kNumGatePart)) * kPartHidden + i] * shared_weight[i*num_hidden + b*blockDim.x + threadIdx.x];
                }
            }
            #pragma unroll
            for(int i=0; i<kGlobal; ++i){
                #pragma unroll
                for(int b=0; b<1; ++b){
                    thread_output[b] += h_wavefront[((blockIdx.x - gridDim.x / 2) / (kNumInputGate * kNumGatePart)) * num_hidden + ((blockIdx.x - gridDim.x / 2) % (kNumInputGate * kNumGatePart)) * kPartHidden + kShared + i] * 
                        weight_state_wavefront[(blockIdx.x - gridDim.x / 2) / kNumGatePart * num_hidden * num_hidden + (((blockIdx.x - gridDim.x / 2) % kNumGatePart) * kPartHidden + kShared + i) * num_hidden + b*blockDim.x + threadIdx.x];
                }
            }
        }
        
        atomicAdd(output_buffer + ((blockIdx.x / kNumGatePart) * num_hidden + (blockIdx.x % kNumGatePart) * kPartHidden + 0*blockDim.x + threadIdx.x), thread_output[0]);
        
        __threadfence();
        grid.sync();
        
        // Each block compute a cell thus we need num_layer blocks
        if(cond_input && (blockIdx.x % (kNumInputGate * kNumGatePart)==0)){
            // Let one block compute input_gate+state_gate+bias of the cell to reduce sync
            #pragma unroll
            for(int i=0; i<kNumInputGate; ++i){
                output_buffer[(blockIdx.x / kNumGatePart + i) * num_hidden + threadIdx.x] += \
                    output_buffer[((blockIdx.x + gridDim.x / 2) / kNumGatePart + i) * num_hidden + threadIdx.x] + bias[(blockIdx.x / (kNumInputGate * kNumGatePart)) + threadIdx.x];
            }

            float* i = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 0) * num_hidden;
            float* j = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 1) * num_hidden;
            float* f = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 2) * num_hidden;
            float* o = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 3) * num_hidden;
            c_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x] = 
                c_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x] * sigmoid(f[threadIdx.x] + 1.0) +
                sigmoid(i[threadIdx.x]) * tanh(j[threadIdx.x]);
            
            h_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x] = 
                tanh(c_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x]) * sigmoid(o[threadIdx.x]);
            
            if(blockIdx.x / (kNumInputGate * kNumGatePart) < num_layer - 1){// Shift h to next layer
                input_wavefront[(blockIdx.x / (kNumInputGate * kNumGatePart) + 1) * num_hidden + threadIdx.x] = h_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x];
            }else if((step < num_timestep) && (blockIdx.x==0)){// Feed inputs[step] to input_wavefront
                input_wavefront[0*num_hidden + threadIdx.x] = inputs_timestep[step*num_hidden + threadIdx.x];
            }else if((step >= num_layer-1) && blockIdx.x / (kNumInputGate * kNumGatePart) == num_layer - 1){// Feed last layer's h to output
                outputs_timestep[(step+1-num_layer)*num_hidden + threadIdx.x]=h_wavefront[(blockIdx.x / (kNumInputGate * kNumGatePart)) * num_hidden + threadIdx.x];
            }
            // if(step>= num_timestep && threadIdx.x==0){
            //     printf("step: %d\n", step);
            //     printf("blockIdx.x %d\n", blockIdx.x);
            // }
        }
        output_buffer[(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x] = 0;
        __threadfence();
        grid.sync();
    }
}


// blockDim(256,1,1)
template<int batch, int num_layer, int num_hidden, int num_timestep>
    __global__ void __launch_bounds__(256, 3) lstm_reuse_shared_memory_v5(float* inputs_timestep, float* outputs_timestep, 
    float* c_wavefront, float* h_wavefront, float* input_wavefront,
    float* weight_input_wavefront, float* weight_state_wavefront, float* bias,
    float* output_buffer){
    
    const int kNumInputGate = 4;
    const int kNumGatesInLstmCell = 8;
    const int kNumGatePart = gridDim.x / (num_layer*kNumGatesInLstmCell);
    const int kPartHidden = num_hidden / kNumGatePart;
    const int kShared = 48; // Note, only for hidden size 256
    const int kGlobal = num_hidden/kNumGatePart - kShared; // Note, only for hidden size 256

    assert(gridDim.x % (num_layer*kNumGatePart)==0);
    assert(num_hidden % kNumGatePart == 0);
    assert(num_hidden * kShared * sizeof(float) <= 48 * 1024);
    if(blockIdx.x >= num_layer*kNumGatesInLstmCell*kNumGatePart){
        return;
    }
    
    extern float __shared__ shared_weight[];
    float thread_output[1];
    float * weight_ptr = NULL;
    float* is_ptr = NULL;

    // Feed inputs_timestep[0] to input_wavefront
    if(blockIdx.x < kNumGatePart){
        input_wavefront[blockIdx.x * kPartHidden + threadIdx.x]=inputs_timestep[0 * num_hidden + blockIdx.x * kPartHidden + threadIdx.x];
    }
    // Load weight to shared memory
    // Weight layout: [hidden_to_reduce(128), hidden_to_out(64)]
    if(blockIdx.x < gridDim.x / 2){
        weight_ptr=weight_input_wavefront;
        // is_ptr = input_wavefront;
        #pragma unroll
        for(int i=0; i<kShared; ++i){
            shared_weight[i*num_hidden + threadIdx.x] = weight_ptr[blockIdx.x / kNumGatePart * num_hidden * num_hidden + (blockIdx.x % kNumGatePart) * kPartHidden * num_hidden + i*num_hidden + threadIdx.x];
        }
    }else{
        weight_ptr = weight_state_wavefront;
        // is_ptr = h_wavefront;
        #pragma unroll
        for(int i=0; i<kShared; ++i){
            shared_weight[i*num_hidden + threadIdx.x] = weight_ptr[(blockIdx.x - gridDim.x / 2) / kNumGatePart * num_hidden * num_hidden + ((blockIdx.x - gridDim.x / 2) %  kNumGatePart) * kPartHidden * num_hidden + i*num_hidden + threadIdx.x];
        }
    }
    output_buffer[(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x] = 0;
    __syncthreads();
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    
    for(int step=0; step<num_timestep+num_layer-1; ++step){
        thread_output[0] = 0;
        // Blocks of first half gridDim.x compute the input gate
        bool cond_input = ((step<num_timestep && blockIdx.x < min(step+1, (int)num_layer) * kNumInputGate * kNumGatePart) 
            || (step>=num_timestep && blockIdx.x >= (step+1-num_timestep) * kNumInputGate * kNumGatePart)) && (blockIdx.x < gridDim.x / 2);
        // Blocks of second half gridDim.x compute the input gate
        bool cond_state = (((blockIdx.x >= gridDim.x / 2) && (blockIdx.x < gridDim.x / 2 + min(step+1, (int)num_layer) * kNumInputGate * kNumGatePart))
            || (step>=num_timestep && (blockIdx.x >= (gridDim.x / 2 + (step+1-num_timestep) * kNumInputGate * kNumGatePart)))) && (blockIdx.x >= gridDim.x / 2);
        
        // GEMV
        if(cond_input){
            // Reduce shared memory part
            #pragma unroll
            for(int i=0; i<kShared; ++i){
                thread_output[0] += input_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) * num_hidden + (blockIdx.x % (kNumInputGate * kNumGatePart)) * kPartHidden + i] * shared_weight[i*num_hidden + threadIdx.x];
            }
            // Reduce global memory part
            #pragma unroll
            for(int i=0; i<kGlobal; ++i){
                thread_output[0] += input_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) * num_hidden + (blockIdx.x % (kNumInputGate * kNumGatePart)) * kPartHidden + kShared + i] * 
                    weight_input_wavefront[blockIdx.x / kNumGatePart * num_hidden * num_hidden + ((blockIdx.x % kNumGatePart) * kPartHidden + kShared + i) * num_hidden + threadIdx.x];
            }
        }else if(cond_state){
            // Reduce shared memory part
            #pragma unroll
            for(int i=0; i<kShared; ++i){
                thread_output[0] += h_wavefront[((blockIdx.x - gridDim.x / 2) / (kNumInputGate * kNumGatePart)) * num_hidden + ((blockIdx.x - gridDim.x / 2) % (kNumInputGate * kNumGatePart)) * kPartHidden + i] * shared_weight[i*num_hidden + threadIdx.x];
            }
            #pragma unroll
            for(int i=0; i<kGlobal; ++i){
                thread_output[0] += h_wavefront[((blockIdx.x - gridDim.x / 2) / (kNumInputGate * kNumGatePart)) * num_hidden + ((blockIdx.x - gridDim.x / 2) % (kNumInputGate * kNumGatePart)) * kPartHidden + kShared + i] * 
                    weight_state_wavefront[(blockIdx.x - gridDim.x / 2) / kNumGatePart * num_hidden * num_hidden + (((blockIdx.x - gridDim.x / 2) % kNumGatePart) * kPartHidden + kShared + i) * num_hidden + threadIdx.x];
            }
        }
        atomicAdd(output_buffer + ((blockIdx.x / kNumGatePart) * num_hidden + (blockIdx.x % kNumGatePart) * kPartHidden + threadIdx.x), thread_output[0]);
        
        __threadfence();
        // grid.sync();
        // continue;
        // Each block compute a cell thus we need num_layer blocks
        if(cond_input && (blockIdx.x % (kNumInputGate * kNumGatePart)==0)){
            // Let one block compute input_gate+state_gate+bias of the cell to reduce sync
            // TODO(xiachunwei) Account for 0.22ms latency in the end-to-end latency, need to improve
            #pragma unroll
            for(int i=0; i<kNumInputGate; ++i){
                output_buffer[(blockIdx.x / kNumGatePart + i) * num_hidden + threadIdx.x] += \
                    output_buffer[((blockIdx.x + gridDim.x / 2) / kNumGatePart + i) * num_hidden + threadIdx.x] + bias[(blockIdx.x / (kNumInputGate * kNumGatePart)) + threadIdx.x];
            }

            float* i = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 0) * num_hidden;
            float* j = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 1) * num_hidden;
            float* f = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 2) * num_hidden;
            float* o = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 3) * num_hidden;
            c_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x] = 
                c_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x] * sigmoid(f[threadIdx.x] + 1.0) +
                sigmoid(i[threadIdx.x]) * tanh(j[threadIdx.x]);
            
            h_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x] = 
                tanh(c_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x]) * sigmoid(o[threadIdx.x]);
            
            if(blockIdx.x / (kNumInputGate * kNumGatePart) < num_layer - 1){// Shift h to next layer
                input_wavefront[(blockIdx.x / (kNumInputGate * kNumGatePart) + 1) * num_hidden + threadIdx.x] = h_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x];
            }else if((step < num_timestep) && (blockIdx.x==0)){// Feed inputs[step] to input_wavefront
                input_wavefront[0*num_hidden + threadIdx.x] = inputs_timestep[step*num_hidden + threadIdx.x];
            }else if((step >= num_layer-1) && blockIdx.x / (kNumInputGate * kNumGatePart) == num_layer - 1){// Feed last layer's h to output
                outputs_timestep[(step+1-num_layer)*num_hidden + threadIdx.x]=h_wavefront[(blockIdx.x / (kNumInputGate * kNumGatePart)) * num_hidden + threadIdx.x];
            }
            // if(step>= num_timestep && threadIdx.x==0){
            //     printf("step: %d\n", step);
            //     printf("blockIdx.x %d\n", blockIdx.x);
            // }
        }
        output_buffer[(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x] = 0;
        __threadfence();
        // grid.sync();
    }
}


// blockDim(256,1,1)
// Using register file to hold weight
template<int batch, int num_layer, int num_hidden, int num_timestep>
    __global__ void __launch_bounds__(256, 3) lstm_reuse_shared_memory_v6(float* inputs_timestep, float* outputs_timestep, 
    float* c_wavefront, float* h_wavefront, float* input_wavefront,
    float* weight_input_wavefront, float* weight_state_wavefront, float* bias,
    float* output_buffer){
    
    const int kNumInputGate = 4;
    const int kNumGatesInLstmCell = 8;
    const int kNumGatePart = gridDim.x / (num_layer*kNumGatesInLstmCell);
    const int kPartHidden = num_hidden / kNumGatePart;
    const int kShared = 48; // Note, only for hidden size 256
    const int kGlobal = num_hidden/kNumGatePart - kShared; // Note, only for hidden size 256

    assert(gridDim.x % (num_layer*kNumGatePart)==0);
    assert(num_hidden % kNumGatePart == 0);
    assert(num_hidden * kShared * sizeof(float) <= 48 * 1024);
    if(blockIdx.x >= num_layer*kNumGatesInLstmCell*kNumGatePart){
        return;
    }
    
    extern float __shared__ shared_weight[];
    float register_weight[16];
    float thread_output[1];
    float * weight_ptr = NULL;

    // Feed inputs_timestep[0] to input_wavefront
    if(blockIdx.x < kNumGatePart){
        input_wavefront[blockIdx.x * kPartHidden + threadIdx.x]=inputs_timestep[0 * num_hidden + blockIdx.x * kPartHidden + threadIdx.x];
    }
    // Load weight to shared memory
    // Weight layout: [hidden_to_reduce(128), hidden_to_out(64)]
    if(blockIdx.x < gridDim.x / 2){
        weight_ptr=weight_input_wavefront;
        // is_ptr = input_wavefront;
        #pragma unroll
        for(int i=0; i<kShared; ++i){
            shared_weight[i*num_hidden + threadIdx.x] = weight_ptr[blockIdx.x / kNumGatePart * num_hidden * num_hidden + (blockIdx.x % kNumGatePart) * kPartHidden * num_hidden + i*num_hidden + threadIdx.x];
        }
        #pragma unroll
        for(int i=0; i<kGlobal; ++i){
            register_weight[i] = weight_ptr[blockIdx.x / kNumGatePart * num_hidden * num_hidden + (blockIdx.x % kNumGatePart) * kPartHidden * num_hidden + (i+kShared)*num_hidden + threadIdx.x];
        }
    }else{
        weight_ptr = weight_state_wavefront;
        // is_ptr = h_wavefront;
        #pragma unroll
        for(int i=0; i<kShared; ++i){
            shared_weight[i*num_hidden + threadIdx.x] = weight_ptr[(blockIdx.x - gridDim.x / 2) / kNumGatePart * num_hidden * num_hidden + ((blockIdx.x - gridDim.x / 2) %  kNumGatePart) * kPartHidden * num_hidden + i*num_hidden + threadIdx.x];
        }
        #pragma unroll
        for(int i=0; i<kGlobal; ++i){
            register_weight[i] = weight_ptr[(blockIdx.x - gridDim.x / 2) / kNumGatePart * num_hidden * num_hidden + ((blockIdx.x - gridDim.x / 2) % kNumGatePart) * kPartHidden * num_hidden + (i+kShared)*num_hidden + threadIdx.x];
        }
    }
    output_buffer[(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x] = 0;
    __syncthreads();
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    
    for(int step=0; step<num_timestep+num_layer-1; ++step){
        thread_output[0] = 0;
        // Blocks of first half gridDim.x compute the input gate
        bool cond_input = ((step<num_timestep && blockIdx.x < min(step+1, (int)num_layer) * kNumInputGate * kNumGatePart) 
            || (step>=num_timestep && blockIdx.x >= (step+1-num_timestep) * kNumInputGate * kNumGatePart)) && (blockIdx.x < gridDim.x / 2);
        // Blocks of second half gridDim.x compute the input gate
        bool cond_state = (((blockIdx.x >= gridDim.x / 2) && (blockIdx.x < gridDim.x / 2 + min(step+1, (int)num_layer) * kNumInputGate * kNumGatePart))
            || (step>=num_timestep && (blockIdx.x >= (gridDim.x / 2 + (step+1-num_timestep) * kNumInputGate * kNumGatePart)))) && (blockIdx.x >= gridDim.x / 2);
        
        // GEMV
        if(cond_input){
            // Reduce shared memory part
            #pragma unroll
            for(int i=0; i<kShared; ++i){
                thread_output[0] += input_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) * num_hidden + (blockIdx.x % (kNumInputGate * kNumGatePart)) * kPartHidden + i] * shared_weight[i*num_hidden + threadIdx.x];
            }
            // Reduce global memory part
            #pragma unroll
            for(int i=0; i<kGlobal; ++i){
                thread_output[0] += input_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) * num_hidden + (blockIdx.x % (kNumInputGate * kNumGatePart)) * kPartHidden + kShared + i] * register_weight[i];
            }
        }else if(cond_state){
            // Reduce shared memory part
            #pragma unroll
            for(int i=0; i<kShared; ++i){
                thread_output[0] += h_wavefront[((blockIdx.x - gridDim.x / 2) / (kNumInputGate * kNumGatePart)) * num_hidden + ((blockIdx.x - gridDim.x / 2) % (kNumInputGate * kNumGatePart)) * kPartHidden + i] * shared_weight[i*num_hidden + threadIdx.x];
            }
            #pragma unroll
            for(int i=0; i<kGlobal; ++i){
                thread_output[0] += h_wavefront[((blockIdx.x - gridDim.x / 2) / (kNumInputGate * kNumGatePart)) * num_hidden + ((blockIdx.x - gridDim.x / 2) % (kNumInputGate * kNumGatePart)) * kPartHidden + kShared + i]  * register_weight[i];
            }
        }
        atomicAdd(output_buffer + ((blockIdx.x / kNumGatePart) * num_hidden + (blockIdx.x % kNumGatePart) * kPartHidden + threadIdx.x), thread_output[0]);
        
        // __threadfence();
        // grid.sync();
        // continue;
        // Each block compute a cell thus we need num_layer blocks
        if(cond_input && (blockIdx.x % (kNumInputGate * kNumGatePart)==0)){
            // Let one block compute input_gate+state_gate+bias of the cell to reduce sync
            // TODO(xiachunwei) Account for 0.22ms latency in the end-to-end latency, need to improve
            #pragma unroll
            for(int i=0; i<kNumInputGate; ++i){
                output_buffer[(blockIdx.x / kNumGatePart + i) * num_hidden + threadIdx.x] += \
                    output_buffer[((blockIdx.x + gridDim.x / 2) / kNumGatePart + i) * num_hidden + threadIdx.x] + bias[(blockIdx.x / (kNumInputGate * kNumGatePart)) + threadIdx.x];
            }

            float* i = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 0) * num_hidden;
            float* j = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 1) * num_hidden;
            float* f = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 2) * num_hidden;
            float* o = output_buffer + (blockIdx.x / (kNumInputGate * kNumGatePart) + 3) * num_hidden;
            c_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x] = 
                c_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x] * sigmoid(f[threadIdx.x] + 1.0) +
                sigmoid(i[threadIdx.x]) * tanh(j[threadIdx.x]);
            
            h_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x] = 
                tanh(c_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x]) * sigmoid(o[threadIdx.x]);
            
            if(blockIdx.x / (kNumInputGate * kNumGatePart) < num_layer - 1){// Shift h to next layer
                input_wavefront[(blockIdx.x / (kNumInputGate * kNumGatePart) + 1) * num_hidden + threadIdx.x] = h_wavefront[blockIdx.x / (kNumInputGate * kNumGatePart) + threadIdx.x];
            }else if((step < num_timestep) && (blockIdx.x==0)){// Feed inputs[step] to input_wavefront
                input_wavefront[0*num_hidden + threadIdx.x] = inputs_timestep[step*num_hidden + threadIdx.x];
            }else if((step >= num_layer-1) && blockIdx.x / (kNumInputGate * kNumGatePart) == num_layer - 1){// Feed last layer's h to output
                outputs_timestep[(step+1-num_layer)*num_hidden + threadIdx.x]=h_wavefront[(blockIdx.x / (kNumInputGate * kNumGatePart)) * num_hidden + threadIdx.x];
            }
            // if(step>= num_timestep && threadIdx.x==0){
            //     printf("step: %d\n", step);
            //     printf("blockIdx.x %d\n", blockIdx.x);
            // }
        }
        output_buffer[(blockIdx.x / kNumGatePart) * num_hidden + threadIdx.x] = 0;
        // __threadfence();
        grid.sync();
    }
}


// blockDim(32,4,1), gridDim(640)
// ERROR! too many blocks in cooperative launch
template<int batch, int num_layer, int num_hidden, int num_timestep>
    __global__ void __launch_bounds__(128, 6) lstm_reuse_shared_memory_v7(float* inputs_timestep, float* outputs_timestep, 
    float* c_wavefront, float* h_wavefront, float* input_wavefront,
    float* weight_input_wavefront, float* weight_state_wavefront, float* bias,
    float* output_buffer){
    
    const int kNumInputGate = 4;    
    const int kNumGatesInLstmCell = 8;
    const int kNumRow = 4;
    const int kNumThreadIter = num_hidden / blockDim.x;
    const int kNumBlockPerCell = num_hidden / kNumRow;// 256/4 = 64

    extern float __shared__ shared_input_weight[];//4*4*256
    extern float __shared__ shared_state_weight[];//4*4*256
    float input_local_sum[kNumInputGate];
    float state_local_sum[kNumInputGate];
    
    if(threadIdx.y * blockDim.x + threadIdx.x >= 256){
        return;
    }
    // Load weight to shared memory
    for(int i=0; i<kNumInputGate; ++i){
        #pragma unroll
        for(int j=0; j<kNumThreadIter; ++j){
            shared_input_weight[i * kNumRow * num_hidden + threadIdx.y * num_hidden + j*blockDim.x + threadIdx.x] = \
                weight_input_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + i * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + j*blockDim.x + threadIdx.x];
        }
    }
    for(int i=0; i<kNumInputGate; ++i){
        #pragma unroll
        for(int j=0; j<kNumThreadIter; ++j){
            shared_state_weight[i * kNumRow * num_hidden + threadIdx.y * num_hidden + j*blockDim.x + threadIdx.x] = \
                weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + i * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + j*blockDim.x + threadIdx.x];
        }
    }
    __syncthreads();
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    for(int step=0; step<num_timestep; ++step){
        bool block_cond = (blockIdx.x < min(step+1, num_layer)*kNumBlockPerCell);
        if(block_cond){
            for(int i=0; i<kNumInputGate; ++i){
                // Input gate GEMV
                input_local_sum[i] = 0;
                #pragma unroll
                for(int j=0; j<kNumThreadIter; ++j){
                    input_local_sum[i] += input_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + j * blockDim.x + threadIdx.x] * \
                        shared_input_weight[threadIdx.y * num_hidden + j * blockDim.x + threadIdx.x];
                }
                #define FULL_MASK 0xffffffff
                for (int offset = blockDim.x; offset > 0; offset /= 2){
                    input_local_sum[i] += __shfl_down_sync(FULL_MASK, input_local_sum[i], offset);
                }
                // State gate GEMV
                state_local_sum[i] = 0;
                #pragma unroll
                for(int j=0; j<kNumThreadIter; ++j){
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + j * blockDim.x + threadIdx.x] * \
                        shared_state_weight[threadIdx.y * num_hidden + j * blockDim.x + threadIdx.x];
                }
                for (int offset = blockDim.x; offset > 0; offset /= 2){
                    state_local_sum[i] += __shfl_down_sync(FULL_MASK, state_local_sum[i], offset);
                }
                // input gate + state gate
                if(threadIdx.x == 0){
                    output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + i * num_hidden + (blockIdx.x % kNumGatesInLstmCell) * kNumRow + threadIdx.y] = input_local_sum[i]+state_local_sum[i];
                    printf("blockIdx.x %d threadIdx.y %d output_buffer %f\n", blockIdx.x, threadIdx.y, input_local_sum[i]+state_local_sum[i]);
                }
            }
            __threadfence_block();
            // Solve here
            if(threadIdx.x == 0){
                float i = output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + 0 * num_hidden + (blockIdx.x % kNumGatesInLstmCell) * kNumRow + threadIdx.y];
                float j = output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + 1 * num_hidden + (blockIdx.x % kNumGatesInLstmCell) * kNumRow + threadIdx.y];
                float f = output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + 2 * num_hidden + (blockIdx.x % kNumGatesInLstmCell) * kNumRow + threadIdx.y];
                float o = output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + 3 * num_hidden + (blockIdx.x % kNumGatesInLstmCell) * kNumRow + threadIdx.y];

                c_wavefront[blockIdx.x / kNumBlockPerCell * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] = 
                    c_wavefront[blockIdx.x / kNumBlockPerCell * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] * sigmoid(f + 1.0) +
                    sigmoid(i) * tanh(j);
                h_wavefront[blockIdx.x / kNumBlockPerCell * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] = 
                    tanh(c_wavefront[blockIdx.x / kNumBlockPerCell * num_hidden + blockIdx.x % kNumBlockPerCell * kNumRow + threadIdx.y]) * sigmoid(o);
            }
            __threadfence();
        }
        grid.sync();
    }
}


// blockDim(32,4,1), gridDim(640)
// ERROR! too many blocks in cooperative launch
template<int batch, int num_layer, int num_hidden, int num_timestep>
    __global__ void __launch_bounds__(256, 3) lstm_reuse_shared_memory_v8(float* inputs_timestep, float* outputs_timestep, 
    float* c_wavefront, float* h_wavefront, float* input_wavefront,
    float* weight_input_wavefront, float* weight_state_wavefront, float* bias,
    float* output_buffer){
    
    const int kNumInputGate = 4;
    const int kNumRow = 8; //Equal to blockDim.y
    const int kNumThreadIter = num_hidden / 32; // 32 equal to blockDim.x
    const int kNumBlockPerCell = num_hidden / kNumRow; // 256/8 = 32

    extern float __shared__ shared_input_weight[]; //4*8*256*4B=32KB

    float input_local_sum[kNumInputGate];
    float state_local_sum[kNumInputGate];
    
    // Load input weight to shared memory
    #pragma unroll
    for(int i=0; i<kNumInputGate; ++i){
        #pragma unroll
        for(int j=0; j<kNumThreadIter; ++j){
            shared_input_weight[i * kNumRow * num_hidden + threadIdx.y * num_hidden + j*blockDim.x + threadIdx.x] = \
                weight_input_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + i * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + j*blockDim.x + threadIdx.x];
        }
    }

    __syncthreads();
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    for(int step=0; step<num_timestep+num_layer-1; ++step){
        bool block_cond = ((step<num_timestep) && (blockIdx.x < min(step+1, num_layer)*kNumBlockPerCell)) || 
            ((step>=num_timestep) && (blockIdx.x >= (step+1-num_timestep)*kNumBlockPerCell));
        if(block_cond){
            #pragma unroll
            for(int i=0; i<kNumInputGate; ++i){
                // Input gate GEMV
                input_local_sum[i] = 0;
                state_local_sum[i] = 0;
                #pragma unroll
                for(int j=0; j<kNumThreadIter; ++j){
                    input_local_sum[i] += input_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + j * blockDim.x + threadIdx.x] * \
                        shared_input_weight[i * kNumRow * num_hidden + threadIdx.y * num_hidden + j * blockDim.x + threadIdx.x];
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + j * blockDim.x + threadIdx.x] * \
                        weight_input_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + i * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + j*blockDim.x + threadIdx.x];
                }
                #define FULL_MASK 0xffffffff
                for (int offset = 16; offset > 0; offset /= 2){
                    input_local_sum[i] += __shfl_down_sync(FULL_MASK, input_local_sum[i], offset);
                    state_local_sum[i] += __shfl_down_sync(FULL_MASK, state_local_sum[i], offset);
                }
                // input gate + state gate
                if(threadIdx.x == 0){
                    output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + i * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] = \
                        input_local_sum[i]+state_local_sum[i]+bias[(blockIdx.x / kNumBlockPerCell)*num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y];
                    // printf("step: %d blockIdx.x %d threadIdx.y %d output_buffer %f\n", step, blockIdx.x, threadIdx.y, output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + i * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y]);
                }
            }
            __syncthreads();
            __threadfence_block();
            // Solve here
            if(threadIdx.x == 0){
                float i = output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + 0 * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y];
                float j = output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + 1 * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y];
                float f = output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + 2 * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y];
                float o = output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + 3 * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y];

                c_wavefront[blockIdx.x / kNumBlockPerCell * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] = 
                    c_wavefront[blockIdx.x / kNumBlockPerCell * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] * sigmoid(f + 1.0) +
                    sigmoid(i) * tanh(j);
                h_wavefront[blockIdx.x / kNumBlockPerCell * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] = 
                    tanh(c_wavefront[blockIdx.x / kNumBlockPerCell * num_hidden + blockIdx.x % kNumBlockPerCell * kNumRow + threadIdx.y]) * sigmoid(o);
                // Shift result for next timestep
                if(blockIdx.x / kNumBlockPerCell < num_layer - 1){
                    input_wavefront[(blockIdx.x / kNumBlockPerCell + 1) * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] = h_wavefront[blockIdx.x / kNumBlockPerCell * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y];
                }
                if(step<num_timestep && blockIdx.x<kNumBlockPerCell){
                    input_wavefront[blockIdx.x * kNumRow + threadIdx.y] = inputs_timestep[step*num_hidden + blockIdx.x * kNumRow + threadIdx.y];
                }
                if((step >= num_layer-1) && blockIdx.x / kNumBlockPerCell == (num_layer-1)){
                   outputs_timestep[(step+1-num_layer)*num_hidden + blockIdx.x * kNumRow + threadIdx.y] = h_wavefront[blockIdx.x / kNumBlockPerCell * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y];
                }
            }
        }
        __threadfence();
        grid.sync();
    }
}



// blockDim(32,4,1), gridDim(640)
template<int batch, int num_layer, int num_hidden, int num_timestep>
    __global__ void __launch_bounds__(256, 3) lstm_reuse_shared_memory_v9(float* inputs_timestep, float* outputs_timestep, 
    float* c_wavefront, float* h_wavefront, float* input_wavefront,
    float* weight_input_wavefront, float* weight_state_wavefront, float* bias,
    float* output_buffer){
    
    const int kNumInputGate = 4;
    const int kNumRow = 8; //Equal to blockDim.y
    const int kNumThreadIter = num_hidden / 32; // 32 equal to blockDim.x
    const int kNumBlockPerCell = num_hidden / kNumRow; // 256/8 = 32

    extern float __shared__ shared_weight[]; //(4+2)*8*256*4B=48KB
    float *shared_input_weight = (float*)&shared_weight[0];
    float *shared_state_weight = (float*)&shared_weight[8*1024];

    float s00=0, s01=0, s02=0, s03=0, s04=0, s05=0, s06=0, s07=0;
    float s10=0, s11=0, s12=0, s13=0, s14=0, s15=0, s16=0, s17=0;

    float input_local_sum[kNumInputGate];
    float state_local_sum[kNumInputGate];
    
    // Load input weight to shared memory
    #pragma unroll
    for(int i=0; i<kNumInputGate; ++i){
        #pragma unroll
        for(int j=0; j<kNumThreadIter; ++j){
            shared_input_weight[i * kNumRow * num_hidden + threadIdx.y * num_hidden + j*blockDim.x + threadIdx.x] = \
                weight_input_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + i * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + j*blockDim.x + threadIdx.x];
        }
    }
    // Load 2 state weight to shared memory
    #pragma unroll
    for(int i=0; i<2; ++i){
        #pragma unroll
        for(int j=0; j<kNumThreadIter; ++j){
            shared_state_weight[i * kNumRow * num_hidden + threadIdx.y * num_hidden + j*blockDim.x + threadIdx.x] = \
                weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + i * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + j*blockDim.x + threadIdx.x];
        }
    }
    // Load last 2 state weight to register
    const int ws2_idx = (blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 2 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden;
    const int ws3_idx = (blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 3 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden;

    s00 = weight_state_wavefront[ws2_idx + 0 * blockDim.x + threadIdx.x];
    s01 = weight_state_wavefront[ws2_idx + 1 * blockDim.x + threadIdx.x];
    s02 = weight_state_wavefront[ws2_idx + 2 * blockDim.x + threadIdx.x];
    s03 = weight_state_wavefront[ws2_idx + 3 * blockDim.x + threadIdx.x];
    s04 = weight_state_wavefront[ws2_idx + 4 * blockDim.x + threadIdx.x];
    s05 = weight_state_wavefront[ws2_idx + 5 * blockDim.x + threadIdx.x];
    s06 = weight_state_wavefront[ws2_idx + 6 * blockDim.x + threadIdx.x];
    s07 = weight_state_wavefront[ws2_idx + 7 * blockDim.x + threadIdx.x];

    s10 = weight_state_wavefront[ws3_idx + 0 * blockDim.x + threadIdx.x];
    s11 = weight_state_wavefront[ws3_idx + 1 * blockDim.x + threadIdx.x];
    s12 = weight_state_wavefront[ws3_idx + 2 * blockDim.x + threadIdx.x];
    s13 = weight_state_wavefront[ws3_idx + 3 * blockDim.x + threadIdx.x];
    s14 = weight_state_wavefront[ws3_idx + 4 * blockDim.x + threadIdx.x];
    s15 = weight_state_wavefront[ws3_idx + 5 * blockDim.x + threadIdx.x];
    s16 = weight_state_wavefront[ws3_idx + 6 * blockDim.x + threadIdx.x];
    s17 = weight_state_wavefront[ws3_idx + 7 * blockDim.x + threadIdx.x];

    __syncthreads();
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    for(int step=0; step<num_timestep+num_layer-1; ++step){
        bool block_cond = ((step<num_timestep) && (blockIdx.x < min(step+1, num_layer)*kNumBlockPerCell)) || 
            ((step>=num_timestep) && (blockIdx.x >= (step+1-num_timestep)*kNumBlockPerCell));
        if(block_cond){
            const int wavefront_stride = (blockIdx.x / kNumBlockPerCell) * num_hidden;
            #pragma unroll
            for(int i=0; i<kNumInputGate; ++i){
                // Input gate GEMV
                input_local_sum[i] = 0;
                state_local_sum[i] = 0;
                #pragma unroll
                for(int j=0; j<kNumThreadIter; ++j){
                    input_local_sum[i] += input_wavefront[wavefront_stride + j * blockDim.x + threadIdx.x] * \
                        shared_input_weight[i * kNumRow * num_hidden + threadIdx.y * num_hidden + j * blockDim.x + threadIdx.x];
                    
                }
                if(i<2){
                    #pragma unroll
                    for(int j=0; j<kNumThreadIter; ++j){
                        state_local_sum[i] += h_wavefront[wavefront_stride + j * blockDim.x + threadIdx.x] * \
                            shared_state_weight[i * kNumRow * num_hidden + threadIdx.y * num_hidden + j * blockDim.x + threadIdx.x];
                    }
                }else if(i==2){
                    state_local_sum[i] += h_wavefront[wavefront_stride + 0 * blockDim.x + threadIdx.x] * s00;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 1 * blockDim.x + threadIdx.x] * s01;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 2 * blockDim.x + threadIdx.x] * s02;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 3 * blockDim.x + threadIdx.x] * s03;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 4 * blockDim.x + threadIdx.x] * s04;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 5 * blockDim.x + threadIdx.x] * s05;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 6 * blockDim.x + threadIdx.x] * s06;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 7 * blockDim.x + threadIdx.x] * s07;
                }else if(i==3){
                    state_local_sum[i] += h_wavefront[wavefront_stride + 0 * blockDim.x + threadIdx.x] * s10;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 1 * blockDim.x + threadIdx.x] * s11;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 2 * blockDim.x + threadIdx.x] * s12;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 3 * blockDim.x + threadIdx.x] * s13;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 4 * blockDim.x + threadIdx.x] * s14;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 5 * blockDim.x + threadIdx.x] * s15;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 6 * blockDim.x + threadIdx.x] * s16;
                    state_local_sum[i] += h_wavefront[wavefront_stride + 7 * blockDim.x + threadIdx.x] * s17;
                }
                
                #define FULL_MASK 0xffffffff
                for (int offset = 16; offset > 0; offset /= 2){
                    input_local_sum[i] += __shfl_down_sync(FULL_MASK, input_local_sum[i], offset);
                    state_local_sum[i] += __shfl_down_sync(FULL_MASK, state_local_sum[i], offset);
                }
                if(i==1){
                    i=i;
                }
                // input gate + state gate
                if(threadIdx.x == 0){
                    output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + i * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] = \
                        input_local_sum[i]+state_local_sum[i]+bias[(blockIdx.x / kNumBlockPerCell)*num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y];
                    // printf("step: %d blockIdx.x %d threadIdx.y %d output_buffer %f\n", step, blockIdx.x, threadIdx.y, output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + i * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y]);
                }
            }
            __syncthreads();
            __threadfence_block();
            // Solve here
            if(threadIdx.x == 0){
                const int output_buffer_block_idx = (blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden;
                const int output_buffer_thread_idx = (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y;
                float i = output_buffer[output_buffer_block_idx + 0 * num_hidden + output_buffer_thread_idx];
                float j = output_buffer[output_buffer_block_idx + 1 * num_hidden + output_buffer_thread_idx];
                float f = output_buffer[output_buffer_block_idx + 2 * num_hidden + output_buffer_thread_idx];
                float o = output_buffer[output_buffer_block_idx + 3 * num_hidden + output_buffer_thread_idx];
                const int idx = (blockIdx.x / kNumBlockPerCell) * num_hidden + output_buffer_thread_idx;
                c_wavefront[idx] = 
                    c_wavefront[idx] * sigmoid(f + 1.0) +
                    sigmoid(i) * tanh(j);
                h_wavefront[idx] = 
                    tanh(c_wavefront[idx]) * sigmoid(o);
                
                // Shift result for next timestep
                if(blockIdx.x / kNumBlockPerCell < num_layer - 1){
                    input_wavefront[idx + num_hidden] = h_wavefront[idx];
                }
                if(step<num_timestep && blockIdx.x<kNumBlockPerCell){
                    input_wavefront[blockIdx.x * kNumRow + threadIdx.y] = inputs_timestep[step*num_hidden + blockIdx.x * kNumRow + threadIdx.y];
                }
                if((step >= num_layer-1) && blockIdx.x / kNumBlockPerCell == (num_layer-1)){
                   outputs_timestep[(step+1-num_layer)*num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] = h_wavefront[idx];
                }
            }
            __syncthreads();
        }
        __syncthreads();
        __threadfence();
        grid.sync();
    }
}

// lstm_reuse_shared_memory.h

// blockDim(32,4,1), gridDim(640)
__device__ int arr_sync[2] = {};
template<int batch, int num_layer, int num_hidden, int num_timestep>
    __global__ void __launch_bounds__(256, 3) lstm_reuse_shared_memory_v9_block_sync(float* inputs_timestep, float* outputs_timestep, 
    float* c_wavefront, float* h_wavefront, float* input_wavefront,
    float* weight_input_wavefront, float* weight_state_wavefront, float* bias,
    float* output_buffer){
    arr_sync[0]=0;arr_sync[1]=0;
    volatile int* b_sync = arr_sync;
    const int kNumInputGate = 4;
    const int kNumRow = 8; //Equal to blockDim.y
    const int kNumThreadIter = num_hidden / 32; // 32 equal to blockDim.x
    const int kNumBlockPerCell = num_hidden / kNumRow; // 256/8 = 32

    extern float __shared__ shared_weight[]; //(4+2)*8*256*4B=48KB
    float *shared_input_weight = (float*)&shared_weight[0];
    float *shared_state_weight = (float*)&shared_weight[8*1024];

    float s00=0, s01=0, s02=0, s03=0, s04=0, s05=0, s06=0, s07=0;
    float s10=0, s11=0, s12=0, s13=0, s14=0, s15=0, s16=0, s17=0;

    float input_local_sum[kNumInputGate];
    float state_local_sum[kNumInputGate];
    
    // Load input weight to shared memory
    #pragma unroll
    for(int i=0; i<kNumInputGate; ++i){
        #pragma unroll
        for(int j=0; j<kNumThreadIter; ++j){
            shared_input_weight[i * kNumRow * num_hidden + threadIdx.y * num_hidden + j*blockDim.x + threadIdx.x] = \
                weight_input_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + i * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + j*blockDim.x + threadIdx.x];
        }
    }
    // Load 2 state weight to shared memory
    #pragma unroll
    for(int i=0; i<2; ++i){
        #pragma unroll
        for(int j=0; j<kNumThreadIter; ++j){
            shared_state_weight[i * kNumRow * num_hidden + threadIdx.y * num_hidden + j*blockDim.x + threadIdx.x] = \
                weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + i * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + j*blockDim.x + threadIdx.x];
        }
    }
    // Load last 2 state weight to register
    s00 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 2 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 0 * blockDim.x + threadIdx.x];
    s01 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 2 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 1 * blockDim.x + threadIdx.x];
    s02 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 2 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 2 * blockDim.x + threadIdx.x];
    s03 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 2 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 3 * blockDim.x + threadIdx.x];
    s04 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 2 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 4 * blockDim.x + threadIdx.x];
    s05 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 2 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 5 * blockDim.x + threadIdx.x];
    s06 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 2 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 6 * blockDim.x + threadIdx.x];
    s07 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 2 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 7 * blockDim.x + threadIdx.x];

    s10 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 3 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 0 * blockDim.x + threadIdx.x];
    s11 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 3 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 1 * blockDim.x + threadIdx.x];
    s12 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 3 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 2 * blockDim.x + threadIdx.x];
    s13 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 3 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 3 * blockDim.x + threadIdx.x];
    s14 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 3 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 4 * blockDim.x + threadIdx.x];
    s15 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 3 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 5 * blockDim.x + threadIdx.x];
    s16 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 3 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 6 * blockDim.x + threadIdx.x];
    s17 = weight_state_wavefront[(blockIdx.x / kNumBlockPerCell) * kNumInputGate * num_hidden * num_hidden + 3 * num_hidden * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow * num_hidden + threadIdx.y * num_hidden + 7 * blockDim.x + threadIdx.x];

    __syncthreads();
    // cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    for(int step=0; step<num_timestep+num_layer-1; ++step){
        bool block_cond = ((step<num_timestep) && (blockIdx.x < min(step+1, num_layer)*kNumBlockPerCell)) || 
            ((step>=num_timestep) && (blockIdx.x >= (step+1-num_timestep)*kNumBlockPerCell));
        if(block_cond){
            #pragma unroll
            for(int i=0; i<kNumInputGate; ++i){
                // Input gate GEMV
                input_local_sum[i] = 0;
                state_local_sum[i] = 0;
                #pragma unroll
                for(int j=0; j<kNumThreadIter; ++j){
                    input_local_sum[i] += input_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + j * blockDim.x + threadIdx.x] * \
                        shared_input_weight[i * kNumRow * num_hidden + threadIdx.y * num_hidden + j * blockDim.x + threadIdx.x];
                    
                }
                if(i<2){
                    #pragma unroll
                    for(int j=0; j<kNumThreadIter; ++j){
                        state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + j * blockDim.x + threadIdx.x] * \
                            shared_state_weight[i * kNumRow * num_hidden + threadIdx.y * num_hidden + j * blockDim.x + threadIdx.x];
                    }
                }else if(i==2){
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 0 * blockDim.x + threadIdx.x] * s00;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 1 * blockDim.x + threadIdx.x] * s01;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 2 * blockDim.x + threadIdx.x] * s02;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 3 * blockDim.x + threadIdx.x] * s03;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 4 * blockDim.x + threadIdx.x] * s04;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 5 * blockDim.x + threadIdx.x] * s05;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 6 * blockDim.x + threadIdx.x] * s06;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 7 * blockDim.x + threadIdx.x] * s07;
                }else if(i==3){
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 0 * blockDim.x + threadIdx.x] * s10;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 1 * blockDim.x + threadIdx.x] * s11;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 2 * blockDim.x + threadIdx.x] * s12;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 3 * blockDim.x + threadIdx.x] * s13;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 4 * blockDim.x + threadIdx.x] * s14;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 5 * blockDim.x + threadIdx.x] * s15;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 6 * blockDim.x + threadIdx.x] * s16;
                    state_local_sum[i] += h_wavefront[(blockIdx.x / kNumBlockPerCell) * num_hidden + 7 * blockDim.x + threadIdx.x] * s17;
                }
                
                #define FULL_MASK 0xffffffff
                for (int offset = 16; offset > 0; offset /= 2){
                    input_local_sum[i] += __shfl_down_sync(FULL_MASK, input_local_sum[i], offset);
                    state_local_sum[i] += __shfl_down_sync(FULL_MASK, state_local_sum[i], offset);
                }
                if(i==1){
                    i=i;
                }
                // input gate + state gate
                if(threadIdx.x == 0){
                    output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + i * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] = \
                        input_local_sum[i]+state_local_sum[i]+bias[(blockIdx.x / kNumBlockPerCell)*num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y];
                    // printf("step: %d blockIdx.x %d threadIdx.y %d output_buffer %f\n", step, blockIdx.x, threadIdx.y, output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + i * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y]);
                }
            }
            __syncthreads();
            __threadfence_block();
            // Solve here
            if(threadIdx.x == 0){
                float i = output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + 0 * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y];
                float j = output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + 1 * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y];
                float f = output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + 2 * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y];
                float o = output_buffer[(blockIdx.x/kNumBlockPerCell) * kNumInputGate * num_hidden + 3 * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y];

                c_wavefront[blockIdx.x / kNumBlockPerCell * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] = 
                    c_wavefront[blockIdx.x / kNumBlockPerCell * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] * sigmoid(f + 1.0) +
                    sigmoid(i) * tanh(j);
                h_wavefront[blockIdx.x / kNumBlockPerCell * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] = 
                    tanh(c_wavefront[blockIdx.x / kNumBlockPerCell * num_hidden + blockIdx.x % kNumBlockPerCell * kNumRow + threadIdx.y]) * sigmoid(o);
                
                // Shift result for next timestep
                if(blockIdx.x / kNumBlockPerCell < num_layer - 1){
                    input_wavefront[(blockIdx.x / kNumBlockPerCell + 1) * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] = h_wavefront[blockIdx.x / kNumBlockPerCell * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y];
                }
                if(step<num_timestep && blockIdx.x<kNumBlockPerCell){
                    input_wavefront[blockIdx.x * kNumRow + threadIdx.y] = inputs_timestep[step*num_hidden + blockIdx.x * kNumRow + threadIdx.y];
                }
                if((step >= num_layer-1) && blockIdx.x / kNumBlockPerCell == (num_layer-1)){
                   outputs_timestep[(step+1-num_layer)*num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y] = h_wavefront[blockIdx.x / kNumBlockPerCell * num_hidden + (blockIdx.x % kNumBlockPerCell) * kNumRow + threadIdx.y];
                }
            }
        }

        __threadfence();
        
        if(threadIdx.x % 32 == 0){
            atomicAdd(&(arr_sync[0]), 1);
        }
        while(b_sync[0]<(step+1)*(gridDim.x * blockDim.x / 32 * blockDim.y)){
        }
        // if(threadIdx.x == 0 && threadIdx.y==0){
        //     atomicAdd(&(arr_sync[0]), 1);
        // }
        // while(b_sync[0]<(step+1)*(gridDim.x)){
        // }
        __threadfence();
    }
}
