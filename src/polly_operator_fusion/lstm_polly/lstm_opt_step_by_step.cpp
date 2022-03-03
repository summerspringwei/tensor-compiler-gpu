
const int num_hidden = 256;
const int num_gate = 4;


// Naive implementation
void lstm_cell_naive(float inputs[num_hidden],
          float c_state[num_hidden],
          float h_state[num_hidden],
          float W[num_gate][num_hidden][num_hidden],
          float U[num_gate][num_hidden][num_hidden],
          float bias[num_hidden]){
    float input_gate[num_gate][num_hidden];
    float state_gate[num_gate][num_hidden];
    float output_buf[num_gate][num_hidden];
    for(int i=0; i<num_gate; ++i){
        for(int j=0; j<num_hidden; ++j){
            input_gate[i][j] = 0;
            for(int k=0; k<num_hidden; ++k){
                input_gate[i][j] += inputs[k] * W[i][j][k];
            }
        }
        for(int j=0; j<num_hidden; ++j){
            state_gate[i][j] = 0;
            for(int k=0; k<num_hidden; ++k){
                state_gate[i][j] += h_state[k] * W[i][j][k];
            }
        }
        for(int j=0; j<num_hidden; ++j){
            output_buf[i][j] = input_gate[i][j] + state_gate[i][j] + bias[j];
        }
    }

    for(int j=0; j<num_hidden; ++j){
        c_state[j] = c_state[j] * (output_buf[2][j] + 1) + 
            output_buf[0][j] * output_buf[1][j];
    }
    for(int j=0; j<num_hidden; ++j){
        h_state[j] = c_state[j] * output_buf[3][j];
    }
}


// Fusion at num_hidden
void lstm_cell_fusion(float inputs[num_hidden],
          float c_state[num_hidden],
          float h_state[num_hidden],
          float W[num_gate][num_hidden][num_hidden],
          float U[num_gate][num_hidden][num_hidden],
          float bias[num_hidden]){
    float input_gate[num_gate][num_hidden];
    float state_gate[num_gate][num_hidden];
    float output_buf[num_gate][num_hidden];
    for(int i=0; i<num_gate; ++i){
        for(int j=0; j<num_hidden; ++j){
            input_gate[i][j] = 0;
            state_gate[i][j] = 0;
            for(int k=0; k<num_hidden; ++k){
                input_gate[i][j] += inputs[k] * W[i][j][k];
                state_gate[i][j] += h_state[k] * W[i][j][k];
            }
            output_buf[i][j] = input_gate[i][j] + state_gate[i][j] + bias[j];
        }
    }
    for(int j=0; j<num_hidden; ++j){
        c_state[j] = c_state[j] * (output_buf[2][j] + 1) + output_buf[0][j] * output_buf[1][j];
        h_state[j] = c_state[j] * output_buf[3][j];
    }
}


// Fusion at num_hidden
void lstm_cell_fusion_interchange(float inputs[num_hidden],
          float c_state[num_hidden],
          float h_state[num_hidden],
          float W[num_gate][num_hidden][num_hidden],
          float U[num_gate][num_hidden][num_hidden],
          float bias[num_hidden]){
    float input_gate[num_gate][num_hidden];
    float state_gate[num_gate][num_hidden];
    float output_buf[num_gate][num_hidden];
    for(int j=0; j<num_hidden; ++j){
        for(int i=0; i<num_gate; ++i){
            input_gate[i][j] = 0;
            state_gate[i][j] = 0;
            for(int k=0; k<num_hidden; ++k){
                input_gate[i][j] += inputs[k] * W[i][j][k];
                state_gate[i][j] += h_state[k] * W[i][j][k];
            }
            output_buf[i][j] = input_gate[i][j] + state_gate[i][j] + bias[j];
        }
    }
    for(int j=0; j<num_hidden; ++j){
        c_state[j] = c_state[j] * (output_buf[2][j] + 1) + output_buf[0][j] * output_buf[1][j];
        h_state[j] = c_state[j] * output_buf[3][j];
    }
}


void lstm_cell_fusion_interchange_fusion(float inputs[num_hidden],
          float c_state[num_hidden],
          float h_state[num_hidden],
          float W[num_gate][num_hidden][num_hidden],
          float U[num_gate][num_hidden][num_hidden],
          float bias[num_hidden]){
    float input_gate[num_gate][num_hidden];
    float state_gate[num_gate][num_hidden];
    float output_buf[num_gate][num_hidden];
    for(int j=0; j<num_hidden; ++j){
        for(int i=0; i<num_gate; ++i){
            input_gate[i][j] = 0;
            state_gate[i][j] = 0;
            for(int k=0; k<num_hidden; ++k){
                input_gate[i][j] += inputs[k] * W[i][j][k];
                state_gate[i][j] += h_state[k] * W[i][j][k];
            }
            output_buf[i][j] = input_gate[i][j] + state_gate[i][j] + bias[j];
        }
        c_state[j] = c_state[j] * (output_buf[2][j] + 1) + output_buf[0][j] * output_buf[1][j];
        h_state[j] = c_state[j] * output_buf[3][j];
    }
}



const int num_layer = 10;
void lstm_cell_layers(float inputs[num_layer][num_hidden],
          float c_state[num_layer][num_hidden],
          float h_state[num_layer][num_hidden],
          float W[num_layer][num_gate][num_hidden][num_hidden],
          float U[num_layer][num_gate][num_hidden][num_hidden],
          float bias[num_layer][num_hidden]){
    float input_gate[num_layer][num_gate][num_hidden];
    float state_gate[num_layer][num_gate][num_hidden];
    float output_buf[num_layer][num_gate][num_hidden];
    for(int x=0; x<num_layer; ++x){
        for(int j=0; j<num_hidden; ++j){
            if(x>0){
                inputs[x][j] = h_state[x-1][j]; //Feed input from last layer
            }
            for(int i=0; i<num_gate; ++i){
                input_gate[x][i][j] = 0;
                state_gate[x][i][j] = 0;
                for(int k=0; k<num_hidden; ++k){
                    input_gate[x][i][j] += inputs[x][k] * W[x][i][j][k];
                    state_gate[x][i][j] += h_state[x][k] * W[x][i][j][k];
                }
                output_buf[x][i][j] = input_gate[x][i][j] + state_gate[x][i][j] + bias[x][j];
            }
            c_state[x][j] = c_state[x][j] * (output_buf[x][2][j] + 1) + output_buf[x][0][j] * output_buf[x][1][j];
            h_state[x][j] = c_state[x][j] * output_buf[x][3][j];
        }
    }
}


const int num_timestep = 10;
void lstm_cell_layers_timesteps(float inputs_timesteps[num_timestep][num_hidden],
          float c_state[num_layer][num_hidden],
          float h_state[num_layer][num_hidden],
          float W[num_layer][num_gate][num_hidden][num_hidden],
          float U[num_layer][num_gate][num_hidden][num_hidden],
          float bias[num_layer][num_hidden],
          float outputs[num_timestep][num_hidden]){
    float inputs[num_hidden];
    float input_gate[num_layer][num_gate][num_hidden];
    float state_gate[num_layer][num_gate][num_hidden];
    float output_buf[num_layer][num_gate][num_hidden];
    
    for(int s=0; s<num_timestep; ++s){
        for(int x=0; x<num_layer; ++x){
            for(int j=0; j<num_hidden; ++j){
                if(x==0){
                    inputs[j] = inputs_timesteps[s][j];
                }
                else if(x>0){
                    inputs[j] = h_state[x-1][j]; //Feed input from previous layer
                }
                for(int i=0; i<num_gate; ++i){
                    input_gate[x][i][j] = 0;
                    state_gate[x][i][j] = 0;
                    for(int k=0; k<num_hidden; ++k){
                        input_gate[x][i][j] += inputs[k] * W[x][i][j][k];
                        state_gate[x][i][j] += h_state[x][k] * W[x][i][j][k];
                    }
                    output_buf[x][i][j] = input_gate[x][i][j] + state_gate[x][i][j] + bias[x][j];
                }
                c_state[x][j] = c_state[x][j] * (output_buf[x][2][j] + 1) + output_buf[x][0][j] * output_buf[x][1][j];
                h_state[x][j] = c_state[x][j] * output_buf[x][3][j];
                if(x==num_layer-1){
                    outputs[s][j] = h_state[x][j];// Feed output
                }
            }
        }
    }
}
