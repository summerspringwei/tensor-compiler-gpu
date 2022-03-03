const int num_hidden = 256;
const int num_gate = 4;

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
