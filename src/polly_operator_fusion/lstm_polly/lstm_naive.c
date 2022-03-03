
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
                state_gate[i][j] += h_state[k] * U[i][j][k];
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

// if (1 && (&MemRef7[256] <= &MemRef4[0] || &MemRef4[256] <= &MemRef7[0]) && (&MemRef2[3][255][256] <= &MemRef4[0] || &MemRef4[256] <= &MemRef2[0][0][0]) && (&MemRef1[256] <= &MemRef4[0] || &MemRef4[256] <= &MemRef1[0]) && (&MemRef5[256] <= &MemRef4[0] || &MemRef4[256] <= &MemRef5[0]) && (&MemRef2[3][255][256] <= &MemRef7[0] || &MemRef7[256] <= &MemRef2[0][0][0]) && (&MemRef1[256] <= &MemRef7[0] || &MemRef7[256] <= &MemRef1[0]) && (&MemRef5[256] <= &MemRef7[0] || &MemRef7[256] <= &MemRef5[0]))

//     {
//       for (int c0 = 0; c0 <= 3; c0 += 1) {
//         for (int c1 = 0; c1 <= 255; c1 += 1) {
//           Stmt1(c0, c1);
//           for (int c2 = 0; c2 <= 255; c2 += 1)
//             Stmt2(c0, c1, c2);
//         }
//         for (int c1 = 0; c1 <= 255; c1 += 1) {
//           Stmt5(c0, c1);
//           for (int c2 = 0; c2 <= 255; c2 += 1)
//             Stmt6(c0, c1, c2);
//         }
//         for (int c1 = 0; c1 <= 255; c1 += 1)
//           Stmt9(c0, c1);
//       }
//       for (int c0 = 0; c0 <= 255; c0 += 1)
//         Stmt12(c0);
//       for (int c0 = 0; c0 <= 255; c0 += 1)
//         Stmt14(c0);
//     }