
const int n = 1;
const int c = 16;
const int h = 32;
const int w = 32;
void maxpool(float input[n][c][h][w], float output[n][c][h][w]){
    for(int i=0; i<n; ++i){
        for(int j=0; j<c; ++j){
            for(int k=0; k<h; ++k){
                for(int x=0; x<w; ++x){
                    for(int a=0; a<3; ++a){
                        for(int b=0; b<3; ++b){
                            int idx_a = k-1+a;
                            int idx_b = x-1+b;
                            float tmp=-10000;
                            if(idx_a>=0 && idx_a<h && idx_b>=0 && idx_b<w){
                                tmp=(input[i][j][k][x]);
                            }
                            output[i][j][k][x] = tmp;
                        }
                    }
                }
            }
        }
    }
}