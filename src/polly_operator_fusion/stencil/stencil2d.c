#include "stdio.h"

const int L=10, M = 10, N = 10;
void stencil2d(float U[M][N]){
    for(int i=0; i<L; ++i){
        for(int j=1; j<M-1; ++j){
            for(int k=1; k<N-1; ++k){
                U[j][k] = (U[j+1][k]+U[j][k+1]+U[j-1][k]+U[j][k-1])*0.25;
                printf("U[%d][%d] = (U[%d][%d]+U[%d][%d]+U[%d][%d]+U[%d][%d])*0.25;\n", j, k, j+1, k, j, k+1, j-1, k, j, k-1);
                // U[j][k] -> U[j+1][k] 依赖statement(i, j, k) -> statement(i, j+1, k)  
                // U[j][k] -> U[j][k+1] 依赖statement(i, j, k) -> statement(i, j, k+1)
                // U[j][k] -> U[j-1][k] 依赖statement(i, j, k) -> 
            }
        }
    }
}


void stencil2d_transformed(float U[M][N]){
    for(int i=4; i<2*L+M+N; ++i){
        for(int j=0; j<L; ++j){
            for(int k=1; k<N-1; ++k){
                if(i-2*j-k>=1 && i-2*j-k<M-1){
                    U[i-2*j-k][k] = (U[i-2*j-k+1][k]+U[i-2*j-k][k+1]+U[i-2*j-k-1][k]+U[i-2*j-k][k-1])*0.25;
                    printf("U[%d][%d] = (U[%d][%d]+U[%d][%d]+U[%d][%d]+U[%d][%d])*0.25;\n", i-2*j-k,k, i-2*j-k+1,k, i-2*j-k,k+1, i-2*j-k-1,k, i-2*j-k,k-1);
                }
                // U[j][k] -> U[j+1][k] 依赖statement(i, j, k) -> statement(i, j+1, k)  
                // U[j][k] -> U[j][k+1] 依赖statement(i, j, k) -> statement(i, j, k+1)
                // U[j][k] -> U[j-1][k] 依赖statement(i, j, k) -> 
            }
        }
    }
}

int main(){
    float U[M][N];
    // stencil2d(U);
    stencil2d_transformed(U);
    return 0;
}
