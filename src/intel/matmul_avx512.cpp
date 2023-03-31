#include <immintrin.h>
#include <stdio.h>
#include <memory>
#include <iostream>

using namespace std;


// GEMM size: M384N768K768, tile size: 64*64
void matmul_bert_base_v0(int8_t* A, int8_t* B, int32_t* C, int M, int N, int K){
  __m512i v1_int8;
  __m512i v2_int8;
  __m512i vresult;
  const int kMTileSize = 64, kNTileSize = 64, kKTileSize = 64;
  for(int io=0; io<M/kMTileSize; ++io){
    for(int jo=0; jo<N/kNTileSize; ++jo){
      for(int ko=0; ko<K/kKTileSize; ++ko){
        for(int ii=0; ii<kMTileSize; ++ii){
          for(int jj=0; jj<kNTileSize; ++jj){
            int mi = io*kMTileSize+ii;
            int ni = jo*kNTileSize+jj;
            int ki = ko*kKTileSize;
            v1_int8 = _mm512_loadu_si512(A + mi*K+ki);
            v2_int8 = _mm512_loadu_si512(B + ni*K+ki);
            vresult = _mm512_set1_epi32(0);
            vresult = _mm512_dpbusds_epi32(vresult, v1_int8, v2_int8);
            auto result_sum = _mm512_reduce_add_epi32(vresult);
            C[mi*N + ni] += result_sum;
          }
        }
      }
    }
  }
}

void matmul(int8_t* A, int8_t* B, int32_t* C, int M, int N, int K){
  for(int i=0; i<M; ++i){
    for(int j=0; j<N; ++j){
      for(int k=0; k<K; ++k){
        C[i*N+j] += (A[i*K+k]*B[j*K+k]);
      }
    }
  }
}

void test_matmul_bert_base_v0(){
  const int M = 384, N = 768, K = 768;
  int8_t *A = (int8_t*)malloc(M*K*sizeof(int8_t));
  int8_t *B = (int8_t*)malloc(N*K*sizeof(int8_t));
  int32_t *C = (int32_t*)malloc(N*M*sizeof(int32_t));
  for(int i=0; i<M; ++i){
    for(int j=0; j<N; ++j){
      C[i*N+j] = 0;
    }
    for(int k=0; k<K; ++k){
      A[i*K+k] = 1;
    }
  }
  for(int j=0; j<N; ++j){
    for(int k=0; k<K; ++k){
      B[j*K+k] = 1;
    }
  }
  matmul_bert_base_v0(A, B, C, M, N, K);
  for(int i=0;i<10;++i){
    for(int j=0; j<10; ++j){
      printf("%d ",C[i*N+j]);
    }printf("\n");
  }
}

int main(){
  test_matmul_bert_base_v0();
  return 0;
}