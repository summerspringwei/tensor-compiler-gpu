#include <immintrin.h>
#include <stdio.h>
#include <iostream>

using namespace std;

int main(){
  uint8_t  op1_int8[64];
  int8_t   op2_int8[64];
  int32_t  op3_int[16];
  int16_t  op4_int16[32];
  int32_t  result[16];
  for(int i=0; i<64; ++i){

    op1_int8[i] = (i/4)+1;
    op2_int8[i] = (i/4)+1;
    if(i<16){
      op3_int[i] = 0;
      result[i] = 0;
    }
  }

  __m512i v1_int8;
  __m512i v2_int8;
  __m512i v3_int;
  __m512i v4_int16;
  __m512i vresult;
     v1_int8 =_mm512_loadu_si512(op1_int8);
   v2_int8 =_mm512_loadu_si512(op2_int8);
   v3_int =_mm512_loadu_si512(op3_int);
   v4_int16 =_mm512_loadu_si512(op4_int16);

  // PERFORM THE DOT PRODUCT OPERATION USING FUSED INSTRUCTION
   vresult = _mm512_dpbusds_epi32(v3_int, v1_int8, v2_int8);

   _mm512_storeu_si512((void *) result, vresult);

   printf("RESULTS USING FUSED INSTRUCTION: \n");
   for (int j = 15; j >= 0; j--){
       cout << result[j]<<" ";
   }
  int sum = _mm512_reduce_add_epi32(vresult);
  printf("sum %d\n", sum);
  return 0;
}