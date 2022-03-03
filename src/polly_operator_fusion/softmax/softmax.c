#include <math.h>

const int n = 256;

void softmax(float input[n], float output[n]){
    float sum = 0;
    for(int i=0; i<n; ++i){
        output[i]=(input[i]);
        sum += output[i];
    }
    for(int i=0; i<n; ++i){
        output[i] = output[i] / sum;
    }
}