
const int n = 256;
void relu(float input[n], float output[n]){
    for(int i=0; i<n; ++i){
        if(input[i]>0){
            output[i] = input[i];
        }else{
            output[i] = 0;
        }
    }
}
