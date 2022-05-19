int main(){
	const int input_size=1024;	float *input = new float[input_size];
	const int weight_size=1024;	float *weight = new float[weight_size];
	const int output_size=256;	float *output = new float[output_size];

	cudaError_t err = cudaSuccess;
	float *d_input=NULL;
	float *d_weight=NULL;
	float *d_output=NULL;
	err=cudaMalloc((void **)&d_input, sizeof(float)*input_size);
	err=cudaMalloc((void **)&d_weight, sizeof(float)*weight_size);
	err=cudaMalloc((void **)&d_output, sizeof(float)*output_size);

	cudaMemcpy(d_input, input, sizeof(float)*input_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight, weight, sizeof(float)*weight_size, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	cudaMemcpy(output, d_output, sizeof(float)*output_size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  	delete[] input;
	delete[] weight;
	delete[] output;
	cudaFree(d_input);
	cudaFree(d_weight);
	cudaFree(d_output);
	return 0;
}