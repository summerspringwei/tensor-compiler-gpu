int main(){
	const int input_size=524288;	half *input = new half[input_size];
	const int weight_size=1048576;	half *weight = new half[weight_size];
	const int output_size=4194304;	half *output = new half[output_size];

	cudaError_t err = cudaSuccess;
	half *d_input=NULL;
	half *d_weight=NULL;
	half *d_output=NULL;
	err=cudaMalloc((void **)&d_input, sizeof(half)*input_size);
	err=cudaMalloc((void **)&d_weight, sizeof(half)*weight_size);
	err=cudaMalloc((void **)&d_output, sizeof(half)*output_size);

	cudaMemcpy(d_input, input, sizeof(half)*input_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight, weight, sizeof(half)*weight_size, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	cudaMemcpy(output, d_output, sizeof(half)*output_size, cudaMemcpyDeviceToHost);
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