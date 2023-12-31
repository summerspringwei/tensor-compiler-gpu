int main(){
	const int input_size=16777216;	half *input = new half[input_size];
	const int weight1_size=65536;	half *weight1 = new half[weight1_size];
	const int weight2_size=65536;	half *weight2 = new half[weight2_size];
	const int weight3_size=65536;	half *weight3 = new half[weight3_size];
	const int weight4_size=65536;	half *weight4 = new half[weight4_size];
	const int output_size=16777216;	half *output = new half[output_size];

	cudaError_t err = cudaSuccess;
	half *d_input=NULL;
	half *d_weight1=NULL;
	half *d_weight2=NULL;
	half *d_weight3=NULL;
	half *d_weight4=NULL;
	half *d_output=NULL;
	err=cudaMalloc((void **)&d_input, sizeof(half)*input_size);
	err=cudaMalloc((void **)&d_weight1, sizeof(half)*weight1_size);
	err=cudaMalloc((void **)&d_weight2, sizeof(half)*weight2_size);
	err=cudaMalloc((void **)&d_weight3, sizeof(half)*weight3_size);
	err=cudaMalloc((void **)&d_weight4, sizeof(half)*weight4_size);
	err=cudaMalloc((void **)&d_output, sizeof(half)*output_size);

	cudaMemcpy(d_input, input, sizeof(half)*input_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight1, weight1, sizeof(half)*weight1_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight2, weight2, sizeof(half)*weight2_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight3, weight3, sizeof(half)*weight3_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight4, weight4, sizeof(half)*weight4_size, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	cudaMemcpy(output, d_output, sizeof(half)*output_size, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  	delete[] input;
	delete[] weight1;
	delete[] weight2;
	delete[] weight3;
	delete[] weight4;
	delete[] output;
	cudaFree(d_input);
	cudaFree(d_weight1);
	cudaFree(d_weight2);
	cudaFree(d_weight3);
	cudaFree(d_weight4);
	cudaFree(d_output);
	return 0;
}