int main(){
	half *input = new half[524288];
	half *weight = new half[524288];
	half *output = new half[1048576];
	half *intermedia_output = new half[1048576];
	half *ori_output = new half[1048576];

	cudaError_t err = cudaSuccess;
	half *d_input=NULL;
	half *d_weight=NULL;
	half *d_output=NULL;
	half *d_intermedia_output=NULL;
	half *d_ori_output=NULL;
	err=cudaMalloc((void **)&d_input, sizeof(half)*524288);
	err=cudaMalloc((void **)&d_weight, sizeof(half)*524288);
	err=cudaMalloc((void **)&d_output, sizeof(half)*1048576);
	err=cudaMalloc((void **)&d_intermedia_output, sizeof(half)*1048576);
	err=cudaMalloc((void **)&d_ori_output, sizeof(half)*1048576);

	cudaMemcpy(d_input, input, sizeof(half)*524288, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight, weight, sizeof(half)*524288, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	cudaMemcpy(output, d_output, sizeof(half)*1048576, cudaMemcpyDeviceToHost);
	cudaMemcpy(intermedia_output, d_intermedia_output, sizeof(half)*1048576, cudaMemcpyDeviceToHost);
	cudaMemcpy(ori_output, d_ori_output, sizeof(half)*1048576, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();

  if (err != cudaSuccess){
    fprintf(stderr, "Failed to allocate device vector d_a (error code %s)!\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  	delete[] input;
	delete[] weight;
	delete[] output;
	delete[] intermedia_output;
	delete[] ori_output;
	cudaFree(d_input);
	cudaFree(d_weight);
	cudaFree(d_output);
	cudaFree(d_intermedia_output);
	cudaFree(d_ori_output);
	return 0;
}