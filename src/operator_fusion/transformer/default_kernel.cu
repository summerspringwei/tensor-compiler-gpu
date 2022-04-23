int main(){
	float *input = new float[1024];
	float *weight = new float[2048];
	float *output = new float[1024];

	cudaError_t err = cudaSuccess;
	float *input=NULL;
	float *weight=NULL;
	float *output=NULL;
	err=cudaMalloc((void **)&d_input, sizeof(float)*1024);
	err=cudaMalloc((void **)&d_weight, sizeof(float)*2048);
	err=cudaMalloc((void **)&d_output, sizeof(float)*1024);

	cudaMemcpy(d_input, input, sizeof(float)*1024, cudaMemcpyHostToDevice);
	cudaMemcpy(d_weight, weight, sizeof(float)*2048, cudaMemcpyHostToDevice);

	cudaDeviceSynchronize();
	cudaMemcpy(d_output, output, sizeof(float)*1024, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	delete[] input;
	delete[] weight;
	delete[] output;
	cudaFree(d_input);
	cudaFree(d_weight);
	cudaFree(d_output);
	return 0;
}