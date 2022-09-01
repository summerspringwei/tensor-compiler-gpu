
// Step guide
// 1. rename parameter name (both in function parameters and in function body) to avoid conflicts between kernels
// 2. change static shared memory to dynamic shared memory
// 3. Warp each of original kernels with if(blockIdx.x < xxx)
// 4. Declare grid and pipe
// 5. Change the second GEMM's load to shared memory with cuda::memcpy_async
// 