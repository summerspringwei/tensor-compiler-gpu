weight_tensor_shared[(((int)threadIdx.x))] = weight_tensor[((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)))];
  weight_tensor_shared[((((int)threadIdx.x) + 128))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 128))];
  weight_tensor_shared[((((int)threadIdx.x) + 256))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 256))];
  weight_tensor_shared[((((int)threadIdx.x) + 384))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 384))];
  weight_tensor_shared[((((int)threadIdx.x) + 512))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 512))];
  weight_tensor_shared[((((int)threadIdx.x) + 640))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 640))];
  weight_tensor_shared[((((int)threadIdx.x) + 768))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 768))];
  weight_tensor_shared[((((int)threadIdx.x) + 896))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 896))];
  weight_tensor_shared[((((int)threadIdx.x) + 1024))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 1024))];
  weight_tensor_shared[((((int)threadIdx.x) + 1152))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 1152))];
  weight_tensor_shared[((((int)threadIdx.x) + 1280))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 1280))];
  weight_tensor_shared[((((int)threadIdx.x) + 1408))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 1408))];
  weight_tensor_shared[((((int)threadIdx.x) + 1536))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 1536))];
  weight_tensor_shared[((((int)threadIdx.x) + 1664))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 1664))];
  weight_tensor_shared[((((int)threadIdx.x) + 1792))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 1792))];
  weight_tensor_shared[((((int)threadIdx.x) + 1920))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 1920))];
  weight_tensor_shared[((((int)threadIdx.x) + 2048))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 2048))];
  weight_tensor_shared[((((int)threadIdx.x) + 2176))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 2176))];
  weight_tensor_shared[((((int)threadIdx.x) + 2304))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 2304))];
  weight_tensor_shared[((((int)threadIdx.x) + 2432))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 2432))];
  weight_tensor_shared[((((int)threadIdx.x) + 2560))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 2560))];
  weight_tensor_shared[((((int)threadIdx.x) + 2688))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 2688))];
  weight_tensor_shared[((((int)threadIdx.x) + 2816))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 2816))];
  weight_tensor_shared[((((int)threadIdx.x) + 2944))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 2944))];
  weight_tensor_shared[((((int)threadIdx.x) + 3072))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 3072))];
  weight_tensor_shared[((((int)threadIdx.x) + 3200))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 3200))];
  weight_tensor_shared[((((int)threadIdx.x) + 3328))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 3328))];
  weight_tensor_shared[((((int)threadIdx.x) + 3456))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 3456))];
  weight_tensor_shared[((((int)threadIdx.x) + 3584))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 3584))];
  weight_tensor_shared[((((int)threadIdx.x) + 3712))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 3712))];
  weight_tensor_shared[((((int)threadIdx.x) + 3840))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 3840))];
  weight_tensor_shared[((((int)threadIdx.x) + 3968))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 3968))];
  weight_tensor_shared[((((int)threadIdx.x) + 4096))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 4096))];
  weight_tensor_shared[((((int)threadIdx.x) + 4224))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 4224))];
  weight_tensor_shared[((((int)threadIdx.x) + 4352))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 4352))];
  weight_tensor_shared[((((int)threadIdx.x) + 4480))] = weight_tensor[(((((((int)blockIdx.x) / 7) * 4608) + ((int)threadIdx.x)) + 4480))];