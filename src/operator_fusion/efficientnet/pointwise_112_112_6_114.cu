//grid=(392,1,1),  block=(38,1,1)

#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(38) pointwise_112_112_6_114(float* __restrict__ input, float* __restrict__ weight, float* __restrict__ output) {
  float output_local[96];
  __shared__ float input_shared[192];
  __shared__ float weight_shared[684];
  output_local[(0)] = 0.000000e+00f;
  output_local[(3)] = 0.000000e+00f;
  output_local[(1)] = 0.000000e+00f;
  output_local[(4)] = 0.000000e+00f;
  output_local[(2)] = 0.000000e+00f;
  output_local[(5)] = 0.000000e+00f;
  output_local[(6)] = 0.000000e+00f;
  output_local[(9)] = 0.000000e+00f;
  output_local[(7)] = 0.000000e+00f;
  output_local[(10)] = 0.000000e+00f;
  output_local[(8)] = 0.000000e+00f;
  output_local[(11)] = 0.000000e+00f;
  output_local[(12)] = 0.000000e+00f;
  output_local[(15)] = 0.000000e+00f;
  output_local[(13)] = 0.000000e+00f;
  output_local[(16)] = 0.000000e+00f;
  output_local[(14)] = 0.000000e+00f;
  output_local[(17)] = 0.000000e+00f;
  output_local[(18)] = 0.000000e+00f;
  output_local[(21)] = 0.000000e+00f;
  output_local[(19)] = 0.000000e+00f;
  output_local[(22)] = 0.000000e+00f;
  output_local[(20)] = 0.000000e+00f;
  output_local[(23)] = 0.000000e+00f;
  output_local[(24)] = 0.000000e+00f;
  output_local[(27)] = 0.000000e+00f;
  output_local[(25)] = 0.000000e+00f;
  output_local[(28)] = 0.000000e+00f;
  output_local[(26)] = 0.000000e+00f;
  output_local[(29)] = 0.000000e+00f;
  output_local[(30)] = 0.000000e+00f;
  output_local[(33)] = 0.000000e+00f;
  output_local[(31)] = 0.000000e+00f;
  output_local[(34)] = 0.000000e+00f;
  output_local[(32)] = 0.000000e+00f;
  output_local[(35)] = 0.000000e+00f;
  output_local[(36)] = 0.000000e+00f;
  output_local[(39)] = 0.000000e+00f;
  output_local[(37)] = 0.000000e+00f;
  output_local[(40)] = 0.000000e+00f;
  output_local[(38)] = 0.000000e+00f;
  output_local[(41)] = 0.000000e+00f;
  output_local[(42)] = 0.000000e+00f;
  output_local[(45)] = 0.000000e+00f;
  output_local[(43)] = 0.000000e+00f;
  output_local[(46)] = 0.000000e+00f;
  output_local[(44)] = 0.000000e+00f;
  output_local[(47)] = 0.000000e+00f;
  output_local[(48)] = 0.000000e+00f;
  output_local[(51)] = 0.000000e+00f;
  output_local[(49)] = 0.000000e+00f;
  output_local[(52)] = 0.000000e+00f;
  output_local[(50)] = 0.000000e+00f;
  output_local[(53)] = 0.000000e+00f;
  output_local[(54)] = 0.000000e+00f;
  output_local[(57)] = 0.000000e+00f;
  output_local[(55)] = 0.000000e+00f;
  output_local[(58)] = 0.000000e+00f;
  output_local[(56)] = 0.000000e+00f;
  output_local[(59)] = 0.000000e+00f;
  output_local[(60)] = 0.000000e+00f;
  output_local[(63)] = 0.000000e+00f;
  output_local[(61)] = 0.000000e+00f;
  output_local[(64)] = 0.000000e+00f;
  output_local[(62)] = 0.000000e+00f;
  output_local[(65)] = 0.000000e+00f;
  output_local[(66)] = 0.000000e+00f;
  output_local[(69)] = 0.000000e+00f;
  output_local[(67)] = 0.000000e+00f;
  output_local[(70)] = 0.000000e+00f;
  output_local[(68)] = 0.000000e+00f;
  output_local[(71)] = 0.000000e+00f;
  output_local[(72)] = 0.000000e+00f;
  output_local[(75)] = 0.000000e+00f;
  output_local[(73)] = 0.000000e+00f;
  output_local[(76)] = 0.000000e+00f;
  output_local[(74)] = 0.000000e+00f;
  output_local[(77)] = 0.000000e+00f;
  output_local[(78)] = 0.000000e+00f;
  output_local[(81)] = 0.000000e+00f;
  output_local[(79)] = 0.000000e+00f;
  output_local[(82)] = 0.000000e+00f;
  output_local[(80)] = 0.000000e+00f;
  output_local[(83)] = 0.000000e+00f;
  output_local[(84)] = 0.000000e+00f;
  output_local[(87)] = 0.000000e+00f;
  output_local[(85)] = 0.000000e+00f;
  output_local[(88)] = 0.000000e+00f;
  output_local[(86)] = 0.000000e+00f;
  output_local[(89)] = 0.000000e+00f;
  output_local[(90)] = 0.000000e+00f;
  output_local[(93)] = 0.000000e+00f;
  output_local[(91)] = 0.000000e+00f;
  output_local[(94)] = 0.000000e+00f;
  output_local[(92)] = 0.000000e+00f;
  output_local[(95)] = 0.000000e+00f;
  input_shared[(((int)threadIdx.x))] = input[(((((int)blockIdx.x) * 192) + ((int)threadIdx.x)))];
  input_shared[((((int)threadIdx.x) + 38))] = input[((((((int)blockIdx.x) * 192) + ((int)threadIdx.x)) + 38))];
  input_shared[((((int)threadIdx.x) + 76))] = input[((((((int)blockIdx.x) * 192) + ((int)threadIdx.x)) + 76))];
  input_shared[((((int)threadIdx.x) + 114))] = input[((((((int)blockIdx.x) * 192) + ((int)threadIdx.x)) + 114))];
  input_shared[((((int)threadIdx.x) + 152))] = input[((((((int)blockIdx.x) * 192) + ((int)threadIdx.x)) + 152))];
  if (((int)threadIdx.x) < 2) {
    input_shared[((((int)threadIdx.x) + 190))] = input[((((((int)blockIdx.x) * 192) + ((int)threadIdx.x)) + 190))];
  }
  ((float2*)(weight_shared + ((((int)threadIdx.x) * 2))))[0] = ((float2*)(weight + ((((int)threadIdx.x) * 2))))[0];
  ((float2*)(weight_shared + (((((int)threadIdx.x) * 2) + 76))))[0] = ((float2*)(weight + (((((int)threadIdx.x) * 2) + 76))))[0];
  ((float2*)(weight_shared + (((((int)threadIdx.x) * 2) + 152))))[0] = ((float2*)(weight + (((((int)threadIdx.x) * 2) + 152))))[0];
  ((float2*)(weight_shared + (((((int)threadIdx.x) * 2) + 228))))[0] = ((float2*)(weight + (((((int)threadIdx.x) * 2) + 228))))[0];
  ((float2*)(weight_shared + (((((int)threadIdx.x) * 2) + 304))))[0] = ((float2*)(weight + (((((int)threadIdx.x) * 2) + 304))))[0];
  ((float2*)(weight_shared + (((((int)threadIdx.x) * 2) + 380))))[0] = ((float2*)(weight + (((((int)threadIdx.x) * 2) + 380))))[0];
  ((float2*)(weight_shared + (((((int)threadIdx.x) * 2) + 456))))[0] = ((float2*)(weight + (((((int)threadIdx.x) * 2) + 456))))[0];
  ((float2*)(weight_shared + (((((int)threadIdx.x) * 2) + 532))))[0] = ((float2*)(weight + (((((int)threadIdx.x) * 2) + 532))))[0];
  ((float2*)(weight_shared + (((((int)threadIdx.x) * 2) + 608))))[0] = ((float2*)(weight + (((((int)threadIdx.x) * 2) + 608))))[0];
  __syncthreads();
  for (int rk_outer_inner = 0; rk_outer_inner < 3; ++rk_outer_inner) {
    output_local[(0)] = (output_local[(0)] + (input_shared[((rk_outer_inner * 2))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(3)] = (output_local[(3)] + (input_shared[(((rk_outer_inner * 2) + 6))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(0)] = (output_local[(0)] + (input_shared[(((rk_outer_inner * 2) + 1))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(3)] = (output_local[(3)] + (input_shared[(((rk_outer_inner * 2) + 7))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(1)] = (output_local[(1)] + (input_shared[((rk_outer_inner * 2))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(4)] = (output_local[(4)] + (input_shared[(((rk_outer_inner * 2) + 6))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(1)] = (output_local[(1)] + (input_shared[(((rk_outer_inner * 2) + 1))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(4)] = (output_local[(4)] + (input_shared[(((rk_outer_inner * 2) + 7))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(2)] = (output_local[(2)] + (input_shared[((rk_outer_inner * 2))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(5)] = (output_local[(5)] + (input_shared[(((rk_outer_inner * 2) + 6))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(2)] = (output_local[(2)] + (input_shared[(((rk_outer_inner * 2) + 1))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(5)] = (output_local[(5)] + (input_shared[(((rk_outer_inner * 2) + 7))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(6)] = (output_local[(6)] + (input_shared[(((rk_outer_inner * 2) + 12))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(9)] = (output_local[(9)] + (input_shared[(((rk_outer_inner * 2) + 18))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(6)] = (output_local[(6)] + (input_shared[(((rk_outer_inner * 2) + 13))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(9)] = (output_local[(9)] + (input_shared[(((rk_outer_inner * 2) + 19))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(7)] = (output_local[(7)] + (input_shared[(((rk_outer_inner * 2) + 12))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(10)] = (output_local[(10)] + (input_shared[(((rk_outer_inner * 2) + 18))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(7)] = (output_local[(7)] + (input_shared[(((rk_outer_inner * 2) + 13))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(10)] = (output_local[(10)] + (input_shared[(((rk_outer_inner * 2) + 19))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(8)] = (output_local[(8)] + (input_shared[(((rk_outer_inner * 2) + 12))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(11)] = (output_local[(11)] + (input_shared[(((rk_outer_inner * 2) + 18))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(8)] = (output_local[(8)] + (input_shared[(((rk_outer_inner * 2) + 13))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(11)] = (output_local[(11)] + (input_shared[(((rk_outer_inner * 2) + 19))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(12)] = (output_local[(12)] + (input_shared[(((rk_outer_inner * 2) + 24))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(15)] = (output_local[(15)] + (input_shared[(((rk_outer_inner * 2) + 30))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(12)] = (output_local[(12)] + (input_shared[(((rk_outer_inner * 2) + 25))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(15)] = (output_local[(15)] + (input_shared[(((rk_outer_inner * 2) + 31))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(13)] = (output_local[(13)] + (input_shared[(((rk_outer_inner * 2) + 24))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(16)] = (output_local[(16)] + (input_shared[(((rk_outer_inner * 2) + 30))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(13)] = (output_local[(13)] + (input_shared[(((rk_outer_inner * 2) + 25))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(16)] = (output_local[(16)] + (input_shared[(((rk_outer_inner * 2) + 31))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(14)] = (output_local[(14)] + (input_shared[(((rk_outer_inner * 2) + 24))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(17)] = (output_local[(17)] + (input_shared[(((rk_outer_inner * 2) + 30))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(14)] = (output_local[(14)] + (input_shared[(((rk_outer_inner * 2) + 25))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(17)] = (output_local[(17)] + (input_shared[(((rk_outer_inner * 2) + 31))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(18)] = (output_local[(18)] + (input_shared[(((rk_outer_inner * 2) + 36))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(21)] = (output_local[(21)] + (input_shared[(((rk_outer_inner * 2) + 42))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(18)] = (output_local[(18)] + (input_shared[(((rk_outer_inner * 2) + 37))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(21)] = (output_local[(21)] + (input_shared[(((rk_outer_inner * 2) + 43))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(19)] = (output_local[(19)] + (input_shared[(((rk_outer_inner * 2) + 36))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(22)] = (output_local[(22)] + (input_shared[(((rk_outer_inner * 2) + 42))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(19)] = (output_local[(19)] + (input_shared[(((rk_outer_inner * 2) + 37))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(22)] = (output_local[(22)] + (input_shared[(((rk_outer_inner * 2) + 43))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(20)] = (output_local[(20)] + (input_shared[(((rk_outer_inner * 2) + 36))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(23)] = (output_local[(23)] + (input_shared[(((rk_outer_inner * 2) + 42))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(20)] = (output_local[(20)] + (input_shared[(((rk_outer_inner * 2) + 37))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(23)] = (output_local[(23)] + (input_shared[(((rk_outer_inner * 2) + 43))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(24)] = (output_local[(24)] + (input_shared[(((rk_outer_inner * 2) + 48))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(27)] = (output_local[(27)] + (input_shared[(((rk_outer_inner * 2) + 54))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(24)] = (output_local[(24)] + (input_shared[(((rk_outer_inner * 2) + 49))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(27)] = (output_local[(27)] + (input_shared[(((rk_outer_inner * 2) + 55))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(25)] = (output_local[(25)] + (input_shared[(((rk_outer_inner * 2) + 48))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(28)] = (output_local[(28)] + (input_shared[(((rk_outer_inner * 2) + 54))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(25)] = (output_local[(25)] + (input_shared[(((rk_outer_inner * 2) + 49))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(28)] = (output_local[(28)] + (input_shared[(((rk_outer_inner * 2) + 55))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(26)] = (output_local[(26)] + (input_shared[(((rk_outer_inner * 2) + 48))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(29)] = (output_local[(29)] + (input_shared[(((rk_outer_inner * 2) + 54))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(26)] = (output_local[(26)] + (input_shared[(((rk_outer_inner * 2) + 49))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(29)] = (output_local[(29)] + (input_shared[(((rk_outer_inner * 2) + 55))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(30)] = (output_local[(30)] + (input_shared[(((rk_outer_inner * 2) + 60))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(33)] = (output_local[(33)] + (input_shared[(((rk_outer_inner * 2) + 66))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(30)] = (output_local[(30)] + (input_shared[(((rk_outer_inner * 2) + 61))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(33)] = (output_local[(33)] + (input_shared[(((rk_outer_inner * 2) + 67))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(31)] = (output_local[(31)] + (input_shared[(((rk_outer_inner * 2) + 60))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(34)] = (output_local[(34)] + (input_shared[(((rk_outer_inner * 2) + 66))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(31)] = (output_local[(31)] + (input_shared[(((rk_outer_inner * 2) + 61))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(34)] = (output_local[(34)] + (input_shared[(((rk_outer_inner * 2) + 67))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(32)] = (output_local[(32)] + (input_shared[(((rk_outer_inner * 2) + 60))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(35)] = (output_local[(35)] + (input_shared[(((rk_outer_inner * 2) + 66))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(32)] = (output_local[(32)] + (input_shared[(((rk_outer_inner * 2) + 61))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(35)] = (output_local[(35)] + (input_shared[(((rk_outer_inner * 2) + 67))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(36)] = (output_local[(36)] + (input_shared[(((rk_outer_inner * 2) + 72))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(39)] = (output_local[(39)] + (input_shared[(((rk_outer_inner * 2) + 78))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(36)] = (output_local[(36)] + (input_shared[(((rk_outer_inner * 2) + 73))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(39)] = (output_local[(39)] + (input_shared[(((rk_outer_inner * 2) + 79))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(37)] = (output_local[(37)] + (input_shared[(((rk_outer_inner * 2) + 72))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(40)] = (output_local[(40)] + (input_shared[(((rk_outer_inner * 2) + 78))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(37)] = (output_local[(37)] + (input_shared[(((rk_outer_inner * 2) + 73))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(40)] = (output_local[(40)] + (input_shared[(((rk_outer_inner * 2) + 79))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(38)] = (output_local[(38)] + (input_shared[(((rk_outer_inner * 2) + 72))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(41)] = (output_local[(41)] + (input_shared[(((rk_outer_inner * 2) + 78))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(38)] = (output_local[(38)] + (input_shared[(((rk_outer_inner * 2) + 73))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(41)] = (output_local[(41)] + (input_shared[(((rk_outer_inner * 2) + 79))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(42)] = (output_local[(42)] + (input_shared[(((rk_outer_inner * 2) + 84))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(45)] = (output_local[(45)] + (input_shared[(((rk_outer_inner * 2) + 90))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(42)] = (output_local[(42)] + (input_shared[(((rk_outer_inner * 2) + 85))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(45)] = (output_local[(45)] + (input_shared[(((rk_outer_inner * 2) + 91))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(43)] = (output_local[(43)] + (input_shared[(((rk_outer_inner * 2) + 84))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(46)] = (output_local[(46)] + (input_shared[(((rk_outer_inner * 2) + 90))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(43)] = (output_local[(43)] + (input_shared[(((rk_outer_inner * 2) + 85))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(46)] = (output_local[(46)] + (input_shared[(((rk_outer_inner * 2) + 91))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(44)] = (output_local[(44)] + (input_shared[(((rk_outer_inner * 2) + 84))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(47)] = (output_local[(47)] + (input_shared[(((rk_outer_inner * 2) + 90))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(44)] = (output_local[(44)] + (input_shared[(((rk_outer_inner * 2) + 85))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(47)] = (output_local[(47)] + (input_shared[(((rk_outer_inner * 2) + 91))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(48)] = (output_local[(48)] + (input_shared[(((rk_outer_inner * 2) + 96))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(51)] = (output_local[(51)] + (input_shared[(((rk_outer_inner * 2) + 102))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(48)] = (output_local[(48)] + (input_shared[(((rk_outer_inner * 2) + 97))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(51)] = (output_local[(51)] + (input_shared[(((rk_outer_inner * 2) + 103))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(49)] = (output_local[(49)] + (input_shared[(((rk_outer_inner * 2) + 96))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(52)] = (output_local[(52)] + (input_shared[(((rk_outer_inner * 2) + 102))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(49)] = (output_local[(49)] + (input_shared[(((rk_outer_inner * 2) + 97))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(52)] = (output_local[(52)] + (input_shared[(((rk_outer_inner * 2) + 103))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(50)] = (output_local[(50)] + (input_shared[(((rk_outer_inner * 2) + 96))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(53)] = (output_local[(53)] + (input_shared[(((rk_outer_inner * 2) + 102))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(50)] = (output_local[(50)] + (input_shared[(((rk_outer_inner * 2) + 97))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(53)] = (output_local[(53)] + (input_shared[(((rk_outer_inner * 2) + 103))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(54)] = (output_local[(54)] + (input_shared[(((rk_outer_inner * 2) + 108))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(57)] = (output_local[(57)] + (input_shared[(((rk_outer_inner * 2) + 114))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(54)] = (output_local[(54)] + (input_shared[(((rk_outer_inner * 2) + 109))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(57)] = (output_local[(57)] + (input_shared[(((rk_outer_inner * 2) + 115))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(55)] = (output_local[(55)] + (input_shared[(((rk_outer_inner * 2) + 108))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(58)] = (output_local[(58)] + (input_shared[(((rk_outer_inner * 2) + 114))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(55)] = (output_local[(55)] + (input_shared[(((rk_outer_inner * 2) + 109))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(58)] = (output_local[(58)] + (input_shared[(((rk_outer_inner * 2) + 115))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(56)] = (output_local[(56)] + (input_shared[(((rk_outer_inner * 2) + 108))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(59)] = (output_local[(59)] + (input_shared[(((rk_outer_inner * 2) + 114))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(56)] = (output_local[(56)] + (input_shared[(((rk_outer_inner * 2) + 109))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(59)] = (output_local[(59)] + (input_shared[(((rk_outer_inner * 2) + 115))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(60)] = (output_local[(60)] + (input_shared[(((rk_outer_inner * 2) + 120))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(63)] = (output_local[(63)] + (input_shared[(((rk_outer_inner * 2) + 126))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(60)] = (output_local[(60)] + (input_shared[(((rk_outer_inner * 2) + 121))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(63)] = (output_local[(63)] + (input_shared[(((rk_outer_inner * 2) + 127))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(61)] = (output_local[(61)] + (input_shared[(((rk_outer_inner * 2) + 120))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(64)] = (output_local[(64)] + (input_shared[(((rk_outer_inner * 2) + 126))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(61)] = (output_local[(61)] + (input_shared[(((rk_outer_inner * 2) + 121))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(64)] = (output_local[(64)] + (input_shared[(((rk_outer_inner * 2) + 127))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(62)] = (output_local[(62)] + (input_shared[(((rk_outer_inner * 2) + 120))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(65)] = (output_local[(65)] + (input_shared[(((rk_outer_inner * 2) + 126))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(62)] = (output_local[(62)] + (input_shared[(((rk_outer_inner * 2) + 121))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(65)] = (output_local[(65)] + (input_shared[(((rk_outer_inner * 2) + 127))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(66)] = (output_local[(66)] + (input_shared[(((rk_outer_inner * 2) + 132))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(69)] = (output_local[(69)] + (input_shared[(((rk_outer_inner * 2) + 138))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(66)] = (output_local[(66)] + (input_shared[(((rk_outer_inner * 2) + 133))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(69)] = (output_local[(69)] + (input_shared[(((rk_outer_inner * 2) + 139))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(67)] = (output_local[(67)] + (input_shared[(((rk_outer_inner * 2) + 132))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(70)] = (output_local[(70)] + (input_shared[(((rk_outer_inner * 2) + 138))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(67)] = (output_local[(67)] + (input_shared[(((rk_outer_inner * 2) + 133))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(70)] = (output_local[(70)] + (input_shared[(((rk_outer_inner * 2) + 139))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(68)] = (output_local[(68)] + (input_shared[(((rk_outer_inner * 2) + 132))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(71)] = (output_local[(71)] + (input_shared[(((rk_outer_inner * 2) + 138))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(68)] = (output_local[(68)] + (input_shared[(((rk_outer_inner * 2) + 133))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(71)] = (output_local[(71)] + (input_shared[(((rk_outer_inner * 2) + 139))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(72)] = (output_local[(72)] + (input_shared[(((rk_outer_inner * 2) + 144))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(75)] = (output_local[(75)] + (input_shared[(((rk_outer_inner * 2) + 150))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(72)] = (output_local[(72)] + (input_shared[(((rk_outer_inner * 2) + 145))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(75)] = (output_local[(75)] + (input_shared[(((rk_outer_inner * 2) + 151))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(73)] = (output_local[(73)] + (input_shared[(((rk_outer_inner * 2) + 144))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(76)] = (output_local[(76)] + (input_shared[(((rk_outer_inner * 2) + 150))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(73)] = (output_local[(73)] + (input_shared[(((rk_outer_inner * 2) + 145))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(76)] = (output_local[(76)] + (input_shared[(((rk_outer_inner * 2) + 151))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(74)] = (output_local[(74)] + (input_shared[(((rk_outer_inner * 2) + 144))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(77)] = (output_local[(77)] + (input_shared[(((rk_outer_inner * 2) + 150))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(74)] = (output_local[(74)] + (input_shared[(((rk_outer_inner * 2) + 145))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(77)] = (output_local[(77)] + (input_shared[(((rk_outer_inner * 2) + 151))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(78)] = (output_local[(78)] + (input_shared[(((rk_outer_inner * 2) + 156))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(81)] = (output_local[(81)] + (input_shared[(((rk_outer_inner * 2) + 162))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(78)] = (output_local[(78)] + (input_shared[(((rk_outer_inner * 2) + 157))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(81)] = (output_local[(81)] + (input_shared[(((rk_outer_inner * 2) + 163))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(79)] = (output_local[(79)] + (input_shared[(((rk_outer_inner * 2) + 156))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(82)] = (output_local[(82)] + (input_shared[(((rk_outer_inner * 2) + 162))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(79)] = (output_local[(79)] + (input_shared[(((rk_outer_inner * 2) + 157))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(82)] = (output_local[(82)] + (input_shared[(((rk_outer_inner * 2) + 163))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(80)] = (output_local[(80)] + (input_shared[(((rk_outer_inner * 2) + 156))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(83)] = (output_local[(83)] + (input_shared[(((rk_outer_inner * 2) + 162))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(80)] = (output_local[(80)] + (input_shared[(((rk_outer_inner * 2) + 157))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(83)] = (output_local[(83)] + (input_shared[(((rk_outer_inner * 2) + 163))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(84)] = (output_local[(84)] + (input_shared[(((rk_outer_inner * 2) + 168))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(87)] = (output_local[(87)] + (input_shared[(((rk_outer_inner * 2) + 174))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(84)] = (output_local[(84)] + (input_shared[(((rk_outer_inner * 2) + 169))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(87)] = (output_local[(87)] + (input_shared[(((rk_outer_inner * 2) + 175))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(85)] = (output_local[(85)] + (input_shared[(((rk_outer_inner * 2) + 168))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(88)] = (output_local[(88)] + (input_shared[(((rk_outer_inner * 2) + 174))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(85)] = (output_local[(85)] + (input_shared[(((rk_outer_inner * 2) + 169))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(88)] = (output_local[(88)] + (input_shared[(((rk_outer_inner * 2) + 175))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(86)] = (output_local[(86)] + (input_shared[(((rk_outer_inner * 2) + 168))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(89)] = (output_local[(89)] + (input_shared[(((rk_outer_inner * 2) + 174))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(86)] = (output_local[(86)] + (input_shared[(((rk_outer_inner * 2) + 169))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(89)] = (output_local[(89)] + (input_shared[(((rk_outer_inner * 2) + 175))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(90)] = (output_local[(90)] + (input_shared[(((rk_outer_inner * 2) + 180))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(93)] = (output_local[(93)] + (input_shared[(((rk_outer_inner * 2) + 186))] * weight_shared[(((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)))]));
    output_local[(90)] = (output_local[(90)] + (input_shared[(((rk_outer_inner * 2) + 181))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(93)] = (output_local[(93)] + (input_shared[(((rk_outer_inner * 2) + 187))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 1))]));
    output_local[(91)] = (output_local[(91)] + (input_shared[(((rk_outer_inner * 2) + 180))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(94)] = (output_local[(94)] + (input_shared[(((rk_outer_inner * 2) + 186))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 6))]));
    output_local[(91)] = (output_local[(91)] + (input_shared[(((rk_outer_inner * 2) + 181))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(94)] = (output_local[(94)] + (input_shared[(((rk_outer_inner * 2) + 187))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 7))]));
    output_local[(92)] = (output_local[(92)] + (input_shared[(((rk_outer_inner * 2) + 180))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(95)] = (output_local[(95)] + (input_shared[(((rk_outer_inner * 2) + 186))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 12))]));
    output_local[(92)] = (output_local[(92)] + (input_shared[(((rk_outer_inner * 2) + 181))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
    output_local[(95)] = (output_local[(95)] + (input_shared[(((rk_outer_inner * 2) + 187))] * weight_shared[((((((int)threadIdx.x) * 18) + (rk_outer_inner * 2)) + 13))]));
  }
  for (int i_inner = 0; i_inner < 32; ++i_inner) {
    for (int j_inner = 0; j_inner < 3; ++j_inner) {
      output[(((((((int)blockIdx.x) * 3648) + (i_inner * 114)) + (((int)threadIdx.x) * 3)) + j_inner))] = output_local[(((i_inner * 3) + j_inner))];
    }
  }
}