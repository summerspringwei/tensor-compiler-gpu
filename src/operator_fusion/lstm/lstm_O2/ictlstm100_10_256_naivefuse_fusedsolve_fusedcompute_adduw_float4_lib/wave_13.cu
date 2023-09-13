#include "include/lstmlib.cuh"
__global__ void __launch_bounds__(256, 1)
    wave13(WaveInputParams *__restrict__ input,
           WaveModelParams *__restrict__ model,
           WaveOutputParams *__restrict__ output) {
    switch (blockIdx.x >> 3) {
    case 0:
        call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_float4(
            13 * LstmScaleParams::kCellNumber10 + 0, 0,
            13 * LstmScaleParams::kCellNumber10 + 0,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 1:
        call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_float4(
            12 * LstmScaleParams::kCellNumber10 + 1, 1,
            12 * LstmScaleParams::kCellNumber10 + 1,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 2:
        call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_float4(
            11 * LstmScaleParams::kCellNumber10 + 2, 2,
            11 * LstmScaleParams::kCellNumber10 + 2,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 3:
        call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_float4(
            10 * LstmScaleParams::kCellNumber10 + 3, 3,
            10 * LstmScaleParams::kCellNumber10 + 3,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 4:
        call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_float4(
            9 * LstmScaleParams::kCellNumber10 + 4, 4,
            9 * LstmScaleParams::kCellNumber10 + 4,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 5:
        call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_float4(
            8 * LstmScaleParams::kCellNumber10 + 5, 5,
            8 * LstmScaleParams::kCellNumber10 + 5,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 6:
        call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_float4(
            7 * LstmScaleParams::kCellNumber10 + 6, 6,
            7 * LstmScaleParams::kCellNumber10 + 6,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 7:
        call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_float4(
            6 * LstmScaleParams::kCellNumber10 + 7, 7,
            6 * LstmScaleParams::kCellNumber10 + 7,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 8:
        call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_float4(
            5 * LstmScaleParams::kCellNumber10 + 8, 8,
            5 * LstmScaleParams::kCellNumber10 + 8,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    case 9:
        call_onekernel_naivefuse_fusedsolve_fusedcompute_adduw_float4(
            4 * LstmScaleParams::kCellNumber10 + 9, 9,
            4 * LstmScaleParams::kCellNumber10 + 9,
            LstmScaleParams::kColumsPerBlock32, LstmScaleParams::kHiddenSize256,
            LstmScaleParams::kInputSize256,
            LstmScaleParams::kThreadNumPerBlock256, LstmScaleParams::kMask7);
        break;
    }
}