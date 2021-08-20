#pragma once

void affine_transform_dets_cuda(float* target_dets, float* dets, float* trans, int batch, int n);
