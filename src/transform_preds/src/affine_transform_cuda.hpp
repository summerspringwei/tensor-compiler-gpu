#ifndef AFFINE_TRANSFORM_HPP
#define AFFINE_TRANSFORM_HPP

#include <torch/extension.h>

template<typename T>
struct Point_
{
    T x;
    T y;
};

typedef Point_<float> Point2f;

void get_affine_transform_cv(double trans[6], Point2f* src, Point2f* dst);
void copy_dets_with_slice_cuda(float* d_coords, float* d_dets, int batch, int n, int slice_from, int slice_to);
void affine_transform(float* target_coords, float* coords, float* trans, int batch, int n, int loop_count);
void affine_transform_dets_cuda(float* target_dets, float* dets, float* trans, int batch, int n);
void affine_transform_dets_cuda(torch::PackedTensorAccessor64<float, 1> target_dets,
    torch::PackedTensorAccessor64<float, 1> dets, float* trans, int batch, int n);


void get_affine_transform(double* trans, Point2f center, Point2f scale, float rot, Point2f output_size, Point2f shift);
void transform_preds(float* target_coords, float* coords, int batch, int n, Point2f center, Point2f scale, Point2f output_size);
void transform_preds_cuda(float* d_target_coords, float* d_coords, int batch, int n, Point2f center, Point2f scale, Point2f output_size);
#endif
