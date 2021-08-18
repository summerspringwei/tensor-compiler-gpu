#ifndef AFFINE_TRANSFORM_HPP
#define AFFINE_TRANSFORM_HPP

template<typename T>
struct Point_
{
    T x;
    T y;
};

typedef Point_<float> Point2f;

void get_affine_transform_cv(double trans[6], Point2f* src, Point2f* dst);
void affine_transform(float* target_coords, float* coords, float* trans, int batch, int n, int loop_count);
void get_affine_transform(double* trans, Point2f center, Point2f scale, float rot, Point2f output_size, Point2f shift);
void transform_preds(float* target_coords, float* coords, int batch, int n, Point2f center, Point2f scale, Point2f output_size);
#endif
