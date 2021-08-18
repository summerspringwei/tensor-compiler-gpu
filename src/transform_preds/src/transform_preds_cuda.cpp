
#include "affine_transform_cuda.hpp"

void transform_preds(float* target_coords, float* coords, int batch, int n, Point2f center, Point2f scale, Point2f output_size){
    double trans[6];
    Point2f shift {0, 0};
    get_affine_transform(trans, center, scale, 0, output_size, shift);
    float transf[6];
    for(int i=0; i<6; ++i){
        transf[i] = (float)trans[i];
    }
    affine_transform(target_coords, coords, transf, batch, n, 1);
}

