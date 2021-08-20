
#pragma once

#include "cv_types.hpp"

void format_affine_transform_cpu(double* a, double* b, 
    Point2f center, Point2f scale, float rot, 
    Point2f output_size, Point2f shift, int inv);
