#include "affine_transform_cuda.hpp"

#include <cmath>
#include <math.h>

void format_affine_transform_cpu(double* a, double* b, 
    Point2f center, Point2f scale, float rot, 
    Point2f output_size, Point2f shift, int inv){
    float src_w = scale.x;
    float dst_w = output_size.x;
    float dst_h = output_size.y;

    float rot_pad = M_PI * rot / 180;
    float sn = std::sin(rot_pad), cs = std::cos(rot_pad);
    Point2f src_dir;
    src_dir.x = 0 * cs - (src_w * (-0.5)) * sn;
    src_dir.y = 0 * sn + (src_w * (-0.5)) * cs;
    Point2f dst_dir{0, dst_w * (-0.5f)};
    Point2f src[3];
    Point2f dst[3];
    src[0].x = center.x + scale.x * shift.x;
    src[0].y = center.y + scale.y * shift.y;
    src[1].x = center.x + src_dir.x + scale.x * shift.x;
    src[1].y = center.y + src_dir.y + scale.y * shift.y;
    dst[0].x = dst_w * 0.5;dst[0].y = dst_h * 0.5;
    dst[1].x = dst_w * 0.5 + dst_dir.x;
    dst[1].y = dst_h * 0.5 + dst_dir.y;
    
    // replace get_3rd_point
    src[2].x = src[1].x + src[1].y - src[0].y;
    src[2].y = src[1].y + src[0].x - src[1].x;
    dst[2].x = dst[1].x + dst[1].y - dst[0].y;
    dst[2].y = dst[1].y + dst[0].x - dst[1].x;
    if(inv){
        for(int i=0; i<3; ++i){
            Point2f tmp;
            tmp = src[i];
            src[i] = dst[i];
            dst[i] = tmp;
        }
    }
    printf("src: %f, %f\n %f, %f\n %f, %f\n", src[0].x, src[0].y, src[1].x, src[1].y, src[2].x, src[2].y);
    printf("dst: %f, %f\n %f, %f\n %f, %f\n", dst[0].x, dst[0].y, dst[1].x, dst[1].y, dst[2].x, dst[2].y);
    // Get Affine transform using cuda linear system
    // Init a
    // for(int i=0; i<6*6; ++i){
    //     a[i] = 0;
    // }
    // for( int i = 0; i < 3; i++ ){
    //     int j = i*6;
    //     int k = i*6+21;
    //     a[j] = a[k] = src[i].x;
    //     a[j+1] = a[k+1] = src[i].y;
    //     a[j+2] = a[k+2] = 1;
    //     b[i] = dst[i].x;
    //     b[i+3] = dst[i].y;
    // }
    
    // transpose<double>(a, 6);
    // Copy from opencv source code
    for( int i = 0; i < 3; i++ )
    {
        int j = i*12;
        int k = i*12+6;
        a[j] = a[k+3] = src[i].x;
        a[j+1] = a[k+4] = src[i].y;
        a[j+2] = a[k+5] = 1;
        a[j+3] = a[j+4] = a[j+5] = 0;
        a[k] = a[k+1] = a[k+2] = 0;
        b[i*2] = dst[i].x;
        b[i*2+1] = dst[i].y;
    }
}
