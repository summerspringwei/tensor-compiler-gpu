/*
 * How to compile (assume cuda is installed at /usr/local/cuda/)
 *   nvcc -c -I/usr/local/cuda/include ormqr_example.cpp 
 *   nvcc -o -fopenmp a.out ormqr_example.o -L/usr/local/cuda/lib64 -lcudart -lcublas -lcusolver
 *
 */

#include "affine_transform_cuda.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#include <chrono>


void printMatrix(int m, int n, const double*A, int lda, const char* name)
{
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row + col*lda];
            printf("%s(%d,%d) = %f\n", name, row+1, col+1, Areg);
        }
    }
}

void printMatrix(int m, int n, const double*A, const char* name)
{
    printf("%s:\n", name);
    for(int row = 0 ; row < m ; row++){
        for(int col = 0 ; col < n ; col++){
            double Areg = A[row*n + col];
            printf("%f, ", Areg);
        }printf("\n");
    }
}

void solve_lu(double* X, double* A, double *B, int m){
    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;

    cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    // const int m = 3;
    const int lda = m;
    const int ldb = m;
    // double A[lda*m] = { 1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0};
    // double B[m] = { 1.0, 2.0, 3.0 };
    // double X[m]; /* X = A\B */
    double LU[lda*m]; /* L and U */
    int Ipiv[m];      /* host copy of pivoting sequence */
    int info = 0;     /* host copy of error info */

    double *d_A = NULL; /* device copy of A */
    double *d_B = NULL; /* device copy of B */
    int *d_Ipiv = NULL; /* pivoting sequence */
    int *d_info = NULL; /* error info */
    int  lwork = 0;     /* size of workspace */
    double *d_work = NULL; /* device workspace for getrf */

    const int pivot_on = 0;

    printf("example of getrf \n");

    if (pivot_on){
        printf("pivot is on : compute P*A = L*U \n");
    }else{
        printf("pivot is off: compute A = L*U (not numerically stable)\n");
    }

    printf("A = (matlab base-1)\n");
    printMatrix(m, m, A, lda, "A");
    printf("=====\n");

    printf("B = (matlab base-1)\n");
    printMatrix(m, 1, B, ldb, "B");
    printf("=====\n");

/* step 1: create cusolver handle, bind a stream */
    status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    assert(cudaSuccess == cudaStat1);

    status = cusolverDnSetStream(cusolverH, stream);
    assert(CUSOLVER_STATUS_SUCCESS == status);

/* step 2: copy A to device */
    cudaStat1 = cudaMalloc ((void**)&d_A, sizeof(double) * lda * m);
    cudaStat2 = cudaMalloc ((void**)&d_B, sizeof(double) * m);
    cudaStat2 = cudaMalloc ((void**)&d_Ipiv, sizeof(int) * m);
    cudaStat4 = cudaMalloc ((void**)&d_info, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(double)*lda*m, cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_B, B, sizeof(double)*m, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    /* step 3: query working space of getrf */
    status = cusolverDnDgetrf_bufferSize(
        cusolverH,
        m,
        m,
        d_A,
        lda,
        &lwork);
    assert(CUSOLVER_STATUS_SUCCESS == status);

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);

/* step 4: LU factorization */
    if (pivot_on){
        status = cusolverDnDgetrf(
            cusolverH,
            m,
            m,
            d_A,
            lda,
            d_work,
            d_Ipiv,
            d_info);
    }else{
        status = cusolverDnDgetrf(
            cusolverH,
            m,
            m,
            d_A,
            lda,
            d_work,
            NULL,
            d_info);
    }
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);

    if (pivot_on){
    cudaStat1 = cudaMemcpy(Ipiv , d_Ipiv, sizeof(int)*m, cudaMemcpyDeviceToHost);
    }
    cudaStat2 = cudaMemcpy(LU   , d_A   , sizeof(double)*lda*m, cudaMemcpyDeviceToHost);
    cudaStat3 = cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);

    if ( 0 > info ){
        printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }
    if (pivot_on){
        printf("pivoting sequence, matlab base-1\n");
        for(int j = 0 ; j < m ; j++){
            printf("Ipiv(%d) = %d\n", j+1, Ipiv[j]);
        }
    }
    printf("L and U = (matlab base-1)\n");
    printMatrix(m, m, LU, lda, "LU");
    printf("=====\n");

    if (pivot_on){
        status = cusolverDnDgetrs(
            cusolverH,
            CUBLAS_OP_N,
            m,
            1, /* nrhs */
            d_A,
            lda,
            d_Ipiv,
            d_B,
            ldb,
            d_info);
    }else{
        status = cusolverDnDgetrs(
            cusolverH,
            CUBLAS_OP_N,
            m,
            1, /* nrhs */
            d_A,
            lda,
            NULL,
            d_B,
            ldb,
            d_info);
    }
    cudaStat1 = cudaDeviceSynchronize();
    assert(CUSOLVER_STATUS_SUCCESS == status);
    assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(X , d_B, sizeof(double)*m, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    printf("X = (matlab base-1)\n");
    printMatrix(m, 1, X, ldb, "X");
    printf("=====\n");

/* free resources */
    if (d_A    ) cudaFree(d_A);
    if (d_B    ) cudaFree(d_B);
    if (d_Ipiv ) cudaFree(d_Ipiv);
    if (d_info ) cudaFree(d_info);
    if (d_work ) cudaFree(d_work);

    if (cusolverH   ) cusolverDnDestroy(cusolverH);
    if (stream      ) cudaStreamDestroy(stream);

    cudaDeviceReset();

}

/*       | 1 2 3 |
 *   A = | 4 5 6 |
 *       | 2 1 1 |
 *
 *   x = (1 1 1)'
 *   b = (6 15 4)'
 */
void solve_cuda(double* XC, double* A, double *B, int m){
    cusolverDnHandle_t cusolverH = NULL;
    cublasHandle_t cublasH = NULL;
    cublasStatus_t cublas_status = CUBLAS_STATUS_SUCCESS;
    cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;    
    cudaError_t cudaStat1 = cudaSuccess;
    cudaError_t cudaStat2 = cudaSuccess;
    cudaError_t cudaStat3 = cudaSuccess;
    cudaError_t cudaStat4 = cudaSuccess;
    // const int m = 3;
    const int lda = m;
    const int ldb = m;
    const int nrhs = 1; // number of right hand side vectors

    // double A[lda*m] = { 1.0, 4.0, 2.0, 2.0, 5.0, 1.0, 3.0, 6.0, 1.0}; 
    // double X[ldb*nrhs] = { 1.0, 1.0, 1.0}; // exact solution
    // double B[ldb*nrhs] = { 6.0, 15.0, 4.0}; 
    // double XC[ldb*nrhs]; // solution matrix from GPU
/* device memory */
    double *d_A = NULL;
    double *d_tau = NULL;
    double *d_B  = NULL;
    int *devInfo = NULL;
    double *d_work = NULL;
    int  lwork_geqrf = 0;
    int  lwork_ormqr = 0;
    int  lwork = 0;

    int info_gpu = 0;

    const double one = 1;

    // printf("A = (matlab base-1)\n");
    // printMatrix(m, m, A, "A");
    // printf("=====\n");
    // printf("B = (matlab base-1)\n");
    // printMatrix(m, nrhs, B, "B");
    // printf("=====\n");

/* step 1: create cudense/cublas handle */
    cusolver_status = cusolverDnCreate(&cusolverH);
    assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);

    cublas_status = cublasCreate(&cublasH);
    assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    
/* step 2: copy A and B to device */
    cudaStat1 = cudaMalloc ((void**)&d_A  , sizeof(double) * lda * m);
    cudaStat2 = cudaMalloc ((void**)&d_tau, sizeof(double) * m);
    cudaStat3 = cudaMalloc ((void**)&d_B  , sizeof(double) * ldb * nrhs);
    cudaStat4 = cudaMalloc ((void**)&devInfo, sizeof(int));
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);
    assert(cudaSuccess == cudaStat3);
    assert(cudaSuccess == cudaStat4);

    cudaStat1 = cudaMemcpy(d_A, A, sizeof(double) * lda * m   , cudaMemcpyHostToDevice);
    cudaStat2 = cudaMemcpy(d_B, B, sizeof(double) * ldb * nrhs, cudaMemcpyHostToDevice);
    assert(cudaSuccess == cudaStat1);
    assert(cudaSuccess == cudaStat2);

    /* step 3: query working space of geqrf and ormqr */
    cusolver_status = cusolverDnDgeqrf_bufferSize(
        cusolverH,
        m,
        m,
        d_A,
        lda,
        &lwork_geqrf);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    cusolver_status= cusolverDnDormqr_bufferSize(
        cusolverH,
        CUBLAS_SIDE_LEFT,
        CUBLAS_OP_T,
        m,
        nrhs,
        m,
        d_A,
        lda,
        d_tau,
        d_B,
        ldb,
        &lwork_ormqr);
    assert (cusolver_status == CUSOLVER_STATUS_SUCCESS);

    lwork = (lwork_geqrf > lwork_ormqr)? lwork_geqrf : lwork_ormqr;

    cudaStat1 = cudaMalloc((void**)&d_work, sizeof(double)*lwork);
    assert(cudaSuccess == cudaStat1);

/* step 4: compute QR factorization */
    cusolver_status = cusolverDnDgeqrf(
        cusolverH, 
        m, 
        m, 
        d_A, 
        lda, 
        d_tau, 
        d_work, 
        lwork, 
        devInfo);
    // cudaStat1 = cudaDeviceSynchronize();
    // assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    // assert(cudaSuccess == cudaStat1);

    /* check if QR is good or not */
    // cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    // assert(cudaSuccess == cudaStat1);
    // printf("after geqrf: info_gpu = %d\n", info_gpu);
    // assert(0 == info_gpu);
/* step 5: compute Q^T*B */
    cusolver_status= cusolverDnDormqr(
        cusolverH,
        CUBLAS_SIDE_LEFT,
        CUBLAS_OP_T,
        m,
        nrhs,
        m,
        d_A,
        lda,
        d_tau,
        d_B,
        ldb,
        d_work,
        lwork,
        devInfo);
    // cudaStat1 = cudaDeviceSynchronize();
    // assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
    // assert(cudaSuccess == cudaStat1);
    /* check if QR is good or not */
    // cudaStat1 = cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost);
    // assert(cudaSuccess == cudaStat1);
    // printf("after ormqr: info_gpu = %d\n", info_gpu);
    // assert(0 == info_gpu);

/* step 6: compute x = R \ Q^T*B */
    cublas_status = cublasDtrsm(
         cublasH,
         CUBLAS_SIDE_LEFT,
         CUBLAS_FILL_MODE_UPPER,
         CUBLAS_OP_N, 
         CUBLAS_DIAG_NON_UNIT,
         m,
         nrhs,
         &one,
         d_A,
         lda,
         d_B,
         ldb);
    // cudaStat1 = cudaDeviceSynchronize();
    // assert(CUBLAS_STATUS_SUCCESS == cublas_status);
    // assert(cudaSuccess == cudaStat1);

    cudaStat1 = cudaMemcpy(XC, d_B, sizeof(double)*ldb*nrhs, cudaMemcpyDeviceToHost);
    assert(cudaSuccess == cudaStat1);

    // printf("X = (matlab base-1)\n");
    // printMatrix(m, nrhs, XC, "X");

/* free resources */
    if (d_A    ) cudaFree(d_A);
    if (d_tau  ) cudaFree(d_tau);
    if (d_B    ) cudaFree(d_B);
    if (devInfo) cudaFree(devInfo);
    if (d_work ) cudaFree(d_work);
    if (cublasH ) cublasDestroy(cublasH);   
    if (cusolverH) cusolverDnDestroy(cusolverH);   
    cudaDeviceReset();
}



void get_affine_transform_cv(double trans[6], Point2f* src, Point2f* dst){
    double a[6*6], b[6];
    // Init a
    for(int i=0; i<6*6; ++i){
        a[i] = 0;
    }
    for( int i = 0; i < 3; i++ ){
        int j = i*6;
        int k = i*6+21;
        a[j] = a[k] = src[i].x;
        a[j+1] = a[k+1] = src[i].y;
        a[j+2] = a[k+2] = 1;
        b[i] = dst[i].x;
        b[i+3] = dst[i].y;
    }
    
    transpose<double>(a, 6);
    solve_cuda(trans, a, b, 6);
}


#define CUDART_PI_F 3.141592654f

// Test pass
__global__ void format_affine_transform_kernel(Point2f center, Point2f scale, float rot, Point2f output_size, Point2f shift, Point2f* src, Point2f* dst){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx != 0){
        return;
    }
    float src_w = scale.x;
    float dst_w = output_size.x;
    float dst_h = output_size.y;

    float rot_pad = CUDART_PI_F * rot / 180;
    float sn = sinf(rot_pad), cs = cosf(rot_pad);
    Point2f src_dir;
    src_dir.x = 0 * cs - (src_w * (-0.5)) * sn;
    src_dir.y = 0 * sn + (src_w * (-0.5)) * cs;
    Point2f dst_dir{0, dst_w * (-0.5f)};
    // Point2f src[3];
    // Point2f dst[3];
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
    // Get Affine transform using cuda linear system
    
}

void get_affine_transform(double* trans, Point2f center, Point2f scale, float rot, Point2f output_size, Point2f shift){
    Point2f* d_src;
    Point2f* d_dst;
    cudaMalloc((Point2f **)&d_src, sizeof(Point2f) * 3);
    cudaMalloc((Point2f **)&d_dst, sizeof(Point2f) * 3);
    int block_size = 256;
    dim3 threadsPerBlock(block_size);
    dim3 numBlocks(1);
    format_affine_transform_kernel<<<threadsPerBlock, numBlocks>>>(center, scale, rot, output_size, shift, d_src, d_dst);
    cudaDeviceSynchronize();
    Point2f src[3], dst[3];
    cudaMemcpy(src, d_src, sizeof(Point2f) * 3, cudaMemcpyDeviceToHost);
    cudaMemcpy(dst, d_dst, sizeof(Point2f) * 3, cudaMemcpyDeviceToHost);
    printf("src: %f, %f\n %f, %f\n %f, %f\n", src[0].x, src[0].y, src[1].x, src[1].y, src[2].x, src[2].y);
    printf("dst: %f, %f\n %f, %f\n %f, %f\n", dst[0].x, dst[0].y, dst[1].x, dst[1].y, dst[2].x, dst[2].y);
    get_affine_transform_cv(trans, dst, src);
    printf("trans: ");
    for(int i=0; i<6; ++i){
        printf("%f ", trans[i]);
    }printf("\n");
    cudaFree(d_src);
    cudaFree(d_dst);
}
