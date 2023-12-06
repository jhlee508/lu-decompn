#include <cublas.h>
#include <time.h>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <sys/time.h>

#define DATA_TYPE double

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + tv.tv_usec * (double)1e-6;
}


int main(int argc, char** argv){
    srand(42);
    int matSize = atoi(argv[1]);

    double* arrA = (double*)malloc(sizeof(double)*matSize*matSize);

    for (int i = 0; i < matSize; i++) {
        for (int j = 0; j < matSize; j++) {
            DATA_TYPE elements = (DATA_TYPE)1.0 + 
            ((DATA_TYPE)rand() / (DATA_TYPE)RAND_MAX);
            arrA[j * matSize + i] = elements;
            // printf("%lf ", elements);
        }
        // printf("\n");
    }

    double *arrADev, *workArray;
    double **matrixArray;
    int *pivotArray;
    int *infoArray;
    double* flat = (double*)calloc(matSize*matSize, sizeof(double));
    
    cublasHandle_t cublasHandle;
    cublasStatus_t cublasStatus;
    cudaError_t error;

    cudaError cudaStatus;
    cusolverStatus_t cusolverStatus;
    cusolverDnHandle_t cusolverHandle;

    double *matrix;

    // Initialization
    error = cudaMalloc(&arrADev,  sizeof(double) * matSize*matSize);
    if (error != cudaSuccess) fprintf(stderr,"\nError: %s\n", cudaGetErrorString(error));

    error = cudaMalloc(&matrixArray,  sizeof(double*) * 2);
    if (error != cudaSuccess) fprintf(stderr,"\nError: %s\n", cudaGetErrorString(error));

    error = cudaMalloc(&pivotArray,  sizeof(int) * matSize*matSize);
    if (error != cudaSuccess) fprintf(stderr,"\nError: %s\n", cudaGetErrorString(error));

    error = cudaMalloc(&infoArray,  sizeof(int) * matSize*matSize);
    if (error != cudaSuccess) fprintf(stderr,"\nError: %s\n",cudaGetErrorString(error));

    cublasStatus = cublasCreate(&cublasHandle);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) fprintf(stderr,"error %i\n", cublasStatus);

    cusolverStatus = cusolverDnCreate(&cusolverHandle);
    if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) fprintf(stderr,"error %i\n", cusolverStatus);

    // maps matrix to flat vector
    for(int i = 0; i < matSize; i++){
        for(int j = 0; j < matSize; j++){
            flat[i + j * matSize] = arrA[i + j * matSize];
        }
    }

    // copy matrix A to device
    cublasStatus = cublasSetMatrix(matSize, matSize, sizeof(double), flat, matSize, arrADev, matSize);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) fprintf(stderr,"error %i\n",cublasStatus);

    // save matrix address
    matrix = arrADev;

    // copy matrices references to device
    cudaStatus = cudaMemcpy(matrixArray, matrix, sizeof(double*)*1, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) fprintf(stderr,"\nError: %s\n",cudaGetErrorString(error));

    // calculate buffer size for cuSOLVER LU factorization
    int Lwork;
    cusolverStatus = cusolverDnDgetrf_bufferSize(cusolverHandle, matSize, matSize, arrADev, matSize, &Lwork);
    if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) fprintf(stderr,"error %i\n", cusolverStatus);
    cudaStatus = cudaMalloc((void**)&workArray, Lwork * sizeof(double));
    if (cudaStatus != cudaSuccess) fprintf(stderr,"\nError: %s\n",cudaGetErrorString(error));

    // cuSOLVER LU factorization
    double cusolver_start = get_time();
    cusolverStatus = cusolverDnDgetrf(cusolverHandle, matSize, matSize, arrADev, matSize, workArray, pivotArray, infoArray);
    double cusolver_end = get_time();

    if (cusolverStatus == CUSOLVER_STATUS_SUCCESS)
        printf("> CuSolver LU Decomposition SUCCESSFUL! \n");
    else
        printf("> CuSolver LU Decomposition FAILED! \n");

    // cuBLAS LU factorization
    double cublas_start = get_time();
    cublasStatus = cublasDgetrfBatched(cublasHandle, matSize, matrixArray, matSize, pivotArray, infoArray, 1); // batch
    double cublas_end = get_time();
    if (cublasStatus != CUBLAS_STATUS_SUCCESS)
        printf("> CuBLAS LU Decomposition SUCCESSFUL! \n");
    else
        printf("> CuBLAS LU Decomposition FAILED! \n");

    printf("\n-------------- LU Decomposition Performance --------------\n");
    printf("> CuSolver Elapsed Time: %f (sec)", cusolver_end - cusolver_start);
    printf("\n");
    printf("> CuBLAS Elapsed Time: %f (sec)", cublas_end - cublas_start);
    printf("\n----------------------------------------------------------\n");

    return 0;
}