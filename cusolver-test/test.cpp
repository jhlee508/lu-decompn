#include <cstdio>
#include <cstring>
#include <time.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <cublas.h>
#include <cublas_v2.h>
#include <cusolverDn.h>


double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + tv.tv_usec * (double)1e-6;
}

int main(int argc, char** argv){
    srand(42);
    if (argc < 2) {
        printf("Usage %s [matrix size]\n", argv[0]);
        exit(0);
    }
    int matSize = atoi(argv[1]);

    double* arrA = (double*)malloc(sizeof(double)*matSize*matSize);

    for (int i = 0; i < matSize; i++) {
        for (int j = 0; j < matSize; j++) {
            double elements = (double)1.0 + 
            ((double)rand() / (double)RAND_MAX);
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
    int Lwork = 0;
    
    cublasHandle_t cublasHandle;
    cublasStatus_t cublasStatus;
    cudaError_t error;

    cusolverDnHandle_t cusolverHandle;
    cusolverStatus_t cusolverStatus;
    cudaError_t cudaStatus;

    double *matrix;

    // Initialization
    printf("> initializing..\n");
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
    printf("> CuBLAS: copying data from host memory to GPU memory..\n");
    cublasStatus = cublasSetMatrix(matSize, matSize, sizeof(double), flat, matSize, arrADev, matSize);
    if (cublasStatus != CUBLAS_STATUS_SUCCESS) fprintf(stderr,"error %i\n",cublasStatus);

    // save matrix address
    matrix = arrADev;

    // copy matrices references to device
    printf("> CuSolver: copying data from host memory to GPU memory..\n");
    cudaStatus = cudaMemcpy(matrixArray, matrix, sizeof(double*)*1, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) fprintf(stderr,"\nError: %s\n",cudaGetErrorString(error));

    // calculate buffer size for cuSOLVER LU factorization
    cusolverStatus = cusolverDnDgetrf_bufferSize(cusolverHandle, matSize, matSize, arrADev, matSize, &Lwork);
    if (cusolverStatus != CUSOLVER_STATUS_SUCCESS) fprintf(stderr,"error %i\n", cusolverStatus);
    cudaStatus = cudaMalloc((void**)&workArray, Lwork * sizeof(double));
    if (cudaStatus != cudaSuccess) fprintf(stderr,"\nError: %s\n",cudaGetErrorString(error));

    // cuSOLVER LU factorization
    printf("> CuSolver: performing LU decomposition..\n");
    double cusolver_start = get_time();
    cusolverStatus = cusolverDnDgetrf(cusolverHandle, matSize, matSize, arrADev, matSize, workArray, pivotArray, infoArray);
    double cusolver_end = get_time();

    if (cusolverStatus == CUSOLVER_STATUS_SUCCESS)
        printf("> CuSolver LU Decomposition SUCCESSFUL! \n");
    else
        printf("> CuSolver LU Decomposition FAILED! \n");

    // // cuBLAS LU factorization
    // printf("> CuBLAS: performing LU decomposition..\n");
    // double cublas_start = get_time();
    // cublasStatus = cublasDgetrfBatched(cublasHandle, matSize, matrixArray, matSize, pivotArray, infoArray, 1); // batch
    // double cublas_end = get_time();
    // if (cublasStatus == CUBLAS_STATUS_SUCCESS)
    //     printf("> CuBLAS LU Decomposition SUCCESSFUL! \n");
    // else
    //     printf("> CuBLAS LU Decomposition FAILED! \n");

    printf("\n-------------- LU Decomposition Performance --------------\n");
    // printf("> CuBLAS Elapsed Time: %f (sec)", cublas_end - cublas_start);
    printf("\n");
    printf("> CuSolver Elapsed Time: %f (sec)", cusolver_end - cusolver_start);
    printf("\n----------------------------------------------------------\n");

    return 0;
}