/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * This example demonstrates how to use the cuBLAS library API
 * for lower-upper (LU) decomposition of a matrix. LU decomposition
 * factors a matrix as the product of upper triangular matrix and
 * lower trianglular matrix.
 *
 * https://en.wikipedia.org/wiki/LU_decomposition
 *
 * This sample uses 10000 matrices of size 4x4 and performs
 * LU decomposition of them using batched decomposition API
 * of cuBLAS library. To test the correctness of upper and lower
 * matrices generated, they are multiplied and compared with the
 * original input matrix.
 *
 */

#include <stdio.h>
#include <stdlib.h>

// cuda libraries and helpers
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <sys/time.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + tv.tv_usec * (double)1e-6;
}

// [configurable parameters]
// batch size
#define BATCH_SIZE 1

// use double precision data type
#define DOUBLE_PRECISION /* comment this to use single precision */

#ifdef DOUBLE_PRECISION
  #define DATA_TYPE double
  #define MAX_ERROR 1e-14 // default: 1e-15
#else
  #define DATA_TYPE float
  #define MAX_ERROR 1e-6
#endif /* DOUBLE_PRECISION */

// use pivot vector while decomposing
#define PIVOT /* comment this to disable pivot use */

// verify the result
// #define VERIFY /* comment this to disable verification */

// print matrices
// #define PRINT /* comment this to diable print */

// [helper functions]
// wrapper around cublas<t>getrfBatched()
cublasStatus_t cublasXgetrfBatched(cublasHandle_t handle, int n,
                                   DATA_TYPE* const A[], int lda, int* P,
                                   int* info, int batchSize) {
#ifdef DOUBLE_PRECISION
  return cublasDgetrfBatched(handle, n, A, lda, P, info, batchSize);
#else
  return cublasSgetrfBatched(handle, n, A, lda, P, info, batchSize);
#endif
}

// wrapper around malloc
// clears the allocated memory to 0
// terminates the program if malloc fails
void* xmalloc(size_t size) {
  void* ptr = malloc(size);
  if (ptr == NULL) {
    printf("> ERROR: malloc for size %zu failed..\n", size);
    exit(EXIT_FAILURE);
  }
  memset(ptr, 0, size);
  return ptr;
}

// initalize identity matrix
void initIdentityMatrix(DATA_TYPE* mat, int N) {
  // clear the matrix
  memset(mat, 0, N * N * sizeof(DATA_TYPE));

  // set all diagonals to 1
  for (int i = 0; i < N; i++) {
    mat[(i * N) + i] = 1.0;
  }
}

// initialize matrix with all elements as 0
void initZeroMatrix(DATA_TYPE* mat, int N) {
  memset(mat, 0, N * N * sizeof(DATA_TYPE));
}

// fill random value in column-major matrix
void initRandomMatrix(DATA_TYPE* mat, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      mat[(j * N) + i] =
          (DATA_TYPE)1.0 + ((DATA_TYPE)rand() / (DATA_TYPE)RAND_MAX);
    }
  }

  // diagonal dominant matrix to insure it is invertible matrix
  for (int i = 0; i < N; i++) {
    mat[(i * N) + i] += (DATA_TYPE)N;
  }
}

// print column-major matrix
void printMatrix(DATA_TYPE* mat, int N) {
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      printf("%0.2f ", mat[(j * N) + i]);
    }
    printf("\n");
  }
  printf("====================\n");
}

// matrix mulitplication
void matrixMultiply(DATA_TYPE* res, DATA_TYPE* mat1, DATA_TYPE* mat2, int N) {
  initZeroMatrix(res, N);

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      for (int k = 0; k < N; k++) {
        res[(j * N) + i] += mat1[(k * N) + i] * mat2[(j * N) + k];
      }
    }
  }
}

// check matrix equality
bool checkRelativeError(DATA_TYPE* mat1, DATA_TYPE* mat2, DATA_TYPE maxError, int N) {
  DATA_TYPE err = (DATA_TYPE)0.0;
  DATA_TYPE refNorm = (DATA_TYPE)0.0;
  DATA_TYPE relError = (DATA_TYPE)0.0;
  DATA_TYPE relMaxError = (DATA_TYPE)0.0;

  for (int i = 0; i < N * N; i++) {
    refNorm = abs(mat1[i]);
    err = abs(mat1[i] - mat2[i]);

    if (refNorm != 0.0 && err > 0.0) {
      relError = err / refNorm;
      relMaxError = MAX(relMaxError, relError);
    }

    if (relMaxError > maxError) {
      printf("> ERROR: relMaxError = %e is greater than maxError = %e "
             "at index %d..\n",
             relMaxError, maxError, i); 
      return false;
    }
  }
  return true;
}

// decode lower and upper matrix from single matrix
// returned by getrfBatched()
void getLUdecoded(DATA_TYPE* mat, DATA_TYPE* L, DATA_TYPE* U, int N) {
  // init L as identity matrix
  initIdentityMatrix(L, N);

  // copy lower triangular values from mat to L (skip diagonal)
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < i; j++) {
      L[(j * N) + i] = mat[(j * N) + i];
    }
  }

  // init U as all zero
  initZeroMatrix(U, N);

  // copy upper triangular values from mat to U
  for (int i = 0; i < N; i++) {
    for (int j = i; j < N; j++) {
      U[(j * N) + i] = mat[(j * N) + i];
    }
  }
}

// generate permutation matrix from pivot vector
void getPmatFromPivot(DATA_TYPE* Pmat, int* P, int N) {
  int pivot[N];

  // pivot vector in base-1
  // convert it to base-0
  for (int i = 0; i < N; i++) {
    P[i]--;
  }

  // generate permutation vector from pivot
  // initialize pivot with identity sequence
  for (int k = 0; k < N; k++) {
    pivot[k] = k;
  }

  // swap the indices according to pivot vector
  for (int k = 0; k < N; k++) {
    int q = P[k];

    // swap pivot(k) and pivot(q)
    int s = pivot[k];
    int t = pivot[q];
    pivot[k] = t;
    pivot[q] = s;
  }

  // generate permutation matrix from pivot vector
  initZeroMatrix(Pmat, N);
  for (int i = 0; i < N; i++) {
    int j = pivot[i];
    Pmat[(j * N) + i] = (DATA_TYPE)1.0;
  }
}

int main(int argc, char** argv) {
  // cuBLAS variables
  cublasStatus_t status;
  cublasHandle_t handle;
  cudaStream_t stream;

  // host variables
  if (argc < 2) {
    printf("Usage %s [matrix dimension]\n", argv[0]);
    printf(" e.g., %s 4\n", argv[0]);
    exit(0);
  }
  int N = atoi(argv[1]);
  size_t matSize = N * N * sizeof(DATA_TYPE);

  DATA_TYPE* h_AarrayInput;
  DATA_TYPE* h_AarrayOutput;
  DATA_TYPE* h_ptr_array[BATCH_SIZE];

  int* h_pivotArray;
  int* h_infoArray;

  // device variables
  DATA_TYPE* d_Aarray;
  DATA_TYPE** d_ptr_array;

  int* d_pivotArray;
  int* d_infoArray;

  int err_count = 0;

  // seed the rand() function with time
  srand(42);

  // find cuda device
  printf("> initializing..\n");
  int dev = findCudaDevice(argc, (const char**)argv);
  if (dev == -1) {
    return (EXIT_FAILURE);
  }

  // initialize cuBLAS
  status = cublasCreate(&handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("> ERROR: cuBLAS initialization failed..\n");
    return (EXIT_FAILURE);
  }

#ifdef DOUBLE_PRECISION
  printf("> using DOUBLE precision..\n");
#else
  printf("> using SINGLE precision..\n");
#endif

#ifdef PIVOT
  printf("> pivot ENABLED..\n");
#else
  printf("> pivot DISABLED..\n");
#endif

  // allocate memory for host variables
  h_AarrayInput = (DATA_TYPE*)xmalloc(BATCH_SIZE * matSize);
  h_AarrayOutput = (DATA_TYPE*)xmalloc(BATCH_SIZE * matSize);

  h_pivotArray = (int*)xmalloc(N * BATCH_SIZE * sizeof(int));
  h_infoArray = (int*)xmalloc(BATCH_SIZE * sizeof(int));

  // allocate memory for device variables
  checkCudaErrors(cudaMalloc((void**)&d_Aarray, BATCH_SIZE * matSize));
  checkCudaErrors(
      cudaMalloc((void**)&d_pivotArray, N * BATCH_SIZE * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**)&d_infoArray, BATCH_SIZE * sizeof(int)));
  checkCudaErrors(
      cudaMalloc((void**)&d_ptr_array, BATCH_SIZE * sizeof(DATA_TYPE*)));

  // fill matrix A with random data
  for (int i = 0; i < BATCH_SIZE; i++) {
    initRandomMatrix(h_AarrayInput + (i * N * N), N);
  }
  
#ifdef PRINT
  // print input matrix A
  printf("> printing matrix A..\n");
  printMatrix(h_AarrayInput, N);
#endif /* PRINT */

  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  cublasSetStream(handle, stream);

  double start_comm = get_time();
  // copy data to device from host
  checkCudaErrors(cudaMemcpyAsync(d_Aarray, h_AarrayInput, BATCH_SIZE * matSize,
                             cudaMemcpyHostToDevice, stream));

  // create pointer array for matrices
  for (int i = 0; i < BATCH_SIZE; i++) h_ptr_array[i] = d_Aarray + (i * N * N);

  // copy pointer array to device memory
  checkCudaErrors(cudaMemcpyAsync(d_ptr_array, h_ptr_array,
                             BATCH_SIZE * sizeof(DATA_TYPE*),
                             cudaMemcpyHostToDevice, stream));

  // perform LU decomposition
  double start_comp = get_time();
#ifdef PIVOT
  status = cublasXgetrfBatched(handle, N, d_ptr_array, N, d_pivotArray,
                               d_infoArray, BATCH_SIZE);
#else
  status = cublasXgetrfBatched(handle, N, d_ptr_array, N, NULL, d_infoArray,
                               BATCH_SIZE);
#endif /* PIVOT */
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("> ERROR: cublasDgetrfBatched() failed with error %s..\n",
           _cudaGetErrorEnum(status));
    return (EXIT_FAILURE);
  }
  double end_comp = get_time();

  // copy data to host from device
  checkCudaErrors(cudaMemcpyAsync(h_AarrayOutput, d_Aarray, BATCH_SIZE * matSize,
                             cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaMemcpyAsync(h_infoArray, d_infoArray, BATCH_SIZE * sizeof(int),
                             cudaMemcpyDeviceToHost, stream));

#ifdef PIVOT
  checkCudaErrors(cudaMemcpyAsync(h_pivotArray, d_pivotArray,
                             N * BATCH_SIZE * sizeof(int),
                             cudaMemcpyDeviceToHost, stream));
#endif /* PIVOT */

  checkCudaErrors(cudaStreamSynchronize(stream));
  double end_comm = get_time();

#ifdef PRINT
  // print output matrix LU
  printf("> printing matrix LU..\n");
  printMatrix(h_AarrayOutput, N);
#endif /* PRINT */

#ifdef VERIFY
  // verify the result
  printf("> verifying the result..\n");
  for (int i = 0; i < BATCH_SIZE; i++) {
    if (h_infoArray[i] == 0) {
      DATA_TYPE* A = h_AarrayInput + (i * N * N);
      DATA_TYPE* LU = h_AarrayOutput + (i * N * N);
      /* Dynamic allocation L and U to avoid stack overflow */
      // DATA_TYPE L[N * N];
      // DATA_TYPE U[N * N];
      DATA_TYPE* L = (DATA_TYPE*)malloc(N * N * sizeof(DATA_TYPE));
      DATA_TYPE* U = (DATA_TYPE*)malloc(N * N * sizeof(DATA_TYPE));
      getLUdecoded(LU, L, U, N);

      // test P * A = L * U
      int* P = h_pivotArray + (i * N);
      /* Dynamic allocation Pmat to avoid stack overflow */
      // DATA_TYPE Pmat[N * N];
      DATA_TYPE* Pmat = (DATA_TYPE*)malloc(N * N * sizeof(DATA_TYPE));
#ifdef PIVOT
      getPmatFromPivot(Pmat, P, N);
#else
      initIdentityMatrix(Pmat, N);
#endif /* PIVOT */

      // perform matrix multiplication
      /* Dynamic allocation PxA and LxA to avoid stack overflow */ 
      //DATA_TYPE PxA[N * N];
      DATA_TYPE* PxA = (DATA_TYPE*)malloc(N * N * sizeof(DATA_TYPE));
      //DATA_TYPE LxU[N * N];
      DATA_TYPE* LxU = (DATA_TYPE*)malloc(N * N * sizeof(DATA_TYPE));
      matrixMultiply(PxA, Pmat, A, N);
      matrixMultiply(LxU, L, U, N);

      // check for equality of matrices
      if (!checkRelativeError(PxA, LxU, (DATA_TYPE)MAX_ERROR, N)) {
        printf("> ERROR: accuracy check failed for matrix number %05d..\n",
               i + 1);
        err_count++;
      }
    } 
    else if (h_infoArray[i] > 0) {
      printf(
          "> execution for matrix %05d is successful, but U is singular and "
          "U(%d,%d) = 0..\n",
          i + 1, h_infoArray[i] - 1, h_infoArray[i] - 1);
    } 
    else  // (h_infoArray[i] < 0)
    {
      printf("> ERROR: matrix %05d have an illegal value at index %d = %lf..\n",
             i + 1, -h_infoArray[i],
             *(h_AarrayInput + (i * N * N) + (-h_infoArray[i])));
    }
  }
#endif /* VERIFY */

  // free device variables
  checkCudaErrors(cudaFree(d_ptr_array));
  checkCudaErrors(cudaFree(d_infoArray));
  checkCudaErrors(cudaFree(d_pivotArray));
  checkCudaErrors(cudaFree(d_Aarray));

  // free host variables
  if (h_infoArray) free(h_infoArray);
  if (h_pivotArray) free(h_pivotArray);
  if (h_AarrayOutput) free(h_AarrayOutput);
  if (h_AarrayInput) free(h_AarrayInput);

  // destroy cuda stream
  checkCudaErrors(cudaStreamDestroy(stream));

  // destroy cuBLAS handle
  status = cublasDestroy(handle);
  if (status != CUBLAS_STATUS_SUCCESS) {
    printf("> ERROR: cuBLAS uninitialization failed..\n");
    return (EXIT_FAILURE);
  }

  if (err_count > 0) {
    printf("> TEST FAILED for %d matrices, with precision: %g\n", err_count,
           MAX_ERROR);
    return (EXIT_FAILURE);
  }
  printf("\n------------- Cublas LU Decomposition Result -------------\n");
  printf("> Validation SUCCESS, with precision: %g\n", MAX_ERROR);
  printf("> LU Decomposition Elapsed Time: %f (sec)\n", end_comp - start_comp);
  printf("> LU Decomposition + Communication Time: %f (sec)", end_comm - start_comm);
  printf("\n----------------------------------------------------------\n");
  return (EXIT_SUCCESS);
}
