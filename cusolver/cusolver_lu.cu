/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "cusolver_utils.h"
#include <sys/time.h>

double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + tv.tv_usec * (double)1e-6;
}

/* print matrices */
#define PRINT /* comment this to diable print */

int main(int argc, char *argv[]) {
    srand(42);
    if (argc < 2) {
        printf("Usage %s [matrix dimension]\n", argv[0]);
        printf(" e.g., %s 4\n", argv[0]);
        exit(0);
    }
    int matSize = atoi(argv[1]);
    const int lda = matSize;

    cusolverDnHandle_t cusolverH = NULL;
    cudaStream_t stream = NULL;

    /*       
     *       | 1 2 3  |
     *   A = | 4 5 6  |
     *       | 7 8 10 |
     *
     * without pivoting: A = L*U
     *       | 1 0 0 |      | 1  2  3 |
     *   L = | 4 1 0 |, U = | 0 -3 -6 |
     *       | 7 2 1 |      | 0  0  1 |
     *
     * with pivoting: P*A = L*U
     *       | 0 0 1 |
     *   P = | 1 0 0 |
     *       | 0 1 0 |
     *
     *       | 1       0     0 |      | 7  8       10     |
     *   L = | 0.1429  1     0 |, U = | 0  0.8571  1.5714 |
     *       | 0.5714  0.5   1 |      | 0  0       -0.5   |
     */

    std::vector<double> A(matSize * matSize);
    for (int i = 0; i < matSize; ++i) {
        for (int j = 0; j < matSize; ++j) {
            double element = (double)1.0 + ((double)rand() / (double)RAND_MAX);
            A[j * matSize + i] = element;
        }
    }
    // diagonal dominant matrix to insure it is invertible matrix
    for (int i = 0; i < matSize; i++) {
        A[(i * matSize) + i] += (double)matSize;
    }
    std::vector<double> X(matSize, 0);
    std::vector<double> LU(lda * matSize, 0);
    std::vector<int> Ipiv(matSize, 0);
    int info = 0;

    double *d_A = nullptr; /* device copy of A */
    int *d_Ipiv = nullptr; /* pivoting sequence */
    int *d_info = nullptr; /* error info */

    int lwork = 0;            /* size of workspace */
    double *d_work = nullptr; /* device workspace for getrf */

    const int pivot_on = 1;

    if (pivot_on) {
        printf("> pivot is ENABLED..\n> compute P*A = L*U\n");
    } else {
        printf("> pivot is DISABLED..\n> compute A = L*U (not numerically stable)\n");
    }

#ifdef PRINT
    printf("A = (matlab base-1)\n");
    print_matrix(matSize, matSize, A.data(), lda);
    printf("====================\n");
#endif /* PRINT */

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUSOLVER_CHECK(cusolverDnSetStream(cusolverH, stream));

    /* step 2: copy A to device */
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_A), sizeof(double) * A.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int) * Ipiv.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    CUDA_CHECK(
        cudaMemcpyAsync(d_A, A.data(), sizeof(double) * A.size(), cudaMemcpyHostToDevice, stream));

    /* step 3: query working space of getrf */
    CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(cusolverH, matSize, matSize, d_A, lda, &lwork));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(double) * lwork));

    /* step 4: LU factorization */
    double start = get_time();
    if (pivot_on) {
        CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, matSize, matSize, d_A, lda, d_work, d_Ipiv, d_info));
    } else {
        CUSOLVER_CHECK(cusolverDnDgetrf(cusolverH, matSize, matSize, d_A, lda, d_work, NULL, d_info));
    }
    double end = get_time();

    if (pivot_on) {
        CUDA_CHECK(cudaMemcpyAsync(Ipiv.data(), d_Ipiv, sizeof(int) * Ipiv.size(),
                                   cudaMemcpyDeviceToHost, stream));
    }
    CUDA_CHECK(
        cudaMemcpyAsync(LU.data(), d_A, sizeof(double) * A.size(), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));

    if (0 > info) {
        printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }
#ifdef PRINT
    if (pivot_on) {
        printf("pivoting sequence, matlab base-1\n");
        for (int j = 0; j < matSize; j++) {
            printf("Ipiv(%d) = %d\n", j + 1, Ipiv[j]);
        }
    }
#endif /* PRINT */

#ifdef PRINT
    printf("L and U = (matlab base-1)\n");
    print_matrix(matSize, matSize, LU.data(), lda);
    printf("====================\n");
#endif /* PRINT */

    /* free resources */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_Ipiv));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    CUDA_CHECK(cudaStreamDestroy(stream));

    CUDA_CHECK(cudaDeviceReset());

    printf("\n------------ CuSolver LU Decomposition Result ------------\n");
    printf("> LU Decomposition Elapsed Time: %f (sec)", end - start);
    printf("\n----------------------------------------------------------\n");
    return EXIT_SUCCESS;
}