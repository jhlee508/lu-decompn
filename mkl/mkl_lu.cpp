#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>

// oneMKL/SYCL includes
#include "oneapi/mkl.hpp"
#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

using namespace std;

/* print matrices */
// #define PRINT /* comment this to diable print */

// Helper function to get time stamp
double get_time() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + tv.tv_usec * (double)1e-6;
}


int main(int argc, char* argv[]) {
    srand(42);
    if (argc < 2) {
        printf("Usage %s [matrix dimension]\n", argv[0]);
        printf(" e.g., %s 4\n", argv[0]);
        exit(0);
    }
    int matSize = atoi(argv[1]);
    const int lda = matSize;

    /* Asynchronous error handler */ 
    auto error_handler = [&](sycl::exception_list exceptions) {
        for (auto const& e : exceptions) {
            try {
                rethrow_exception(e);
            }
            catch (oneapi::mkl::lapack::exception const& e) {
                // Handle LAPACK related exceptions that happened during asynchronous call
                cerr << "Caught asynchronous LAPACK exception during GETRF:"
                        << endl;
                cerr << "\t" << e.what() << endl;
                cerr << "\tinfo: " << e.info() << endl;
            }
            catch (sycl::exception const& e) {
                // Handle not LAPACK related exceptions that happened during asynchronous call
                cerr << "Caught asynchronous SYCL exception during GETRF:"
                        << endl;
                cerr << "\t" << e.what() << endl;
            }
        }
        exit(2);
    };

    sycl::device device = sycl::device();
    sycl::queue queue(device, error_handler);

    vector<double> A(matSize * matSize);
    vector<double> LU(lda * matSize, 0);
    vector<int64_t> ipiv(matSize);
    
    // initialize random matrix A
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

#ifdef PRINT
    printf("> printing matrix A..\n");
    for (int i = 0; i < matSize; i++) {
        for (int j = 0; j < matSize; j++) {
            printf("%0.2f ", A[j * matSize + i]);
        }
        printf("\n");
    } 
    printf("====================\n");
#endif /* PRINT */

    double start_comp = get_time();
    // computes size of scratchpad memory required for getrf function.
    auto iwork = oneapi::mkl::lapack::getrf_scratchpad_size<double>(queue, matSize, matSize, matSize);
    vector<double> work(iwork);
    
    // computes the LU factorization of a general m by n matrix
    oneapi::mkl::lapack::getrf(queue, matSize, matSize, A.data(), lda, ipiv.data(), work.data(), iwork);
    double end_comp = get_time();
    
    // copy the result to LU
    LU = A;

    for (auto p : ipiv) {
        if (p < 0) {
            printf("LU factorization failed: matrix is singular\n");
            exit(1);
        }
    }

#ifdef PRINT
    printf("> printing matrix LU..\n");
    for (int i = 0; i < matSize; i++) {
        for (int j = 0; j < matSize; j++) {
            printf("%0.2f ", LU[j * matSize + i]);
        }
        printf("\n");
    } 
    printf("====================\n");
#endif /* PRINT */

    printf("\n------------- oneMKL LU Decomposition Result -------------\n");
    printf("> LU Decomposition Elapsed Time: %f (sec)", end_comp - start_comp);
    printf("\n----------------------------------------------------------\n");
        
    return (EXIT_SUCCESS);
}