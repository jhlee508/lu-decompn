# oneMKL LU Decomposition
This code is a usage of Intel oneAPI MKL for **dense LU decomposition**.

## Setup
Refer to `SETUP_GUIDE.md` file.

## Build and Run
```bash
# Set oneAPI environment variables
source $HOME/intel/oneapi/setvars.sh
```

### Build
```bash
# Use 'icpx -fsycl' instead, since 'dpcpp' is deprecated. 
icpx -fsycl mkl_lu.cpp -o mkl_lu -DMKL_ILP64 -I${MKLROOT}/include -L${MKLROOT}/lib/intel64 -lmkl_sycl -lmkl_intel_ilp64 -lmkl_tbb_thread -lmkl_core -lsycl -lOpenCL -lpthread -lm -ldl
```

### Run
```bash
./mkl_lu [matrix dimension] # e.g., ./mkl_lu 4
```

## Reference
- [Intel oneAPI MKL Link Line Advisor](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-link-line-advisor.html)