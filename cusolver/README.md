# cuSOLVER LU Decomposition

This code is a usage of cuSOLVER API for `dense LU decomposition` from [NVIDIA repository](https://github.com/NVIDIA/CUDALibrarySamples)

_P * A = L * U_

## Build (make)

### Prerequisites
- A Linux system with recent NVIDIA driver
- [CMake](https://cmake.org/download) version >= 3.18

### Build command on Linux

```bash
mkdir build
cd build
cmake ..
make
```

## Usage (run)

```bash
./cusolver_lu [matrix dimension] # e.g., ./cusolver_lu 4
```

## Reference
- https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-getrf
- [cusolverDnDgetrf API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-getrf)