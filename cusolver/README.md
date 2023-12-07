# CuSolver LU Factorization

This code is a usage of cuSOLVER `getrf` API for `dense LU factorization` from [NVIDIA repository](https://github.com/NVIDIA/CUDALibrarySamples)

_PA = LU_

## Build (make)

### Prerequisites
- A Linux system with recent NVIDIA driver
- [CMake](https://cmake.org/download) version >= 3.18

### Build command on Linux
```
$ mkdir build
$ cd build
$ cmake ..
$ make
```

## Usage (run)
```
$  ./cusolver_lu
```

## Reference
- https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-getrf
- [cusolverDnDgetrf API](https://docs.nvidia.com/cuda/cusolver/index.html#cuSolverDN-lt-t-gt-getrf)