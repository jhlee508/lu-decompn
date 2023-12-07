# Cublas_LU

This code is a usage of CuBLAS API for `dense LU decomposition` from [NVIDIA repository](https://github.com/NVIDIA/cuda-samples)

## Build and Run (Linux)

### Build
```bash
make TARGET_ARCH=x86_64 SMS="75" # dbg=1 HOST_COMPILER=g++
```

### Run
```bash
./cublas_lu [matrix dimension] # e.g., ./cublas_lu 4
```

## Reference
https://github.com/NVIDIA/cuda-samples
- [cublasDgetrfBatched API](https://docs.nvidia.com/cuda/cublas/index.html?highlight=cublasDgetrfBatched#cublas-t-getrfbatched)