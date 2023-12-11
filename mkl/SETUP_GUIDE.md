# Intel oneAPI Math Kernel Library (oneMKL)

This demonstrates how to install and setup the [Intel oneAPI Math Kernel Library (oneMKL)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html#gs.1cwhp0) and the [Intel oneAPI DPC++/C++ Compiler (DPCPP)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html#gs.1cx3wv) to use SYCL interfaces for some routines (e.g., **BLAS**, **LAPACK**) on both CPU and GPU.

You must choose a compiler according to the required backend of your application.
- If only **Intel CPU** is required, you can use either Intel oneAPI DPC++ Compiler **DPCPP** on Linux.
- If your application requires **NVIDIA GPU**, use the latest release of **Clang++**.

## Install oneMKL
Select [oneMKL (Stand-ALone Version)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html) **online installer** for your **OS** (Linux/Window) and install as below.

```bash
# Download installer for Linux
wget https://registrationcenter-download.intel.com/akdlm/IRC_NAS/86d6a4c1-c998-4c6b-9fff-ca004e9f7455/l_onemkl_p_2024.0.0.49673.sh

# Install oneMKL
sh ./l_onemkl_p_2024.0.0.49673.sh
```
- Installation location: `$HOME/intel/oneapi`

Initialize oneAPI environment by sourcing the `setsvars.sh` script. 
```bash
# This adds all the PATHs needed for oneAPI environment
source $HOME/intel/oneapi/setvars.sh
```

Or you can alternatively setup only the **MKL** by sourcing `mkl/<version>/env/vars.sh` script.
```bash
# This adds some PATHs such as MKLROOT="$HOME/intel/oneapi/mkl/2024.0"
source $HOME/intel/oneapi/mkl/2024.0/env/vars.sh
```


## Install DPC++ (Compiler)
Select [DPC++ (Stand-ALone Version)](https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#dpcpp-cpp) **online installer** for your **OS** (Linux/Window) and install as below.

```bash
# Install DPC++ Compiler
sh l_dpcpp-cpp-compiler_p_2024.0.0.49524.sh
```
- Installation location: `$HOME/intel/oneapi`

If **DPC++** or **ICX** compiler not found, setup the compilers by sourcing the `compiler/<version>/env/vars.sh` script.
```bash
# This adds some PATHs such as MKLROOT="$HOME/intel/oneapi/mkl/2024.0"
source $HOME/intel/oneapi/compiler/2024.0/env/vars.sh
```

## Install LAPACK
This installation includes not only **LAPACK**, but also **BLAS**, **CBLAS**, and **LAPACKE** which are all neccessary for building oneMKL project.
```bash
# Clone LAPACK github repository
git clone https://github.com/Reference-LAPACK/lapack

cd lapack

# Building LAPACK 
# at '$HOME/.local/lapack' path 
# with necessary options (e.g., CBLAS, LAPACKE)
mkdir build && cd build

cmake .. -DCMAKE_INSTALL_LIBDIR=$HOME/.local/lapack -DCMAKE_INSTALL_PREFIX=$HOME/.local/lapack -DBUILD_SHARED_LIBS=ON -DCBLAS=ON -DLAPACKE=ON

cmake --build . -j --target install
```
- If **LAPACKE64** file is not found, re-build with adding enabling option as `-DBUILD_INDEX64=ON`.


## Build the Project ([oneMKL Github](https://github.com/oneapi-src/oneMKL))
This is to build the [Intel oneMKL Interface Project](https://github.com/oneapi-src/oneMKL) which is a open-source implementation of the [oneAPI Specification for oneMKL](https://spec.oneapi.io/versions/latest/index.html).

### Build Setup
1. Install Intel oneAPI DPC++ Compiler
2. Clone the [oneMKL project](https://github.com/oneapi-src/oneMKL).
3. Download and install the required dependencies manually, and build with **CMake** directly.

### Building with CMake

MKL and Compilers will be included in environment variable PATH if already sourced the `setsvars.sh` script. To disable **testing**, set `-DBUILD_FUNCTIONAL_TESTS=OFF` option.

```bash
# Inside the <path to onemkl>
mkdir build && cd build

cmake .. [-DMKL_ROOT=<mkl_install_prefix>]                          # required only if environment variable MKLROOT is not set
         [-DCMAKE_CXX_COMPILER=<path_to_dpcpp_compiler>/bin/dpcpp]  # required only if dpcpp is not found in environment variable PATH
         [-DCMAKE_C_COMPILER=<path_to_icx_compiler>/bin/icx]        # required only if icx is not found in environment variable PATH
         [-DREF_BLAS_ROOT=<reference_blas_install_prefix>]          # required only for testing
         [-DREF_LAPACK_ROOT=<reference_lapack_install_prefix>]      # required only for testing

# Functional(Unit) tests will run automatically after the entire project is built successfully
cmake --build . 

# Re-run tests without re-building the entire project
cmake --build . --target test  
```

### Install
You can additionally install **include files** and **libraries** on configured path.
```bash
# install libraries and set environment variable
cmake --install . --prefix <path_to_install_dir>
export LD_LIBRARY_PATH="<path_to_install_dir>:$LD_LIBRARY_PATH"
```

### Run Tests and Examples
```bash
# Run tests
ctest
# Test filter use case - runs only GPU specific tests
ctest -R Gpu
# Exclude filtering use case - excludes CPU tests
ctest -E Cpu
```
```bash
# Run examples
./bin/example...
```


## Reference
- [Intel oneAPI MKL Product](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html)
- [Intel oneAPI MKL Specification](https://spec.oneapi.io/versions/latest/index.html)
- [Intel oneAPI MKL Interfaces Project (Github)](https://github.com/oneapi-src/oneMKL)
- [Intel oneAPI MKL Interfaces Project (Document)](https://oneapi-src.github.io/oneMKL/index.html)
