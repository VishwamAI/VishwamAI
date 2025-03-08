#!/bin/bash
set -e

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_FLAGS="--expt-relaxed-constexpr --expt-extended-lambda -O3" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON

# Build with all available cores
make -j$(nproc)

# Copy library to parent directory
cp libdeep_ep.so ../

# Cleanup build directory
cd ..
rm -rf build

echo "Build completed successfully. Library is available at ./libdeep_ep.so"
