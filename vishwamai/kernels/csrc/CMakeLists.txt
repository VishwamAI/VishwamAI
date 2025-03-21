cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(flash_mla_cuda LANGUAGES CXX CUDA)

# Find PyTorch
execute_process(
    COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PREFIX_PATH})
find_package(Torch REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89" CACHE STRING "CUDA architectures")

# Add CUDA extension library
add_library(flash_mla_cuda SHARED
    flash_mla_cuda.cpp
    flash_mla.cu
)

# Set compilation options
target_compile_options(flash_mla_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        --ptxas-options=-v
        -lineinfo
        --extended-lambda
        --expt-relaxed-constexpr
    >
)

# Link against PyTorch
target_link_libraries(flash_mla_cuda PRIVATE ${TORCH_LIBRARIES})
target_include_directories(flash_mla_cuda PRIVATE ${TORCH_INCLUDE_DIRS})

# Installation rules
install(TARGETS flash_mla_cuda
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)