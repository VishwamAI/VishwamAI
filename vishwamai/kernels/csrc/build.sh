#!/bin/bash

# Remove old build files
rm -rf build/
rm -rf *.so

# Build the CUDA extension
python setup.py build_ext --inplace