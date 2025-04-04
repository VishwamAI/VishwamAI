#!/bin/bash

# Set TPU-related environment variables
export TPU_NAME="local"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/local/cuda"
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Install required packages if not present
pip install --upgrade pip
pip install -r requirements.txt

# Configure JAX for TPU
python3 -c "import jax; print(f'Found {jax.local_device_count()} TPU devices')"

# Run the import test
python3 importtest.py -v
