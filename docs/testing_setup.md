# Testing Setup Guide for VishwamAI

## Prerequisites for All Platforms

1. Google Cloud Account Setup:
```bash
# Install Google Cloud SDK
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-latest-linux-x86_64.tar.gz
tar -xf google-cloud-cli-latest-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh

# Initialize and authenticate
gcloud init
gcloud auth application-default login
```

## TPU Testing Setup

1. Environment Requirements:
```bash
# TPU VM specific requirements
pip install "jax[tpu]>=0.4.1" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
pip install tensorflow==2.15.0
pip install -e ".[tpu]"
```

2. TPU VM Configuration:
```bash
# Create TPU VM
gcloud compute tpus tpu-vm create vishwamai-tpu \
    --zone=us-central1-a \
    --accelerator-type=v4-8 \
    --version=tpu-vm-base

# SSH into TPU VM
gcloud compute tpus tpu-vm ssh vishwamai-tpu --zone=us-central1-a

# Set environment variables
export TPU_NAME="local"
export XRT_TPU_CONFIG="localservice;0;localhost:51011"
```

3. Test Execution:
```bash
# Run TPU-specific tests
./tests/run_tests.py --platform tpu
```

## GPU Testing Setup

1. Environment Requirements:
```bash
# GPU specific requirements
pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 -f https://download.pytorch.org/whl/cu118
pip install nvidia-cuda-runtime-cu11 nvidia-cuda-nvrtc-cu11
pip install -e ".[gpu]"
```

2. Google Cloud GPU Configuration:
```bash
# Create GPU instance
gcloud compute instances create vishwamai-gpu \
    --zone=us-central1-a \
    --machine-type=n1-standard-8 \
    --accelerator="type=nvidia-tesla-v100,count=1" \
    --maintenance-policy=TERMINATE \
    --image-family=debian-11-gpu \
    --image-project=debian-cloud

# SSH into GPU instance
gcloud compute ssh vishwamai-gpu --zone=us-central1-a

# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0
```

3. Test Execution:
```bash
# Run GPU-specific tests
./tests/run_tests.py --platform gpu
```

## CPU Testing Setup

1. Environment Requirements:
```bash
# CPU specific requirements
pip install numpy>=1.20.0 scipy>=1.8.0
pip install -e ".[cpu]"
```

2. Google Cloud CPU Configuration:
```bash
# Create CPU instance
gcloud compute instances create vishwamai-cpu \
    --zone=us-central1-a \
    --machine-type=n2-standard-16 \
    --image-family=debian-11 \
    --image-project=debian-cloud

# SSH into CPU instance
gcloud compute ssh vishwamai-cpu --zone=us-central1-a
```

3. Test Execution:
```bash
# Run CPU-specific tests
./tests/run_tests.py --platform cpu
```

## Running All Tests

1. Test Environment Setup:
```bash
# Install all dependencies
pip install -e ".[all]"

# Configure environment
source setup_test_env.sh
```

2. Full Test Suite:
```bash
# Run all platform tests
./tests/run_tests.py

# Run with benchmarks
./tests/run_tests.py --benchmark
```

## Performance Profiling

1. TPU Profiling:
```bash
# Enable TPU profiling
export TPU_PROFILE=1
./tests/run_tests.py --platform tpu --benchmark
```

2. GPU Profiling:
```bash
# Enable CUDA profiling
export CUDA_LAUNCH_BLOCKING=1
nvprof ./tests/run_tests.py --platform gpu --benchmark
```

3. CPU Profiling:
```bash
# Run with cProfile
python -m cProfile -o cpu_profile.stats ./tests/run_tests.py --platform cpu
```

## Cost Estimates

1. TPU Testing:
- v4-8: ~$4.50/hour
- Test suite runtime: ~30 minutes
- Estimated cost: ~$2.25 per full test run

2. GPU Testing:
- V100: ~$2.48/hour
- Test suite runtime: ~45 minutes
- Estimated cost: ~$1.86 per full test run

3. CPU Testing:
- n2-standard-16: ~$0.76/hour
- Test suite runtime: ~60 minutes
- Estimated cost: ~$0.76 per full test run

## Cleanup

```bash
# Delete TPU VM
gcloud compute tpus tpu-vm delete vishwamai-tpu --zone=us-central1-a

# Delete GPU instance
gcloud compute instances delete vishwamai-gpu --zone=us-central1-a

# Delete CPU instance
gcloud compute instances delete vishwamai-cpu --zone=us-central1-a
```

## Troubleshooting

1. TPU Issues:
```bash
# Check TPU status
gcloud compute tpus tpu-vm describe vishwamai-tpu --zone=us-central1-a

# Verify JAX TPU setup
python -c "import jax; print(jax.devices())"
```

2. GPU Issues:
```bash
# Check CUDA setup
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

3. Common Problems:
- TPU initialization failures: Check JAX version compatibility
- GPU out of memory: Reduce batch sizes
- Test timeouts: Increase timeout in pytest.ini
