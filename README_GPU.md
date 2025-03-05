# VishwamAI GPU Setup Guide

This guide explains how to set up and run VishwamAI's efficient pre-training system on NVIDIA GPUs.

## System Requirements

- NVIDIA GPU (GTX 1650 or better)
- CUDA 12.2 or compatible version
- Python 3.8+
- 4GB+ GPU memory

## Installation

1. Clone the repository:
```bash
git clone https://github.com/kasinadhsarma/VishwamAI.git
cd VishwamAI
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

4. Install additional GPU-specific packages:
```bash
pip install nvidia-ml-py3
```

## GPU-Specific Configuration

The system will automatically detect your GPU and optimize accordingly. For the GTX 1650 with 4GB memory:

1. Update the configuration in `vishwamai/configs/training/efficient_pretrain.yaml`:
```yaml
training:
  batch_size: 8  # Reduced for 4GB GPU
  gradient_accumulation_steps: 8  # Increased to compensate
  mixed_precision:
    enabled: true
    dtype: "float16"  # Using float16 for NVIDIA GPU
    
hydra:
  device:
    type: "gpu"
    precision: "float16"
    memory_allocation: 0.85
```

## Running Pre-training

1. For command-line training:
```bash
python -m vishwamai.pretrain_efficient
```

2. For interactive notebook experimentation:
```bash
jupyter notebook notebooks/efficient_pretraining.ipynb
```

## Memory Management

For GTX 1650 (4GB GPU):
- Batch size is set to 8 to fit in memory
- Gradient accumulation is used to achieve effective batch size
- Mixed precision training (FP16) reduces memory usage by ~50%
- Gradient checkpointing saves additional memory at cost of compute

## Monitoring GPU Usage

Monitor GPU usage during training:
```bash
nvidia-smi -l 1  # Updates every 1 second
```

Expected memory usage for default configuration:
- Model parameters: ~1.5GB
- Training buffers: ~1GB
- Gradients: ~0.5GB
- Remaining memory reserved for CUDA runtime

## Troubleshooting

1. Out of Memory (OOM) Errors:
   - Reduce batch_size in config
   - Increase gradient_accumulation_steps
   - Enable gradient checkpointing
   - Reduce model size if needed

2. CUDA Version Mismatch:
   ```bash
   # Check CUDA version
   nvidia-smi
   # Install matching JAX version if needed
   pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   ```

3. Performance Issues:
   - Ensure no other processes are using GPU
   - Monitor GPU utilization with nvidia-smi
   - Try different batch sizes to find optimal configuration

## Example Training Session

Here's a minimal example to verify your setup:

```python
from vishwamai.pretrain_efficient import main

# Run with reduced steps for testing
result = main()
print(f"Training status: {result['status']}")
print(f"Best metrics: {result['best_metrics']}")
```

This should run without errors and show gradually improving metrics.

For full training on GTX 1650, expect:
- ~2GB GPU memory usage during training
- Training speed: ~100-200 samples/second
- Efficient pre-training with curriculum learning
- Mixed precision optimization
