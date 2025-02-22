# VishwamAI Setup Guide

This guide explains how to set up VishwamAI for different GPU configurations.

## Prerequisites

- Python 3.8+
- CUDA compatible GPU (GTX 1650 or better)
- Git

## Quick Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/VishwamAI.git
cd VishwamAI
```

2. **Run the setup script:**
```bash
./setup_gpu_env.sh
```

This will:
- Create a new virtual environment
- Install all required dependencies
- Configure PyTorch with CUDA support
- Run initial GPU compatibility tests

## Manual Setup

If you prefer to set up manually:

1. **Create virtual environment:**
```bash
python3 -m venv venv
source venv/bin/activate
```

2. **Install dependencies:**
```bash
pip install -r requirements-gpu.txt
```

## Troubleshooting

### DeepSpeed Installation Issues

If you encounter DeepSpeed/pydantic compatibility issues:

1. **Clean existing installations:**
```bash
pip uninstall -y deepspeed pydantic
```

2. **Install specific versions:**
```bash
pip install "pydantic>=1.10.0,<2.0.0"
pip install deepspeed==0.10.0
```

### CUDA Issues

1. **Verify CUDA installation:**
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

2. **Check CUDA version:**
```bash
nvidia-smi
```

3. **If CUDA is not detected:**
- Ensure NVIDIA drivers are installed
- Install CUDA toolkit matching your PyTorch version
- Add CUDA paths to your environment

## Testing Your Setup

1. **Basic GPU Test:**
```bash
python vishwamai/training/gpu_testing.py
```

2. **Full Model Test (GTX 1650):**
```bash
# This will automatically adjust settings for 4GB VRAM
python vishwamai/training/gpu_testing.py --full-test
```

3. **A100 Configuration (if available):**
```bash
deepspeed --num_gpus=8 vishwamai/training/a100_pretrain.py
```

## Configuration Files

- `requirements-gpu.txt`: Core dependencies for GPU support
- `setup_gpu_env.sh`: Automated setup script
- `vishwamai/training/gpu_testing.py`: GPU compatibility testing
- `vishwamai/training/a100_pretrain.py`: A100-optimized training

## Environment Variables

Set these if needed:

```bash
export CUDA_VISIBLE_DEVICES=0  # Specify GPU device
export TORCH_CUDA_ARCH_LIST="7.5"  # For GTX 1650
```

For A100:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # All GPUs
export TORCH_CUDA_ARCH_LIST="8.0"  # For A100
```

## Memory Requirements

| GPU Model | VRAM | Max Batch Size | Max Sequence Length |
|-----------|------|----------------|-------------------|
| GTX 1650  | 4GB  | 4             | 128               |
| A100      | 80GB | 2048          | 2048              |

## Next Steps

- See `docs/GPU_TESTING.md` for testing guidelines
- See `docs/A100_OPTIMIZATIONS.md` for A100-specific optimizations
- Start with small configurations and gradually scale up
- Monitor GPU memory usage with `nvidia-smi`

## Support

For issues:
1. Check common solutions in `docs/GPU_TESTING.md`
2. Verify CUDA compatibility
3. Check GPU memory usage
4. Ensure all dependencies are correctly installed

## References

1. [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
2. [DeepSpeed Documentation](https://www.deepspeed.ai/)
3. [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
