# VishwamAI Setup Guide

This guide covers setting up VishwamAI in different environments.

## Official Repository
https://github.com/VishwamAI/VishwamAI

## System Requirements

### Minimum (CPU Only)
- Python 3.8+
- 16GB RAM
- 50GB disk space
- CPU with AVX2 support

### Recommended (GPU)
- NVIDIA GPU with 8GB+ VRAM
- CUDA 11.8+
- 32GB RAM
- 100GB disk space

## Installation

1. Clone official repository:
```bash
git clone https://github.com/VishwamAI/VishwamAI.git
cd VishwamAI
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# OR
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
# For GPU
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Environment Setup

### CPU Environment
```python
import torch
from vishwamai.model_utils import load_model

# Load model in CPU mode
model = load_model(
    config_path="configs/config_optimized.json",
    device="cpu"
)
```

### GPU Environment
```python
import torch
from vishwamai.model_utils import load_model

# Check GPU
assert torch.cuda.is_available(), "GPU not available"
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Load model in GPU mode
model = load_model(
    config_path="configs/config_optimized.json",
    device="cuda"
)
```

## Google Colab Setup

1. Clone repository:
```python
!git clone https://github.com/VishwamAI/VishwamAI.git
%cd VishwamAI
```

2. Install dependencies:
```bash
!pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install -r requirements.txt
```

3. Choose notebook:
- `colab_train.ipynb`: For fine-tuning
- `vishwamai_colab_pretrain.ipynb`: For pretraining

## Common Issues

### Out of Memory
```python
# Enable memory efficient settings
model = load_model(
    config_path="configs/config_optimized.json",
    device="cuda",
    use_cache=False
)
model.gradient_checkpointing_enable()
```

### CPU Performance
For better CPU performance:
```python
# Set number of threads
import torch
torch.set_num_threads(8)  # Adjust based on your CPU

# Use efficient forward pass
with torch.inference_mode():
    output = model(input_tokens)
```

### GPU Utilization
Monitor GPU usage:
```bash
# Linux
watch -n 1 nvidia-smi

# Windows PowerShell
while(1) {nvidia-smi; Start-Sleep -s 1; cls}
```

## Quick Tests

1. Basic functionality:
```python
from vishwamai.examples.model_examples import basic_usage
basic_usage()
```

2. Memory efficiency:
```python
from vishwamai.examples.model_examples import memory_efficient_usage
memory_efficient_usage()
```

3. CPU fallback:
```python
from vishwamai.examples.model_examples import cpu_fallback_usage
cpu_fallback_usage()
```

## Directory Structure
```
vishwamai/
├── configs/              # Configuration files
├── examples/             # Usage examples
├── model.py             # Core model
├── model_utils.py       # Utilities
└── scripts/             # Training scripts
```

## Additional Resources

- Official Repository: https://github.com/VishwamAI/VishwamAI
- Training Guide: See `TRAINING.md`
- Model Details: See `MODEL_CARD.md`
- Example Usage: See `vishwamai/examples/`

## Support

For issues and questions:
1. Check the documentation
2. Search existing issues on GitHub
3. Create a new issue in the official repository

For contributions, please see the contributing guidelines in the README.
