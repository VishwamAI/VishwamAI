# Vishwamai: Efficient T4-Optimized Machine Learning Model

Vishwamai is a PyTorch-based machine learning framework optimized for NVIDIA T4 GPUs, featuring:

- Flexible precision modes (FP16, FP32, BF16, TF32)
- Mixed precision training with automatic optimization
- Flash Attention and kernel optimizations
- Mixture of Experts (MoE) support
- Tree-based planning capabilities

## Features

### Precision Modes

```python
from vishwamai.model import create_model, ModelArgs

# Create model with FP16 mixed precision
model = create_model(
    ModelArgs(
        dtype="fp16",
        use_mixed_precision=True,
        use_flash_attention=True
    )
)
```

### Model Presets

```python
from vishwamai.model import (
    create_tiny_model,
    create_base_model,
    create_efficient_model,
    create_expert_model
)

# Create different model variants
tiny_model = create_tiny_model(dtype="fp16")
base_model = create_base_model(dtype="bf16")
efficient_model = create_efficient_model()
expert_model = create_expert_model(num_experts=8)
```

### Available Configurations

- **VISHWAMAI_TINY**: Small model for quick experiments
- **VISHWAMAI_BASE**: Standard balanced model
- **VISHWAMAI_LARGE**: Larger model with enhanced capacity
- **VISHWAMAI_XL**: Extra large model for complex tasks
- **VISHWAMAI_EXPERT**: Model with Mixture of Experts
- **VISHWAMAI_EFFICIENT**: Memory and compute optimized model
- **VISHWAMAI_QUANTIZED**: Quantization-aware training model
- **VISHWAMAI_EXTENDED**: Long sequence model with dynamic scaling

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import torch
from vishwamai.model import create_model, ModelArgs

# Configure model
config = ModelArgs(
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=12,
    dtype="fp16",
    use_mixed_precision=True
)

# Create and use model
model = create_model(config=config)
inputs = {
    "input_ids": torch.randint(0, config.vocab_size, (1, 128)),
    "attention_mask": torch.ones(1, 128)
}
outputs = model(**inputs)
```

## T4 Optimizations

The framework includes several T4-specific optimizations:

1. Automatic precision selection based on hardware capabilities
2. Flash Attention support for memory-efficient attention
3. Gradient checkpointing for large models
4. Kernel fusion optimizations
5. Memory-efficient mixed precision training

## Development

### Testing

```bash
# Run precision tests
cd tests
./setup_tests.sh  # or setup_tests.bat on Windows
./run_precision_tests.sh
```

### Benchmarking

```bash
cd examples
python benchmark_precision.py
```

## Hardware Requirements

- NVIDIA T4 GPU (recommended)
- CUDA 11.0+
- 16GB+ GPU memory for large models
- 32GB+ system RAM recommended

## Citation

If you use Vishwamai in your research, please cite:

```bibtex
@software{vishwamai2025,
  title={Vishwamai: T4-Optimized Machine Learning Model},
  author={Vishwamai Team},
  year={2025},
  url={https://github.com/yourusername/vishwamai}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) to get started.

## Acknowledgments

- NVIDIA for T4 GPU support
- PyTorch team for the deep learning framework
- HuggingFace team for transformer architecture inspirations
