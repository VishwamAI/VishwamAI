# VishwamAI

VishwamAI is a high-performance transformer model with cache augmentation, neural memory, and tree of thoughts reasoning capabilities.

## Features

- Multi-head Linear Attention with cache augmentation
- Neural Memory Transformer integration
- Tree of Thoughts reasoning
- GPU-optimized implementation
- Automatic configuration based on GPU type
- Support for T4, V100, and A100 GPUs

## Installation

```bash
git clone https://github.com/kasinadhsarma/VishwamAI.git
cd VishwamAI
pip install -e .
```

## Training

### Training on Google Colab

1. Open `colab_train.ipynb` in Google Colab
2. Select GPU runtime: Runtime > Change runtime type > GPU
3. Run all cells in order

### Local Training

1. Make the training script executable:
```bash
chmod +x vishwamai/scripts/pretrain.sh
```

2. Start training:
```bash
./vishwamai/scripts/pretrain.sh \
    -b 4 \           # Batch size
    -e 3 \           # Number of epochs
    -o ./output \    # Output directory
    -c configs/config_optimized.json  # Model config
```

### GPU-Specific Configurations

The model automatically detects your GPU and applies optimized settings:

- T4 GPU (16GB):
  - Dimension: 1536
  - Batch size: 2
  - Sequence length: 1024
  - FP16 precision

- V100 GPU (32GB):
  - Dimension: 2048
  - Batch size: 4
  - Sequence length: 2048
  - FP16 precision

- A100 GPU (40/80GB):
  - Dimension: 2048
  - Batch size: 8
  - Sequence length: 4096
  - FP8 precision

## Model Usage

```python
from vishwamai.model_utils import load_model

# Load model with optimized settings
model = load_model(
    config_path="configs/config_optimized.json",
    device="cuda"
)

# Example inference
import torch
tokens = torch.randint(0, model.args.vocab_size, (1, 128)).cuda()
with torch.inference_mode():
    output = model(tokens)
```

For more examples, see `vishwamai/examples/model_usage.py`.

## File Structure

```
vishwamai/
├── configs/              # Model configurations
├── examples/             # Usage examples
├── scripts/             # Training scripts
├── model.py             # Core model implementation
├── model_utils.py       # Model utilities
├── cache_augmentation.py # Cache augmentation
├── neural_memory.py     # Neural memory
└── tree_of_thoughts.py  # Tree of thoughts
```

## Training Data

The model can be trained on various datasets:
- GSM8K for mathematical reasoning
- MMLU for multi-task learning
- Custom datasets (see examples)

## Performance Optimization

1. Memory Optimization:
```python
# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Disable caching for training
model = load_model(config_path, use_cache=False)
```

2. Training Speed:
```bash
# Use larger batch size with gradient accumulation
./vishwamai/scripts/pretrain.sh -b 8 --gradient_accumulation_steps 4
```

3. GPU Utilization:
```python
# Enable flash attention and optimized kernels
config['optimization_config']['use_flash_attention'] = True
config['optimization_config']['use_kernel_optimizations'] = True
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Implement your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
