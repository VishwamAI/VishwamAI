# VishwamAI Model Card

## Model Description

VishwamAI is a transformer-based language model enhanced with cache augmentation, neural memory, and tree of thoughts reasoning capabilities. It is optimized for both local and cloud GPU environments.

## Official Repository
https://github.com/VishwamAI/VishwamAI

## Specifications

- **Architecture**: Transformer-based with MLA (Multi-head Linear Attention)
- **Parameters**: Varies based on GPU configuration (1.5B - 6B)
- **Context Length**: 1024-4096 tokens (GPU dependent)
- **Training Data**: GSM8K, MMLU
- **License**: MIT

## Hardware Requirements

### Minimum Requirements
- NVIDIA GPU with 8GB VRAM
- 16GB System RAM
- CUDA 11.8+

### Recommended Requirements
- NVIDIA T4, V100, or A100 GPU
- 32GB+ System RAM
- CUDA 12.0+

## Performance

| GPU Type | Parameters | Batch Size | Sequence Length | Precision |
|----------|------------|------------|-----------------|-----------|
| T4       | 1.5B      | 2          | 1024           | FP16      |
| V100     | 2.7B      | 4          | 2048           | FP16      |
| A100     | 6.7B      | 8          | 4096           | FP8       |

## Features

1. Cache Augmentation
   - Dynamic context caching
   - Memory-efficient processing
   - Adaptive cache size

2. Neural Memory
   - Long-term information retention
   - Memory-guided reasoning
   - Multi-layer memory transformer

3. Tree of Thoughts
   - Branching reasoning paths
   - Self-reflective thinking
   - Dynamic depth exploration

## Usage Examples

```python
from vishwamai.model_utils import load_model

# Load model
model = load_model(
    config_path="configs/config_optimized.json",
    device="cuda"
)

# Run inference
import torch

tokens = torch.randint(0, model.args.vocab_size, (1, 128)).cuda()
with torch.inference_mode():
    output = model(tokens)
```

## Training

### Local Training
```bash
./vishwamai/scripts/pretrain.sh -b 4 -e 3 -o ./output
```

### Google Colab
Use the provided `colab_train.ipynb` notebook.

## Performance Optimization Tips

1. Memory Usage
   - Enable gradient checkpointing for large models
   - Use appropriate batch size for your GPU
   - Utilize flash attention when available

2. Training Speed
   - Use mixed precision training (FP16/BF16)
   - Enable GPU kernel optimizations
   - Adjust sequence length based on available memory

3. Inference
   - Use torch.inference_mode()
   - Keep reasonable sequence lengths
   - Utilize model caching when appropriate

## Installation & Setup

For detailed installation instructions, see `SETUP.md` in the official repository: 
https://github.com/VishwamAI/VishwamAI

## Limitations

1. Hardware Constraints
   - Requires NVIDIA GPU
   - Memory usage scales with sequence length
   - Performance dependent on GPU type

2. Training Requirements
   - Significant computational resources needed
   - Long training times on smaller GPUs
   - Memory constraints on large models

## Future Development

- Enhanced CPU support
- Improved memory efficiency
- Additional reasoning capabilities
- Broader dataset support

## Citation

If you use this model in your research, please cite:

```bibtex
@software{vishwamai2025,
  title = {VishwamAI: A High-Performance Transformer with Enhanced Reasoning},
  author = {Kasinadh Sarma},
  year = {2025},
  url = {https://github.com/VishwamAI/VishwamAI}
}
