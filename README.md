# VishwamAI

Efficient pre-training and fine-tuning framework with curriculum learning support for resource-constrained environments.

## Features

- Curriculum learning for efficient training progression
- Mixed precision support for both GPU and TPU
- Memory-efficient training with gradient checkpointing
- Flexible architecture supporting both TPU and GPU deployments
- Comprehensive monitoring and metrics tracking

## Training Optimizations

### Curriculum Learning
- Dynamic sequence length progression
- Automated difficulty adjustment
- Memory-efficient training strategy
- Configurable update intervals

### Hardware-Specific Optimizations
- **GPU (GTX 1650)**:
  - Optimized batch sizes for 4GB VRAM
  - FP16 precision training
  - Gradient accumulation
  - Memory-efficient model configuration
- **TPU**:
  - BFloat16 precision support
  - XLA optimization
  - Efficient data pipeline
  - Dynamic batch sizing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/VishwamAI/VishwamAI.git
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
```

4. Install hardware-specific dependencies:

For NVIDIA GPU:
```bash
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install nvidia-ml-py3
```

For TPU:
```bash
pip install --upgrade "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

5. Update dependencies:
```bash
poetry update
```

## Hardware-Specific Setup

### NVIDIA GPU Setup (GTX 1650)

1. Use the optimized GTX 1650 configuration:
```bash
python -m vishwamai.pretrain_efficient --config vishwamai/configs/training/gtx1650.yaml
```

For detailed GPU setup instructions, see [README_GPU.md](README_GPU.md)

### TPU Setup

1. Use the TPU-optimized configuration:
```bash
python -m vishwamai.pretrain_efficient --config vishwamai/configs/training/efficient_pretrain.yaml
```

## Interactive Development

1. Launch Jupyter notebook:
```bash
jupyter notebook notebooks/efficient_pretraining.ipynb
```

## Project Structure

```
vishwamai/
├── configs/              # Configuration files
│   ├── training/        # Training configurations
│   └── model/          # Model architectures
├── vishwamai/           # Core implementation
│   ├── model.py        # Model architecture
│   ├── training.py     # Training pipeline
│   └── tokenizer.py    # Tokenization utilities
├── notebooks/           # Interactive examples
└── docs/               # Documentation
```

## Configuration

The system supports different hardware configurations through YAML files:

- `configs/training/gtx1650.yaml`: Optimized for NVIDIA GTX 1650 (4GB VRAM)
- `configs/training/efficient_pretrain.yaml`: General TPU configuration

Key configuration sections:
```yaml
training:
  curriculum:      # Curriculum learning settings
  mixed_precision: # Precision optimization
  batch_size:      # Hardware-specific batch sizes
  
model:
  hidden_size:     # Model architecture parameters
  num_layers:      # Adjusted for hardware constraints
```

## Running Tests in Parallel

To run tests in parallel using `pytest-xdist`, use the following command:
```bash
pytest -n auto
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## Citation

If you use VishwamAI in your research, please cite:

```bibtex
@software{vishwamai2025,
  title = {VishwamAI: Efficient Pre-training Framework},
  author = {Kasinadh Sarma},
  year = {2025},
  url = {https://github.com/VishwamAI/VishwamAI}
}
```

## Support

For support and questions:
- Open an issue on GitHub
- Check existing documentation in `/docs`
- Refer to hardware-specific guides:
  - [README_GPU.md](README_GPU.md) for GPU setup
  - [HUGGINGFACE_SETUP.md](HUGGINGFACE_SETUP.md) for HuggingFace integration
- [Quick Start Guide](QUICKSTART.md)
- [Technical Documentation](docs/technical.md)
- [Advanced Training Guide](docs/advanced_training.md)
- [Error Correction System](docs/errorcorrection.md)
- [Tree of Thoughts](docs/tot.md)
- [Architecture Overview](docs/architecture.mermaid)

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Research Papers

The implementation is based on several research papers which can be found in the Research/ directory:
- Tree of Thoughts reasoning
- Mixture of Experts architectures
- Attention mechanism optimizations
- Efficient large language model training


importtest results 

```
kasinadhsarma@bgentech:~/VishwamAI$ python3 importtest.py 

Testing Core Dependencies:
✓ import jax
✓ import jax.numpy as jnp
✓ import flax.linen as nn
✓ import optax
✓ import numpy as np
✓ import torch
/home/kasinadhsarma/.local/lib/python3.12/site-packages/transformers/utils/generic.py:441: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  _torch_pytree._register_pytree_node(
✓ from transformers import AutoTokenizer
✓ from safetensors import safe_open

Testing Data Processing:
✓ import datasets
✓ import sentencepiece
✓ import tokenizers
✓ from huggingface_hub import snapshot_download

Testing Training Utilities:
✓ import wandb
✓ import duckdb
✓ import tqdm
✓ import pyarrow

Testing Memory Optimization:
✓ import einops
✓ import chex
✓ import jaxtyping
✓ import optree
✓ import orbax.checkpoint

Testing Additional Libraries:
✓ import scipy
✓ from ml_collections import ConfigDict
✓ import typing_extensions

Testing VishwamAI Modules:
2025-03-19 00:34:24.580806: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1742324664.601755   51753 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1742324664.607933   51753 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
✓ from vishwamai.transformer import EnhancedTransformerModel
✓ from vishwamai.layers.layers import TPUGEMMLinear, TPULayerNorm, TPUMultiHeadAttention, TPUMoELayer
✓ from vishwamai.layers.attention import FlashAttention
✓ from vishwamai.kernels.kernel import fp8_gemm_optimized
✓ from vishwamai.thoughts.tot import TreeOfThoughts, ThoughtNode
✓ from vishwamai.thoughts.cot import ChainOfThoughtPrompting
✓ from vishwamai.distill import compute_distillation_loss, create_student_model, initialize_from_teacher

Testing SONAR Dependencies:
✓ import fairseq2
✓ import editdistance
✓ import importlib_metadata
✓ import importlib_resources
✓ import sacrebleu

Testing Multimodal Dependencies:
✓ import PIL
✓ from PIL import Image
✓ import torchvision
✓ import timm
/home/kasinadhsarma/.local/lib/python3.12/site-packages/transformers/utils/generic.py:309: FutureWarning: `torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.
  _torch_pytree._register_pytree_node(
✓ from transformers import CLIPProcessor, CLIPModel
✓ import cv2
✓ import albumentations
✓ import kornia
✓ from vishwamai.multimodal.vision import ViTEncoder, CLIPAdapter
✓ from vishwamai.multimodal.fusion import CrossAttentionFuser, MultimodalProjector
✓ from vishwamai.multimodal.processor import ImageProcessor, MultimodalBatchProcessor

Testing TPU Kernels:
✓ from vishwamai.kernels.kernel import fp8_gemm_optimized, act_quant
✓ from vishwamai.kernels.fp8_cast_bf16 import bf16_cast_to_fp8
✓ from vishwamai.kernels.activation import gelu_approx, silu_optimized
✓ from vishwamai.kernels.quantization import dynamic_quant, static_quant
✓ from vishwamai.kernels.tensor_parallel import shard_params, all_gather, all_reduce
✓ from vishwamai.kernels.sparse import sparse_gemm, sparse_attention
✓ from vishwamai.kernels.moe_dispatch import load_balance_loss, compute_routing_prob

Testing TPU Optimized Layers:
✓ from vishwamai.layers.layers import TPUGEMMLinear, TPULayerNorm, TPUMultiHeadAttention
✓ from vishwamai.layers.moe import TPUMoELayer, TPUSparseMoEDispatch
✓ from vishwamai.layers.rotary import TPURotaryEmbedding, apply_rotary_pos_emb
✓ from vishwamai.layers.activation import GELUActivation, SwiGLUActivation
✓ from vishwamai.layers.normalization import RMSNorm, AdaNorm
✓ from vishwamai.layers.attention import FlashAttention

Import Test Summary:
-------------------
Core Dependencies: 8/8 successful
Data Processing: 4/4 successful
Training Utilities: 4/4 successful
Memory Optimization: 5/5 successful
Additional Libraries: 3/3 successful
VishwamAI Modules: 7/7 successful
SONAR Dependencies: 5/5 successful
Multimodal Dependencies: 11/11 successful
TPU Kernels: 7/7 successful
TPU Optimized Layers: 6/6 successful

Overall: 60/60 imports successful (100.0%)

Testing multimodal functionality:
1. Testing image processing...
WARNING:2025-03-19 00:34:30,296:jax._src.xla_bridge:966: An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
WARNING:jax._src.xla_bridge:An NVIDIA GPU may be present on this machine, but a CUDA-enabled jaxlib is not installed. Falling back to cpu.
✗ Image processor failed: convolution dimension_numbers[1] must contain the characters 'O' and 'I' exactly once, got HWCO.
2. Testing vision encoder...
✗ Vision encoder failed: FlashAttention.__init__() got an unexpected keyword argument 'use_fp8'
3. Testing multimodal fusion...
✗ Multimodal fusion failed: Module construction attributes are frozen. (https://flax.readthedocs.io/en/latest/api_reference/flax.errors.html#flax.errors.SetAttributeInModuleSetupError)

Multimodal functionality tests completed

Testing kernel performance:
1. Testing GEMM performance...
✓ GEMM performance - Standard: 0.0000s, Optimized: 0.0001s, Speedup: 0.21x
2. Testing activation functions...
✓ Activation performance - Standard: 0.1619s, Optimized: 0.0078s, Speedup: 20.82x

Kernel performance tests completed
kasinadhsarma@bgentech:~/VishwamAI$ 
```
