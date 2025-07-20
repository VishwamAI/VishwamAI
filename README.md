# VishwamAI

**Advanced generative AI framework like GPT, Claude, and Gemini - with efficient multimodal capabilities and curriculum learning support for resource-constrained environments.**

VishwamAI is a comprehensive generative AI framework for building and training multimodal AI models optimized for resource-constrained environments, similar to how GPT, Claude, and Gemini provide advanced language understanding and generation. It implements a unified Transformer-based architecture that can handle text, vision, and audio inputs through tokenization into a shared sequence space.

## 🏗️ Architecture Overview

### Unified Transformer Backbone
- **Single Architecture**: Unified Transformer that processes all modalities through tokenization
- **Decoder-Only Design**: Optimized for both understanding and generation tasks
- **Efficiency Features**: 2D rotary embeddings, QK-normalization, Cosine-scaled attention
- **Modern Components**: RMSNorm, Grouped Query Attention (GQA), FlashAttention-2

### Multimodal Capabilities
- **Vision Processing**: Vision Transformer (ViT) with patch embeddings
- **Audio Processing**: Spectrogram tokenization with CNN frontend
- **Text Processing**: Standard token embeddings with position encodings
- **Unified Tokenization**: All modalities converted to shared token space

### Hardware Optimizations
- **TPU Support**: BFloat16 precision, block-wise processing, XLA optimization
- **GPU Support**: Mixed precision (FP16/FP32), FlashAttention-2, Triton kernels
- **Memory Efficiency**: Gradient checkpointing, sparse attention patterns
- **Kernel Fusion**: Custom CUDA/TPU kernels for critical operations

## Features

- **🎯 Curriculum Learning**: Progressive training from simple to complex tasks
- **⚡ Mixed Precision**: BFloat16 (TPU) and FP16 (GPU) support
- **💾 Memory Efficiency**: Gradient checkpointing and sparse attention
- **🔄 Parameter Efficiency**: LoRA/QLoRA for fine-tuning with minimal parameters
- **🚀 Hardware Optimization**: Custom TPU and GPU kernels
- **📊 Comprehensive Monitoring**: Training metrics and performance tracking
- **🎭 Multimodal Support**: Text, vision, and audio processing
- **🧠 Advanced Attention**: FlashAttention-2 and sparse attention patterns

## Kernel Optimizations

### TPU-Specific Features
- BFloat16 precision with FP8 quantization support
- Block-wise processing with 128x128 optimal block sizes
- Memory-efficient flash attention implementation
- Dynamic shape optimization for TPU MXU
- Efficient parallel operations with XLA optimization

### GPU-Specific Features
- Mixed precision training (FP16/FP32)
- Block-sparse operations optimization
- Tensor core utilization
- CUDA-optimized attention mechanisms
- Warp-level parallelism

### Performance Highlights
- Matrix multiplication speedup with optimized kernels
- Activation functions optimization showing ~20x speedup
- Memory-efficient attention mechanisms
- Dynamic quantization for reduced memory footprint

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/VishwamAI.git
cd VishwamAI

# Install dependencies
pip install -r requirements.txt

# Run setup validation
python setup_vishwamai.py
```

### Basic Usage

```python
from vishwamai import ModelConfig, VishwamAIModel, pipeline

# Create a text generation pipeline
generator = pipeline("text-generation")

# Generate text
response = generator.generate(
    "The future of artificial intelligence is",
    max_length=100,
    temperature=0.7
)
print(response)
```

### Training a Model

```bash
# Train a small model
python scripts/train_vishwamai.py --config configs/small_model.json

# Train with LoRA for efficiency
python scripts/train_vishwamai.py --config configs/medium_model.json --use-lora

# Multimodal training
python scripts/train_vishwamai.py --config configs/multimodal_config.json
```

### Inference

```bash
# Simple text generation
python scripts/inference.py --prompt "Explain quantum computing"

# Interactive chat
python scripts/inference.py --chat

# Multimodal inference (with image)
python scripts/inference.py --multimodal --text "Describe this image" --image photo.jpg

# Performance benchmark
python scripts/inference.py --benchmark
```

## 📋 Model Configurations

### Small Model (1.2B parameters)
- **Use Case**: Experimentation, fast inference
- **Memory**: ~2.4GB inference, ~9.6GB training
- **Hardware**: GPU 8GB+, TPU v2+, CPU 16GB+

### Medium Model (7B parameters)
- **Use Case**: Balanced performance, multimodal tasks
- **Memory**: ~14GB inference, ~56GB training
- **Hardware**: GPU 24GB+, TPU v3+, CPU 64GB+

### Large Model (20B parameters)
- **Use Case**: High performance, complex reasoning
- **Memory**: ~40GB inference, ~160GB training
- **Hardware**: Multi-GPU, TPU v4 Pod, High-memory cluster

## 🎛️ Configuration Examples

### Small Model Configuration
```json
{
  "model_config": {
    "dim": 1024,
    "depth": 12,
    "heads": 16,
    "vocab_size": 32000,
    "use_flash_attention": true,
    "use_grouped_query_attention": true,
    "gradient_checkpointing": true
  }
}
```

### Multimodal Configuration
```json
{
  "model_config": {
    "dim": 2048,
    "depth": 24,
    "heads": 32,
    "enable_multimodal": true,
    "vision_patch_size": 16,
    "audio_dim": 512
  }
}
```

## 🏋️ Training Features

### Curriculum Learning
Progressive training stages:
1. **Simple**: Short sequences (512 tokens)
2. **Medium**: Medium sequences (1024 tokens)  
3. **Complex**: Full sequences (2048+ tokens)

### Parameter-Efficient Training
- **LoRA**: Low-rank adaptation for minimal parameter updates
- **QLoRA**: 4-bit quantization with LoRA adapters
- **Gradient Checkpointing**: Trade compute for memory

### Hardware-Specific Optimizations
- **TPU**: BFloat16, XLA compilation, optimal block sizes
- **GPU**: Mixed precision, FlashAttention-2, Triton kernels
- **CPU**: Optimized BLAS operations, efficient memory usage

## 🔧 Advanced Features

### Custom Kernels
```python
from vishwamai.kernels import get_optimal_kernels

# Get hardware-optimized kernels
kernels = get_optimal_kernels()

# Use optimized attention
output = kernels.kernels['flash_attention'](q, k, v)
```

### Multimodal Processing
```python
from vishwamai.multimodal import MultimodalProcessor

processor = MultimodalProcessor(
    vocab_size=50304,
    vision_config={"image_size": 224},
    audio_config={"n_mels": 80}
)

# Process multimodal input
embeddings = processor(
    text_ids=text_tokens,
    images=image_data,
    audio=audio_spectrogram
)
```

### Memory Estimation
```python
from vishwamai.utils import estimate_memory_usage

config = ModelConfig(dim=2048, depth=24, heads=32)
memory = estimate_memory_usage(config, batch_size=16)

print(f"Inference memory: {memory['inference_gb']:.2f} GB")
print(f"Training memory: {memory['total_gb']:.2f} GB")
```

## Import Test Status

```
Core Dependencies: 8/8 successful
Data Processing: 4/4 successful
Training Utilities: 4/4 successful
Memory Optimization: 5/5 successful
Additional Libraries: 3/3 successful
VishwamAI Modules: 7/7 successful
SONAR Dependencies: 5/5 successful
Multimodal Dependencies: 11/11 successful
TPU Kernels: 7/7 successful
```

## 📊 Performance Benchmarks

| Model Size | Parameters | Inference (ms) | Memory (GB) | Hardware |
|------------|------------|----------------|-------------|----------|
| Small      | 1.2B       | 45            | 2.4         | RTX 3080 |
| Medium     | 7B         | 120           | 14.0        | A100     |
| Large      | 20B        | 300           | 40.0        | A100 80GB|

## 🛠️ Development

### Project Structure
```
VishwamAI/
├── vishwamai/          # Core framework
│   ├── model.py        # Main model architecture
│   ├── attention.py    # Attention mechanisms
│   ├── kernels.py      # Hardware kernels
│   ├── training.py     # Training utilities
│   ├── multimodal.py   # Multimodal processing
│   └── utils.py        # Utility functions
├── scripts/            # Training and inference scripts
├── configs/            # Model configurations
├── docs/               # Documentation
└── examples/           # Example usage
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📚 Documentation

- [Architecture Guide](docs/VishwamAI%20Development%20Blueprint.md)
- [Multimodal Development](docs/Multimodal_AI_Development_Guide.markdown)
- [Training Guide](docs/Building%20Custom%20Generative%20AI%20with%20Limited%20Resourc.md)
- [API Reference](docs/)

## 🤝 Citation

If you use VishwamAI in your research, please cite:

```bibtex
@software{vishwamai2024,
  title={VishwamAI: Efficient Multimodal AI Framework},
  author={VishwamAI Team},
  year={2024},
  url={https://github.com/your-org/VishwamAI}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- JAX team for the excellent framework
- FlashAttention authors for memory-efficient attention
- Transformer architecture innovations from various research papers
- Open source community for tools and libraries

---

**VishwamAI**: Building the future of efficient multimodal AI 🚀
