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
