# VishwamAI

VishwamAI is a state-of-the-art transformer-based language model implementation featuring advanced architectures like Mixture of Experts (MoE), Tree of Thoughts (ToT), and sophisticated attention mechanisms.

## Core Features

- **Advanced Architecture**:
  - Mixture of Experts (MoE) with dynamic expert selection
  - Tree of Thoughts (ToT) reasoning capabilities
  - Mixture of Depth (MoD) networks
  - Grouped Query Attention (GQA)
  - Rotary Position Embeddings (RoPE)
  - Flash Attention support

- **Efficient Training**:
  - Knowledge distillation framework
  - Multi-level attention mechanisms
  - Expert capacity management
  - Load balancing optimization
  - Gradient checkpointing

- **Advanced Components**:
  - Error detection and correction
  - Sliding window attention
  - Token-wise complexity weighting
  - Dynamic expert routing

## Architecture Details

### Mixture of Experts (MoE)
- Dynamic expert selection with routing
- Load balancing optimization
- Expert capacity management
- Parallel MLP implementation
- Adaptive computation paths

### Tree of Thoughts (ToT)
- Multi-level thought integration
- Beam search for reasoning paths
- Thought feature collection
- Integration with main transformer
- Attention-based thought selection

### Mixture of Depth (MoD)
- Dynamic depth adaptation
- Layer-wise complexity routing
- Adaptive computation time
- Efficient forward pass
- Per-token depth control

### Attention Mechanisms
- Flash Attention support
- Grouped Query Attention (GQA)
- RoPE (Rotary Position Embeddings)
- ALiBi (Attention with Linear Biases)
- Sliding window attention with global tokens

## Project Structure

```
vishwamai/
├── configs/               # Configuration files
│   ├── model/            # Model architectures
│   └── training/         # Training settings
├── core/
│   ├── model.py          # Main model implementation
│   ├── transformer.py    # Transformer architecture
│   └── attention.py      # Attention mechanisms
├── training/
│   ├── distillation.py   # Knowledge distillation
│   └── error_correction.py # Error correction
├── reasoning/
│   └── tot.py           # Tree of Thoughts implementation
└── utils/
    ├── convert.py        # Model conversion
    └── tokenizer.py      # Tokenization utilities
```

## Configuration

### Model Configuration
```python
config = ModelConfig(
    hidden_size=4096,          # Model dimension
    num_attention_heads=32,    # Number of attention heads
    num_key_value_heads=8,     # Number of KV heads for GQA
    n_experts=8,              # Number of experts
    expert_dim=4096,          # Expert dimension
    use_flash_attention=True,  # Enable Flash Attention
    use_rope=True,            # Enable RoPE
    use_alibi=False,          # Disable ALiBi
)
```

### Expert Configuration
```python
moe_config = {
    "n_experts": 8,
    "expert_dim": 4096,
    "expert_pruning_threshold": 0.1,
    "min_active_experts": 4,
    "dynamic_expert_selection": True,
    "expert_capacity_factor": 1.25
}
```

## Installation

```bash
pip install -e .
```

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/VishwamAI/VishwamAI.git
cd VishwamAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start with the distillation training notebook:
```bash
jupyter notebook notebooks/train_vishwamai_distillation.ipynb
```

## Documentation

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
