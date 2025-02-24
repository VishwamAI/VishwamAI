# VishwamAI

VishwamAI is an open-source implementation of a mixture-of-experts (MoE) transformer with multi-level attention (MLA) for efficient large language model training on TPUs.

## Features

- Mixture of Experts (MoE) architecture for parameter efficiency
- Multi-Level Attention (MLA) mechanism for adaptive computation
- Knowledge distillation support for model compression
- TPU-optimized training infrastructure
- Support for HuggingFace ecosystem
- Efficient distributed training on TPU pods

## Project Structure

```
vishwamai/
├── configs/               # Configuration files
│   ├── model/            # Model-specific configurations
│   └── training/         # Training configurations including distillation
├── convert.py            # Model conversion utilities
├── data_utils.py         # Data loading and preprocessing
├── distillation.py       # Knowledge distillation implementation
├── error_correction.py   # Error correction mechanisms
├── generate.py          # Text generation utilities
├── model.py             # Core model implementation
├── tokenizer.py         # Tokenization utilities
├── tot.py               # Tree of Thoughts implementation
├── training.py          # Training pipeline
└── transformer.py       # Transformer architecture components
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

The distillation notebook provides a complete training pipeline including:
- Knowledge distillation setup and configuration
- Dataset loading and preprocessing
- Model initialization with MoE-MLA architecture
- TPU-optimized training setup
- Performance evaluation and model saving
- Integration with HuggingFace Hub

## TPU Training

For TPU training, first configure your TPU environment:

```bash
export TPU_NAME="your-tpu-name"
export TPU_ZONE="your-tpu-zone"
```

Then run the training script:

```bash
./huggingface_pretrain.sh \
    --model_type moe_mla_transformer \
    --model_size base \
    --data_path data/pretrain \
    --tokenizer_path tokenizer \
    --output_dir checkpoints/base_moe \
    --tpu_name $TPU_NAME \
    --tpu_zone $TPU_ZONE \
    --batch_size 32
```

## Model Configurations

Available model sizes:
- `small`: 768 hidden, 12 layers, 12 heads
- `base`: 1024 hidden, 24 layers, 16 heads
- `large`: 1536 hidden, 36 layers, 24 heads
- `xl`: 2048 hidden, 48 layers, 32 heads

Configuration files (in `vishwamai/configs/`):
```yaml
# Model configurations (model/10B.yaml)
model:
  num_experts: 8
  num_attention_levels: 3
  expert_capacity_factor: 1.25
  use_adaptive_computation: true

# Training configurations (training/distillation.yaml)
training:
  teacher_model: "path/to/teacher"
  temperature: 2.0
  alpha: 0.5  # Balance between distillation and task loss
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

```bibtex
@software{vishwamai2025,
  author = {VishwamAI Team},
  title = {VishwamAI: Efficient Large Language Models with MoE-MLA},
  year = {2025},
  url = {https://github.com/VishwamAI/VishwamAI}
}
```

## Acknowledgments

- Thanks to all contributors who helped shape this project
- Special thanks to the TPU Research Cloud for compute resources
