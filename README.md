# VishwamAI

VishwamAI is an open-source implementation of a mixture-of-experts (MoE) transformer with multi-level attention (MLA) for efficient large language model training on TPUs.

## Features

- Mixture of Experts (MoE) architecture for parameter efficiency
- Multi-Level Attention (MLA) mechanism for adaptive computation
- TPU-optimized training infrastructure
- Support for HuggingFace ecosystem
- Efficient distributed training on TPU pods

## Project Structure

```
vishwamai/
├── configs/               # Configuration files
├── data/                 # Data loading and preprocessing
├── model/               
│   ├── attention/        # Attention mechanisms
│   ├── embeddings/       # Token and positional embeddings
│   ├── moe/             # Mixture of Experts implementation
│   ├── mla/             # Multi-Level Attention implementation
│   └── transformer/      # Core transformer components
├── training/
│   ├── callbacks/        # Training callbacks
│   ├── distributed/      # TPU distribution utilities
│   ├── optimizer/        # Optimizers and schedulers
│   └── scheduling/       # Learning rate scheduling
└── utils/               # Utility functions
```

## Installation

```bash
pip install -e .
```

## Quick Start: GSM8K Training

We provide a notebook for training on the GSM8K (Grade School Math 8K) dataset:

1. Clone the repository:
```bash
git clone https://github.com/VishwamAI/VishwamAI.git
cd VishwamAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Open `train_gsm8k.ipynb` in Jupyter:
```bash
jupyter notebook train_gsm8k.ipynb
```

The notebook contains a complete training pipeline including:
- Dataset loading and preprocessing
- Model initialization with MoE-MLA architecture
- TPU-optimized training setup
- Evaluation and model saving
- Automatic upload to HuggingFace Hub

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

MoE-MLA specific settings:
```yaml
num_experts: 8
num_attention_levels: 3
expert_capacity_factor: 1.25
use_adaptive_computation: true
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
