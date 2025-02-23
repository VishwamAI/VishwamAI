# Quickstart Guide

This guide helps you get started with training models using the Vishwamai framework.

## Installation

First, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

The training process is controlled by several YAML configuration files:

1. `model_config.yaml` - Model architecture configuration
2. `data_config.yaml` - Dataset and preprocessing configuration
3. `training_config.yaml` - Training hyperparameters and settings
4. `moe_config.yaml` (optional) - Mixture of Experts configuration
5. `mla_config.yaml` (optional) - Multi-Level Attention configuration

Example configurations are provided in the `configs/` directory.

### Basic Model Configuration

```yaml
# configs/model_config.yaml
model_type: "transformer"
hidden_size: 768
num_layers: 12
num_heads: 12
mlp_ratio: 4
dropout: 0.1
attention_dropout: 0.1
use_bias: true
vocab_size: 50000
max_position_embeddings: 2048
```

### Data Configuration

```yaml
# configs/data_config.yaml
dataset_type: "text"
tokenizer_type: "bpe"
vocab_file: "path/to/vocab.json"
max_seq_length: 1024
batch_size: 32
num_workers: 4
```

### Training Configuration

```yaml
# configs/training_config.yaml
num_epochs: 100
learning_rate: 1e-4
weight_decay: 0.01
warmup_steps: 1000
gradient_accumulation_steps: 4
early_stopping_patience: 3
```

## Training

To train a basic transformer model:

```bash
python -m vishwamai.train \
    --model_config configs/model_config.yaml \
    --data_config configs/data_config.yaml \
    --training_config configs/training_config.yaml \
    --train_data path/to/train.txt \
    --val_data path/to/val.txt \
    --output_dir experiments \
    --experiment_name my_experiment
```

### Distributed Training

To enable distributed training across multiple GPUs:

```bash
python -m vishwamai.train \
    --model_config configs/model_config.yaml \
    --data_config configs/data_config.yaml \
    --training_config configs/training_config.yaml \
    --train_data path/to/train.txt \
    --val_data path/to/val.txt \
    --distributed \
    --world_size 4 \
    --output_dir experiments \
    --experiment_name distributed_training
```

### Training with MoE

To train a Mixture of Experts model:

```bash
python -m vishwamai.train \
    --model_config configs/model_config.yaml \
    --moe_config configs/moe_config.yaml \
    --data_config configs/data_config.yaml \
    --training_config configs/training_config.yaml \
    --train_data path/to/train.txt \
    --val_data path/to/val.txt \
    --distributed \
    --world_size 8 \
    --output_dir experiments \
    --experiment_name moe_training
```

## Experiment Tracking

Training logs and artifacts are saved in the specified output directory:

```
experiments/
└── my_experiment/
    ├── args.json              # Command line arguments
    ├── model_config.yaml      # Model configuration
    ├── data_config.yaml       # Data configuration  
    ├── training_config.yaml   # Training configuration
    ├── checkpoints/          # Model checkpoints
    ├── tensorboard/         # TensorBoard logs
    └── training.log         # Training logs
```

To monitor training progress:

```bash
# View training logs
tail -f experiments/my_experiment/training.log

# Launch TensorBoard
tensorboard --logdir experiments/my_experiment/tensorboard
```

## Advanced Features

### Mixed Precision Training

Enable automatic mixed precision:

```bash
python -m vishwamai.train \
    [...]  # Other arguments
    --use_amp
```

### TPU Training

Train on Cloud TPUs:

```bash
python -m vishwamai.train \
    [...]  # Other arguments  
    --use_tpu
```

### Resuming Training

Resume training from a checkpoint:

```bash
python -m vishwamai.train \
    [...]  # Other arguments
    --resume_from experiments/my_experiment/checkpoints/checkpoint-latest.pt
```

## Configuration Reference

See [CONFIG_REFERENCE.md](docs/CONFIG_REFERENCE.md) for detailed documentation of all configuration options.
