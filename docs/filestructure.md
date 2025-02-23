# Project File Structure

## Overview
The Vishwamai project follows a modular structure organized by functionality. Below is a detailed breakdown of each directory and its purpose.

```plaintext
vishwamai/
├── configs/              # Configuration files
│   ├── model_config.yaml     # Model architecture settings
│   ├── moe_config.yaml      # MoE-specific settings
│   ├── mla_config.yaml      # Multi-Level Attention settings
│   ├── training_config.yaml # Training parameters
│   ├── data_config.yaml    # Data processing settings
│   └── tpu_config.yaml     # TPU and distribution settings
│
├── data/                # Data processing modules
│   ├── preprocessing.py     # Text preprocessing utilities
│   ├── tokenization.py     # Tokenizer implementation
│   ├── dataloader.py       # Data loading utilities
│   ├── dataset/            # Dataset implementations
│   │   ├── base.py           # Base dataset class
│   │   └── implementations/  # Specific dataset implementations
│   │       ├── mmlu.py
│   │       ├── mmmu.py
│   │       └── gsm8k.py
│   └── augmentation/      # Data augmentation utilities
│       └── text_augment.py
│
├── model/               # Model architecture components
│   ├── attention/         # Attention mechanisms
│   │   ├── self_attention.py
│   │   ├── cross_attention.py
│   │   └── flash_attention.py
│   ├── moe/              # Mixture of Experts implementation
│   │   ├── expert.py
│   │   ├── router.py
│   │   ├── moe_layer.py
│   │   └── gating/
│   │       └── __init__.py
│   ├── mla/              # Multi-Level Attention
│   │   ├── attention.py
│   │   ├── layer_manager.py
│   │   ├── mla_block.py
│   │   └── residual.py
│   ├── transformer/      # Transformer architecture
│   │   ├── block.py
│   │   ├── layer.py
│   │   ├── config.py
│   │   └── model.py
│   ├── embeddings/       # Embedding layers
│   │   ├── token_embedding.py
│   │   └── positional.py
│   └── initialization/   # Weight initialization
│       ├── expert_init.py
│       └── weight_init.py
│
├── training/            # Training utilities
│   ├── optimizer/         # Optimization algorithms
│   │   ├── adamw.py
│   │   └── fairscale.py
│   ├── scheduling/       # Learning rate scheduling
│   │   ├── lr_scheduler.py
│   │   └── warmup.py
│   ├── distributed/      # Distributed training
│   │   ├── tpu_utils.py
│   │   ├── expert_sharding.py
│   │   └── comm_ops.py
│   └── callbacks/        # Training callbacks
│       ├── checkpoint.py
│       ├── early_stopping.py
│       └── lr_scheduler_cb.py
│
├── utils/               # Utility functions
│   ├── logging.py         # Logging utilities
│   ├── profiling/        # Performance profiling
│   │   ├── memory.py
│   │   └── performance.py
│   └── visualization/    # Visualization tools
│       ├── training_viz.py
│       ├── attention_viz.py
│       └── expert_viz.py
│
├── scripts/             # Command-line scripts
│   ├── preprocess_data.py  # Data preprocessing script
│   ├── train_tokenizer.py  # Tokenizer training script
│   ├── train_model.py      # Model training script
│   ├── evaluate_model.py   # Model evaluation script
│   ├── serve_model.py      # Model serving script
│   └── export_model.py     # Model export script
│
└── docs/                # Documentation
    ├── technical.md       # Technical documentation
    ├── architecture.mermaid # Architecture diagram
    └── filestructure.md   # This file

```

## Key Components

### 1. Configurations (`configs/`)
Contains YAML configuration files for different aspects of the model and training process.

### 2. Data Processing (`data/`)
Modules for data preprocessing, tokenization, and dataset implementations.

### 3. Model Architecture (`model/`)
Core model components including attention mechanisms, MoE layers, and transformer blocks.

### 4. Training (`training/`)
Training utilities, optimizers, schedulers, and distributed training components.

### 5. Utilities (`utils/`)
Helper functions for logging, profiling, and visualization.

### 6. Scripts (`scripts/`)
Command-line tools for various tasks in the training pipeline.

### 7. Documentation (`docs/`)
Project documentation and technical details.

## Organization Principles

1. **Modularity**:
   - Each component is self-contained
   - Clear separation of concerns
   - Minimal interdependencies

2. **Configuration**:
   - All parameters are configurable via YAML files
   - Separate configs for different aspects
   - Easy experiment management

3. **Extensibility**:
   - Base classes for key components
   - Easy to add new implementations
   - Pluggable architecture

4. **Documentation**:
   - Inline documentation
   - Architecture diagrams
   - Technical specifications

## Adding New Components

1. **New Dataset**:
   ```plaintext
   data/dataset/implementations/
   └── new_dataset.py  # Implement BaseDataset
   ```

2. **New Expert Type**:
   ```plaintext
   model/moe/experts/
   └── new_expert.py   # Implement BaseExpert
   ```

3. **New Attention Mechanism**:
   ```plaintext
   model/attention/
   └── new_attention.py  # Implement BaseAttention
   ```

## Best Practices

1. **Code Organization**:
   - Follow the established directory structure
   - Keep related files together
   - Use appropriate subdirectories

2. **Documentation**:
   - Document all public interfaces
   - Include examples in docstrings
   - Keep documentation up-to-date

3. **Configuration**:
   - Add new parameters to appropriate config files
   - Document configuration options
   - Provide default values

4. **Testing**:
   - Add tests for new components
   - Follow existing test patterns
   - Include integration tests
