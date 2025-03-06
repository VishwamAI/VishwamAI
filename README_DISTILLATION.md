# VishwamAI Knowledge Distillation from Qwen-32B

This document describes the optimized knowledge distillation process from Qwen-32B to VishwamAI-7B using JAX/Flax.

## Features

- Efficient partial model loading for Qwen-32B
- TPU-optimized training with bfloat16 precision
- Intermediate layer feature matching
- Attention pattern matching with GQA support
- Dynamic temperature scaling
- EPLB (Enhanced Progressive Layer Balancing)
- Aim experiment tracking
- Automatic layer mapping and alignment

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure distillation settings in `configs/distillation_config.yaml`:
```yaml
distillation:
  teacher_model:
    path: "Qwen/Qwen-32B"
    # Teacher model configuration...
  student_model:
    config:
      # Student model configuration...
training:
  # Training parameters...
distillation_params:
  # Distillation hyperparameters...
```

## Usage

1. Start training using the provided notebook:
```bash
jupyter notebook train_vishwamai_distillation.ipynb
```

2. Monitor training with Aim:
```bash
aim up
```

## Implementation Details

### Layer Mapping

We use an optimized layer mapping strategy that aligns Qwen's architecture with the student model:
- Early layers: Dense mapping for low-level features
- Middle layers: Progressive mapping
- Final layers: Attention pattern matching

### Loss Components

1. Knowledge Distillation Loss:
   - Soft target matching with dynamic temperature
   - Feature matching across mapped layers
   - Attention pattern alignment

2. Cross-Entropy Loss:
   - Hard target supervision
   - Label smoothing for regularization

3. Feature Matching Loss:
   - Intermediate layer alignment
   - Hidden state projection when dimensions differ

4. Attention Matching Loss:
   - GQA-aware attention pattern matching
   - Head-wise correspondence

## Memory Optimization

- Partial model loading (5 shards at a time)
- Gradient checkpointing
- TPU sharding strategies
- bfloat16 precision by default

## TPU Training

The implementation is optimized for TPU training with:
- Automatic device sharding
- XLA compilation
- Efficient tensor operations
- Multi-host training support

## Monitoring & Checkpointing

Training progress is tracked using Aim:
- Loss components
- Feature matching metrics
- Attention alignment scores
- Resource utilization
- Model checkpoints
- Layer mapping artifacts

## Results

Expected distillation metrics:
- KL divergence reduction over time
- Feature alignment improvement
- Attention pattern convergence
- Perplexity comparison with teacher

## Recommendations

1. Start with a small number of shards (5) to verify setup
2. Monitor feature matching losses carefully
3. Adjust temperature dynamically based on training progress
4. Use gradient accumulation for larger effective batch sizes

## Troubleshooting

Common issues and solutions:
- OOM errors: Reduce number of shards or batch size
- Gradient instability: Adjust learning rate or gradient clipping
- Feature mismatch: Verify layer mapping configuration
- TPU compilation errors: Check tensor shapes and types

## License

See LICENSE file for details.
