# Advanced Training Features

This document outlines the advanced training capabilities added to VishwamAI and how to use them effectively.

## Configuration Management

VishwamAI now uses Hydra for configuration management, providing a more flexible and maintainable way to handle hyperparameters and settings.

```bash
# Train with default configuration
python -m vishwamai.training_v2

# Override specific values
python -m vishwamai.training_v2 training.optimizer.lr=3e-4 model.dropout_rate=0.2

# Use a different model configuration
python -m vishwamai.training_v2 model=7B
```

Configuration files are organized in `vishwamai/configs/`:
- `default_config.yaml`: Base configuration
- `model/`: Model-specific configurations
- `training/`: Training-specific configurations

## Hyperparameter Tuning

Automated hyperparameter tuning is supported through both Optuna and Ray Tune.

```bash
# Run hyperparameter tuning with Optuna
python -m vishwamai.training_v2 tuning.enabled=true tuning.framework=optuna

# Use Ray Tune instead
python -m vishwamai.training_v2 tuning.enabled=true tuning.framework=ray
```

Key tuning features:
- Automated early stopping
- Parallel trial execution
- Pruning of underperforming trials
- Support for various search algorithms (TPE, BOHB, etc.)

## Distributed Training

Enhanced distributed training support with both data and model parallelism:

```bash
# Data parallel training across 8 GPUs
python -m vishwamai.training_v2 distributed.strategy=data_parallel

# Model parallel training
python -m vishwamai.training_v2 distributed.strategy=model_parallel
```

Features:
- Automatic batch sharding
- Gradient all-reduce optimization
- Dynamic batch sizing
- Gradient accumulation

## Advanced Mixed Precision

Improved AMP support with dynamic loss scaling:

```yaml
# In training/default.yaml
amp:
  enabled: true
  dtype: "bfloat16"
  opt_level: "O2"
  loss_scale:
    enabled: true
    init_scale: 65536
    growth_interval: 2000
```

## Monitoring and Profiling

Enhanced monitoring capabilities with TensorBoard and Weights & Biases integration:

```yaml
# In default_config.yaml
monitoring:
  enabled: true
  wandb:
    enabled: true
    project: "vishwamai"
  tensorboard:
    enabled: true
```

Features:
- Real-time training metrics
- GPU utilization tracking
- Memory profiling
- Model parameter statistics
- Training speed monitoring

## Checkpointing and Recovery

Robust checkpointing system:

```yaml
# In training/default.yaml
checkpointing:
  save_optimizer_state: true
  keep_last_n: 5
  save_best: true
  metric: "validation_loss"
  mode: "min"
  save_zero_redundancy: true
```

Features:
- Best model tracking
- Optimizer state saving
- Zero-redundancy optimizer state saving
- Automatic checkpoint cleanup

## Dynamic Batch Sizing

Automatic batch size adjustment based on available memory and training stability:

```yaml
dynamic_batch_size:
  enabled: true
  initial_batch_size: 32
  target_batch_size: 128
  min_batch_size: 8
  growth_factor: 2
  shrink_factor: 0.5
  stable_steps: 100
```

The system will automatically:
- Start with a conservative batch size
- Gradually increase if training is stable
- Reduce if out-of-memory or gradient issues occur

## Usage Examples

### Full Training Run with All Features

```bash
python -m vishwamai.training_v2 \
  model=10B \
  training.optimizer.lr=2e-4 \
  training.max_steps=50000 \
  monitoring.wandb.enabled=true \
  distributed.strategy=data_parallel \
  amp.enabled=true \
  dynamic_batch_size.enabled=true
```

### Hyperparameter Search

```bash
python -m vishwamai.training_v2 \
  tuning.enabled=true \
  tuning.framework=optuna \
  tuning.num_trials=100 \
  tuning.metric=validation_loss \
  monitoring.wandb.enabled=true
```

### Multi-Node Training

On each node:
```bash
python -m vishwamai.training_v2 \
  distributed.strategy=model_parallel \
  distributed.world_size=16 \
  distributed.node_rank=${NODE_RANK} \
  model.parallel_factor=2
```

## Best Practices

1. **Mixed Precision Training**
   - Always enable AMP when training on modern GPUs
   - Use bfloat16 for better stability than float16

2. **Distributed Training**
   - Start with data parallel for smaller models
   - Use model parallel when model size exceeds single GPU memory
   - Enable gradient accumulation for better optimization

3. **Monitoring**
   - Enable both TensorBoard and W&B for comprehensive tracking
   - Monitor gradient norms and learning rates
   - Track GPU memory usage and throughput

4. **Hyperparameter Tuning**
   - Start with a small number of trials to validate setup
   - Use pruning to terminate unpromising trials early
   - Consider population-based training for long runs

5. **Checkpointing**
   - Enable best model tracking
   - Keep multiple recent checkpoints
   - Use zero-redundancy saving for large models

## Model Distillation

VishwamAI now supports advanced knowledge distillation with multiple distillation strategies:

```bash
# Basic distillation training
python -m vishwamai.training_v2 \
  distillation.enabled=true \
  distillation.teacher_model.path=/path/to/teacher/checkpoint

# With feature distillation
python -m vishwamai.training_v2 \
  distillation.enabled=true \
  distillation.teacher_model.path=/path/to/teacher/checkpoint \
  distillation.feature_distillation.enabled=true \
  distillation.feature_distillation.layers=[0,6,11]

# With attention distillation
python -m vishwamai.training_v2 \
  distillation.enabled=true \
  distillation.teacher_model.path=/path/to/teacher/checkpoint \
  distillation.attention_distillation.enabled=true
```

### Distillation Features

1. **Multiple Distillation Types**
   - Standard logit-based knowledge distillation
   - Feature map distillation
   - Attention map distillation
   - Hidden state distillation

2. **Model Compression**
   - Gradual pruning during training
   - Post-training quantization
   - Configurable compression schedules

3. **Advanced Loss Functions**
   - KL divergence for logits
   - MSE for feature matching
   - Cosine similarity for embeddings
   - Customizable loss weights

### Example Configurations

```yaml
# Enable pruning with distillation
distillation:
  enabled: true
  pruning:
    enabled: true
    target_sparsity: 0.5
    pruning_schedule: cubic
    begin_step: 1000
    end_step: 10000

# Enable quantization
distillation:
  enabled: true
  quantization:
    enabled: true
    precision: "int8"
    calibration_steps: 100

# Full distillation setup
distillation:
  enabled: true
  teacher_model:
    path: "/path/to/teacher"
    temperature: 2.0
    alpha: 0.5
  feature_distillation:
    enabled: true
    layers: [0, 6, 11]
    loss_weight: 0.1
  attention_distillation:
    enabled: true
    loss_weight: 0.1
  hidden_distillation:
    enabled: true
    loss_weight: 0.1
  pruning:
    enabled: true
    target_sparsity: 0.5
  quantization:
    enabled: true
    precision: "int8"
```

### Best Practices for Distillation

1. **Teacher Model Selection**
   - Choose a well-trained teacher model
   - Consider using ensemble of teachers
   - Validate teacher performance first

2. **Training Strategy**
   - Start with higher learning rates
   - Use gradual pruning if enabled
   - Monitor student-teacher feature alignment

3. **Compression Pipeline**
   - Begin with knowledge distillation
   - Apply pruning during training
   - Finish with quantization
   - Validate at each step

4. **Loss Balancing**
   - Adjust alpha for KD vs task loss
   - Tune feature matching weights
   - Monitor individual loss components
