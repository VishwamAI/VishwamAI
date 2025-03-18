# TPU Optimization Guide for VishwamAI

This guide provides strategies for maximizing training efficiency with VishwamAI on TPU setups, particularly when working with 8 TPU devices and memory constraints.

## Quick Start

Run the optimized training script with:

```bash
python optimize_train_tpu.py --config configs/tpu_optimized_config.yaml
```

For additional profiling information to further optimize your setup:

```bash
python optimize_train_tpu.py --config configs/tpu_optimized_config.yaml --profile
```

## Memory Optimization Techniques

The `optimize_train_tpu.py` script implements several memory-efficient techniques:

### 1. Gradient Accumulation

The script uses gradient accumulation to simulate larger batch sizes while keeping memory usage low. With 8 TPU devices and a per-device batch size of 8, gradient accumulation with 4 steps gives you an effective batch size of 256.

```yaml
training:
  initial_batch_size: 8
  gradient_accumulation_steps: 4  # Effective batch size = 8 * 4 * 8_devices = 256
```

### 2. Mixed Precision Training

All computation is performed in `bfloat16` precision, which is optimal for TPU and reduces memory usage by half compared to `float32`.

```yaml
model:
  dtype: "bfloat16"  # TPU-optimized format
```

### 3. Dynamic Batch Sizing

The training script automatically determines the optimal batch size for your TPU setup and can gradually increase it as training stabilizes:

```yaml
training:
  dynamic_batch_size: true
  batch_size_increase_step: 2000
  batch_size_increase_factor: 1.2
```

### 4. Memory-Efficient Model Architecture

- **Grouped Query Attention (GQA)**: Reduces KV cache memory usage
- **Gradient Checkpointing**: Trades computation for memory by not saving all activations
- **Chunked Processing**: Processes sequences in smaller chunks to reduce peak memory

### 5. Optimized Tree of Thoughts

The ToT implementation has been optimized for TPU with:
- Reduced memory footprint for thought generation
- Efficient parallel thought evaluation
- TPU-optimized search strategies
- Adaptive priority to most promising paths

## Configuration Best Practices

For an 8-TPU setup with memory constraints:

1. **Scaling Model Size**:
   - `hidden_size`: 768-1024 (depending on sequence length)
   - `num_layers`: 18-24
   - `num_attention_heads`: 12-16

2. **Sequence Length Tradeoffs**:
   - 2048 tokens: Use smaller hidden size (768)
   - 1024 tokens: Can increase to hidden size 1024
   - 512 tokens: Can add more layers (up to 32)

3. **GQA Configuration**:
   - For memory efficiency: `num_key_value_heads: 4`
   - For accuracy with more memory: `num_key_value_heads: 8`

4. **Gradient Accumulation Recommendations**:
   - 2-4 steps for most workloads
   - 8 steps for very large models
   - Higher values help stability but slow training

## Monitoring & Debugging

The training script includes built-in memory tracking:

```
2023-05-15 14:20:32 - __main__ - INFO - Host memory usage: 15.42 GB
2023-05-15 14:20:32 - __main__ - INFO - TPU 0 peak memory: 6.78 GB
```

To optimize further, look for:
- High memory spikes during forward/backward passes
- Unbalanced memory usage across TPU devices
- Excessive host memory usage

## Tree of Thoughts Memory Optimization

When using Tree of Thoughts reasoning with memory constraints:

```yaml
tot:
  max_thoughts: 5       # Reduce if memory limited
  thoughts_per_step: 3  # Controls memory used per step
  memory_efficient: true
```

## Error Correction Module

The error correction module can be tuned for memory vs accuracy:

```yaml
error_correction:
  num_layers: 2         # Increase up to 4 if memory allows
  use_mixture_density: true
  memory_efficient: true
```

## Example Resource Profiles

| Configuration | TPU Memory per Device | Training Throughput | Recommended Batch Size |
|---------------|----------------------|---------------------|----------------------|
| Small (6L, 512h) | ~2.5 GB | ~1200 tokens/sec | 32 per device |  
| Medium (12L, 768h) | ~4.5 GB | ~700 tokens/sec | 16 per device |
| Large (24L, 1024h) | ~7.5 GB | ~300 tokens/sec | 8 per device |

## Troubleshooting

**OOM Errors**: If encountering out-of-memory errors:

1. Reduce `initial_batch_size` (try halving it)
2. Increase `gradient_accumulation_steps` to maintain effective batch size
3. Reduce model size (`hidden_size`, `intermediate_size`, or `num_layers`)
4. Disable either `use_tot` or `use_error_correction` temporarily
5. Decrease sequence length if possible

**Slow Training**: If training is slower than expected:

1. Check if TPUs are being fully utilized (should see >85% usage)
2. Ensure data loading isn't a bottleneck (increase `num_workers`)
3. Try decreasing `gradient_accumulation_steps`
4. Verify all TPU devices are being used with `jax.device_count()`

## Detailed Documentation on TPU Optimization

### TPUMeshContext

The `TPUMeshContext` class in `vishwamai/device_mesh.py` manages TPU device mesh and data/model parallelism. It includes methods for dynamic mesh reshaping and optimal sharding strategies.

#### Dynamic Mesh Reshaping

The `dynamic_mesh_reshaping` method allows for reshaping the device mesh dynamically based on the current workload and device availability. This helps in optimizing the computation and memory usage across the TPU devices.

#### Optimal Sharding Strategies

The `optimal_sharding_strategy` method determines the best sharding strategy for a given tensor based on its shape and type. This ensures efficient data distribution and computation across the TPU devices.

### DistillationTrainer

The `DistillationTrainer` class in `vishwamai/distill.py` implements TPU-optimized knowledge distillation. It includes advanced loss functions and training strategies for improved distillation performance.

#### Advanced Loss Functions

The `advanced_loss_functions` method provides enhanced loss computation for better distillation results. It includes additional terms for embedding path loss and hard target cross-entropy loss.

#### Improved Training Strategies

The `improved_training_strategies` method incorporates advanced techniques for better training performance, such as chunked processing and gradient accumulation.

### TPUProfiler

The `TPUProfiler` class in `vishwamai/profiler.py` provides detailed metrics and actionable recommendations for optimizing TPU performance.

#### Detailed Metrics

The `add_detailed_metrics` method adds comprehensive metrics for TPU performance, including compute efficiency, memory bandwidth, latency, and energy consumption.

#### Actionable Recommendations

The `get_actionable_recommendations` method generates specific recommendations for improving TPU performance based on the collected metrics.

### VishwamAITrainer

The `VishwamAITrainer` class in `vishwamai/training.py` manages the training process and integrates seamlessly with TPU-optimized components, ensuring efficient training and evaluation.

#### Integration with TPU-Optimized Components

The `VishwamAITrainer` class leverages the TPU-optimized components such as `TPUMeshContext`, `DistillationTrainer`, and `TPUProfiler` to provide a streamlined training experience.

#### Efficient Training and Evaluation

The `VishwamAITrainer` class ensures efficient training and evaluation by utilizing advanced techniques such as dynamic batch sizing, gradient accumulation, and mixed precision training.

## Examples and Best Practices

### Using TPUMeshContext

```python
from vishwamai.device_mesh import TPUMeshContext

config = {
    "training": {
        "data_parallel": True,
        "model_parallel": True,
        "pipeline_parallel": False
    },
    "tpu": {
        "tpu_cores": 8,
        "tpu_topology": "2x2x2"
    }
}

mesh_context = TPUMeshContext(config)
mesh_context.dynamic_mesh_reshaping((4, 2, 2))
sharding_strategy = mesh_context.optimal_sharding_strategy((1024, 1024), "weights")
```

### Using DistillationTrainer

```python
from vishwamai.distill import DistillationTrainer

teacher_model = ...  # Load pre-trained teacher model
student_config = {
    "vocab_size": 30522,
    "num_layers": 12,
    "num_heads": 12,
    "head_dim": 64,
    "hidden_dim": 768,
    "mlp_dim": 3072,
    "max_seq_len": 512
}

distillation_trainer = DistillationTrainer(teacher_model, student_config)
loss = distillation_trainer.advanced_loss_functions(student_logits, teacher_logits, student_embeddings, teacher_embeddings, labels, mask)
```

### Using TPUProfiler

```python
from vishwamai.profiler import TPUProfiler

config = {
    "training": {
        "learning_rate": 1e-4,
        "max_steps": 100000
    }
}

profiler = TPUProfiler(config)
profiler.start_step()
# Perform training step
profiler.end_step()
metrics_summary = profiler.get_metrics_summary()
recommendations = profiler.get_actionable_recommendations()
```

### Using VishwamAITrainer

```python
from vishwamai.training import VishwamAITrainer
from vishwamai.pipeline import VishwamAIPipeline

config = {
    "training": {
        "learning_rate": 1e-4,
        "max_steps": 100000,
        "eval_every": 1000,
        "save_every": 5000,
        "log_every": 100
    }
}

pipeline = VishwamAIPipeline(config)
train_loader = ...  # Initialize training data loader
eval_loader = ...  # Initialize evaluation data loader

trainer = VishwamAITrainer(pipeline, config, train_loader, eval_loader)
trainer.train()
```

## Best Practices

1. **Dynamic Mesh Reshaping**: Use the `dynamic_mesh_reshaping` method in `TPUMeshContext` to adapt the device mesh based on the current workload and device availability.
2. **Optimal Sharding**: Utilize the `optimal_sharding_strategy` method in `TPUMeshContext` to determine the best sharding strategy for tensors.
3. **Advanced Loss Functions**: Implement the `advanced_loss_functions` method in `DistillationTrainer` for improved distillation performance.
4. **Detailed Metrics**: Leverage the `add_detailed_metrics` method in `TPUProfiler` to collect comprehensive performance metrics.
5. **Actionable Recommendations**: Follow the `get_actionable_recommendations` method in `TPUProfiler` to optimize TPU performance.
6. **Efficient Training**: Use the `VishwamAITrainer` class to manage the training process and integrate TPU-optimized components for efficient training and evaluation.

By following these best practices and utilizing the TPU-optimized components provided in VishwamAI, you can achieve efficient and effective training on TPU hardware.
