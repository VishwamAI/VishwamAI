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