# VishwamAI QwQ-32B Distillation Guide

This guide explains how to efficiently distill knowledge from QwQ-32B to VishwamAI using memory-optimized implementation.

## System Requirements

- TPU v3-8 or higher
- 128GB RAM minimum
- 200GB disk space
- Python 3.9+

## Setup

1. **Install Dependencies**
```bash
pip install -r requirements.txt
```

2. **Environment Setup**
```bash
# Set TPU environment variables
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_PLATFORM_NAME="tpu"
export PYTHONPATH=$PYTHONPATH:/path/to/vishwamai
```

3. **QwQ Model Setup**
```bash
# Set model path
export QWQ_PATH=/path/to/qwq/model

# Verify setup
python test_qwq_loading.py
```

## Memory-Optimized Implementation

The implementation uses several strategies to handle QwQ-32B efficiently:

1. **Chunked Loading**
- Processes model weights in chunks (default 32)
- Auto-adjusts chunk size based on available memory
- Caches frequently used chunks

2. **Gradient Accumulation**
- Per-device batch size of 1
- Accumulates gradients over 16 steps
- Effective batch size = 16 * num_devices

3. **Memory Management**
- Automatic cache clearing
- Memory monitoring
- TPU-optimized tensor operations

## Running Distillation

1. **Quick Test**
```bash
# Test memory efficiency
python test_memory_loading.py

# Verify full setup
python test_distillation_setup.py
```

2. **Start Training**
```bash
# Run full distillation
./run_distillation.sh
```

3. **Monitor Progress**
```bash
# Start monitoring dashboard
aim up --host 0.0.0.0 --port 43800
```

## Implementation Details

### Memory Usage Pattern

For 14 QwQ shards:
```
Per Shard (~2.3GB):
- Load: 2.3GB
- Process: ~8GB peak
- Training: ~32GB steady state
```

### Checkpointing

Automatic checkpoints:
```
- Every 1000 steps
- End of each epoch
- On interruption
- Keeps last 3 by default
```

### TPU Optimization

1. **Data Loading**
```python
loader = QwenDataLoader(
    safetensor_dir=QWQ_PATH,
    batch_size=1,
    gradient_accumulation_steps=16,
    chunk_size=32  # Auto-adjusts if needed
)
```

2. **Training Settings**
```yaml
memory_optimization:
  chunk_size: 32
  clear_cache_steps: 10
  prefetch_blocks: 2
  max_memory_gb: 80

training:
  batch_size: 1
  gradient_accumulation_steps: 16
```

3. **TPU Configuration**
```yaml
tpu_config:
  use_bfloat16: true
  use_dynamic_scale: true
  use_scanned_attention: true
  memory_fraction: 0.95
```

## Troubleshooting

### Memory Issues

1. **Out of Memory**
```python
# Reduce chunk size
loader = QwenDataLoader(..., chunk_size=16)

# Increase gradient accumulation
config.training.gradient_accumulation_steps = 32
```

2. **Slow Training**
```python
# Check memory usage
python -c "from vishwamai.tensor_utils import get_memory_usage; print(f'Memory: {get_memory_usage():.2f}GB')"

# Monitor TPU utilization
python -c "import jax; print(jax.devices())"
```

3. **Loading Errors**
```bash
# Verify shards
python test_qwq_loading.py --verify-shards

# Check permissions
ls -l $QWQ_PATH/*.safetensors
```

### Recovery

If training is interrupted:

1. **Find Last Checkpoint**
```python
from vishwamai.qwen_distiller import QwQDistillationTrainer

trainer = QwQDistillationTrainer(...)
state = trainer.load_checkpoint("checkpoints/step_X")
```

2. **Resume Training**
```bash
# Set resume point
export RESUME_STEP=X
./run_distillation.sh --resume
```

## Monitoring

1. **Memory Usage**
```python
from vishwamai.tensor_utils import get_memory_usage

print(f"Current memory: {get_memory_usage():.2f}GB")
```

2. **Training Metrics**
- Visit `http://localhost:43800`
- Monitor:
  - Memory usage per shard
  - Loss components
  - TPU utilization
  - Gradient norms

## Contact

For issues:
- Open GitHub issue with:
  - Error message
  - Memory usage stats
  - TPU configuration
  - Test results from `test_qwq_loading.py`
