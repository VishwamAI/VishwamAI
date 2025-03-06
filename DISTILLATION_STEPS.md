# VishwamAI-QwQ Distillation Guide

Step-by-step guide for distilling knowledge from [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) to VishwamAI-7B.

## Prerequisites

1. **Hardware Requirements**
   - TPU v3-8 or higher recommended
   - At least 128GB RAM
   - 200GB disk space

2. **Environment Setup**
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. **TPU Configuration**
```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_PLATFORM_NAME="tpu"
```

## Verify Setup

Before starting the distillation, verify your setup:

```bash
# Run setup verification
python test_distillation_setup.py

# Expected output should show:
# - JAX devices detected
# - Data loader working with TPU sharding
# - Trainer initialization successful
# - Gradient accumulation working
```

## QwQ-32B Setup

1. **Model Files**
The 14 safetensor shards should be present:
```
qwen_model/
├── model-00001-of-00014.safetensors
├── model-00002-of-00014.safetensors
...
└── model-00014-of-00014.safetensors
```

2. **Verify Shards**
```python
python -c "
from vishwamai.qwen_data import QwenDataLoader
loader = QwenDataLoader('path/to/qwen_model', batch_size=1, gradient_accumulation_steps=16)
print('Shards verified successfully')
"
```

## Running Distillation

1. **Start Training**
```bash
jupyter notebook train_vishwamai_distillation.ipynb
```

2. **Monitor Progress**
```bash
# Start Aim dashboard
aim up --host 0.0.0.0 --port 43800
```

## Memory Management

The implementation is optimized for QwQ's 14 shards:
- Per-shard loading (no full model in memory)
- Gradient accumulation (effective batch size = 16)
- bfloat16 precision
- TPU memory optimization

Memory usage:
```
Per Shard (~2.3GB each):
- Load: ~2.3GB
- Process: ~8GB peak
- Training: ~32GB steady state
```

## Checkpointing

Automatic saves:
- Every 1000 steps
- At end of each epoch
- On interruption

## TPU-Specific Settings

1. **Batch Size Setup**
```python
# In the notebook:
BATCH_SIZE = 1  # Per device
GRAD_ACCUM_STEPS = 16  # For effective batch size of 16
```

2. **Memory Configuration**
```bash
# Recommended TPU memory settings
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
```

## Troubleshooting

1. **TPU OOM**
Try:
```python
# Reduce per-device batch size
config.training.batch_size = 1
config.training.gradient_accumulation_steps = 32
```

2. **Slow Training**
Check:
```python
# Verify TPU usage
print(jax.devices())
```

3. **Corrupted Shards**
Run:
```bash
python test_distillation_setup.py --verify-shards
```

## Recovery

If training is interrupted:

1. Note last checkpoint:
```python
RESUME_FROM = "checkpoints/step_X"
```

2. Restart training:
```python
state = trainer.load_checkpoint(RESUME_FROM)
```

## Validation

After training completes:

1. Check model size (~7GB)
2. Run inference test:
```python
from vishwamai.generate import load_model, generate_text

model = load_model("final_model")
print(generate_text(model, "Once upon a time"))
```

## Common Issues

1. **Device Assignment**
```
Problem: "ValueError: Not enough devices for batch"
Solution: Adjust batch_size and gradient_accumulation_steps
```

2. **Memory Leaks**
```python
# Add to training loop:
if step % 10 == 0:
    jax.clear_caches()
    gc.collect()
```

3. **Shard Loading**
```
Problem: "FileNotFoundError: Shard X not found"
Solution: Verify all 14 shards are present and readable
```

## Contact

For issues:
- Open GitHub issue with:
  - Error message
  - TPU configuration
  - Memory usage stats
  - Training logs
  - test_distillation_setup.py output
