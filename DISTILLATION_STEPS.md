# VishwamAI-QwQ Distillation Guide

Step-by-step guide for distilling knowledge from [QwQ-32B](https://huggingface.co/Qwen/QwQ-32B) to VishwamAI-7B.

## Prerequisites

1. **Hardware Requirements**
   - TPU v3-8 or higher recommended
   - At least 128GB RAM
   - 200GB disk space

2. **Python Environment**
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. **TPU Setup**
```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export JAX_PLATFORM_NAME="tpu"
```

## QwQ-32B Setup

1. **Download Model**
```python
from huggingface_hub import snapshot_download

model_path = snapshot_download(
    "Qwen/QwQ-32B",
    allow_patterns=["*.safetensors", "config.json", "tokenizer.model"]
)
```

Expected files:
```
model_path/
├── model-00001-of-00014.safetensors
├── model-00002-of-00014.safetensors
...
└── model-00014-of-00014.safetensors
```

2. **Verify Files**
```python
import os
shard_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
assert len(shard_files) == 14, f"Expected 14 safetensor files, found {len(shard_files)}"
```

## Distillation Process

1. **Configuration**
- Update `configs/distillation_config.yaml`:
  - Verify QwQ-32B architecture settings
  - Adjust student model size (7B)
  - Set training hyperparameters

2. **Run Training**
```bash
jupyter notebook train_vishwamai_distillation.ipynb
```

3. **Monitor Progress**
```bash
# Start Aim dashboard
aim up --host 0.0.0.0 --port 43800
```

Key metrics to monitor:
- Distillation loss
- Feature alignment quality
- Memory usage per shard
- Training speed

## Memory Management

The process is optimized for QwQ-32B's 14 safetensor shards:
- Sequential shard loading
- Automatic memory cleanup
- Gradient checkpointing
- TPU memory optimization

Memory usage pattern:
```
Per Shard:
- Load: ~2.3GB
- Process: ~8GB peak
- Training: ~32GB steady state
```

## Checkpointing

Automatic checkpoints:
- Every epoch
- Every 1000 steps
- On interruption

Checkpoint structure:
```
checkpoints/
├── epoch_0/
├── epoch_1/
...
└── final_model/
```

## Troubleshooting

1. **TPU OOM**
Solution:
```python
# Reduce batch size in config
config.training.batch_size = 2
config.training.gradient_accumulation_steps = 32
```

2. **Slow Training**
Check:
```python
# Verify TPU is being used
print(jax.devices())
# Should show TPU devices
```

3. **Poor Convergence**
Check:
```python
# Monitor loss components
print(metrics['kd_loss'], metrics['feature_loss'])
# Should decrease steadily
```

## Recovery

If training is interrupted:
```python
# In the notebook
RESUME_FROM = "checkpoints/step_X"
state = trainer.load_checkpoint(RESUME_FROM)
```

## Validation

After training:
1. Model size should be ~7GB
2. Run inference test:
```python
from vishwamai.generate import load_model, generate_text
model = load_model("final_model")
print(generate_text(model, "Once upon a"))
```

## Common Issues

1. **Safetensor Loading**
```python
# Verify shard integrity
import safetensors.flax as stf
for shard in shard_files:
    try:
        _ = stf.load_file(os.path.join(model_path, shard))
    except:
        print(f"Corrupted shard: {shard}")
```

2. **TPU Compilation**
```python
# Clear TPU cache if needed
jax.clear_caches()
```

3. **Memory Leaks**
```python
# Add to training loop
if step % 10 == 0:
    jax.clear_caches()
    gc.collect()
```

## Contact

For issues:
- Open GitHub issue
- Include:
  - Error message
  - TPU configuration
  - Memory usage stats
  - Training logs
