# VishwamAI Quick Start Guide

This guide helps you get started with training VishwamAI on the GSM8K dataset using TPUs.

## Prerequisites

1. TPU VM Access
   - Request access through [TPU Research Cloud](https://sites.research.google/trc/about/)
   - Set up a TPU VM instance (v3-8 or higher recommended)

2. HuggingFace Account
   - Create account at [HuggingFace](https://huggingface.co/join)
   - Generate access token: https://huggingface.co/settings/tokens

3. Weights & Biases Account
   - Sign up at [WandB](https://wandb.ai/login)
   - Get your API key

## Quick Setup

1. Clone the repository and run setup script:
```bash
git clone https://github.com/VishwamAI/VishwamAI.git
cd VishwamAI
chmod +x setup_tpu_training.sh
./setup_tpu_training.sh
```

2. Configure credentials:
```bash
# Login to WandB
wandb login

# Login to HuggingFace
huggingface-cli login
```

3. Start training:
```bash
# Activate environment
source vishwamai_env/bin/activate

# Launch Jupyter
jupyter notebook --ip=0.0.0.0 --port=8888
```

4. Open `GSM8K_Training.ipynb` in Jupyter and follow the notebook.

## Common Issues & Solutions

### TPU Issues

1. TPU Not Detected
```bash
# Check TPU status
sudo lsof -w /dev/accel0

# Restart TPU if needed
sudo service tpu restart
```

2. Out of Memory
```python
# Reduce batch size in training_args:
training_args = {
    "per_device_train_batch_size": 4,  # Decrease this
    "gradient_accumulation_steps": 8    # Increase this
}
```

3. CUDA Error
```bash
# Ensure using TPU device
device = xm.xla_device()
model = model.to(device)
```

### Dataset Issues

1. Download Failed
```bash
# Manual download
from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main", use_auth_token=True)
dataset.save_to_disk("gsm8k_data")
```

2. Memory Error
```python
# Load in streaming mode
dataset = load_dataset("openai/gsm8k", "main", streaming=True)
```

## Training Configuration

### Basic Configuration
```yaml
# In vishwamai/configs/training_config.yaml
training:
  num_epochs: 3
  batch_size: 8
  learning_rate: 5e-4
  warmup_steps: 500
```

### Expert Configuration
```yaml
# In vishwamai/configs/moe_config.yaml
moe:
  num_experts: 8
  expert_capacity: 1.25
  min_expert_capacity: 4
```

### TPU Configuration
```yaml
# In vishwamai/configs/tpu_config.yaml
tpu:
  topology: "2x2"
  num_cores: 8
  precision: "bfloat16"
```

## Monitoring Training

1. Access WandB dashboard:
```python
wandb.init(project="vishwamai-gsm8k")
```

2. View TPU metrics:
```python
from vishwamai.utils.profiling import get_tpu_metrics
metrics = get_tpu_metrics()
print(metrics.summary())
```

## Model Export

Save and upload trained model:
```python
# Save locally
model.save_pretrained("gsm8k_trained_model")

# Upload to HuggingFace
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="gsm8k_trained_model",
    repo_id="VishwamAI/VishwamAI"
)
```

## Performance Tips

1. Use Gradient Checkpointing
```python
model.gradient_checkpointing_enable()
```

2. Enable TPU Optimizations
```python
model = tpu_utils.optimize_tpu_execution(model, tpu_config)
```

3. Use BFloat16
```python
model = model.to(torch.bfloat16)
```

4. Optimize Dataset Loading
```python
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=8,
    num_workers=4,
    prefetch_factor=2
)
```

## Next Steps

1. Check the [Documentation](docs/technical.md) for detailed information
2. Explore model configurations in `configs/`
3. View examples in `examples/` directory
4. Read [Contributing Guidelines](CONTRIBUTING.md) to contribute

## Support

- Open an issue on GitHub
- Check TPU documentation
- Visit HuggingFace forums
- Join our Discord community

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.
