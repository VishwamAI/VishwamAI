# HuggingFace Integration Setup Guide

This guide explains how to set up and use the HuggingFace integration for training VishwamAI models on the GSM8K dataset.

## Prerequisites

1. HuggingFace Account
- Sign up at [HuggingFace](https://huggingface.co/join)
- Generate an access token at https://huggingface.co/settings/tokens

2. TPU Access
- Request TPU access through [TPU Research Cloud](https://sites.research.google/trc/about/)
- Set up your TPU VM environment

## Repository Setup

1. Clone the VishwamAI repository:
```bash
git clone https://github.com/VishwamAI/VishwamAI.git
cd VishwamAI
```

2. Login to HuggingFace:
```bash
huggingface-cli login
# Enter your access token when prompted
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## GSM8K Training Notebook

### Option 1: Google Colab (Recommended)

1. Open the notebook in Colab:
   - Visit [GSM8K_Training.ipynb](https://colab.research.google.com/github/VishwamAI/VishwamAI/blob/main/GSM8K_Training.ipynb)
   - Or click the "Open in Colab" button in the notebook

2. Set up TPU Runtime:
   - Runtime → Change runtime type
   - Hardware accelerator → TPU
   - TPU version → v3-8 or higher

3. Configure HuggingFace credentials:
```python
from huggingface_hub import notebook_login
notebook_login()
```

### Option 2: Local TPU VM

1. Copy your notebook to TPU VM:
```bash
gcloud compute scp GSM8K_Training.ipynb [VM_NAME]:~/
```

2. Start Jupyter server:
```bash
jupyter notebook --ip=0.0.0.0 --port=8888
```

3. Access the notebook via provided URL

## Training Configuration

### Model Settings

```python
config = get_pretrained_config(
    model_size="base",  # Options: "small", "base", "large", "xl"
    model_type="moe_mla_transformer"
)
```

### TPU Settings

```yaml
# vishwamai/configs/tpu_config.yaml
tpu:
  topology: "2x2"  # Match your TPU setup
  num_tpu_cores: 8
  use_bfloat16: true
```

### Training Parameters

```python
training_args = {
    "num_train_epochs": 3,
    "per_device_train_batch_size": 8,
    "gradient_accumulation_steps": 4,
    "learning_rate": 5e-4,
    "weight_decay": 0.01,
    "warmup_steps": 500
}
```

## Monitoring Training

1. Set up Weights & Biases:
```python
import wandb
wandb.init(project="vishwamai-gsm8k")
```

2. Access TPU metrics:
```python
from vishwamai.utils.profiling import get_tpu_metrics
metrics = get_tpu_metrics()
```

3. View training progress in W&B dashboard

## Model Upload

The trained model will be automatically uploaded to the HuggingFace Hub:
- Repository: [VishwamAI/VishwamAI](https://huggingface.co/VishwamAI/VishwamAI)
- Model files will be saved in the `gsm8k_trained_model` directory

## Custom Dataset Training

To train on your own math problems:

1. Prepare your data in GSM8K format:
```python
data = {
    "question": "Your math question here",
    "answer": "Step-by-step solution"
}
```

2. Create a custom dataset:
```python
from vishwamai.data.dataset import CustomMathDataset
dataset = CustomMathDataset(data_path, format="gsm8k")
```

3. Update training configuration accordingly

## Troubleshooting

### Common Issues

1. TPU Initialization Failed
```bash
# Check TPU status
gcloud compute tpus tpu-vm describe [TPU_NAME]

# Restart TPU if needed
gcloud compute tpus tpu-vm stop [TPU_NAME]
gcloud compute tpus tpu-vm start [TPU_NAME]
```

2. Out of Memory
- Reduce batch size
- Enable gradient checkpointing
- Use BFloat16 precision

3. HuggingFace Upload Failed
- Verify access token
- Check repository permissions
- Ensure stable internet connection

### Support

For additional help:
1. Open an issue on GitHub
2. Check TPU documentation
3. Visit HuggingFace forums

## Best Practices

1. Regular Checkpointing
```python
# Save every N steps
if step % save_interval == 0:
    model.save_checkpoint("checkpoint_{step}")
```

2. Gradient Accumulation
- Use larger effective batch sizes
- Maintain memory efficiency
- Improve training stability

3. Performance Optimization
- Enable TPU profiling
- Monitor memory usage
- Use optimized data loading

## References

- [GSM8K Dataset](https://huggingface.co/datasets/openai/gsm8k)
- [TPU Documentation](https://cloud.google.com/tpu/docs)
- [HuggingFace Model Hub](https://huggingface.co/models)
- [TPU Research Cloud](https://sites.research.google/trc/)
