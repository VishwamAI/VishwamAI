# Training VishwamAI Model

This document provides detailed instructions for training the VishwamAI model on different GPU configurations.

## Official Repository
https://github.com/VishwamAI/VishwamAI

## Google Colab Training

1. Access the repository:
```python
!git clone https://github.com/VishwamAI/VishwamAI.git
%cd VishwamAI
```

2. Open `colab_train.ipynb` in Google Colab
3. Set runtime to GPU: Runtime > Change runtime type > GPU
4. Run the setup cells:
```python
# Install dependencies
!pip install torch==2.4.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers==4.34.0 datasets accelerate

# Install package
!pip install -e .
```

5. The notebook will automatically detect your GPU type and apply optimized settings

## Local Training

### Prerequisites
- NVIDIA GPU with CUDA support
- Python 3.8 or higher
- PyTorch 2.0 or higher

### Setup

1. Clone the repository:
```bash
git clone https://github.com/VishwamAI/VishwamAI.git
cd VishwamAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Make training script executable:
```bash
chmod +x vishwamai/scripts/pretrain.sh
```

### Training Options

1. Basic Training:
```bash
./vishwamai/scripts/pretrain.sh \
    -b 4 \           # Batch size
    -e 3 \           # Number of epochs
    -o ./output \    # Output directory
    -c configs/config_optimized.json  # Model config
```

2. GPU-Specific Training:

For T4 GPU:
```bash
./vishwamai/scripts/pretrain.sh \
    -b 2 \
    -e 3 \
    -o ./output \
    -c configs/config_optimized.json \
    --gpu_type T4_optimized
```

For V100 GPU:
```bash
./vishwamai/scripts/pretrain.sh \
    -b 4 \
    -e 3 \
    -o ./output \
    -c configs/config_optimized.json \
    --gpu_type V100_optimized
```

For A100 GPU:
```bash
./vishwamai/scripts/pretrain.sh \
    -b 8 \
    -e 3 \
    -o ./output \
    -c configs/config_optimized.json \
    --gpu_type A100_optimized
```

## Memory Optimization

1. Enable gradient checkpointing:
```python
model.gradient_checkpointing_enable()
```

2. Use smaller batch size with gradient accumulation:
```bash
./vishwamai/scripts/pretrain.sh \
    -b 2 \
    --gradient_accumulation_steps 8
```

3. Disable caching during training:
```bash
./vishwamai/scripts/pretrain.sh --disable_cache
```

## Training Monitoring

The training script saves:
- Model checkpoints in the output directory
- Training logs in `output/status.txt`
- Training metrics in `output/trainer_state.json`

Monitor GPU usage during training:
```bash
watch -n 1 nvidia-smi
```

## Advanced Configuration

Edit `configs/config_optimized.json` to customize:
- Model architecture
- Training hyperparameters
- Optimization settings
- GPU-specific configurations

Example configuration adjustments:
```json
{
    "model_config": {
        "dim": 2048,
        "n_heads": 16,
        "n_layers": 24
    },
    "optimization_config": {
        "use_flash_attention": true,
        "gradient_checkpointing": true
    }
}
```

## Training Data

The model uses two main datasets:
1. GSM8K for mathematical reasoning
2. MMLU for multi-task learning

Load custom datasets:
```python
from datasets import load_dataset

datasets = {
    "train": load_dataset("your_dataset", split="train"),
    "validation": load_dataset("your_dataset", split="validation")
}
```

## Common Issues

1. Out of Memory (OOM):
   - Reduce batch size
   - Enable gradient checkpointing
   - Reduce model size for your GPU

2. Slow Training:
   - Enable mixed precision (FP16/BF16)
   - Use flash attention
   - Optimize sequence length

3. GPU Utilization:
   - Adjust batch size
   - Enable kernel optimizations
   - Monitor with nvidia-smi

## Training Results

Expected training metrics:
- Loss convergence in 3-5 epochs
- GPU utilization > 90%
- Memory usage ~85% of available VRAM

Save trained model:
```python
torch.save(model.state_dict(), "final_model.pt")
```

Load trained model:
```python
model = load_model(config_path, pretrained_path="final_model.pt")
```

## Support

For training issues and questions:
1. Check the documentation
2. Search existing issues in the [official repository](https://github.com/VishwamAI/VishwamAI)
3. Create a new issue if needed
