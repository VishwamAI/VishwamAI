# Quick Start Guide

## Installation

### 1. Basic Installation
```bash
pip install vishwamai
```

### 2. Development Installation
```bash
git clone https://github.com/organization/vishwamai.git
cd vishwamai
pip install -e ".[dev]"
```

### 3. TPU Setup
```bash
# Install TPU dependencies
pip install torch-xla cloud-tpu-client

# Configure TPU
export TPU_NAME="v4-8"
export TPU_ZONE="us-central1-a"
export PROJECT_ID="your-project-id"
```

## Basic Usage

### 1. Training a Model

#### a. Using Command Line
```bash
# Preprocess data
vishwamai-preprocess \
    --config configs/data_config.yaml \
    --input-dir data/raw \
    --output-dir data/processed

# Train tokenizer
vishwamai-tokenizer \
    --config configs/data_config.yaml \
    --input-dir data/processed \
    --output-dir models/tokenizer

# Train model
vishwamai-train \
    --model-config configs/model_config.yaml \
    --training-config configs/training_config.yaml \
    --data-config configs/data_config.yaml \
    --tpu-config configs/tpu_config.yaml \
    --tokenizer-path models/tokenizer \
    --output-dir models/vishwamai
```

#### b. Using Python API
```python
from vishwamai.model import VishwamaiModel
from vishwamai.training import Trainer
from vishwamai.data import create_dataloaders
from vishwamai.utils import load_config

# Load configurations
model_config = load_config("configs/model_config.yaml")
training_config = load_config("configs/training_config.yaml")
data_config = load_config("configs/data_config.yaml")

# Initialize model
model = VishwamaiModel(model_config)

# Create dataloaders
train_loader, val_loader = create_dataloaders(
    train_path="data/train",
    val_path="data/val",
    config=data_config
)

# Initialize trainer
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=training_config
)

# Start training
trainer.train()
```

### 2. Evaluation

#### a. Command Line
```bash
vishwamai-eval \
    --model-path models/vishwamai \
    --data-dir data/test \
    --output-dir results \
    --benchmark mmlu
```

#### b. Python API
```python
from vishwamai.model import VishwamaiModel
from vishwamai.utils import load_model, evaluate_model

# Load model
model = load_model("models/vishwamai")

# Run evaluation
results = evaluate_model(
    model,
    dataset="mmlu",
    data_path="data/test",
    batch_size=32
)

print(results)
```

### 3. Inference

#### a. Command Line Service
```bash
# Start model server
vishwamai-serve \
    --model-path models/vishwamai \
    --port 8000

# Make request
curl -X POST http://localhost:8000/generate \
    -H "Content-Type: application/json" \
    -d '{"text": "Your input text here"}'
```

#### b. Python API
```python
from vishwamai.model import VishwamaiModel
from vishwamai.utils import load_model

# Load model
model = load_model("models/vishwamai")

# Generate text
output = model.generate(
    "Your input text here",
    max_length=100,
    top_k=50,
    top_p=0.9,
    temperature=0.7
)

print(output)
```

## Common Tasks

### 1. Fine-tuning on Custom Data
```python
from vishwamai.data import CustomDataset
from vishwamai.training import Trainer
from vishwamai.utils import load_model

# Prepare custom dataset
dataset = CustomDataset("path/to/data")

# Load pre-trained model
model = load_model("models/vishwamai")

# Fine-tune
trainer = Trainer(model)
trainer.train(
    train_dataset=dataset,
    num_epochs=3,
    learning_rate=1e-5
)
```

### 2. Model Export
```bash
# Export model for deployment
vishwamai-export \
    --model-path models/vishwamai \
    --format onnx \
    --output-dir exported_model \
    --quantize
```

### 3. TPU Training Configuration
```yaml
# configs/tpu_config.yaml
device:
  num_cores: 8
  topology: 2x2x2
optimization:
  batch_processing:
    pipelining: true
    prefetch_depth: 3
```

## Troubleshooting

### 1. Memory Issues
```python
# Use gradient checkpointing
model.enable_gradient_checkpointing()

# Enable memory efficient attention
model.config.use_flash_attention = True
```

### 2. TPU Errors
```bash
# Check TPU status
gcloud compute tpus tpu-vm describe $TPU_NAME --zone=$TPU_ZONE

# SSH into TPU VM
gcloud compute tpus tpu-vm ssh $TPU_NAME --zone=$TPU_ZONE

# Monitor TPU metrics
vishwamai-monitor --tpu-name=$TPU_NAME
```

## Resource Management

### 1. Memory Profiling
```python
from vishwamai.utils.profiling import MemoryProfiler

with MemoryProfiler() as profiler:
    model.train()
    
print(profiler.summary())
```

### 2. Performance Monitoring
```python
from vishwamai.utils.profiling import PerformanceTracker

tracker = PerformanceTracker()
tracker.start()
model.train()
tracker.stop()
tracker.plot_metrics()
```

For more detailed information, please refer to:
- [Technical Documentation](docs/technical.md)
- [API Reference](docs/api.md)
- [Configuration Guide](docs/configuration.md)
