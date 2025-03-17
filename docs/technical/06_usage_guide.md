# VishwamAI Usage Guide

This document provides comprehensive examples and best practices for using VishwamAI in various scenarios.

## Quick Start

### Basic Model Usage

```python
from vishwamai import VishwamAI, VishwamAITokenizer

# Load model and tokenizer
model = VishwamAI.from_pretrained("vishwamai/base")
tokenizer = VishwamAITokenizer.from_pretrained("vishwamai/base")

# Generate text
text = "Explain the concept of recursion in programming:"
input_ids = tokenizer.encode(text, return_tensors="jax")
outputs = model.generate(input_ids, max_length=512)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

## Training

### Fine-tuning Example

```python
from vishwamai import VishwamAITrainer, TrainingConfig

# Training configuration
config = TrainingConfig(
    batch_size=8,
    learning_rate=1e-4,
    warmup_steps=100,
    total_steps=1000,
    gradient_checkpointing=True
)

# Initialize trainer
trainer = VishwamAITrainer(
    model=model,
    config=config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset
)

# Start training
trainer.train()
```

### Knowledge Distillation

```python
from vishwamai import DistillationTrainer

# Initialize teacher model
teacher_model = VishwamAI.from_pretrained("gemma-7b")

# Setup distillation
distill_trainer = DistillationTrainer(
    student_model=model,
    teacher_model=teacher_model,
    temperature=2.0,
    alpha=0.5
)

# Train with distillation
distill_trainer.train(
    train_dataset=train_dataset,
    num_epochs=3
)
```

## Advanced Usage

### Tree of Thoughts Reasoning

```python
from vishwamai import TreeOfThoughts

# Initialize ToT
tot = TreeOfThoughts(
    model=model,
    max_branches=3,
    max_depth=3
)

# Solve complex problem
problem = """
Design a system to efficiently manage traffic flow in a smart city.
Consider factors like:
1. Real-time traffic monitoring
2. Emergency vehicle priority
3. Public transport optimization
"""

solution = tot.solve(
    problem=problem,
    evaluation_metric="completeness",
    max_steps=5
)
```

### Chain of Thought Processing

```python
from vishwamai import ChainOfThought

# Initialize CoT
cot = ChainOfThought(
    model=model,
    max_steps=5
)

# Solve mathematical problem
problem = "If a train travels 120 km in 2 hours, and another train travels 180 km in 3 hours, which train is faster?"

solution = cot.solve(
    problem=problem,
    show_work=True
)
```

## Model Configuration

### Memory Optimization

```python
# Configure for memory efficiency
config = {
    "model": {
        "hidden_dim": 768,
        "num_layers": 24
    },
    "memory": {
        "gradient_checkpointing": True,
        "kv_cache_fp8": True
    },
    "optimization": {
        "use_fp8": True,
        "mixed_precision": True
    }
}

model = VishwamAI.from_config(config)
```

### High Performance Setup

```python
# Configure for maximum performance
config = {
    "model": {
        "hidden_dim": 1024,
        "num_layers": 32
    },
    "tpu": {
        "tpu_cores": 8,
        "device_strategy": "data_parallel"
    },
    "optimization": {
        "use_fp8": False,
        "block_size": 256
    }
}

model = VishwamAI.from_config(config)
```

## Common Use Cases

### Text Generation

```python
def generate_with_prompt(prompt: str, max_length: int = 512) -> str:
    """Generate text with specified prompt"""
    input_ids = tokenizer.encode(prompt, return_tensors="jax")
    
    outputs = model.generate(
        input_ids,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Chat Completion

```python
def chat_completion(messages: List[Dict[str, str]]) -> str:
    """Generate chat responses"""
    response = model.generate_chat(
        messages=messages,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9
    )
    return response
```

## Performance Optimization

### Batch Processing

```python
def process_batch(texts: List[str]) -> List[str]:
    """Process multiple inputs efficiently"""
    # Tokenize batch
    inputs = tokenizer.batch_encode(
        texts,
        padding=True,
        truncation=True,
        return_tensors="jax"
    )
    
    # Generate responses
    outputs = model.generate(
        inputs,
        max_length=512,
        num_beams=4
    )
    
    # Decode outputs
    return tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )
```

### Memory-Efficient Inference

```python
def stream_generate(
    prompt: str,
    chunk_size: int = 512
) -> Generator[str, None, None]:
    """Stream generation for long sequences"""
    input_ids = tokenizer.encode(prompt, return_tensors="jax")
    
    for i in range(0, 2048, chunk_size):
        outputs = model.generate(
            input_ids,
            max_new_tokens=chunk_size,
            temperature=0.7
        )
        
        # Get new tokens
        new_tokens = outputs[0][input_ids.shape[1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        yield text
        
        # Update input for next iteration
        input_ids = outputs
```

## Error Handling

### Common Issues and Solutions

```python
def safe_generate(
    prompt: str,
    max_retries: int = 3
) -> Optional[str]:
    """Generate text with error handling"""
    for attempt in range(max_retries):
        try:
            return generate_with_prompt(prompt)
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"Failed after {max_retries} attempts: {e}")
                return None
            print(f"Attempt {attempt + 1} failed: {e}")
```

## Best Practices

1. Model Initialization
   - Use appropriate precision for task
   - Enable memory optimizations when needed
   - Configure batch size based on hardware

2. Input Processing
   - Properly handle special tokens
   - Use batch processing when possible
   - Implement proper error handling

3. Generation Settings
   - Adjust temperature for creativity
   - Use appropriate max length
   - Enable sampling for diverse outputs

4. Resource Management
   - Implement proper cleanup
   - Monitor memory usage
   - Use efficient batch sizes

## Monitoring and Logging

### Basic Monitoring

```python
from vishwamai import DuckDBLogger

logger = DuckDBLogger(
    db_path="logs.db",
    experiment_name="production_inference"
)

def monitored_generate(
    prompt: str,
    **kwargs
) -> str:
    start_time = time.time()
    result = generate_with_prompt(prompt, **kwargs)
    duration = time.time() - start_time
    
    logger.log_metrics({
        "duration": duration,
        "input_length": len(prompt),
        "output_length": len(result)
    })
    
    return result
