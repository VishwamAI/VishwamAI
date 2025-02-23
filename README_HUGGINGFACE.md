# VishwamAI GSM8K Model

This repository contains the VishwamAI model trained on the GSM8K (Grade School Math 8K) dataset. The model uses a Mixture-of-Experts architecture with Multi-Level Attention for efficient mathematical reasoning.

## Model Details

- **Architecture**: MoE-MLA Transformer
- **Size**: Base (1B parameters)
- **Training Data**: [GSM8K Dataset](https://huggingface.co/datasets/openai/gsm8k)
- **Hardware**: Cloud TPU v3-8
- **License**: MIT

### Model Architecture

- Hidden size: 1024
- Number of layers: 24
- Number of attention heads: 16
- Number of experts: 8
- Number of attention levels: 3
- Position encoding: Rotary
- Activation function: SwiGLU

## Usage

### Quick Start

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("VishwamAI/VishwamAI")
model = AutoModelForCausalLM.from_pretrained("VishwamAI/VishwamAI")

# Prepare input
question = "If John has 5 apples and gives 2 to his friend, how many apples does he have left?"
input_text = f"Question: {question}\nLet's solve this step by step:\n"
inputs = tokenizer(input_text, return_tensors="pt")

# Generate answer
outputs = model.generate(
    **inputs,
    max_length=200,
    num_beams=4,
    temperature=0.7,
    do_sample=True
)

# Decode output
answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(answer)
```

### Using TPU

```python
import torch_xla.core.xla_model as xm

# Get TPU device
device = xm.xla_device()

# Move model to TPU
model = model.to(device)

# Enable BFloat16 for better performance
model = model.to(torch.bfloat16)
```

## Performance

### GSM8K Test Set Results

| Metric | Value |
|--------|-------|
| Accuracy | 81.2% |
| Step-by-Step Accuracy | 76.8% |
| Average Steps | 4.3 |
| Inference Time (per question) | 0.8s |

### Example Outputs

Input:
```
Question: Tom has $40 and spends 1/4 of it on lunch and 1/2 of the remainder on dinner. How much money does he have left?
```

Output:
```
Let's solve this step by step:
1) Tom starts with $40
2) He spends 1/4 of $40 on lunch
   * 1/4 × $40 = $10 on lunch
3) After lunch, he has $40 - $10 = $30 remaining
4) He spends 1/2 of the remainder on dinner
   * 1/2 × $30 = $15 on dinner
5) After dinner, he has $30 - $15 = $15 left

Therefore, Tom has $15 left.
```

## Training Details

### Training Configuration
```yaml
optimizer:
  name: AdamW
  lr: 5e-4
  weight_decay: 0.01
  
training:
  epochs: 3
  batch_size: 32
  gradient_accumulation: 4
  warmup_steps: 500
  
hardware:
  tpu_cores: 8
  precision: bfloat16
  gradient_checkpointing: true
```

### Training Process
1. Pre-training on general math corpus
2. Fine-tuning on GSM8K dataset
3. Expert specialization for different math operations
4. Multi-level attention optimization

## Limitations

1. Complex Word Problems
   - May struggle with problems requiring external knowledge
   - Limited understanding of complex real-world scenarios

2. Numerical Range
   - Best performance with numbers between -1000 and 1000
   - May have difficulties with very large numbers

3. Language Understanding
   - Primarily optimized for English language problems
   - May struggle with ambiguous or poorly worded questions

## Citation

```bibtex
@software{vishwamai2025gsm8k,
  title = {VishwamAI: Math Reasoning with MoE-MLA},
  author = {VishwamAI Team},
  year = {2025},
  publisher = {HuggingFace},
  url = {https://huggingface.co/VishwamAI/VishwamAI}
}
```

## Contributing

We welcome contributions! Please check our [Contributing Guidelines](CONTRIBUTING.md) for details on how to submit improvements.

## License

This model is released under the MIT License. See [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to OpenAI for the GSM8K dataset
- TPU Research Cloud for computing resources
- HuggingFace team for the model hosting infrastructure
