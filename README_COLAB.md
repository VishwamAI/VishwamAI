# Training VishwamAI on Google Colab

This guide explains how to train VishwamAI on Google Colab's T4 GPU using the provided notebook.

## Setup Instructions

1. Open `vishwamai_colab_pretrain.ipynb` in Google Colab
2. Select Runtime -> Change runtime type -> GPU
3. Run the installation cell to set up dependencies
4. Verify GPU availability in the setup cell

## Memory Management

The notebook is optimized for T4's ~16GB memory:
- Uses 8-bit quantization
- Reduced sequence length (512)
- Batch size of 4 with gradient accumulation
- Gradient checkpointing enabled
- FP16 training

## T5 Integration

The model uses T5-base architecture:
- Pre-trained T5 weights are loaded
- Architecture matches T5-base dimensions
- Uses T5 tokenizer and preprocessing
- T5-style prompt formatting

## Training Datasets

The model trains on:
- GSM8K for mathematical reasoning
- MMLU for general knowledge

Data is formatted in T5 style:
```python
# Math problems:
Input: "solve: What is 7 * 12?"
Target: "84"

# MMLU questions:
Input: "answer: What is the capital of France?\n\nOptions:\nA) London\nB) Paris\nC) Berlin\nD) Madrid"
Target: "The answer is Paris"
```

## Running Training

1. Enter your Hugging Face token when prompted
2. Training progress is tracked in Weights & Biases
3. Model checkpoints are saved every 200 steps
4. Final model is pushed to Hugging Face Hub

## Memory Usage Tips

- Clear GPU cache between major operations
- Use streaming datasets for large data
- Monitor GPU memory usage with `nvidia-smi`
- Reduce batch size if OOM errors occur

## Expected Results

- Training Time: ~6-8 hours on T4
- GPU Memory Usage: ~14GB
- Final Model Size: ~1.5GB

The trained model will be available at: https://huggingface.co/kasinadhsarma/vishwamai-model
