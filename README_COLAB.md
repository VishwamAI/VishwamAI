# Training Vishwamai on Google Colab T4 GPUs

This guide explains how to train and use Vishwamai models on Google Colab's T4 GPUs.

## Getting Started

1. Open the [Vishwamai Colab Notebook](vishwamai_colab_pretrain.ipynb) in Google Colab
2. Select Runtime -> Change runtime type -> GPU
3. Verify you have a T4 GPU allocated (shown in notebook output)

## Features

The Colab notebook demonstrates:
- Mixed precision training (FP16)
- Flash Attention optimizations
- Memory-efficient attention
- Gradient scaling
- T4-specific optimizations

## Hardware Requirements

Google Colab T4 GPUs provide:
- 16GB GPU memory
- CUDA compute capability 7.5
- Support for Tensor Cores
- FP16 mixed precision

## Optimization Features

### 1. Mixed Precision Training
```python
config = ModelArgs(
    dtype="fp16",
    use_mixed_precision=True,
    gradient_checkpointing=True
)
```

### 2. Memory Optimizations
```python
config.update(
    use_flash_attention=True,
    unified=UnifiedConfig(
        transformer=dict(
            use_memory_efficient_attention=True,
            fused_qkv=True,
            fused_mlp=True
        )
    )
)
```

### 3. T4-Specific Features
```python
from vishwamai.utils.t4_utils import enable_t4_optimizations

# Enable T4 optimizations
enable_t4_optimizations()
```

## Training Tips

1. **Batch Size Selection**
   - Start with batch_size=32
   - Increase if memory allows
   - Use gradient accumulation for larger effective batches

2. **Memory Management**
   - Enable gradient checkpointing
   - Use Flash Attention
   - Monitor memory usage with get_memory_stats()

3. **Performance Optimization**
   - Use mixed precision training
   - Enable fused operations
   - Optimize sequence lengths

## Example Usage

1. **Basic Training**
```python
# Create model with T4 optimizations
model = create_model(
    ModelArgs(
        dtype="fp16",
        use_mixed_precision=True,
        use_flash_attention=True
    )
)

# Training loop with mixed precision
scaler = GradScaler()
for batch in dataloader:
    with torch.cuda.amp.autocast():
        outputs = model(**batch)
        loss = outputs['loss']
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

2. **Memory-Efficient Training**
```python
config = ModelArgs(
    hidden_size=768,
    num_attention_heads=12,
    num_hidden_layers=12,
    gradient_checkpointing=True,
    unified=UnifiedConfig(
        transformer=dict(
            use_memory_efficient_attention=True
        )
    )
)
```

3. **Inference Optimization**
```python
model.eval()
with torch.no_grad(), torch.cuda.amp.autocast():
    outputs = model.generate(
        input_ids,
        max_length=100,
        use_cache=True
    )
```

## Common Issues

1. **Out of Memory (OOM)**
   - Reduce batch size
   - Enable gradient checkpointing
   - Use Flash Attention
   - Monitor with get_memory_stats()

2. **Training Instability**
   - Adjust gradient scaling
   - Monitor loss values
   - Check gradient norms
   - Use gradient clipping

3. **Performance Issues**
   - Verify T4 optimizations are enabled
   - Check batch size utilization
   - Monitor GPU utilization
   - Profile with torch.profiler

## Saving and Loading

```python
# Save to Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'config': config.to_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, '/content/drive/MyDrive/vishwamai_model.pt')
```

## Performance Metrics

Typical performance on T4 GPU:
- Training throughput: ~10K tokens/sec
- Memory usage: 12-14GB with mixed precision
- Maximum sequence length: 2048 tokens
- Batch size range: 16-64

## Additional Resources

- [Google Colab Pro Features](https://colab.research.google.com/signup)
- [T4 GPU Documentation](https://www.nvidia.com/en-us/data-center/tesla-t4/)
- [PyTorch AMP Guide](https://pytorch.org/docs/stable/amp.html)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)

## Support

For issues and questions:
- Open an issue on GitHub
- Join our Discord community
- Check the troubleshooting guide in documentation

## References

1. NVIDIA T4 Deep Learning Guide
2. PyTorch Mixed Precision Training
3. Flash Attention Implementation
4. Memory-Efficient Transformers

## Updates

We regularly update the Colab notebook with:
- New optimization techniques
- Performance improvements
- Bug fixes
- Additional examples

Check the repository for the latest version of the notebook.
