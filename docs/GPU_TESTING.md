# VishwamAI GPU Testing Guide

This guide covers testing VishwamAI optimizations on consumer GPUs (e.g. GTX 1650) before A100 deployment.

## GTX 1650 Testing Configuration

The testing script automatically adjusts parameters based on available VRAM:

| Parameter | A100 Value | GTX 1650 Value | Notes |
|-----------|------------|----------------|--------|
| Hidden Size | 4096 | 512 | Reduced for 4GB VRAM |
| Num Heads | 32 | 8 | Scaled proportionally |
| Num Layers | 32 | 4 | Minimum viable depth |
| Batch Size | 2048 | 4 | Adjusted for memory |
| Sequence Length | 2048 | 128 | Reduced for testing |

## Running Tests

1. **Basic Compatibility Test**
```bash
python vishwamai/training/gpu_testing.py
```

The script will:
- Auto-detect GPU specifications
- Adjust model size for available VRAM
- Run a test forward pass
- Report memory usage and performance metrics

2. **Expected Output**
```
INFO: Testing on NVIDIA GeForce GTX 1650 with 4.0GB VRAM
WARNING: Low VRAM detected (4.0GB), reducing model size
INFO: Test successful! Peak memory usage: 2048.3MB
INFO: Loss: 2.1234
INFO: MoE loss: 0.0123
```

## Memory Management

The testing implementation includes several memory optimizations:
- Automatic VRAM detection and model scaling
- Intermediate tensor cleanup
- Conditional mixed precision (disabled if <4GB VRAM)
- Memory monitoring with detailed logging

## Common Issues & Solutions

1. **Out of Memory (OOM)**
   ```
   ERROR: GPU OOM. Try reducing batch_size or model size.
   ```
   Solutions:
   - Reduce batch_size (default: 4)
   - Decrease hidden_size (default: 512)
   - Reduce num_layers (default: 4)

2. **Slow Performance**
   - Disable mixed precision (fp16=False)
   - Reduce sequence length
   - Clean VRAM between runs: `torch.cuda.empty_cache()`

3. **High Memory Usage**
   - Monitor with `nvidia-smi`
   - Use `torch.cuda.memory_summary()`
   - Enable garbage collection

## Testing Flow

1. **Initial Setup**
```python
config = TestConfig()
model = GPUTestModel(config)
```

2. **Memory Checks**
```python
gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
if gpu_mem < 6:  # GTX 1650 has 4GB
    # Reduce model size automatically
```

3. **Forward Pass**
```python
outputs = model(input_ids, attention_mask, labels)
```

4. **Memory Monitoring**
```python
peak_mem = torch.cuda.max_memory_allocated() / 1024**2
logging.info(f"Peak memory usage: {peak_mem:.1f}MB")
```

## Advanced Testing

1. **Stress Testing**
```bash
python vishwamai/training/gpu_testing.py --stress-test
```

2. **Memory Profiling**
```python
from torch.profiler import profile
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    outputs = model(input_ids)
print(prof.key_averages().table())
```

3. **Gradient Checks**
```python
torch.autograd.gradcheck(model, input_ids)
```

## Best Practices

1. **Before Testing**
   - Close other GPU applications
   - Monitor temperature with `nvidia-smi`
   - Clear VRAM: `torch.cuda.empty_cache()`

2. **During Testing**
   - Start with small configurations
   - Monitor memory usage
   - Check for memory leaks
   - Log performance metrics

3. **After Testing**
   - Document peak memory usage
   - Record performance metrics
   - Note any stability issues

## Moving to A100

After successful testing on GTX 1650:
1. Remove memory restrictions
2. Enable all optimizations
3. Scale up model parameters
4. Enable multi-GPU training
5. Verify A100 performance

## References

1. [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)
2. [NVIDIA GPU Monitoring](https://developer.nvidia.com/nvidia-system-management-interface)
3. [GTX 1650 Specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/gtx-1650/)
