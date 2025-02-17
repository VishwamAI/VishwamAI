# VishwamAI Training on Google Colab

## Setup and Training Process

1. **Environment Setup**
   ```bash
   # Clone repository
   git clone https://github.com/VishwamAI/VishwamAI.git
   cd VishwamAI
   
   # Install dependencies
   pip install -r requirements.txt
   ```

2. **Dataset Preparation**
   - GSM8K is loaded from parquet files if available
   - MMLU and code datasets are loaded directly
   - Each dataset type has custom preprocessing

3. **Training Configuration**
   - Initial model: 7B parameters
   - Batch size: 4 (Colab optimized)
   - Gradient accumulation: 256 steps
   - Mixed precision training enabled
   - Gradient checkpointing enabled

4. **Running Training**
   - Open `vishwamai_colab_pretrain.ipynb` in Google Colab
   - Mount Google Drive for checkpoints
   - Run cells sequentially
   - Monitor training through WandB

5. **Checkpoints**
   - Saved every 1000 steps
   - Stored in Google Drive
   - Automatically uploaded to HuggingFace

6. **Evaluation**
   - Regular evaluation during training
   - Final evaluation on test sets
   - Results saved to Drive and HuggingFace

## Handling Errors

1. **Dataset Loading**
   ```python
   try:
       dataset = load_dataset_with_type(dataset_name)
   except Exception as e:
       print(f"Error loading {dataset_name}: {str(e)}")
   ```

2. **Out of Memory**
   - Reduce batch size
   - Increase gradient accumulation
   - Enable gradient checkpointing

3. **Training Interruption**
   - Checkpoints are saved automatically
   - Can resume from last checkpoint

## Monitoring

1. **WandB Integration**
   - Real-time loss tracking
   - Memory usage monitoring
   - Training metrics visualization

2. **Evaluation Metrics**
   - Regular validation checks
   - Test set performance
   - Dataset-specific metrics

## HuggingFace Integration

```python
# Upload model
trainer.push_to_hub(
    "VishwamAI/VishwamAI",
    commit_message="Training update"
)
```

## Known Limitations

1. **Colab Constraints**
   - 12-hour runtime limit
   - Limited GPU memory (16GB)
   - Potential connection drops

2. **Dataset Size**
   - GSM8K: ~7K examples
   - MMLU: ~17K examples
   - Need to manage memory efficiently

## Best Practices

1. **Save Frequently**
   - Regular checkpoints
   - Back up to Drive
   - Push to HuggingFace

2. **Monitor Resources**
   - Watch GPU memory
   - Check training metrics
   - Handle errors gracefully

3. **Gradient Accumulation**
   - Use larger effective batch size
   - Maintain training stability
   - Optimize memory usage
