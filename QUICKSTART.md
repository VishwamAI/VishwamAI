# VishwamAI Colab Training Quick Start

## Prerequisites
1. Google Colab account with GPU runtime
2. Google Drive for checkpoint storage
3. HuggingFace account and API token
4. Weights & Biases account (for monitoring)

## Architecture Overview

1. **Base Architecture**
   - Custom Transformer implementation
   - 27 layers, 16 attention heads
   - 2048 hidden dimension
   - 102,400 vocabulary size
   - 16,384 context length

2. **Advanced Features**
   - Mixture of Experts (MoE)
     - 64 routed experts
     - 2 shared experts
     - 6 activated experts per token
   - ALiBi positional embeddings
   - RoPE scaling
   - LoRA ranks optimization
   - Neural memory augmentation

## Setup Steps

1. **Open in Colab**
   - Upload `vishwamai_colab_pretrain.ipynb` to Colab
   - Or open directly from GitHub

2. **Enable GPU**
   ```
   Runtime -> Change runtime type -> GPU
   ```

3. **Mount Drive**
   - Run first cell to mount Google Drive
   - Accept the authorization prompt

4. **Install Dependencies**
   - Repository will be cloned automatically
   - All required packages will be installed
   - MoE and custom components will be set up

## Training Process

1. **Initialization**
   ```python
   config = {
       "model_type": "moe",
       "hidden_size": 2048,
       "num_hidden_layers": 27,
       "num_attention_heads": 16,
       "num_experts": 64,
       ...
   }
   ```

2. **Dataset Processing**
   - GSM8K for mathematical reasoning
   - MMLU for knowledge evaluation
   - Code repositories for programming
   - Automatic curriculum learning

3. **Training Features**
   - Mixed precision (BF16/FP8)
   - Gradient checkpointing
   - Dynamic batch sizing
   - Expert load balancing
   - Tree of Thoughts reasoning

## Monitoring

1. **Training Metrics**
   - Loss and learning rate
   - Expert utilization
   - Memory usage
   - Curriculum progression

2. **WandB Integration**
   ```python
   wandb.log({
       "loss": stats["loss"],
       "expert_usage": stats["moe_metrics"],
       "curriculum_level": stats["curriculum_stats"]["current_difficulty"]
   })
   ```

3. **Checkpointing**
   - Every 1000 steps to Drive
   - Auto-upload to HuggingFace
   - Evaluation every 5000 steps

## Common Issues

1. **Memory Management**
   ```python
   config.update({
       "batch_size": 4,  # Reduce if OOM
       "gradient_accumulation_steps": 256,
       "memory_size": 512,  # Adjust based on GPU
       "cache_size": 256
   })
   ```

2. **Expert Balance**
   - Monitor expert usage metrics
   - Adjust load balancing weights
   - Check expert capacity settings

3. **Training Stability**
   - Use gradient clipping
   - Monitor loss curves
   - Adjust learning rate if needed

## Performance Tips

1. **GPU Optimization**
   - Enable gradient checkpointing
   - Use mixed precision training
   - Monitor memory fragmentation

2. **Dataset Balance**
   - Proper curriculum progression
   - Balanced expert routing
   - Regular evaluation

3. **Checkpointing Strategy**
   - Save model state
   - Save optimizer state
   - Save curriculum state

## Support
- GitHub Issues: [VishwamAI/VishwamAI](https://github.com/VishwamAI/VishwamAI/issues)
- Model Card: [HuggingFace](https://huggingface.co/VishwamAI/VishwamAI)
- Documentation: Under development
