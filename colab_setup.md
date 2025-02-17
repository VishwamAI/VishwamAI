# Running VishwamAI Training on Google Colab

## Setup Instructions

1. Open the `vishwamai_colab_pretrain.ipynb` notebook in Google Colab
2. Enable GPU runtime:
   - Runtime -> Change runtime type -> GPU

3. Mount Google Drive:
   - The notebook will automatically mount your Drive
   - Checkpoints will be saved to `/content/drive/MyDrive/VishwamAI/checkpoints`

4. The notebook will:
   - Clone the VishwamAI repository
   - Install dependencies
   - Set up HuggingFace access
   - Initialize training with Colab-optimized settings

## Memory Management

The notebook is configured for Colab's T4 GPU:
- Batch size: 8 (reduced from standard)
- Gradient accumulation steps: 128
- Gradient checkpointing enabled
- FP8/BF16 mixed precision training

## Training Flow

1. Model and data setup is automatic
2. Checkpoints are saved every 1000 steps to Google Drive
3. Models are automatically uploaded to HuggingFace
4. Training can be safely interrupted and resumed

## Monitoring

- Training progress visible in Colab UI
- WandB integration for metrics tracking
- Regular evaluation on benchmark datasets
- Results saved to your Google Drive

## Known Limitations

- Colab sessions have time limits (12 hours)
- T4 GPU has 16GB memory
- Drive storage needed for checkpoints
