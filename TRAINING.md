# Training VishwamAI Model

This guide explains how to train the VishwamAI model using Google Colab and HuggingFace.

## Prerequisites

1. Google Account (for Colab access)
2. HuggingFace Account (for model hosting)
3. HuggingFace API Token with write access

## Steps to Train

1. **Setup Google Colab**
   - Open [Google Colab](https://colab.research.google.com)
   - Click File > Upload Notebook
   - Upload the `colab_train.ipynb` file

2. **Configure Environment**
   - In Colab menu: Runtime > Change runtime type
   - Set "Hardware accelerator" to GPU
   - Set "Runtime shape" to High-RAM if available

3. **Run Training**
   - Run each cell in sequence by clicking the play button or pressing Shift+Enter
   - When prompted, login to HuggingFace using your token
   - The notebook will:
     - Install required dependencies
     - Clone the repository
     - Load and prepare datasets
     - Initialize model with Colab-optimized settings
     - Train the model
     - Save and upload to HuggingFace

## Training Configuration

The notebook uses these optimized settings for Colab:
- Model size: 2048 hidden size, 12 layers
- Batch size: 8 with gradient accumulation
- Mixed precision (FP16)
- Gradient checkpointing
- 3 training epochs

## Security Notes

- Never share or commit your HuggingFace token
- Use HuggingFace's built-in notebook_login() for authentication
- The training script expects tokens as environment variables
- All credentials are handled securely through HuggingFace's auth system
