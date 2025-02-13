import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.multiprocessing as mp
import torch.distributed as dist

from vishwamai.model_utils import load_model
from vishwamai.trainer import Trainer, TrainingArgs

def setup_distributed(rank: int, world_size: int):
    """Initialize distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()

def create_dataloaders(args: TrainingArgs):
    """
    Create training and evaluation dataloaders
    This is a placeholder - replace with your actual data loading logic
    """
    # Dummy data for demonstration
    train_data = torch.randint(0, 64000, (1000, 128))
    train_labels = torch.randint(0, 64000, (1000, 128))
    eval_data = torch.randint(0, 64000, (100, 128))
    eval_labels = torch.randint(0, 64000, (100, 128))
    
    train_dataloader = DataLoader(
        list(zip(train_data, train_labels)),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True
    )
    
    eval_dataloader = DataLoader(
        list(zip(eval_data, eval_labels)),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True
    )
    
    return train_dataloader, eval_dataloader

def train_process(rank: int, world_size: int):
    """Main training process"""
    # Initialize distributed training
    if world_size > 1:
        setup_distributed(rank, world_size)
    
    # Training arguments
    train_args = TrainingArgs(
        output_dir="checkpoints",
        num_epochs=3,
        batch_size=32,
        gradient_accumulation_steps=4,
        learning_rate=1e-4,
        warmup_steps=100,
        weight_decay=0.1,
        max_grad_norm=1.0,
        save_steps=500,
        logging_steps=50,
        use_fsdp=True if world_size > 1 else False,
        mixed_precision=True,
        cpu_offload=False,
        gradient_checkpointing=True
    )
    
    # Load model
    config_path = Path(__file__).parent.parent / "configs" / "config_optimized.json"
    model = load_model(
        config_path=config_path,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Create dataloaders
    train_dataloader, eval_dataloader = create_dataloaders(train_args)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        args=train_args
    )
    
    if rank == 0:
        print("\nTraining Configuration:")
        print(f"Number of GPUs: {world_size}")
        print(f"Mixed Precision: {train_args.mixed_precision}")
        print(f"Gradient Checkpointing: {train_args.gradient_checkpointing}")
        print(f"Number of Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Train
    trainer.train()
    
    # Clean up
    if world_size > 1:
        cleanup_distributed()

def main():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available, running on CPU")
        world_size = 1
    else:
        world_size = torch.cuda.device_count()
    
    if world_size > 1:
        # Multi-GPU training
        mp.spawn(
            train_process,
            args=(world_size,),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU or CPU training
        train_process(0, 1)

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    main()
