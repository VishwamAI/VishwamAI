"""Main training script with command line interface."""
import os
import sys
import json
import argparse
import yaml
from pathlib import Path
from datetime import datetime
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vishwamai.model import build_model
from vishwamai.data import build_dataset, build_dataloader
from vishwamai.training import Trainer
from vishwamai.training.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LRSchedulerCallback
)
from vishwamai.utils import (
    MetricLogger,
    DistributedLogger,
    print_gpu_utilization
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Vishwamai model')
    
    # Model configuration
    parser.add_argument('--model_config', type=str, required=True,
                       help='Path to model configuration YAML')
    parser.add_argument('--moe_config', type=str,
                       help='Path to MoE configuration YAML')
    parser.add_argument('--mla_config', type=str,
                       help='Path to MLA configuration YAML')
                       
    # Data configuration
    parser.add_argument('--data_config', type=str, required=True,
                       help='Path to data configuration YAML')
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training data')
    parser.add_argument('--val_data', type=str,
                       help='Path to validation data')
                       
    # Training configuration
    parser.add_argument('--training_config', type=str, required=True,
                       help='Path to training configuration YAML')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--experiment_name', type=str,
                       help='Name for experiment tracking')
                       
    # Distributed training
    parser.add_argument('--distributed', action='store_true',
                       help='Enable distributed training')
    parser.add_argument('--world_size', type=int, default=1,
                       help='Number of processes for distributed training')
    parser.add_argument('--dist_url', type=str, default='tcp://localhost:23456',
                       help='URL for distributed training')
    parser.add_argument('--dist_backend', type=str, default='nccl',
                       help='Distributed backend')
                       
    # Hardware
    parser.add_argument('--use_tpu', action='store_true',
                       help='Use TPU for training')
    parser.add_argument('--use_amp', action='store_true',
                       help='Use automatic mixed precision')
                       
    # Misc
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--resume_from', type=str,
                       help='Path to checkpoint to resume from')
                       
    return parser.parse_args()

def load_config(path: str) -> dict:
    """Load YAML configuration file.
    
    Args:
        path: Path to YAML file
        
    Returns:
        Configuration dictionary
    """
    with open(path) as f:
        config = yaml.safe_load(f)
    return config

def setup_experiment(args) -> Path:
    """Setup experiment output directory.
    
    Args:
        args: Command line arguments
        
    Returns:
        Path to experiment directory
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_name = args.experiment_name or f"experiment_{timestamp}"
    exp_dir = Path(args.output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configurations
    configs = {
        'model_config.yaml': load_config(args.model_config),
        'data_config.yaml': load_config(args.data_config),
        'training_config.yaml': load_config(args.training_config)
    }
    if args.moe_config:
        configs['moe_config.yaml'] = load_config(args.moe_config)
    if args.mla_config:
        configs['mla_config.yaml'] = load_config(args.mla_config)
        
    for name, config in configs.items():
        with open(exp_dir / name, 'w') as f:
            yaml.dump(config, f)
            
    # Save command line args
    with open(exp_dir / 'args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
        
    return exp_dir

def setup_distributed(
    rank: int,
    world_size: int,
    dist_url: str,
    dist_backend: str
) -> None:
    """Initialize distributed training.
    
    Args:
        rank: Process rank
        world_size: Number of processes
        dist_url: URL for distributed coordination
        dist_backend: Backend type
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = dist_url.split(':')[-1]
    
    dist.init_process_group(
        backend=dist_backend,
        init_method=dist_url,
        world_size=world_size,
        rank=rank
    )

def train(
    rank: int,
    world_size: int,
    args,
    exp_dir: Path
) -> None:
    """Training function for each process.
    
    Args:
        rank: Process rank
        world_size: Number of processes
        args: Command line arguments
        exp_dir: Experiment directory
    """
    if args.distributed:
        setup_distributed(rank, world_size, args.dist_url, args.dist_backend)
        
    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        
    # Load configurations
    model_config = load_config(args.model_config)
    data_config = load_config(args.data_config)
    training_config = load_config(args.training_config)
    moe_config = load_config(args.moe_config) if args.moe_config else None
    mla_config = load_config(args.mla_config) if args.mla_config else None
    
    # Setup logging
    logger_cls = DistributedLogger if args.distributed else MetricLogger
    logger = logger_cls(
        log_dir=str(exp_dir),
        experiment_name=args.experiment_name or exp_dir.name,
        rank=rank if args.distributed else 0,
        world_size=world_size if args.distributed else 1
    )
    
    # Build datasets and dataloaders
    train_dataset = build_dataset(
        args.train_data,
        data_config,
        split='train'
    )
    train_loader = build_dataloader(
        train_dataset,
        training_config,
        distributed=args.distributed
    )
    
    if args.val_data:
        val_dataset = build_dataset(
            args.val_data,
            data_config,
            split='val'
        )
        val_loader = build_dataloader(
            val_dataset,
            training_config,
            distributed=args.distributed
        )
    else:
        val_loader = None
        
    # Build model
    model = build_model(
        model_config,
        moe_config=moe_config,
        mla_config=mla_config
    )
    
    # Setup training callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=str(exp_dir / 'checkpoints'),
            monitor='val_loss' if val_loader else 'train_loss',
            save_top_k=2,
            every_n_epochs=1
        ),
        LRSchedulerCallback(
            monitor='val_loss' if val_loader else 'train_loss',
            interval='epoch'
        )
    ]
    
    if val_loader:
        callbacks.append(
            EarlyStopping(
                monitor='val_loss',
                patience=training_config.get('early_stopping_patience', 3)
            )
        )
        
    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        max_epochs=training_config['num_epochs'],
        max_steps=training_config.get('max_steps'),
        accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
        callbacks=callbacks,
        use_tpu=args.use_tpu,
        use_amp=args.use_amp,
        expert_parallel=moe_config is not None
    )
    
    # Log initial info
    if rank == 0:
        logger.log_model_info(model)
        logger.log_hyperparameters({
            **model_config,
            **training_config,
            **(moe_config or {}),
            **(mla_config or {})
        })
        print_gpu_utilization()
        
    # Train
    trainer.train(resume_from=args.resume_from)
    
    # Cleanup
    if args.distributed:
        dist.destroy_process_group()
    logger.close()

def main():
    """Main entry point."""
    args = parse_args()
    exp_dir = setup_experiment(args)
    
    if args.distributed:
        mp.spawn(
            train,
            args=(args.world_size, args, exp_dir),
            nprocs=args.world_size
        )
    else:
        train(0, 1, args, exp_dir)

if __name__ == '__main__':
    main()
