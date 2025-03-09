"""
Parallel training utilities for VishwamAI.
Supports multi-GPU (DDP) and model parallelism with smallpond integration.
"""

import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import logging
from typing import Optional, Dict, Any, Type, List
from dataclasses import dataclass
import math
import smallpond

from vishwamai.models.transformer import VishwamAITransformer
from vishwamai.training.dataset_loader import VishwamAIDataset, create_dataloader
from vishwamai.optimisation.memory_optimization import MemoryOptimizer

logger = logging.getLogger(__name__)

@dataclass
class ParallelConfig:
    """Configuration for parallel training"""
    backend: str = 'nccl'              # 'nccl' or 'gloo'
    world_size: int = 1                # Total number of processes
    local_rank: int = -1               # Local process rank
    device_ids: Optional[List[int]] = None  # GPU device IDs to use
    master_addr: str = 'localhost'     # Master node address
    master_port: str = '12355'         # Master node port
    model_parallel_size: int = 1       # Number of model parallel shards
    pipeline_parallel_size: int = 1    # Number of pipeline parallel stages
    grad_accumulation: int = 1         # Gradient accumulation steps
    mixed_precision: bool = True       # Whether to use mixed precision training
    optimize_memory: bool = True       # Whether to use memory optimization
    seed: int = 42                     # Random seed
    smallpond_config: Optional[Dict[str, Any]] = None

class ParallelTrainer:
    """
    Manager for parallel training across multiple devices.
    Supports DDP and model parallelism.
    """
    
    def __init__(
        self,
        model_class: Type,
        model_config: Dict[str, Any],
        train_dataset: VishwamAIDataset,
        val_dataset: Optional[VishwamAIDataset] = None,
        parallel_config: Optional[ParallelConfig] = None,
        training_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize parallel trainer.
        
        Args:
            model_class: Model class to instantiate
            model_config: Model configuration
            train_dataset: Training dataset
            val_dataset: Validation dataset
            parallel_config: Parallel training configuration
            training_config: Training hyperparameters
        """
        self.model_class = model_class
        self.model_config = model_config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.parallel_config = parallel_config or ParallelConfig()
        self.training_config = training_config or {}
        
        # Set random seed
        self._set_seed(self.parallel_config.seed)
        
        # Initialize process group
        self._setup_distributed()
        
    def _set_seed(self, seed: int):
        """Set random seed across all processes"""
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
    def _init_smallpond(self):
        """Initialize smallpond for distributed data processing"""
        if self.parallel_config.smallpond_config:
            config = self.parallel_config.smallpond_config
            self.sp_session = smallpond.init(
                num_executors=config.get('num_executors', self.parallel_config.world_size),
                ray_address=config.get('ray_address'),
                bind_numa_node=config.get('bind_numa_node', True),
                data_root=config.get('data_root', '/tmp/vishwamai/smallpond'),
                platform=config.get('platform'),
            )
            
    def _setup_distributed(self):
        """Setup distributed training environment"""
        # GPU setup
        if self.parallel_config.local_rank != -1:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available for distributed training")
                
            # Set environment variables
            os.environ['MASTER_ADDR'] = self.parallel_config.master_addr
            os.environ['MASTER_PORT'] = self.parallel_config.master_port
            
            # Initialize process group
            dist.init_process_group(
                backend=self.parallel_config.backend,
                world_size=self.parallel_config.world_size,
                rank=self.parallel_config.local_rank
            )
            
            torch.cuda.set_device(self.parallel_config.local_rank)
            self.device = torch.device(f"cuda:{self.parallel_config.local_rank}")
            self.is_main_process = self.parallel_config.local_rank == 0
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.is_main_process = True
                
        # Initialize smallpond if configured
        self._init_smallpond()

    def _setup_model_parallel(self):
        """Setup model parallelism"""
        if self.parallel_config.model_parallel_size > 1:
            # Split model across devices
            devices = self.parallel_config.device_ids or list(range(torch.cuda.device_count()))
            num_devices = len(devices)
            
            if num_devices < self.parallel_config.model_parallel_size:
                raise ValueError(f"Not enough devices ({num_devices}) for model parallel size {self.parallel_config.model_parallel_size}")
                
            # Split model into chunks
            self.model_chunks = []
            chunk_size = len(list(self.model.parameters())) // self.parallel_config.model_parallel_size
            params = list(self.model.parameters())
            
            for i in range(self.parallel_config.model_parallel_size):
                start_idx = i * chunk_size
                end_idx = start_idx + chunk_size if i < self.parallel_config.model_parallel_size - 1 else None
                chunk_params = params[start_idx:end_idx]
                
                chunk = torch.nn.ParameterList(chunk_params).to(devices[i])
                self.model_chunks.append(chunk)
                
    def _setup_pipeline_parallel(self):
        """Setup pipeline parallelism"""
        if self.parallel_config.pipeline_parallel_size > 1:
            # Split model into stages
            self.model_stages = []
            num_layers = len(self.model.transformer.layers)
            layers_per_stage = num_layers // self.parallel_config.pipeline_parallel_size
            
            for i in range(self.parallel_config.pipeline_parallel_size):
                start_idx = i * layers_per_stage
                end_idx = start_idx + layers_per_stage if i < self.parallel_config.pipeline_parallel_size - 1 else None
                stage_layers = self.model.transformer.layers[start_idx:end_idx]
                
                stage = torch.nn.ModuleList(stage_layers)
                self.model_stages.append(stage)
                
    def _setup_optimizer(self):
        """Setup optimizer with proper parameter groups"""
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.training_config.get('weight_decay', 0.01)
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.training_config.get('learning_rate', 1e-4)
        )
        
        # Setup mixed precision training
        if self.parallel_config.mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            
    def _setup_data_parallel(self):
        """Setup data parallelism"""
        if self.parallel_config.local_rank != -1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.parallel_config.local_rank]
            )

    def prepare(self):
        """Prepare for parallel training"""
        # Initialize model
        self.model = self.model_class(**self.model_config)
        
        # Setup parallelism
        self._setup_model_parallel()
        self._setup_pipeline_parallel()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup data parallel training
        self._setup_data_parallel()
        
        # Setup optimizer
        self._setup_optimizer()
        
        # Setup memory optimization if enabled
        if self.parallel_config.optimize_memory:
            self.memory_optimizer = MemoryOptimizer(
                self.model,
                device=self.device
            )
            
        # Create data loaders
        self.train_dataloader = self._get_dataloader(
            batch_size=self.training_config.get('batch_size', 32),
            shuffle=True
        )
        
        if self.val_dataset:
            self.val_dataloader = self._get_dataloader(
                batch_size=self.training_config.get('batch_size', 32),
                shuffle=False
            )
            
    def _get_dataloader(self, batch_size: int, shuffle: bool = True):
        """Create dataloader with appropriate sampler"""
        if self.sp_session:
            # Use smallpond for data loading
            df = self.sp_session.read_parquet(self.train_dataset.data_path)
            df = df.repartition(self.parallel_config.world_size)
            return create_dataloader(
                df.to_pandas(), 
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=self.parallel_config.num_workers
            )
        else:
            # Standard PyTorch dataloader
            sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.parallel_config.world_size,
                rank=self.parallel_config.local_rank,
                shuffle=shuffle
            ) if self.parallel_config.local_rank != -1 else None
            
            return create_dataloader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=(sampler is None and shuffle),
                sampler=sampler,
                num_workers=self.parallel_config.num_workers
            )
            
    def train_step(self, batch):
        """Execute single training step"""
        self.model.train()
        
        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        # Forward pass with mixed precision
        with torch.cuda.amp.autocast(enabled=self.parallel_config.mixed_precision):
            outputs = self.model(**batch)
            loss = outputs.loss / self.parallel_config.grad_accumulation
            
        # Backward pass
        if self.parallel_config.mixed_precision:
            self.scaler.scale(loss).backward()
            if (self.global_step + 1) % self.parallel_config.grad_accumulation == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            loss.backward()
            if (self.global_step + 1) % self.parallel_config.grad_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
        return loss.item()
        
    def train(self, num_epochs: int):
        """Main training loop"""
        self.global_step = 0
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            num_steps = 0
            
            # Enable memory optimization if configured
            if self.parallel_config.optimize_memory:
                self.memory_optimizer.enable_gradient_checkpointing()
                if self.parallel_config.mixed_precision:
                    self.memory_optimizer.enable_mixed_precision(self.scaler)
                    
            # Training epoch
            for batch in self.train_dataloader:
                loss = self.train_step(batch)
                epoch_loss += loss
                num_steps += 1
                self.global_step += 1
                
            # Log metrics
            if self.is_main_process:
                logger.info(f"Epoch {epoch + 1}, Loss: {epoch_loss / num_steps:.4f}")
                
            # Synchronize processes
            if self.parallel_config.local_rank != -1:
                dist.barrier()
                
    def cleanup(self):
        """Cleanup distributed training"""
        if self.parallel_config.local_rank != -1:
            dist.destroy_process_group()
            
        if self.sp_session:
            self.sp_session.shutdown()

def run_parallel_training(
    rank: int,
    world_size: int,
    model_class: Type,
    model_config: Dict[str, Any],
    train_dataset: VishwamAIDataset,
    val_dataset: Optional[VishwamAIDataset] = None,
    parallel_config: Optional[ParallelConfig] = None,
    training_config: Optional[Dict[str, Any]] = None,
    num_epochs: int = 10
):
    """
    Run parallel training on a single process.
    """
    # Update parallel config with process info
    if parallel_config is None:
        parallel_config = ParallelConfig()
    parallel_config.local_rank = rank
    parallel_config.world_size = world_size
    
    # Initialize trainer
    trainer = ParallelTrainer(
        model_class=model_class,
        model_config=model_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        parallel_config=parallel_config,
        training_config=training_config
    )
    
    # Prepare and train
    trainer.prepare()
    trainer.train(num_epochs)
    trainer.cleanup()

def main():
    """Main function to start parallel training"""
    # Configuration
    model_config = {
        'vocab_size': 50000,
        'embed_dim': 768,
        'num_layers': 12,
        'num_heads': 12,
        'ff_dim': 3072
    }
    
    parallel_config = ParallelConfig(
        backend='nccl',
        world_size=torch.cuda.device_count(),
        model_parallel_size=2,
        pipeline_parallel_size=2,
        mixed_precision=True,
        optimize_memory=True
    )
    
    training_config = {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'weight_decay': 0.01
    }
    
    # Initialize datasets
    train_dataset = VishwamAIDataset(
        data_path='path/to/train.json',
        tokenizer=None,  # Add your tokenizer here
        mode='normal'
    )
    
    val_dataset = VishwamAIDataset(
        data_path='path/to/val.json',
        tokenizer=None,  # Add your tokenizer here
        mode='normal'
    )
    
    # GPU training
    mp.spawn(
        run_parallel_training,
        args=(
            parallel_config.world_size,
            VishwamAITransformer,
            model_config,
            train_dataset,
            val_dataset,
            parallel_config,
            training_config,
            10  # num_epochs
        ),
        nprocs=parallel_config.world_size
    )

if __name__ == "__main__":
    main()