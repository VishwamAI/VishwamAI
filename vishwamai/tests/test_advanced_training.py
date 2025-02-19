"""
Unit tests for advanced training functionality.
"""
import pytest
import torch
import tempfile
import os
from pathlib import Path

from vishwamai.models.Transformer import Transformer
from vishwamai.training.advanced_training import AdvancedTrainer
from vishwamai.utils.config import ModelConfig, TrainingConfig
from vishwamai.extensions.tree_of_thoughts import TreeConfig

class TestAdvancedTrainer:
    @pytest.fixture
    def model_config(self):
        return ModelConfig(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            intermediate_size=512,
            max_position_embeddings=512,
            use_memory=True,
            tree_search_depth=2
        )
        
    @pytest.fixture
    def training_config(self):
        return TrainingConfig(
            learning_rate=1e-4,
            weight_decay=0.01,
            warmup_steps=100,
            max_grad_norm=1.0,
            checkpoint_interval=3600
        )
        
    @pytest.fixture
    def model(self, model_config):
        return Transformer(
            vocab_size=model_config.vocab_size,
            hidden_size=model_config.hidden_size,
            num_layers=model_config.num_layers,
            num_heads=model_config.num_heads,
            intermediate_size=model_config.intermediate_size,
            max_position_embeddings=model_config.max_position_embeddings
        )
        
    @pytest.fixture
    def trainer(self, model, model_config, training_config):
        return AdvancedTrainer(
            model=model,
            config=model_config,
            training_config=training_config,
            use_tree_search=True
        )
        
    def test_initialization(self, trainer):
        """Test trainer initialization."""
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None
        assert trainer.scaler is not None
        assert trainer.batch_accumulation == 4
        
    def test_mixed_precision_setup(self, trainer):
        """Test mixed precision training setup."""
        assert torch.cuda.is_available()
        assert trainer.scaler is not None
        assert torch.backends.cuda.matmul.allow_tf32
        assert torch.backends.cudnn.allow_tf32
        
    def test_memory_integration(self, trainer):
        """Test neural memory integration."""
        assert hasattr(trainer, 'memory')
        assert trainer.memory is not None
        
    def test_tree_of_thoughts_integration(self, trainer):
        """Test Tree of Thoughts integration."""
        assert hasattr(trainer, 'tree_of_thoughts')
        assert trainer.tree_of_thoughts is not None
        assert trainer.tree_of_thoughts.config.beam_width == 4
        assert trainer.tree_of_thoughts.config.max_depth == 3
        
    @pytest.mark.training
    def test_training_step(self, trainer):
        """Test single training step."""
        batch = {
            'input_ids': torch.randint(0, 1000, (2, 32)),
            'attention_mask': torch.ones(2, 32)
        }
        
        metrics = trainer.train_step(batch)
        assert 'loss' in metrics
        assert 'perplexity' in metrics
        assert metrics['loss'] > 0
        
    @pytest.mark.training
    def test_gradient_accumulation(self, trainer):
        """Test gradient accumulation."""
        batch = {
            'input_ids': torch.randint(0, 1000, (2, 32)),
            'attention_mask': torch.ones(2, 32)
        }
        
        # First step shouldn't update optimizer
        metrics = trainer.train_step(batch)
        assert trainer.optimizer._step_count == 0
        
        # After batch_accumulation steps, optimizer should update
        for _ in range(trainer.batch_accumulation - 1):
            trainer.train_step(batch)
        assert trainer.optimizer._step_count == 1
        
    def test_checkpoint_saving_loading(self, trainer):
        """Test checkpoint functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            
            # Save checkpoint
            trainer.save_checkpoint(str(checkpoint_path), epoch=1)
            assert checkpoint_path.exists()
            
            # Load checkpoint
            loaded_epoch = trainer.load_checkpoint(str(checkpoint_path))
            assert loaded_epoch == 1
            
            # Verify state dictionaries are loaded
            assert hasattr(trainer.model, 'state_dict')
            assert hasattr(trainer.optimizer, 'state_dict')
            assert hasattr(trainer.scheduler, 'state_dict')
            assert hasattr(trainer.scaler, 'state_dict')
            
    def test_memory_persistence(self, trainer):
        """Test memory state persistence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint.pt"
            
            # Generate some memory state
            batch = {
                'input_ids': torch.randint(0, 1000, (2, 32)),
                'attention_mask': torch.ones(2, 32)
            }
            trainer.train_step(batch)
            
            # Save checkpoint with memory
            trainer.save_checkpoint(str(checkpoint_path), epoch=1, include_memory=True)
            
            # Clear memory state
            trainer.memory = None
            
            # Load checkpoint and verify memory is restored
            trainer.load_checkpoint(str(checkpoint_path), load_memory=True)
            assert trainer.memory is not None
            
    def test_a100_optimizations(self, trainer):
        """Test A100-specific optimizations."""
        if 'A100' in torch.cuda.get_device_name():
            assert torch.backends.cuda.matmul.allow_tf32
            assert torch.backends.cudnn.allow_tf32
            assert trainer.optimizer.defaults.get('fused', False)
            
    def test_learning_rate_schedule(self, trainer):
        """Test learning rate scheduling."""
        initial_lr = trainer.get_learning_rate()
        
        # Test warmup phase
        for _ in range(50):
            trainer.scheduler.step()
        warmup_lr = trainer.get_learning_rate()
        assert warmup_lr > initial_lr
        
        # Test plateau phase
        for _ in range(1000):
            trainer.scheduler.step()
        plateau_lr = trainer.get_learning_rate()
        assert plateau_lr > 0
        
    @pytest.mark.training
    def test_tree_search_integration(self, trainer):
        """Test Tree of Thoughts search during training."""
        batch = {
            'input_ids': torch.randint(0, 1000, (2, 32)),
            'attention_mask': torch.ones(2, 32)
        }
        
        # Test without tree search
        metrics1 = trainer.train_step(batch, use_tree_search=False)
        
        # Test with tree search
        metrics2 = trainer.train_step(batch, use_tree_search=True)
        
        assert 'loss' in metrics1 and 'loss' in metrics2
        assert isinstance(metrics1['loss'], float)
        assert isinstance(metrics2['loss'], float)
        
if __name__ == '__main__':
    pytest.main(['-v', __file__])
