import torch
import pytest
import numpy as np
from vishwamai.advanced_training import AdvancedTrainer
from vishwamai.neural_memory import NeuralMemory
from vishwamai.cache_augmentation import CacheAugmentation
from vishwamai.MoE import MoEConfig, create_moe_layer

class DummyModel(torch.nn.Module):
    def __init__(self, moe_enabled: bool = True):
        super().__init__()
        self.linear = torch.nn.Linear(768, 768)
        if moe_enabled:
            self.moe_layer = create_moe_layer(
                input_dim=768,
                parallel=False,
                num_experts=4,
                expert_dim=1024,
                num_experts_per_token=2
            )
        self.moe_enabled = moe_enabled
        
    def forward(self, input_ids, attention_mask=None, labels=None, 
                output_attentions=False, output_hidden_states=False):
        outputs = self.linear(input_ids)
        if self.moe_enabled:
            moe_out, aux_info = self.moe_layer(outputs)
            self.moe_loss = aux_info.get('aux_loss', None)
            outputs = moe_out
        
        class ModelOutput:
            def __init__(self, logits, loss, hidden_states, attentions):
                self.logits = logits
                self.loss = loss
                self.hidden_states = hidden_states
                self.attentions = attentions
                
        if labels is not None:
            loss = torch.nn.functional.mse_loss(outputs, labels)
        else:
            loss = None
            
        hidden_states = [outputs] * 3 if output_hidden_states else None
        attentions = torch.ones(1, 8, 32, 32) if output_attentions else None
        
        return ModelOutput(outputs, loss, hidden_states, attentions)

@pytest.fixture
def trainer():
    model = DummyModel(moe_enabled=True)
    config = {
        'gradient_accumulation_steps': 2,
        'mixed_precision': True,
        'gradient_checkpointing': True,
        'tree_of_thoughts_depth': 2,
        'memory_lr': 1e-4,
        'num_training_steps': 1000,
        'num_warmup_steps': 100,
        'max_grad_norm': 1.0,
        'moe_aux_loss_weight': 0.01,
        'expert_load_balance_weight': 0.01,
        'hidden_dim': 768,
        'num_heads': 8,
        'memory_sparsity': 0.9
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return AdvancedTrainer(model, config, device)

def test_trainer_initialization(trainer):
    assert trainer.gradient_accumulation_steps == 2
    assert trainer.mixed_precision == True
    assert trainer.gradient_checkpointing == True
    assert trainer.tree_of_thoughts_depth == 2
    assert isinstance(trainer.memory, NeuralMemory)
    assert isinstance(trainer.cache, CacheAugmentation)
    assert trainer.moe_aux_loss_weight == 0.01
    assert trainer.expert_load_balance_weight == 0.01

def test_moe_forward_pass(trainer):
    batch_size = 4
    seq_len = 32
    hidden_dim = 768
    
    input_ids = torch.randn(batch_size, seq_len, hidden_dim)
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randn(batch_size, seq_len, hidden_dim)
    
    # Run forward pass
    loss, logits = trainer.deep_calculation_step(
        input_ids,
        attention_mask,
        labels
    )
    
    # Check outputs
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
    assert logits.size() == (batch_size, seq_len, hidden_dim)
    
    # Verify MoE loss exists
    assert hasattr(trainer.model, 'moe_loss')
    assert trainer.model.moe_loss is not None

def test_expert_load_balancing(trainer):
    batch_size = 4
    seq_len = 32
    hidden_dim = 768
    
    # Run several batches
    for _ in range(5):
        batch = {
            'input_ids': torch.randn(batch_size, seq_len, hidden_dim),
            'attention_mask': torch.ones(batch_size, seq_len),
            'labels': torch.randn(batch_size, seq_len, hidden_dim)
        }
        stats = trainer.train_step(batch)
        
        # Check MoE metrics
        assert 'moe_metrics' in stats
        expert_metrics = stats['moe_metrics']
        for layer_metrics in expert_metrics.values():
            # Check expert utilization metrics
            assert 'usage_cv' in layer_metrics
            assert 'max_usage' in layer_metrics
            assert 'min_usage' in layer_metrics
            
            # Verify load balancing is working
            assert layer_metrics['usage_cv'] < 1.0  # Coefficient of variation should be reasonable

def test_expert_parallelism():
    model = DummyModel(moe_enabled=True)
    config = {
        'gradient_accumulation_steps': 2,
        'mixed_precision': True,
        'tree_of_thoughts_depth': 2,
        'memory_lr': 1e-4,
        'moe_aux_loss_weight': 0.01,
        'expert_load_balance_weight': 0.01
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create trainer with parallel experts
    config['use_parallel_moe'] = True
    trainer = AdvancedTrainer(model, config, device)
    
    # Check that experts are configured for parallel execution
    assert hasattr(trainer.model.moe_layer, 'parallel_experts')
    assert trainer.model.moe_layer.parallel_experts == True

def test_expert_metrics_collection(trainer):
    batch_size = 4
    seq_len = 32
    hidden_dim = 768
    
    batch = {
        'input_ids': torch.randn(batch_size, seq_len, hidden_dim),
        'attention_mask': torch.ones(batch_size, seq_len),
        'labels': torch.randn(batch_size, seq_len, hidden_dim)
    }
    
    # Run training step
    stats = trainer.train_step(batch)
    
    # Check expert usage history
    assert hasattr(trainer, 'expert_usage_history')
    assert len(trainer.expert_usage_history) > 0
    
    # Verify metrics
    assert 'moe_metrics' in stats
    expert_metrics = stats['moe_metrics']
    assert len(expert_metrics) > 0
    
    # Check specific metrics
    for layer_metrics in expert_metrics.values():
        assert 0 <= layer_metrics['min_usage'] <= layer_metrics['max_usage'] <= 1.0
        assert 0 <= layer_metrics['usage_cv'] <= 2.0  # Reasonable CV range

def test_checkpoint_with_moe(trainer, tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    
    # Save checkpoint
    trainer.save_checkpoint(str(checkpoint_path))
    assert checkpoint_path.exists()
    
    # Save initial state
    original_lr = trainer.scheduler.get_last_lr()[0]
    original_batch_size = trainer.current_batch_size
    original_expert_history = trainer.expert_usage_history.copy()
    
    # Modify training state
    for _ in range(10):
        trainer._update_batch_size(0.5)  # Simulate loss updates
        trainer.scheduler.step()
        trainer.expert_usage_history.append({'test': 1.0})
    
    # Load checkpoint
    trainer.load_checkpoint(str(checkpoint_path))
    loaded_lr = trainer.scheduler.get_last_lr()[0]
    
    # Verify state restoration
    assert original_lr == loaded_lr
    assert original_batch_size == trainer.current_batch_size
    assert trainer.expert_usage_history == original_expert_history

def test_train_step_with_moe(trainer):
    batch_size = 4
    seq_len = 32
    hidden_dim = 768
    
    batch = {
        'input_ids': torch.randn(batch_size, seq_len, hidden_dim),
        'attention_mask': torch.ones(batch_size, seq_len),
        'labels': torch.randn(batch_size, seq_len, hidden_dim)
    }
    
    output = trainer.train_step(batch)
    
    # Check outputs
    assert 'loss' in output
    assert isinstance(output['loss'], float)
    assert 'logits' in output
    assert output['logits'].size() == (batch_size, seq_len, hidden_dim)
    assert 'lr' in output
    assert isinstance(output['lr'], float)
    
    # Check MoE specific outputs
    assert 'moe_metrics' in output
    assert 'moe_loss' in output
    assert isinstance(output['moe_loss'], float)

if __name__ == '__main__':
    pytest.main([__file__])
