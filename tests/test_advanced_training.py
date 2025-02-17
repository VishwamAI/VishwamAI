import torch
import pytest
from vishwamai.advanced_training import AdvancedTrainer
from vishwamai.neural_memory import NeuralMemory
from vishwamai.cache_augmentation import CacheAugmentation

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(768, 768)
        
    def forward(self, input_ids, attention_mask=None, labels=None, 
                output_attentions=False, output_hidden_states=False):
        outputs = self.linear(input_ids)
        
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
    model = DummyModel()
    config = {
        'gradient_accumulation_steps': 2,
        'mixed_precision': True,
        'tree_of_thoughts_depth': 2,
        'memory_lr': 1e-4,
        'num_training_steps': 1000,
        'num_warmup_steps': 100,
        'max_grad_norm': 1.0
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return AdvancedTrainer(model, config, device)

def test_trainer_initialization(trainer):
    assert trainer.gradient_accumulation_steps == 2
    assert trainer.mixed_precision == True
    assert trainer.tree_of_thoughts_depth == 2
    assert isinstance(trainer.memory, NeuralMemory)
    assert isinstance(trainer.cache, CacheAugmentation)

def test_deep_calculation_step(trainer):
    batch_size = 4
    seq_len = 32
    hidden_dim = 768
    
    input_ids = torch.randn(batch_size, seq_len, hidden_dim)
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randn(batch_size, seq_len, hidden_dim)
    
    loss, logits = trainer.deep_calculation_step(
        input_ids,
        attention_mask,
        labels
    )
    
    assert isinstance(loss, torch.Tensor)
    assert loss.ndim == 0  # scalar
    assert logits.size() == (batch_size, seq_len, hidden_dim)

def test_train_step(trainer):
    batch_size = 4
    seq_len = 32
    hidden_dim = 768
    
    batch = {
        'input_ids': torch.randn(batch_size, seq_len, hidden_dim),
        'attention_mask': torch.ones(batch_size, seq_len),
        'labels': torch.randn(batch_size, seq_len, hidden_dim)
    }
    
    output = trainer.train_step(batch)
    
    assert 'loss' in output
    assert isinstance(output['loss'], float)
    assert 'logits' in output
    assert output['logits'].size() == (batch_size, seq_len, hidden_dim)
    assert 'lr' in output
    assert isinstance(output['lr'], float)

def test_memory_integration(trainer):
    memory = trainer.memory
    
    # Test memory forward pass
    input_tensor = torch.randn(4, 32, 768)
    output = memory(input_tensor)
    
    assert output.size() == input_tensor.size()
    
    # Test memory state management
    keys, values = memory.get_memory_state()
    assert keys.size() == (1024, 768)  # memory_size x hidden_dim
    assert values.size() == (1024, 768)
    
    memory.reset_memory()
    new_keys, new_values = memory.get_memory_state()
    assert not torch.allclose(keys, new_keys)
    assert not torch.allclose(values, new_values)

def test_cache_integration(trainer):
    cache = trainer.cache
    
    # Test cache forward pass
    input_tensor = torch.randn(4, 32, 768)
    output = cache(input_tensor)
    
    assert output.size() == input_tensor.size()
    
    # Test cache retrieval
    query = torch.randn(4, 768)
    retrieved = cache.retrieve(query)
    
    if retrieved is not None:
        assert retrieved.size() == (4, 768)
    
    # Test cache statistics
    stats = cache.get_cache_stats()
    assert 'average_age' in stats
    assert 'max_age' in stats
    assert 'unused_entries' in stats
    assert 'total_entries' in stats

def test_checkpoint_management(trainer, tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    
    # Save checkpoint
    trainer.save_checkpoint(str(checkpoint_path))
    assert checkpoint_path.exists()
    
    # Modify some parameters
    original_lr = trainer.scheduler.get_last_lr()[0]
    for _ in range(10):
        trainer.scheduler.step()
    
    # Load checkpoint
    trainer.load_checkpoint(str(checkpoint_path))
    loaded_lr = trainer.scheduler.get_last_lr()[0]
    
    assert original_lr == loaded_lr

if __name__ == '__main__':
    pytest.main([__file__])
