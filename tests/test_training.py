import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from vishwamai.training import VishwamaiTrainer
from vishwamai.model import VishwamaiModel, VishwamaiConfig
from vishwamai.conceptual_tokenizer import ConceptualTokenizer, ConceptualTokenizerConfig

class DummyDataset(Dataset):
    def __init__(self, size=100, seq_length=32):
        self.size = size
        self.seq_length = seq_length
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        return {
            'input_ids': torch.randint(0, 100, (self.seq_length,)),
            'attention_mask': torch.ones(self.seq_length),
            'labels': torch.randint(0, 100, (self.seq_length,))
        }

@pytest.fixture
def trainer_components():
    config = VishwamaiConfig(
        vocab_size=100,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4
    )
    model = VishwamaiModel(config)
    
    tokenizer_config = ConceptualTokenizerConfig(vocab_size=100)
    tokenizer = ConceptualTokenizer(tokenizer_config)
    
    train_dataset = DummyDataset()
    eval_dataset = DummyDataset(size=10)
    
    train_loader = DataLoader(train_dataset, batch_size=4)
    eval_loader = DataLoader(eval_dataset, batch_size=4)
    
    return model, tokenizer, train_loader, eval_loader

def test_trainer_initialization(trainer_components):
    model, tokenizer, train_loader, eval_loader = trainer_components
    trainer = VishwamaiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_loader,
        eval_dataset=eval_loader,
        device="cpu"
    )
    
    assert trainer.model is not None
    assert trainer.tokenizer is not None
    assert trainer.train_dataset is not None
    assert trainer.eval_dataset is not None
    assert trainer.optimizer is not None
    assert trainer.scheduler is not None

def test_compute_loss(trainer_components):
    model, tokenizer, train_loader, _ = trainer_components
    trainer = VishwamaiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_loader,
        device="cpu"
    )
    
    batch = next(iter(train_loader))
    loss = trainer.compute_loss(batch)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar loss
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

def test_train_step(trainer_components):
    model, tokenizer, train_loader, _ = trainer_components
    trainer = VishwamaiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_loader,
        device="cpu"
    )
    
    batch = next(iter(train_loader))
    loss = trainer.train_step(batch)
    
    assert isinstance(loss, float)
    assert not torch.isnan(torch.tensor(loss))
    assert not torch.isinf(torch.tensor(loss))

def test_evaluation(trainer_components):
    model, tokenizer, train_loader, eval_loader = trainer_components
    trainer = VishwamaiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_loader,
        eval_dataset=eval_loader,
        device="cpu"
    )
    
    eval_results = trainer.evaluate()
    
    assert isinstance(eval_results, dict)
    assert "eval_loss" in eval_results
    assert isinstance(eval_results["eval_loss"], float)

def test_save_load_model(trainer_components, tmp_path):
    model, tokenizer, train_loader, _ = trainer_components
    trainer = VishwamaiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_loader,
        device="cpu"
    )
    
    # Save model
    save_path = tmp_path / "test_model"
    trainer.save_model(save_path)
    
    # Verify files exist
    assert (save_path / "model.pt").exists()
    assert (save_path / "training_state.pt").exists()

def test_training_loop(trainer_components, tmp_path):
    model, tokenizer, train_loader, eval_loader = trainer_components
    trainer = VishwamaiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_loader,
        eval_dataset=eval_loader,
        device="cpu"
    )
    
    # Run short training loop
    trainer.train(
        num_epochs=1,
        save_dir=tmp_path / "test_training",
        evaluation_steps=5,
        save_steps=10,
        logging_steps=2
    )
    
    # Verify training artifacts
    assert (tmp_path / "test_training" / "final_model").exists()
    assert (tmp_path / "test_training" / "final_model" / "model.pt").exists()

def test_gradient_accumulation(trainer_components):
    model, tokenizer, train_loader, _ = trainer_components
    trainer = VishwamaiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_loader,
        device="cpu"
    )
    
    # Get initial parameters
    initial_params = [param.clone() for param in model.parameters()]
    
    # Train with gradient accumulation
    batch = next(iter(train_loader))
    for _ in range(4):  # Accumulate gradients 4 times
        trainer.compute_loss(batch).backward()
    
    trainer.optimizer.step()
    trainer.optimizer.zero_grad()
    
    # Verify parameters have been updated
    current_params = [param.clone() for param in model.parameters()]
    assert any(not torch.equal(p1, p2) for p1, p2 in zip(initial_params, current_params))

def test_fp16_training(trainer_components):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    model, tokenizer, train_loader, _ = trainer_components
    trainer = VishwamaiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_loader,
        device="cuda"
    )
    
    # Set up GradScaler for FP16 training
    scaler = torch.amp.GradScaler('cuda')
    
    # Verify FP16 training works
    batch = next(iter(train_loader))
    batch = {k: v.cuda() for k, v in batch.items()}
    
    # Test mixed precision training
    with torch.amp.autocast('cuda'):
        loss = trainer.compute_loss(batch)
    
    # Test gradient scaling
    scaler.scale(loss).backward()
    scaler.unscale_(trainer.optimizer)
    scaler.step(trainer.optimizer)
    scaler.update()
    
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    
    # Verify optimizer state is maintained
    assert trainer.optimizer.state_dict()['param_groups'][0]['lr'] > 0
