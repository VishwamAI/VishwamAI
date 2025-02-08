import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from vishwamai.training import VishwamaiTrainer
from vishwamai.model import VishwamaiModel, VishwamaiConfig
from vishwamai.conceptual_tokenizer import ConceptualTokenizer, ConceptualTokenizerConfig

class DummyDataset(Dataset):
    def __init__(self, size=100, seq_length=32, device='cpu'):
        self.size = size
        self.seq_length = seq_length
        self.device = device

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        input_ids = torch.randint(0, 100, (self.seq_length,), device=self.device)
        attention_mask = torch.ones(self.seq_length, device=self.device)
        labels = torch.randint(0, 100, (self.seq_length,), device=self.device)
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

@pytest.fixture
def trainer_components():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = VishwamaiConfig(
        vocab_size=100,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # Added to match model config
        intermediate_size=512
    )
    model = VishwamaiModel(config, device=device)
    
    tokenizer_config = ConceptualTokenizerConfig(vocab_size=100)
    tokenizer = ConceptualTokenizer(tokenizer_config)
    
    # Ensure datasets use correct sequence length
    seq_length = 32
    train_dataset = DummyDataset(size=100, seq_length=seq_length, device=device)
    eval_dataset = DummyDataset(size=10, seq_length=seq_length, device=device)
    
    # Use smaller batch size and pin memory for GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        pin_memory=False if torch.cuda.is_available() else True
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=4,
        pin_memory=False if torch.cuda.is_available() else True
    )
    
    return model, tokenizer, train_loader, eval_loader, device

def test_trainer_initialization(trainer_components):
    model, tokenizer, train_loader, eval_loader, device = trainer_components
    trainer = VishwamaiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_loader,
        eval_dataset=eval_loader,
        device=device
    )
    
    assert trainer.model is not None
    assert trainer.tokenizer is not None
    assert trainer.train_dataset is not None
    assert trainer.eval_dataset is not None
    assert trainer.optimizer is not None
    assert trainer.scheduler is not None

def test_compute_loss(trainer_components):
    model, tokenizer, train_loader, _, device = trainer_components
    trainer = VishwamaiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_loader,
        device=device
    )
    
    batch = next(iter(train_loader))
    loss = trainer.compute_loss(batch)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # scalar loss
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

def test_train_step(trainer_components):
    model, tokenizer, train_loader, _, device = trainer_components
    trainer = VishwamaiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_loader,
        device=device
    )
    
    batch = next(iter(train_loader))
    loss = trainer.train_step(batch)
    
    assert isinstance(loss, float)
    assert not torch.isnan(torch.tensor(loss))
    assert not torch.isinf(torch.tensor(loss))

def test_evaluation(trainer_components):
    model, tokenizer, train_loader, eval_loader, device = trainer_components
    trainer = VishwamaiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_loader,
        eval_dataset=eval_loader,
        device=device
    )
    
    eval_results = trainer.evaluate()
    
    assert isinstance(eval_results, dict)
    assert "eval_loss" in eval_results
    assert isinstance(eval_results["eval_loss"], float)

def test_save_load_model(trainer_components, tmp_path):
    model, tokenizer, train_loader, _, device = trainer_components
    trainer = VishwamaiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_loader,
        device=device
    )
    
    # Save model
    save_path = tmp_path / "test_model"
    trainer.save_model(save_path)
    
    # Verify files exist
    assert (save_path / "model.pt").exists()
    assert (save_path / "training_state.pt").exists()

def test_training_loop(trainer_components, tmp_path):
    model, tokenizer, train_loader, eval_loader, device = trainer_components
    trainer = VishwamaiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_loader,
        eval_dataset=eval_loader,
        device=device
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
    model, tokenizer, train_loader, _, device = trainer_components
    trainer = VishwamaiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_loader,
        device=device
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
        
    model, tokenizer, train_loader, _, device = trainer_components
    trainer = VishwamaiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_loader,
        device="cuda"
    )
    
    # Set up GradScaler for FP16 training
    scaler = torch.amp.GradScaler('cuda')
    
    # Get batch and pin memory before CUDA transfer
    batch = next(iter(train_loader))
    
    # Test mixed precision training on device
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
