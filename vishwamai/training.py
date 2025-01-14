import torch
import torch.nn.functional as F
from typing import List, Optional, Union, Dict
from dataclasses import dataclass
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.cuda.amp import autocast
import json
import os

from .architecture import VishwamaiV1
from vishwamai.parquet_handling import ParquetConfig, ParquetDataset, create_parquet_dataloader

@dataclass
class GenerationConfig:
    max_length: int = 2048
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    num_return_sequences: int = 1

class VishwamaiTokenizer:
    def __init__(self, vocab_file: str):
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)
        self.ids_to_tokens = {v: k for k, v in self.vocab.items()}
        self.pad_token_id = self.vocab.get("[PAD]", 0)
        self.eos_token_id = self.vocab.get("[EOS]", 1)
        self.bos_token_id = self.vocab.get("[BOS]", 2)
        
    def encode(self, text: str) -> List[int]:
        # Implement BPE tokenization here
        # This is a simplified version
        tokens = text.split()
        return [self.vocab.get(token, self.vocab["[UNK]"]) for token in tokens]
        
    def decode(self, token_ids: List[int]) -> str:
        return " ".join([self.ids_to_tokens.get(id, "[UNK]") for id in token_ids])

def select_device(preferred_device: str = "auto") -> str:
    """
    Checks for available devices (CPU, GPU, TPU, NPU, etc.) and 
    returns the most suitable one based on user preference or auto-detect.
    """
    # Pseudocode checking logic, expand or adjust as needed
    if preferred_device != "auto":
        return preferred_device
    
    if torch.cuda.is_available():
        return "cuda"
    # ...add checks for TPU/NPU/LPU/XPU if integrated...
    return "cpu"

def enable_distributed_training(model):
    """
    Sets up Distributed Data Parallel for multi-GPU training.
    """
    import torch.distributed as dist
    import torch
    from torch.nn.parallel import DistributedDataParallel as DDP

    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    model = DDP(model, device_ids=[torch.cuda.current_device()])
    return model

from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

class VishwamaiTrainer:
    def __init__(
        self,
        model,
        tokenizer,
        train_dataset,
        use_parquet: bool,
        parquet_config: Optional[ParquetConfig],
        eval_dataset=None,
        device=None,
        train_batch_size=16,  # Reduced from 32
        eval_batch_size=16,   # Reduced from 32
        gradient_accumulation_steps=2  # To simulate larger batch size
    ):
        chosen_device = select_device(device or "auto")
        self.model = model.to(chosen_device)
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.device = chosen_device
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=3e-5,  # Adjusted for smaller model
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        self.scaler = GradScaler()
        
        if use_parquet:
            if not isinstance(train_dataset, ParquetDataset):
                raise ValueError("When use_parquet=True, train_dataset must be ParquetDataset")
            self.train_loader = create_parquet_dataloader(
                train_dataset,
                parquet_config or ParquetConfig(),
                distributed=torch.distributed.is_initialized()
            )
            
            if self.eval_dataset:
                self.eval_loader = create_parquet_dataloader(
                    eval_dataset,
                    parquet_config or ParquetConfig(),
                    distributed=False
                )
        else:
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=self.train_batch_size,
                shuffle=True,
                num_workers=2,  # Optimized number of workers
                pin_memory=True
            )
            if self.eval_dataset:
                self.eval_loader = DataLoader(
                    eval_dataset,
                    batch_size=self.eval_batch_size,
                    shuffle=False,
                    num_workers=2,
                    pin_memory=True
                )
    
    def train(
        self,
        num_epochs: int,
        save_dir: str,
        evaluation_steps: int = 100,
        save_steps: int = 1000,
        logging_steps: int = 10
    ):
        self.model.train()
        global_step = 0
        total_loss = 0
        
        os.makedirs(save_dir, exist_ok=True)
        
        for epoch in range(num_epochs):
            for step, batch in enumerate(self.train_loader):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                with autocast():
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"]
                    )
                    loss = outputs.loss / self.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
                total_loss += loss.item()
                
                if (step + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    
                    global_step += 1
                    
                    if global_step % logging_steps == 0:
                        print(f"Step {global_step}: Average loss = {total_loss/logging_steps:.4f}")
                        total_loss = 0
                    
                    if global_step % evaluation_steps == 0 and self.eval_dataset is not None:
                        eval_loss = self.evaluate()
                        print(f"Step {global_step}: Evaluation loss = {eval_loss:.4f}")
                        self.model.train()
                    
                    if global_step % save_steps == 0:
                        self.save_model(os.path.join(save_dir, f"checkpoint-{global_step}"))
            
            # Save after each epoch
            self.save_model(os.path.join(save_dir, f"checkpoint-epoch-{epoch}"))
    
    def evaluate(self):
        self.model.eval()
        total_eval_loss = 0
        eval_steps = 0
        
        for batch in self.eval_loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"]
                )
                
                total_eval_loss += outputs.loss.item()
                eval_steps += 1
        
        return total_eval_loss / eval_steps
    
    def save_model(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        
        # Save model
        torch.save(self.model.state_dict(), os.path.join(output_dir, "model.pt"))
        
        # Save training args
        training_args = {
            "train_batch_size": self.train_batch_size,
            "eval_batch_size": self.eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps
        }
        
        with open(os.path.join(output_dir, "training_args.json"), "w") as f:
            json.dump(training_args, f)

class VishwamaiInference:
    def __init__(
        self,
        model,
        tokenizer,
        device="cuda",
        generation_config: Optional[GenerationConfig] = None
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.generation_config = generation_config or GenerationConfig()
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> List[str]:
        # Override generation config with provided parameters
        max_length = max_length or self.generation_config.max_length
        temperature = temperature or self.generation_config.temperature
        top_p = top_p or self.generation_config.top_p
        top_k = top_k or self.generation_config.top_k
        
        # Encode prompt
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
        attention_mask = torch.ones_like(input_ids)
        
        # Set model to eval mode
        self.model.eval()
        
        generated_sequences = []
        
        for _ in range(self.generation_config.num_return_sequences):
            current_input_ids = input_ids.clone()
            current_attention_mask = attention_mask.clone()
            
            while current_input_ids.shape[1] < max_length:
                outputs = self.model(
                    input_ids=current_input_ids,
                    attention_mask=current_attention_mask
                )
                next_token_logits = outputs.logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append next token
                current_input_ids = torch.cat([current_input_ids, next_token], dim=1)
                current_attention_mask = torch.cat([
                    current_attention_mask,
                    torch.ones((current_attention_mask.shape[0], 1), device=self.device)
                ], dim=1)
                
                # Check for EOS token
                if next_token[0, 0].item() == self.tokenizer.eos_token_id:
                    break
            
            # Decode generated sequence
            generated_sequence = self.tokenizer.decode(current_input_ids[0].tolist())
            generated_sequences.append(generated_sequence)
        
        return generated_sequences

def load_model_from_checkpoint(checkpoint_path: str, config, device="cuda"):
    model = VishwamaiV1(config)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device)

def train_model(model, optimizer, criterion, dataloader, device, dtype):
    model.train()
    for batch in dataloader:
        inputs, targets = batch
        inputs = inputs.to(device=device, dtype=dtype)
        targets = targets.to(device=device, dtype=torch.long)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    # ...existing code...

import os
import torch
from torch.utils.data import DataLoader
from vishwamai.parquet_handling import ParquetDataset, ParquetConfig
from vishwamai.model import VishwamaiModel, init_model
from vishwamai.conceptual_tokenizer import ConceptualTokenizer, ConceptualTokenizerConfig
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import Trainer, TrainingArguments

def load_data(config):
    train_data_path = config.get("train_data")
    val_data_path = config.get("val_data")
    
    if not os.path.isfile(train_data_path):
        raise FileNotFoundError(f"Training data file not found at path: {train_data_path}")
    
    if not os.path.isfile(val_data_path):
        raise FileNotFoundError(f"Validation data file not found at path: {val_data_path}")
    
    # ...existing data loading logic...

def train_model(train_data_path: str, val_data_path: str, config: dict, output_dir: str, epochs: int, batch_size: int):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize tokenizer
    tokenizer_config = ConceptualTokenizerConfig(
        vocab_size=config.get('vocab_size', 32000),
        max_length=config.get('max_length', 512),
        concept_tokens=["math", "logic", "science"],
        reasoning_tokens=["if", "then", "equals", "because"]
    )
    tokenizer = ConceptualTokenizer(tokenizer_config)
    
    # Initialize datasets
    train_config = ParquetConfig(chunk_size=config.get('chunk_size', 10000), batch_size=batch_size)
    val_config = ParquetConfig(chunk_size=config.get('chunk_size', 10000), batch_size=batch_size)
    
    train_dataset = ParquetDataset(
        parquet_path=train_data_path,
        config=train_config,
        tokenizer=tokenizer,
        max_length=tokenizer_config.max_length
    )
    
    val_dataset = ParquetDataset(
        parquet_path=val_data_path,
        config=val_config,
        tokenizer=tokenizer,
        max_length=tokenizer_config.max_length
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model_config = init_model(config)
    model = VishwamaiModel(model_config)
    model.train()
    
    # Setup optimizer and loss
    optimizer = Adam(model.parameters(), lr=config.get('learning_rate', 1e-4))
    criterion = CrossEntropyLoss()
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        evaluation_strategy="steps",
        save_steps=1000,
        eval_steps=100,
        logging_steps=10,
        learning_rate=config['training_config']['learning_rate'],
        weight_decay=config['training_config']['weight_decay'],
        warmup_steps=config['training_config']['warmup_steps'],
        fp16=True if config['model_config']['dtype'] == "float16" else False,
        push_to_hub=True,
        hub_model_id="YourHuggingFaceUsername/VishwamaiModel",
        hub_token=os.getenv("HF_TOKEN")
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_loader,
        eval_dataset=val_loader,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    trainer.push_to_hub()
    
    # Training loop
    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(config.get('device', 'cpu'))
            attention_mask = batch['attention_mask'].to(config.get('device', 'cpu'))
            labels = batch['input_ids'].to(config.get('device', 'cpu'))  # Assuming labels are the same as input_ids
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Average Training Loss: {avg_loss}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(config.get('device', 'cpu'))
                attention_mask = batch['attention_mask'].to(config.get('device', 'cpu'))
                labels = batch['input_ids'].to(config.get('device', 'cpu'))
                
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"Average Validation Loss: {avg_val_loss}")
        model.train()
        
        # Save model checkpoint
        checkpoint_path = os.path.join(output_dir, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Model saved to {checkpoint_path}")

    # Save the tokenizer after training
    tokenizer.save_tokenizer()
