"""
Device-agnostic Chain of Thought (CoT) model for VishwamAI.
Supports both GPU (PyTorch) and TPU (JAX) execution with unified interface.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import math

try:
    import jax
    import jax.numpy as jnp
    from jax import random, grad, jit, vmap
    import flax.linen as flax_nn
    import optax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from vishwamai.models.transformer import VishwamAITransformer
from vishwamai.models.attention import DeviceAgnosticModule

def get_device_type():
    """Determine the available device type."""
    if torch.cuda.is_available():
        return "gpu"
    elif HAS_JAX and len(jax.devices("tpu")) > 0:
        return "tpu"
    return "cpu"

class CoTModel(DeviceAgnosticModule, nn.Module):
    """
    CoT model extending a transformer to generate reasoning steps and answers.
    """
    def __init__(self, embed_dim=512, num_layers=12, num_heads=8, ff_dim=2048, 
                 vocab_size=50000, max_seq_len=512, num_experts=7, force_device=None):
        """
        Initialize the CoT model.
        
        Args:
            embed_dim (int): Embedding dimension.
            num_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            ff_dim (int): Feed-forward dimension.
            vocab_size (int): Vocabulary size including special tokens.
            max_seq_len (int): Maximum sequence length.
            num_experts (int): Number of attention experts for MoE.
        """
        super(CoTModel, self).__init__()
        DeviceAgnosticModule.__init__(self)
        
        if force_device:
            self.device_type = force_device
        
        # Special tokens for CoT structure
        self.special_tokens = {
            "think_start": "<think>",
            "think_end": "</think>",
            "answer_start": "<answer>",
            "answer_end": "</answer>"
        }
        
        # Base transformer with device-specific configuration
        attention_kwargs = {"num_experts": num_experts, "taa_kwargs": {"k": 10, "kernel_dim": 256}}
        self.transformer = VishwamAITransformer(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            max_seq_len=max_seq_len,
            attention_kwargs=attention_kwargs,
            force_device=self.device_type
        )
        
        # Device-specific setup
        if self.device_type == "gpu":
            self.device = torch.device("cuda")
            self.to(self.device)
        elif self.device_type == "tpu" and HAS_JAX:
            # JAX/TPU-specific initialization if needed
            self.device = jax.devices("tpu")[0]
        else:
            self.device = torch.device("cpu")
            self.to(self.device)
            
        self.max_seq_len = max_seq_len

    def forward(self, input_ids, target_ids=None):
        # Convert inputs to appropriate device format
        input_ids = self.to_device(input_ids)
        if target_ids is not None:
            target_ids = self.to_device(target_ids)
        """
        Forward pass for training or inference.
        
        Args:
            input_ids (torch.Tensor): Input token IDs (batch_size, seq_len).
            target_ids (torch.Tensor, optional): Target token IDs for training (batch_size, seq_len).
        
        Returns:
            torch.Tensor: Logits (batch_size, seq_len, vocab_size) or loss if target_ids provided.
        """
        if target_ids is not None:
            # Shift targets for teacher forcing
            input_to_transformer = torch.cat([input_ids, target_ids[:, :-1]], dim=1)
            logits = self.transformer(input_to_transformer)
            # Compute loss over the target sequence
            loss = F.cross_entropy(
                logits[:, input_ids.size(1)-1:, :].reshape(-1, self.transformer.vocab_size),
                target_ids[:, 1:].reshape(-1),
                ignore_index=-1  # Padding token
            )
            return loss
        else:
            return self.transformer(input_ids)

    def generate_cot(self, input_text, tokenizer, max_length=512, temperature=0.6, top_p=0.95):
        """
        Generate a CoT response with reasoning and answer.
        
        Args:
            input_text (str): Input prompt (e.g., math problem).
            tokenizer: Tokenizer with special tokens added.
            max_length (int): Maximum generation length.
            temperature (float): Sampling temperature.
            top_p (float): Top-p sampling threshold.
        
        Returns:
            str: Generated CoT response.
        """
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(self.device)
        generated_ids = self._sample(input_ids, max_length, temperature, top_p)
        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        return output_text

    def _sample(self, input_ids, max_length, temperature, top_p):
        # Convert input to appropriate device format
        input_ids = self.to_device(input_ids)
        """
        Sample tokens with temperature and top-p filtering.
        """
        generated = input_ids
        end_answer_id = None
        
        # Try to get the answer_end token ID from the transformer's tokenizer
        if hasattr(self.transformer, 'tokenizer'):
            end_answer_id = self.transformer.tokenizer.encode(self.special_tokens["answer_end"])[0]
        else:
            # Fallback to an estimated token ID near the vocab size
            end_answer_id = self.transformer.vocab_size - 1
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                if generated.size(1) >= max_length:
                    break
                    
                # Forward pass through the model
                logits = self.transformer(generated)
                next_logits = logits[:, -1, :] / temperature
                
                # Top-p filtering
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[:, indices_to_remove] = float('-inf')
                
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                
                if next_token.item() == end_answer_id:
                    break
        
        return generated

def extract_answer(output_text):
    """
    Extract the answer from the CoT output.
    
    Args:
        output_text (str): Generated text with <think> and <answer> tags.
    
    Returns:
        str: Content between <answer> and </answer>, or error message.
    """
    start = output_text.find("<answer>") + len("<answer>")
    end = output_text.find("</answer>")
    if start != -1 and end != -1 and start < end:
        return output_text[start:end].strip()
    return "Answer not found"

# Training function
def train_cot_model(model, dataloader, optimizer, num_epochs):
    """
    Train the CoT model with supervised learning.
    
    Args:
        model (CoTModel): The CoT model instance.
        dataloader: DataLoader with (input_ids, target_ids) pairs.
        optimizer: Optimizer (e.g., Adam).
        num_epochs (int): Number of training epochs.
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            # Handle device-specific training
            if model.device_type == "gpu":
                input_ids, target_ids = input_ids.to(model.device), target_ids.to(model.device)
                optimizer.zero_grad()
                loss = model(input_ids, target_ids)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            elif model.device_type == "tpu" and HAS_JAX:
                # TPU training using JAX
                input_ids = jnp.array(input_ids.numpy())
                target_ids = jnp.array(target_ids.numpy())
                
                @jit
                def update_step(params, inputs, targets):
                    def loss_fn(params):
                        logits = model.apply({'params': params}, inputs)
                        return jnp.mean(
                            optax.softmax_cross_entropy_with_integer_labels(
                                logits[:, model.transformer.embed_dim:],
                                targets
                            )
                        )
                    loss, grads = jax.value_and_grad(loss_fn)(params)
                    params = jax.tree_map(
                        lambda p, g: p - 0.01 * g,  # Simple SGD for example
                        params,
                        grads
                    )
                    return params, loss
                
                model.params, loss = update_step(model.params, input_ids, target_ids)
                total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

# Example usage
if __name__ == "__main__":
    # Mock tokenizer (replace with actual VishwamAI tokenizer)
    class MockTokenizer:
        def __init__(self, vocab_size=50000):
            self.vocab_size = vocab_size
            self.special_tokens = {
                "<think>": vocab_size-4, "</think>": vocab_size-3,
                "<answer>": vocab_size-2, "</answer>": vocab_size-1
            }
            self.inverse_vocab = {v: k for k, v in self.special_tokens.items()}
            self.inverse_vocab.update({i: f"token_{i}" for i in range(vocab_size-4)})

        def encode(self, text, return_tensors="pt"):
            tokens = [self.special_tokens.get(text, i) for i in range(5)]  # Simplified
            if return_tensors == "pt":
                return torch.tensor([tokens], dtype=torch.long)
            return tokens

        def decode(self, token_ids, skip_special_tokens=False):
            text = ""
            for token in token_ids:
                if token.item() in self.inverse_vocab:
                    if not skip_special_tokens or token.item() < self.vocab_size-4:
                        text += self.inverse_vocab[token.item()] + " "
            return text.strip()

    # Initialize model and tokenizer
    tokenizer = MockTokenizer()
    model = CoTModel(vocab_size=tokenizer.vocab_size)
    
    # Mock dataset (replace with actual data)
    from torch.utils.data import DataLoader, TensorDataset
    input_ids = torch.randint(0, tokenizer.vocab_size-4, (10, 20))
    target_ids = torch.cat([
        torch.tensor([[tokenizer.special_tokens["<think>"]]] * 10),
        torch.randint(0, tokenizer.vocab_size-4, (10, 18)),
        torch.tensor([[tokenizer.special_tokens["</think>"]]] * 10),
        torch.tensor([[tokenizer.special_tokens["<answer>"]]] * 10),
        torch.randint(0, tokenizer.vocab_size-4, (10, 5)),
        torch.tensor([[tokenizer.special_tokens["</answer>"]]] * 10)
    ], dim=1)
    dataset = TensorDataset(input_ids, target_ids)
    dataloader = DataLoader(dataset, batch_size=2)
    
    # Train the model
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_cot_model(model, dataloader, optimizer, num_epochs=3, device=model.device)
    
    # Generate a sample CoT
    input_text = "Solve 2x + 3 = 7"
    output = model.generate_cot(input_text, tokenizer)
    print("Generated CoT:", output)
    print("Extracted Answer:", extract_answer(output))
