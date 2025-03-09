"""
Chain of Thought (CoT) model with device-agnostic implementation.
"""
from typing import Optional, Dict, Any
import torch
import torch.nn.functional as F
from jax import jit
from vishwamai.models.transformer import VishwamAITransformer, DeviceAgnosticModule
try:
    import jax
    import jax.numpy as jnp
    from jax import random
    import flax.linen as flax_nn
    import optax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

class CoTModel(DeviceAgnosticModule):
    """Chain of Thought model with hardware-specific optimizations."""
    def __init__(
        self,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        ff_dim: int = 2048,
        vocab_size: int = 50000,
        max_seq_len: int = 512,
        num_experts: int = 7,
        force_device: str = None
    ):
        super().__init__()
        if force_device:
            self.device_type = force_device
            
        # Special tokens
        self.special_tokens = {
            "think_start": "<think>",
            "think_end": "</think>",
            "answer_start": "<answer>",
            "answer_end": "</answer>"
        }
        
        # Initialize transformer with device-specific optimizations
        attention_kwargs = {
            "num_experts": num_experts,
            "taa_kwargs": {"k": 10, "kernel_dim": 256}
        }
        
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
        
    def __call__(self, input_ids, target_ids=None, training=False):
        """Forward pass with device-specific optimizations."""
        if target_ids is not None:
            # Training mode
            if self.device_type == "tpu" and HAS_JAX:
                # Concatenate for teacher forcing
                input_to_transformer = jnp.concatenate(
                    [input_ids, target_ids[:, :-1]], axis=1
                )
                logits = self.transformer(input_to_transformer, training=True)
                
                # Compute loss over target sequence
                L = input_ids.shape[1]
                loss_logits = logits[:, L-1:, :]
                loss = optax.softmax_cross_entropy_with_integer_labels(
                    loss_logits, target_ids[:, 1:]
                )
                return jnp.mean(loss)
            else:
                import torch
                import torch.nn.functional as F
                
                # PyTorch implementation
                input_to_transformer = torch.cat(
                    [input_ids, target_ids[:, :-1]], dim=1
                )
                logits = self.transformer(input_to_transformer, training=True)
                
                L = input_ids.size(1)
                loss = F.cross_entropy(
                    logits[:, L-1:, :].reshape(-1, self.transformer.vocab_size),
                    target_ids[:, 1:].reshape(-1),
                    ignore_index=-1
                )
                return loss
        else:
            # Inference mode
            return self.transformer(input_ids, training=False)
            
    def generate_cot(self, input_text, tokenizer, max_length=512, 
                    temperature=0.6, top_p=0.95, rng=None):
        """Generate CoT response with unified device handling."""
        if self.device_type == "tpu" and HAS_JAX:
            if rng is None:
                rng = random.PRNGKey(0)
                
            input_ids = tokenizer.encode(input_text, return_tensors="jax")
            generated_ids = self._sample_tpu(
                input_ids, max_length, temperature, top_p, rng
            )
        else:
            import torch
            if rng is not None:
                torch.manual_seed(rng)
                
            input_ids = tokenizer.encode(
                input_text, return_tensors="pt"
            ).to(self.transformer.device)
            generated_ids = self._sample_gpu(
                input_ids, max_length, temperature, top_p
            )
            
        return tokenizer.decode(generated_ids[0], skip_special_tokens=False)
        
    def _sample_tpu(self, input_ids, max_length, temperature, top_p, rng):
        """TPU-optimized sampling."""
        generated = input_ids
        end_answer_id = self.transformer.vocab_size - 1
        
        def body_fn(state):
            i, generated, rng = state
            logits = self.transformer(generated, training=False)
            next_logits = logits[:, -1, :] / temperature
            
            # Top-p filtering
            sorted_logits = jnp.sort(next_logits, axis=-1)[::-1]
            cumulative_probs = jnp.cumsum(
                jax.nn.softmax(sorted_logits, axis=-1), axis=-1
            )
            sorted_mask = cumulative_probs <= top_p
            sorted_mask = sorted_mask.at[0].set(True)
            
            next_logits = jnp.where(
                sorted_mask, next_logits, jnp.full_like(next_logits, -1e10)
            )
            
            # Sample
            rng, sampling_rng = random.split(rng)
            next_token = random.categorical(
                sampling_rng, next_logits, shape=(1,)
            )
            generated = jnp.concatenate([generated, next_token[:, None]], axis=1)
            
            return i + 1, generated, rng
            
        def cond_fn(state):
            i, generated, _ = state
            if i >= max_length - input_ids.shape[1]:
                return False
            last_token = generated[:, -1]
            return jnp.logical_not(jnp.any(last_token == end_answer_id))
            
        _, generated, _ = jax.lax.while_loop(
            cond_fn, body_fn, (0, generated, rng)
        )
        return generated
        
    def _sample_gpu(self, input_ids, max_length, temperature, top_p):
        """GPU-optimized sampling."""
        import torch
        import torch.nn.functional as F
        
        generated = input_ids
        end_answer_id = self.transformer.vocab_size - 1
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                if generated.size(1) >= max_length:
                    break
                    
                logits = self.transformer(generated)
                next_logits = logits[:, -1, :] / temperature
                
                # Top-p filtering
                sorted_logits, sorted_indices = torch.sort(
                    next_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_logits[:, indices_to_remove] = float('-inf')
                
                # Sample
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                
                if next_token.item() == end_answer_id:
                    break
                    
        return generated

def extract_answer(output_text: str) -> str:
    """Extract answer from CoT output."""
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
    train_cot_model(model, dataloader, optimizer, num_epochs=3)
    
    # Generate a sample CoT
    input_text = "Solve 2x + 3 = 7"
    output = model.generate_cot(input_text, tokenizer)
    print("Generated CoT:", output)
    print("Extracted Answer:", extract_answer(output))
