"""
Chain of Thought (CoT) model for VishwamAI, designed to generate reasoning steps before answers.
Inspired by DeepSeek-R1, outputs are structured with <think> and <answer> tags.
Supports deep calculations for tasks like mathematics and coding.
Updated for JAX/Flax/Optax, optimized for TPUs.
"""

import jax
import jax.numpy as jnp
from jax import random, jit
import flax.linen as nn
import optax
from typing import Optional, Tuple

from vishwamai.models.transformer import VishwamAITransformer

class CoTModel(nn.Module):
    """
    CoT model extending a transformer to generate reasoning steps and answers.
    """
    embed_dim: int = 512
    num_layers: int = 12
    num_heads: int = 8
    ff_dim: int = 2048
    vocab_size: int = 50000
    max_seq_len: int = 512
    num_experts: int = 7

    def setup(self):
        """
        Initialize the CoT model with a VishwamAI Transformer.
        """
        # Special tokens for CoT structure (will be mapped to IDs by tokenizer)
        self.special_tokens = {
            "think_start": "<think>",
            "think_end": "</think>",
            "answer_start": "<answer>",
            "answer_end": "</answer>"
        }

        # Base transformer with enhanced VishwamAI architecture
        attention_kwargs = {"num_experts": self.num_experts, "taa_kwargs": {"k": 10, "kernel_dim": 256}}
        self.transformer = VishwamAITransformer(
            vocab_size=self.vocab_size,
            embed_dim=self.embed_dim,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            ff_dim=self.ff_dim,
            max_seq_len=self.max_seq_len,
            attention_kwargs=attention_kwargs
        )

    @nn.compact
    def __call__(self, input_ids: jnp.ndarray, target_ids: Optional[jnp.ndarray] = None, train: bool = False) -> jnp.ndarray:
        """
        Forward pass for training or inference.
        
        Args:
            input_ids: Input token IDs (batch_size, seq_len).
            target_ids: Target token IDs for training (batch_size, seq_len).
            train: Whether in training mode.
        
        Returns:
            Logits (batch_size, seq_len, vocab_size) or loss if target_ids provided.
        """
        if target_ids is not None:
            # Concatenate input and shifted targets for teacher forcing
            input_to_transformer = jnp.concatenate([input_ids, target_ids[:, :-1]], axis=1)
            logits = self.transformer(input_to_transformer, train=train)
            # Compute loss over the target sequence
            L = input_ids.shape[1]
            loss_logits = logits[:, L-1:, :]  # Predict target positions
            loss = optax.softmax_cross_entropy_with_integer_labels(loss_logits, target_ids[:, 1:])
            return jnp.mean(loss)
        else:
            return self.transformer(input_ids, train=train)

def generate_cot(model, params, input_text: str, tokenizer, max_length: int = 512, 
                 temperature: float = 0.6, top_p: float = 0.95, rng: jnp.ndarray = None) -> str:
    """
    Generate a CoT response with reasoning and answer.
    
    Args:
        model: CoTModel instance.
        params: Model parameters.
        input_text: Input prompt (e.g., math problem).
        tokenizer: Tokenizer with special tokens added.
        max_length: Maximum generation length.
        temperature: Sampling temperature.
        top_p: Top-p sampling threshold.
        rng: Random number generator key.
    
    Returns:
        Generated CoT response as a string.
    """
    if rng is None:
        rng = random.PRNGKey(0)
    
    input_ids = tokenizer.encode(input_text, return_tensors="jax")
    generated_ids = sample(model, params, input_ids, tokenizer, max_length, temperature, top_p, rng)
    output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=False)
    return output_text

def sample(model, params, input_ids: jnp.ndarray, tokenizer, max_length: int, 
           temperature: float, top_p: float, rng: jnp.ndarray) -> jnp.ndarray:
    """
    Sample tokens with temperature and top-p filtering.
    
    Args:
        model: CoTModel instance.
        params: Model parameters.
        input_ids: Input token IDs (batch_size, seq_len).
        tokenizer: Tokenizer instance.
        max_length: Maximum generation length.
        temperature: Sampling temperature.
        top_p: Top-p sampling threshold.
        rng: Random number generator key.
    
    Returns:
        Generated token IDs (batch_size, generated_seq_len).
    """
    generated = input_ids
    end_answer_id = tokenizer.special_tokens["</answer>"]

    def body_fn(val):
        i, generated, rng = val
        if i >= max_length - input_ids.shape[1]:
            return i + 1, generated, rng

        logits = model.apply({'params': params}, generated, train=False)
        next_logits = logits[:, -1, :] / temperature

        # Top-p filtering
        probs = jax.nn.softmax(next_logits, axis=-1)
        sorted_indices = jnp.argsort(probs, axis=-1, descending=True)
        sorted_probs = jnp.take_along_axis(probs, sorted_indices, axis=-1)
        cumulative_probs = jnp.cumsum(sorted_probs, axis=-1)
        cutoff = jnp.argmax(cumulative_probs >= top_p, axis=-1, keepdims=True)
        mask = jnp.arange(probs.shape[-1]) <= cutoff
        filtered_probs = jnp.where(mask, sorted_probs, 0)
        filtered_probs = filtered_probs / jnp.sum(filtered_probs, axis=-1, keepdims=True)
        filtered_probs = jnp.take_along_axis(filtered_probs, jnp.argsort(sorted_indices, axis=-1), axis=-1)

        # Sample next token
        rng, sub_rng = random.split(rng)
        next_token = random.categorical(sub_rng, jnp.log(filtered_probs + 1e-10), axis=-1)
        generated = jnp.concatenate([generated, next_token[:, None]], axis=1)

        return i + 1, generated, rng

    def cond_fn(val):
        i, generated, _ = val
        if i >= max_length - input_ids.shape[1]:
            return False
        last_token = generated[:, -1]
        return jnp.logical_not(jnp.any(last_token == end_answer_id))

    # Use jax.lax.while_loop for generation
    i, generated, rng = jax.lax.while_loop(
        cond_fn, body_fn, (0, generated, rng)
    )
    return generated

def extract_answer(output_text: str) -> str:
    """
    Extract the answer from the CoT output.
    
    Args:
        output_text: Generated text with <think> and <answer> tags.
    
    Returns:
        Content between <answer> and </answer>, or error message.
    """
    start = output_text.find("<answer>") + len("<answer>")
    end = output_text.find("</answer>")
    if start != -1 and end != -1 and start < end:
        return output_text[start:end].strip()
    return "Answer not found"

@jax.jit
def train_step(params, opt_state, batch, model, optimizer, rng):
    """
    Perform a single training step.
    
    Args:
        params: Model parameters.
        opt_state: Optimizer state.
        batch: Tuple of (input_ids, target_ids).
        model: CoTModel instance.
        optimizer: Optax optimizer.
        rng: Random number generator key.
    
    Returns:
        Updated params, opt_state, and loss.
    """
    input_ids, target_ids = batch

    def loss_fn(params):
        loss = model.apply({'params': params}, input_ids, target_ids, train=True)
        return loss

    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

def train_cot_model(model, params, dataloader, optimizer, num_epochs: int, rng: jnp.ndarray) -> dict:
    """
    Train the CoT model with supervised learning.
    
    Args:
        model: CoTModel instance.
        params: Initial model parameters.
        dataloader: Iterable of (input_ids, target_ids) pairs.
        optimizer: Optax optimizer.
        num_epochs: Number of training epochs.
        rng: Random number generator key.
    
    Returns:
        Trained model parameters.
    """
    opt_state = optimizer.init(params)
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            rng, sub_rng = random.split(rng)
            params, opt_state, loss = train_step(params, opt_state, batch, model, optimizer, sub_rng)
            total_loss += loss.item()

            if batch_idx % 2 == 0:  # Reduced logging for demo
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss:.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
    return params

# Example usage
if __name__ == "__main__":
    # Mock tokenizer (same as original but adapted for JAX)
    class MockTokenizer:
        def __init__(self, vocab_size=50000):
            self.vocab_size = vocab_size
            self.special_tokens = {
                "<think>": vocab_size-4, "</think>": vocab_size-3,
                "<answer>": vocab_size-2, "</answer>": vocab_size-1
            }
            self.inverse_vocab = {v: k for k, v in self.special_tokens.items()}
            self.inverse_vocab.update({i: f"token_{i}" for i in range(vocab_size-4)})

        def encode(self, text, return_tensors="jax"):
            tokens = [self.special_tokens.get(text, i) for i in range(5)]  # Simplified
            if return_tensors == "jax":
                return jnp.array([tokens], dtype=jnp.int32)
            return tokens

        def decode(self, token_ids, skip_special_tokens=False):
            text = []
            for token in token_ids:
                token = int(token)
                if token in self.inverse_vocab:
                    if not skip_special_tokens or token < self.vocab_size-4:
                        text.append(self.inverse_vocab[token])
            return " ".join(text)

    # Initialize model and tokenizer
    rng = random.PRNGKey(0)
    rng, init_rng = random.split(rng)
    tokenizer = MockTokenizer()
    model = CoTModel(vocab_size=tokenizer.vocab_size)
    params = model.init(init_rng, jnp.ones((1, 5), dtype=jnp.int32))['params']

    # Mock dataset (adapted for JAX)
    rng, data_rng = random.split(rng)
    input_ids = random.randint(data_rng, (10, 20), 0, tokenizer.vocab_size-4)
    target_ids = jnp.concatenate([
        jnp.full((10, 1), tokenizer.special_tokens["<think>"], dtype=jnp.int32),
        random.randint(data_rng, (10, 18), 0, tokenizer.vocab_size-4),
        jnp.full((10, 1), tokenizer.special_tokens["</think>"], dtype=jnp.int32),
        jnp.full((10, 1), tokenizer.special_tokens["<answer>"], dtype=jnp.int32),
        random.randint(data_rng, (10, 5), 0, tokenizer.vocab_size-4),
        jnp.full((10, 1), tokenizer.special_tokens["</answer>"], dtype=jnp.int32)
    ], axis=1)

    # Mock dataloader (list of batches for simplicity)
    batch_size = 2
    dataloader = [
        (input_ids[i:i+batch_size], target_ids[i:i+batch_size])
        for i in range(0, len(input_ids), batch_size)
    ]

    # Train the model
    optimizer = optax.adam(1e-4)
    trained_params = train_cot_model(model, params, dataloader, optimizer, num_epochs=3, rng=rng)

    # Generate a sample CoT
    input_text = "Solve 2x + 3 = 7"
    rng, gen_rng = random.split(rng)
    output = generate_cot(model, trained_params, input_text, tokenizer, max_length=512, temperature=0.6, top_p=0.95, rng=gen_rng)
    print("Generated CoT:", output)
    print("Extracted Answer:", extract_answer(output))