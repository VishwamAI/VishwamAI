import os
import json
import logging
from typing import List, Optional, Dict, Any, NamedTuple
from dataclasses import dataclass
from tqdm import tqdm

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from .model import VishwamAIModel, ModelConfig
from .tokenizer import VishwamAITokenizer
from .error_correction import compute_error_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.2
    do_sample: bool = True
    num_beams: int = 1
    early_stopping: bool = True
    batch_size: int = 4
    min_temperature: float = 0.1
    max_temperature: float = 2.0
    dynamic_temperature: bool = True
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    length_penalty: float = 1.0

class GenerationState(NamedTuple):
    """State maintained during generation."""
    tokens: jnp.ndarray
    scores: jnp.ndarray
    temperature: float
    token_counts: Dict[int, int]

def adjust_temperature(
    state: GenerationState,
    config: GenerationConfig
) -> float:
    """Dynamically adjust temperature based on generation state."""
    if not config.dynamic_temperature:
        return config.temperature
    
    # Compute token diversity
    unique_tokens = len(set(state.token_counts.keys()))
    total_tokens = sum(state.token_counts.values())
    diversity = unique_tokens / max(total_tokens, 1)
    
    # Adjust temperature based on diversity
    if diversity < 0.3:  # Low diversity
        temperature = min(state.temperature * 1.1, config.max_temperature)
    elif diversity > 0.7:  # High diversity
        temperature = max(state.temperature * 0.9, config.min_temperature)
    else:
        temperature = state.temperature
    
    return temperature

def compute_repetition_penalty(
    logits: jnp.ndarray,
    prev_tokens: jnp.ndarray,
    config: GenerationConfig
) -> jnp.ndarray:
    """Apply enhanced repetition, presence, and frequency penalties."""
    # Get unique previous tokens and their counts
    unique_tokens, counts = jnp.unique(prev_tokens, return_counts=True)
    
    # Initialize penalty matrix
    penalty = jnp.ones_like(logits)
    
    # Apply repetition penalty
    if config.repetition_penalty != 1.0:
        rep_mask = jnp.zeros_like(logits, dtype=bool)
        rep_mask = rep_mask.at[unique_tokens].set(True)
        penalty = jnp.where(rep_mask, config.repetition_penalty, penalty)
    
    # Apply frequency penalty
    if config.frequency_penalty != 0.0:
        freq_penalty = jnp.zeros_like(logits)
        freq_penalty = freq_penalty.at[unique_tokens].set(counts * config.frequency_penalty)
        logits = logits - freq_penalty
    
    # Apply presence penalty
    if config.presence_penalty != 0.0:
        presence_mask = jnp.zeros_like(logits, dtype=bool)
        presence_mask = presence_mask.at[unique_tokens].set(True)
        logits = jnp.where(presence_mask, logits - config.presence_penalty, logits)
    
    return jnp.where(logits < 0, logits * penalty, logits / penalty)

def top_k_top_p_filtering(
    logits: jnp.ndarray,
    top_k: int = 0,
    top_p: float = 1.0,
    min_tokens_to_keep: int = 1
) -> jnp.ndarray:
    """Enhanced logits filtering with minimum token guarantee."""
    if top_k > 0:
        # Keep at least min_tokens_to_keep
        top_k = max(top_k, min_tokens_to_keep)
        # Get top k values and create mask
        top_k_values = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))[0]
        threshold = top_k_values[..., -1, None]
        logits = jnp.where(logits < threshold, float('-inf'), logits)
    
    if top_p < 1.0:
        sorted_logits = jnp.sort(logits, axis=-1)[:, ::-1]
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
        
        # Ensure we keep at least min_tokens_to_keep tokens
        sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
        sorted_indices = sorted_indices[:, :max(min_tokens_to_keep, (cumulative_probs <= top_p).sum())]
        
        mask = jnp.zeros_like(logits, dtype=bool)
        mask = mask.at[jnp.arange(logits.shape[0])[:, None], sorted_indices].set(True)
        logits = jnp.where(~mask, float('-inf'), logits)
    
    return logits

def sample_tokens(
    logits: jnp.ndarray,
    state: GenerationState,
    config: GenerationConfig,
    rng: jax.random.PRNGKey
) -> jnp.ndarray:
    """Enhanced token sampling with improved diversity and control."""
    # Apply penalties
    if state.tokens is not None:
        logits = compute_repetition_penalty(logits, state.tokens, config)
    
    # Adjust temperature dynamically
    temperature = adjust_temperature(state, config)
    
    # Apply temperature scaling
    if temperature > 0:
        logits = logits / jnp.maximum(temperature, 1e-5)
    
    # Apply filtering
    filtered_logits = top_k_top_p_filtering(
        logits,
        config.top_k,
        config.top_p,
        min_tokens_to_keep=2
    )
    
    if config.do_sample:
        # Split PRNG key for sampling
        rng, sampling_rng = jax.random.split(rng)
        probs = jax.nn.softmax(filtered_logits, axis=-1)
        next_tokens = jax.random.categorical(sampling_rng, probs)
    else:
        next_tokens = jnp.argmax(filtered_logits, axis=-1)
    
    return next_tokens, rng

def generate(
    model: VishwamAIModel,
    tokenizer: VishwamAITokenizer,
    prompts: List[str],
    config: Optional[GenerationConfig] = None,
    callback: Optional[callable] = None,
    rng: Optional[jax.random.PRNGKey] = None
) -> Dict[str, Any]:
    """Enhanced generation with improved control and monitoring."""
    if config is None:
        config = GenerationConfig()
    
    if rng is None:
        rng = jax.random.PRNGKey(0)
    
    # Batch processing
    all_generated = []
    all_metrics = []
    
    for i in range(0, len(prompts), config.batch_size):
        batch_prompts = prompts[i:i + config.batch_size]
        
        # Tokenize inputs
        input_ids = tokenizer.encode(batch_prompts)
        if not isinstance(input_ids, list):
            input_ids = [input_ids]
        
        # Initialize generation tensors
        max_prompt_len = max(len(ids) for ids in input_ids)
        batch_size = len(batch_prompts)
        
        tokens = jnp.full(
            (batch_size, max_prompt_len + config.max_new_tokens),
            tokenizer.pad_id,
            dtype=jnp.int32
        )
        
        # Fill input tokens
        for j, ids in enumerate(input_ids):
            tokens = tokens.at[j, :len(ids)].set(jnp.array(ids))
        
        # Initialize generation state
        state = GenerationState(
            tokens=None,
            scores=jnp.zeros((batch_size,)),
            temperature=config.temperature,
            token_counts={}
        )
        
        # Generate tokens with progress tracking
        generated_tokens = []
        for pos in tqdm(range(max_prompt_len, max_prompt_len + config.max_new_tokens)):
            # Split PRNG key for model and sampling
            rng, model_rng, sampling_rng = jax.random.split(rng, 3)
            
            # Get model outputs
            outputs = model(tokens[:, :pos], rngs={'dropout': model_rng})
            next_token_logits = outputs['logits'][:, -1, :]
            
            # Apply error correction
            corrected_logits, error_gates = model.error_corrector(
                next_token_logits.reshape(-1, 1, next_token_logits.shape[-1])
            )
            
            # Update state with current tokens
            state = state._replace(tokens=tokens[:, :pos])
            
            # Sample next tokens
            next_tokens, sampling_rng = sample_tokens(
                corrected_logits,
                state,
                config,
                sampling_rng
            )
            
            # Update tokens and state
            tokens = tokens.at[:, pos].set(next_tokens)
            generated_tokens.append(next_tokens)
            
            # Update token counts for diversity tracking
            for token in next_tokens:
                state.token_counts[int(token)] = state.token_counts.get(int(token), 0) + 1
            
            # Compute metrics
            metrics = compute_error_metrics(corrected_logits, next_token_logits)
            all_metrics.append(metrics)
            
            # Early stopping check with minimum length consideration
            min_length_reached = pos - max_prompt_len >= config.max_new_tokens // 2
            if config.early_stopping and min_length_reached:
                if all((next_tokens == tokenizer.eos_id).all() for next_tokens in generated_tokens[-5:]):
                    break
            
            # Progress callback
            if callback:
                callback(pos - max_prompt_len, config.max_new_tokens)
        
        # Decode generated tokens
        for j in range(batch_size):
            output_ids = tokens[j, max_prompt_len:pos].tolist()
            generated_text = tokenizer.decode(output_ids)
            all_generated.append(generated_text)
    
    return {
        'generated_texts': all_generated,
        'error_metrics': all_metrics
    }

def main():
    """Enhanced main function with better configuration and error handling."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dynamic-temperature", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        # Load configurations
        with open(args.config) as f:
            config = ModelConfig(**json.load(f))
        
        # Create generation config with all parameters
        gen_config = GenerationConfig(
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
            dynamic_temperature=args.dynamic_temperature
        )
        
        # Initialize model and tokenizer
        model = VishwamAIModel(config)
        model.load_weights(args.model_path)
        tokenizer = VishwamAITokenizer.from_pretrained(args.tokenizer_path)
        
        # Set random seed
        rng = jax.random.PRNGKey(args.seed)
        
        # Process inputs
        if args.input_file:
            with open(args.input_file) as f:
                prompts = [line.strip() for line in f]
        else:
            prompts = []
            print("Enter prompts (type 'quit' to finish):")
            while True:
                prompt = input(">>> ")
                if prompt.lower() == 'quit':
                    break
                prompts.append(prompt)
        
        # Generate text
        def progress_callback(current, total):
            logger.info(f"Generation progress: {current}/{total}")
        
        results = generate(
            model,
            tokenizer,
            prompts,
            gen_config,
            progress_callback,
            rng=rng
        )
        
        # Save or display results
        if args.output_file:
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
        else:
            for i, (prompt, generated) in enumerate(zip(prompts, results['generated_texts'])):
                print(f"\nPrompt {i+1}: {prompt}")
                print(f"Generated: {generated}")
                print("-" * 50)
        
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        raise

if __name__ == "__main__":
    main()
