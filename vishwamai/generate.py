"""Text generation optimized for TPU with advanced batching and sampling."""
import os
import json
import logging
from typing import List, Optional, Dict, Any, NamedTuple
from dataclasses import dataclass
from tqdm import tqdm
from functools import partial

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer
from google.cloud import storage

from .model import VishwamAIModel, ModelConfig
from .tokenizer import VishwamAITokenizer
from .error_correction import compute_error_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for TPU-optimized text generation."""
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
    use_dualpipe: bool = True  # Enable dualpipe-style generation
    device_batch_size: Optional[int] = None  # Batch size per TPU device

class GenerationState(NamedTuple):
    """State maintained during generation, optimized for TPU."""
    tokens: jnp.ndarray
    scores: jnp.ndarray
    temperature: float
    token_counts: Dict[int, int]
    device_mesh: Optional[Any] = None  # TPU device mesh for sharding

@partial(jax.jit, static_argnums=(1,))
def adjust_temperature(
    state: GenerationState,
    config: GenerationConfig
) -> float:
    """TPU-optimized temperature adjustment."""
    if not config.dynamic_temperature:
        return config.temperature
    
    # Compute token diversity with TPU optimization
    unique_tokens = jnp.unique(state.tokens).shape[0]
    total_tokens = state.tokens.size
    diversity = unique_tokens / jnp.maximum(total_tokens, 1)
    
    # Vectorized temperature adjustment
    return jnp.clip(
        jnp.where(
            diversity < 0.3,
            state.temperature * 1.1,
            jnp.where(
                diversity > 0.7,
                state.temperature * 0.9,
                state.temperature
            )
        ),
        config.min_temperature,
        config.max_temperature
    )

@partial(jax.jit, static_argnums=(2,))
def compute_repetition_penalty(
    logits: jnp.ndarray,
    prev_tokens: jnp.ndarray,
    config: GenerationConfig
) -> jnp.ndarray:
    """TPU-optimized token penalty computation."""
    # Get unique previous tokens and their counts
    unique_tokens, counts = jnp.unique(prev_tokens, return_counts=True)
    
    # Initialize penalty matrix
    penalty = jnp.ones_like(logits)
    
    # Vectorized penalty application
    if config.repetition_penalty != 1.0:
        penalty = penalty.at[unique_tokens].multiply(config.repetition_penalty)
    
    if config.frequency_penalty != 0.0:
        freq_penalty = jnp.zeros_like(logits)
        freq_penalty = freq_penalty.at[unique_tokens].add(counts * config.frequency_penalty)
        logits = logits - freq_penalty
    
    if config.presence_penalty != 0.0:
        presence_penalty = jnp.zeros_like(logits)
        presence_penalty = presence_penalty.at[unique_tokens].add(config.presence_penalty)
        logits = logits - presence_penalty
    
    return jnp.where(logits < 0, logits * penalty, logits / penalty)

@partial(jax.jit, static_argnums=(1, 2, 3))
def top_k_top_p_filtering(
    logits: jnp.ndarray,
    top_k: int = 0,
    top_p: float = 1.0,
    min_tokens_to_keep: int = 1
) -> jnp.ndarray:
    """TPU-optimized logits filtering."""
    if top_k > 0:
        top_k = jnp.maximum(top_k, min_tokens_to_keep)
        top_k_values = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))[0]
        threshold = jnp.expand_dims(top_k_values[..., -1], axis=-1)
        logits = jnp.where(logits < threshold, jnp.full_like(logits, float('-inf')), logits)
    
    if top_p < 1.0:
        sorted_logits = jnp.sort(logits, axis=-1)[..., ::-1]
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
        
        # Ensure minimum tokens with TPU optimization
        sorted_indices = jnp.argsort(logits, axis=-1)[..., ::-1]
        min_tokens = jnp.maximum(min_tokens_to_keep, (cumulative_probs <= top_p).sum())
        sorted_indices = sorted_indices[..., :min_tokens]
        
        mask = jnp.zeros_like(logits, dtype=bool)
        batch_indices = jnp.arange(logits.shape[0])[:, None]
        mask = mask.at[batch_indices, sorted_indices].set(True)
        logits = jnp.where(~mask, jnp.full_like(logits, float('-inf')), logits)
    
    return logits

@partial(jax.jit, static_argnums=(2,))
def sample_tokens(
    logits: jnp.ndarray,
    state: GenerationState,
    config: GenerationConfig,
    rng: jax.random.PRNGKey
) -> tuple[jnp.ndarray, jax.random.PRNGKey]:
    """TPU-optimized token sampling."""
    if state.tokens is not None:
        logits = compute_repetition_penalty(logits, state.tokens, config)
    
    temperature = adjust_temperature(state, config)
    temperature = jnp.maximum(temperature, 1e-5)
    
    # Apply temperature scaling with TPU optimization
    logits = logits / temperature
    
    # Filter logits
    filtered_logits = top_k_top_p_filtering(
        logits,
        config.top_k,
        config.top_p,
        min_tokens_to_keep=2
    )
    
    if config.do_sample:
        rng, sampling_rng = jax.random.split(rng)
        probs = jax.nn.softmax(filtered_logits, axis=-1)
        next_tokens = jax.random.categorical(sampling_rng, probs)
    else:
        next_tokens = jnp.argmax(filtered_logits, axis=-1)
    
    return next_tokens, rng

@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(3,))
def parallel_generate_step(
    model_state: Any,
    tokens: jnp.ndarray,
    rng: jax.random.PRNGKey,
    generation_config: GenerationConfig
) -> tuple[jnp.ndarray, jax.random.PRNGKey]:
    """TPU-parallel generation step."""
    outputs = model_state.apply_fn(
        {'params': model_state.params},
        tokens,
        deterministic=True
    )
    
    logits = outputs['logits'][:, -1, :]
    
    # Error correction if available
    if hasattr(model_state, 'error_corrector'):
        corrected_logits, _ = model_state.error_corrector(
            logits.reshape(-1, 1, logits.shape[-1])
        )
    else:
        corrected_logits = logits
    
    # Sample tokens in parallel
    next_tokens, new_rng = sample_tokens(
        corrected_logits,
        GenerationState(tokens=tokens, scores=None, temperature=generation_config.temperature, token_counts={}),
        generation_config,
        rng
    )
    
    return next_tokens, new_rng

def generate(
    model: VishwamAIModel,
    tokenizer: VishwamAITokenizer,
    prompts: List[str],
    config: Optional[GenerationConfig] = None,
    callback: Optional[callable] = None,
    rng: Optional[jax.random.PRNGKey] = None
) -> Dict[str, Any]:
    """Enhanced text generation with TPU optimizations."""
    if config is None:
        config = GenerationConfig()
    
    if rng is None:
        rng = jax.random.PRNGKey(0)
    
    # Set device batch size for TPU
    num_devices = jax.device_count()
    if config.device_batch_size is None:
        config.device_batch_size = max(1, config.batch_size // num_devices)
    
    # Initialize TPU mesh
    devices = jnp.array(jax.devices()).reshape(-1)
    mesh = jax.sharding.Mesh(devices, ('batch',))
    
    all_generated = []
    all_metrics = []
    
    # Process prompts in TPU-optimized batches
    for i in range(0, len(prompts), config.batch_size):
        batch_prompts = prompts[i:i + config.batch_size]
        batch_size = len(batch_prompts)
        
        # Tokenize with padding
        input_ids = tokenizer.encode(batch_prompts)
        if not isinstance(input_ids, list):
            input_ids = [input_ids]
        
        max_prompt_len = max(len(ids) for ids in input_ids)
        
        # Create padded input tensor
        tokens = jnp.full(
            (batch_size, max_prompt_len + config.max_new_tokens),
            tokenizer.pad_id,
            dtype=jnp.int32
        )
        
        # Fill input tokens
        for j, ids in enumerate(input_ids):
            tokens = tokens.at[j, :len(ids)].set(jnp.array(ids))
        
        # Reshape for TPU devices
        if num_devices > 1:
            pad_size = (num_devices - (batch_size % num_devices)) % num_devices
            if pad_size > 0:
                tokens = jnp.pad(tokens, ((0, pad_size), (0, 0)))
            
            tokens = tokens.reshape(num_devices, -1, tokens.shape[-1])
        
        # Initialize generation state
        state = GenerationState(
            tokens=None,
            scores=jnp.zeros((batch_size,)),
            temperature=config.temperature,
            token_counts={},
            device_mesh=mesh
        )
        
        # Generate tokens with dualpipe if enabled
        if config.use_dualpipe:
            forward_size = batch_size // 2
            forward_tokens = tokens[:forward_size]
            backward_tokens = tokens[forward_size:batch_size]
            
            # Process both streams
            for pos in tqdm(range(max_prompt_len, max_prompt_len + config.max_new_tokens)):
                rng, forward_rng, backward_rng = jax.random.split(rng, 3)
                
                # Forward pass
                forward_next, forward_rng = parallel_generate_step(
                    model, forward_tokens[:, :pos], forward_rng, config
                )
                
                # Backward pass
                backward_next, backward_rng = parallel_generate_step(
                    model, backward_tokens[:, :pos], backward_rng, config
                )
                
                # Update tokens
                forward_tokens = forward_tokens.at[:, pos].set(forward_next)
                backward_tokens = backward_tokens.at[:, pos].set(backward_next)
                
                # Update state and check stopping condition
                if _check_early_stopping(
                    tokenizer, forward_next, backward_next, pos - max_prompt_len, config
                ):
                    break
                
                if callback:
                    callback(pos - max_prompt_len, config.max_new_tokens)
            
            # Combine results
            generated_tokens = jnp.concatenate([forward_tokens, backward_tokens], axis=0)
        
        else:
            # Standard generation
            generated_tokens = tokens
            for pos in tqdm(range(max_prompt_len, max_prompt_len + config.max_new_tokens)):
                rng, step_rng = jax.random.split(rng)
                
                next_tokens, step_rng = parallel_generate_step(
                    model, generated_tokens[:, :pos], step_rng, config
                )
                
                generated_tokens = generated_tokens.at[:, pos].set(next_tokens)
                
                if callback:
                    callback(pos - max_prompt_len, config.max_new_tokens)
        
        # Process results
        for j in range(min(batch_size, len(batch_prompts))):
            output_ids = generated_tokens[j, max_prompt_len:].tolist()
            # Remove padding and end tokens
            output_ids = [
                token for token in output_ids 
                if token not in {tokenizer.pad_id, tokenizer.eos_id}
            ]
            generated_text = tokenizer.decode(output_ids)
            all_generated.append(generated_text)
    
    result = {'generated_texts': all_generated}
    if all_metrics:
        result['error_metrics'] = all_metrics
    return result

def _check_early_stopping(
    tokenizer: VishwamAITokenizer,
    forward_tokens: jnp.ndarray,
    backward_tokens: jnp.ndarray,
    current_length: int,
    config: GenerationConfig
) -> bool:
    """Check early stopping conditions for dualpipe generation."""
    if not config.early_stopping:
        return False
    
    min_length_reached = current_length >= config.max_new_tokens // 2
    if not min_length_reached:
        return False
    
    # Check if both streams have generated EOS tokens
    forward_done = (forward_tokens == tokenizer.eos_id).any()
    backward_done = (backward_tokens == tokenizer.eos_id).any()
    
    return forward_done and backward_done

def main():
    """Main function for TPU-optimized text generation."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device-batch-size", type=int)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--dynamic-temperature", action="store_true")
    parser.add_argument("--use-dualpipe", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        # Handle GCS paths
        if args.model_path.startswith('gs://'):
            client = storage.Client()
            bucket_name, blob_path = args.model_path.replace('gs://', '').split('/', 1)
            bucket = client.bucket(bucket_name)
            local_path = '/tmp/model'
            os.makedirs(local_path, exist_ok=True)
            
            for blob in bucket.list_blobs(prefix=blob_path):
                if blob.name.endswith(('.safetensors', 'config.json')):
                    local_file = os.path.join(local_path, os.path.basename(blob.name))
                    blob.download_to_filename(local_file)
            args.model_path = local_path
        
        # Load configurations
        with open(args.config) as f:
            config = ModelConfig(**json.load(f))
        
        # Create TPU-optimized generation config
        gen_config = GenerationConfig(
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size,
            device_batch_size=args.device_batch_size,
            dynamic_temperature=args.dynamic_temperature,
            use_dualpipe=args.use_dualpipe
        )
        
        # Initialize model and tokenizer
        model = VishwamAIModel(config)
        model.load_weights(args.model_path)
        tokenizer = VishwamAITokenizer.from_pretrained(args.tokenizer_path)
        
        # Set TPU-optimized random seed
        rng = jax.random.PRNGKey(args.seed)
        if jax.device_count() > 1:
            rng = jax.random.split(rng, jax.device_count())
        
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
        
        def progress_callback(current, total):
            logger.info(f"Generation progress: {current}/{total}")
        
        # Generate text with TPU optimization
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
            if args.output_file.startswith('gs://'):
                client = storage.Client()
                bucket_name, blob_path = args.output_file.replace('gs://', '').split('/', 1)
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                
                # Save to temporary file first
                temp_path = '/tmp/generation_results.json'
                with open(temp_path, 'w') as f:
                    json.dump(results, f, indent=2)
                
                # Upload to GCS
                blob.upload_from_filename(temp_path)
                os.remove(temp_path)
            else:
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
