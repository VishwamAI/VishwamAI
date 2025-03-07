"""Text generation optimized for TPU with advanced batching and sampling."""
from typing import List, Optional, Dict, Any, NamedTuple
from dataclasses import dataclass
import os
import argparse
import json
import logging
from tqdm import tqdm
from functools import partial
import numpy as np

import jax
import jax.numpy as jnp
from flax.training.common_utils import shard

from .model import VishwamAIModel, ModelConfig
from .tokenizer import VishwamAITokenizer
from .error_correction import compute_error_metrics, ErrorCorrectionTrainer
from .integration import ToTModelIntegrator
from .tot import TreeOfThoughts

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
    chunk_size: int = 32  # Chunk size for memory-efficient processing
    use_error_correction: bool = True  # Use error correction module
    use_tot: bool = False  # Use Tree of Thoughts reasoning
    tot_search_strategy: str = "beam"  # ToT search strategy

class GenerationState(NamedTuple):
    """State maintained during generation, optimized for TPU."""
    tokens: jnp.ndarray
    scores: jnp.ndarray
    temperature: float
    token_counts: Dict[int, int]
    device_mesh: Optional[Any] = None  # TPU device mesh for sharding
    chunk_idx: int = 0  # Current chunk being processed
    active_chunks: List[int] = None  # List of active chunks

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
    
    # Initialize penalty matrix with zeros instead of ones for TPU efficiency
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
    # Use JAX's builtin optimizations for top-k selection
    if top_k > 0:
        top_k = jnp.maximum(top_k, min_tokens_to_keep)
        # Use TPU-optimized topk operation
        top_k_values, _ = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))
        threshold = jnp.expand_dims(top_k_values[..., -1], axis=-1)
        logits = jnp.where(logits < threshold, jnp.full_like(logits, float('-inf')), logits)
    
    # Use JAX's vectorized operations for top-p filtering
    if top_p < 1.0:
        # Sort logits and compute cumulative probabilities in a TPU-friendly way
        sorted_indices = jnp.argsort(logits, axis=-1)[..., ::-1]
        sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
        
        # Create mask for tokens below top_p threshold
        sorted_indices_to_keep = cumulative_probs <= top_p
        # Always keep at least min_tokens_to_keep
        sorted_indices_to_keep = jnp.concatenate([
            jnp.ones_like(sorted_indices_to_keep[..., :min_tokens_to_keep]),
            sorted_indices_to_keep[..., min_tokens_to_keep:]
        ], axis=-1)
        
        # Scatter the mask back to original indices
        indices_to_keep = jnp.take_along_axis(sorted_indices_to_keep, jnp.argsort(sorted_indices, axis=-1), axis=-1)
        
        # Apply the mask - set all tokens outside top_p to -inf
        logits = jnp.where(indices_to_keep, logits, jnp.full_like(logits, float('-inf')))
    
    return logits

@partial(jax.jit, static_argnums=(2,))
def sample_tokens(
    logits: jnp.ndarray,
    state: GenerationState,
    config: GenerationConfig,
    rng: jax.random.PRNGKey
) -> tuple[jnp.ndarray, jax.random.PRNGKey]:
    """TPU-optimized token sampling."""
    # Apply repetition penalty if we have previous tokens
    if state.tokens is not None:
        logits = compute_repetition_penalty(logits, state.tokens, config)
    
    # Get dynamic temperature
    temperature = adjust_temperature(state, config)
    temperature = jnp.maximum(temperature, 1e-5)
    
    # Apply temperature scaling with TPU optimization
    logits = logits / temperature
    
    # Filter logits with top-k and top-p sampling
    filtered_logits = top_k_top_p_filtering(
        logits,
        config.top_k,
        config.top_p,
        min_tokens_to_keep=2
    )
    
    # Sample or take argmax
    if config.do_sample:
        probs = jax.nn.softmax(filtered_logits, axis=-1)
        rng, sampling_rng = jax.random.split(rng)
        next_tokens = jax.random.categorical(sampling_rng, probs)
    else:
        next_tokens = jnp.argmax(filtered_logits, axis=-1)
    
    return next_tokens, rng

@partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(3,), in_axes=(0, 0, 0, None))
def parallel_generate_step(
    model_state: Any,
    tokens: jnp.ndarray,
    rng: jax.random.PRNGKey,
    generation_config: GenerationConfig
) -> tuple[jnp.ndarray, jax.random.PRNGKey]:
    """TPU-parallel generation step."""
    # Prepare tokens with shape that model expects
    if len(tokens.shape) == 2:  # [batch, seq_len]
        tokens_input = tokens
    else:  # [batch, max_len, seq_len]
        active_tokens = tokens[:, tokens.shape[1]-1, :]  # Get latest chunk
        tokens_input = active_tokens
    
    # Forward pass with TPU optimization
    outputs = model_state.apply_fn(
        {'params': model_state.params},
        tokens_input,
        deterministic=True,
        output_hidden_states=True
    )
    
    # Get logits for next token prediction
    logits = outputs['logits'][:, -1, :]
    
    # Use error correction if available
    error_corrected_logits = None
    if hasattr(model_state, 'error_corrector') and generation_config.use_error_correction:
        error_outputs = model_state.error_corrector(
            hidden_states=outputs['hidden_states'][-1][:, -1:, :],
            deterministic=True
        )
        error_corrected_logits = outputs['logits'][:, -1:, :]  # Original shape for correction
        error_mask = error_outputs['correction_mask']
        error_corrected_logits = jnp.where(
            error_mask,
            error_outputs['corrected_logits'],
            error_corrected_logits
        )
        error_corrected_logits = error_corrected_logits[:, -1, :]  # Back to [batch, vocab]
    else:
        error_corrected_logits = logits
    
    # Use Tree of Thoughts if enabled
    tot_logits = None
    if 'tot_logits' in outputs and generation_config.use_tot:
        tot_logits = outputs['tot_logits'][:, -1, :]
        # Blend logits based on integration info if available
        if 'integration_info' in outputs:
            tot_weight = outputs['integration_info'].get('tot_weight', 0.3)
            error_weight = outputs['integration_info'].get('error_weight', 0.3)
            model_weight = outputs['integration_info'].get('model_weight', 0.4)
            
            final_logits = (
                tot_weight * tot_logits +
                error_weight * error_corrected_logits +
                model_weight * logits
            )
        else:
            # Simple weighted average if no integration info
            final_logits = 0.7 * error_corrected_logits + 0.3 * tot_logits
    else:
        final_logits = error_corrected_logits
    
    # Create dummy state for sampling
    dummy_state = GenerationState(
        tokens=tokens_input,
        scores=None,
        temperature=generation_config.temperature,
        token_counts={},
        device_mesh=None
    )
    
    # Sample tokens in parallel
    next_tokens, new_rng = sample_tokens(
        final_logits,
        dummy_state,
        generation_config,
        rng
    )
    
    # All-reduce to ensure consistent results across replicas
    next_tokens = jax.lax.all_gather(next_tokens, axis_name='batch')[:, 0]
    
    return next_tokens, new_rng

def _setup_tpu_mesh():
    """Set up TPU mesh for optimal sharding."""
    try:
        # Get available TPU devices
        devices = jax.devices("tpu")
        num_devices = len(devices)
        
        if num_devices > 0:
            # Create device mesh for sharding
            device_mesh = np.array(devices).reshape(-1)
            mesh = jax.sharding.Mesh(device_mesh, ('batch',))
            return mesh
        else:
            logger.warning("No TPU devices found, falling back to CPU")
            return None
    except Exception as e:
        logger.error(f"Failed to initialize TPU mesh: {str(e)}")
        return None

def generate(
    model: VishwamAIModel,
    tokenizer: VishwamAITokenizer,
    prompts: List[str],
    config: Optional[GenerationConfig] = None,
    callback: Optional[callable] = None,
    rng: Optional[jax.random.PRNGKey] = None,
    tot_model: Optional[TreeOfThoughts] = None
) -> Dict[str, Any]:
    """Enhanced text generation with TPU optimizations."""
    if config is None:
        config = GenerationConfig()
    
    if rng is None:
        rng = jax.random.PRNGKey(0)
    
    # Set up TPU mesh for sharding
    mesh = _setup_tpu_mesh()
    
    # Set device batch size for TPU
    num_devices = jax.device_count()
    if config.device_batch_size is None:
        config.device_batch_size = max(1, config.batch_size // num_devices)
    
    logger.info(f"Generating with {num_devices} TPU devices, "
               f"batch size: {config.batch_size}, "
               f"per-device: {config.device_batch_size}")
    
    # Initialize ToT integration if enabled
    tot_integrator = None
    if config.use_tot and tot_model is not None:
        from .integration import ToTModelIntegrator
        tot_integrator = ToTModelIntegrator(
            model=model,
            tot_model=tot_model,
            config=model.config,
            use_error_correction=config.use_error_correction,
            use_dualpipe=config.use_dualpipe
        )
        logger.info(f"Tree of Thoughts enabled with {config.tot_search_strategy} search strategy")
    
    all_generated = []
    all_metrics = []
    
    # Process prompts in TPU-optimized batches
    for batch_start in range(0, len(prompts), config.batch_size):
        batch_prompts = prompts[batch_start:batch_start + config.batch_size]
        batch_size = len(batch_prompts)
        
        # Tokenize with padding
        input_ids = tokenizer.batch_encode(batch_prompts, add_special_tokens=True)
        max_prompt_len = input_ids.shape[1]
        
        # Calculate maximum possible output length for this batch
        max_output_len = max_prompt_len + config.max_new_tokens
        
        # Create padded input tensor with efficient shape for TPU
        tokens = jnp.full(
            (batch_size, max_output_len),
            tokenizer.pad_id,
            dtype=jnp.int32
        )
        
        # Fill input tokens
        tokens = tokens.at[:, :max_prompt_len].set(input_ids)
        
        # Reshape for TPU devices with padding to match device count
        if num_devices > 1:
            # Calculate padding needed to make batch size divisible by device count
            pad_size = (num_devices - (batch_size % num_devices)) % num_devices
            if pad_size > 0:
                tokens = jnp.pad(tokens, ((0, pad_size), (0, 0)), constant_values=tokenizer.pad_id)
            
            # Reshape for multi-device processing
            dev_batch_size = tokens.shape[0] // num_devices
            tokens = tokens.reshape(num_devices, dev_batch_size, tokens.shape[-1])
            
            # Create a device array to enable pmap
            sharded_tokens = shard(tokens)
        else:
            sharded_tokens = tokens[None, ...]  # Add dummy device dimension
        
        # Initialize generation state
        state = GenerationState(
            tokens=None,
            scores=jnp.zeros((batch_size,)),
            temperature=config.temperature,
            token_counts={},
            device_mesh=mesh
        )
        
        # Create sharded random keys for each device
        sharded_rng = jax.random.split(rng, num_devices)
        
        # Generate tokens with chunked processing for memory efficiency
        chunk_size = config.chunk_size
        num_chunks = (max_output_len + chunk_size - 1) // chunk_size
        
        # Allocate full output tensor
        full_output = np.full((batch_size, max_output_len), tokenizer.pad_id, dtype=np.int32)
        full_output[:, :max_prompt_len] = input_ids.astype(np.int32)
        
        # Process chunks to generate full output
        for chunk_idx in range(max_prompt_len // chunk_size, num_chunks):
            chunk_start = chunk_idx * chunk_size
            chunk_end = min(chunk_start + chunk_size, max_output_len)
            
            # Skip completed chunks
            if chunk_start < max_prompt_len:
                continue
                
            # Process tokens for this chunk
            for pos in range(max(chunk_start, max_prompt_len), chunk_end):
                # Extract active sequence up to current position
                active_seq = full_output[:, :pos]
                
                # Reshape for device processing
                if num_devices > 1:
                    # Add padding if needed
                    if pad_size > 0:
                        active_seq_padded = np.pad(
                            active_seq, 
                            ((0, pad_size), (0, 0)), 
                            constant_values=tokenizer.pad_id
                        )
                    else:
                        active_seq_padded = active_seq
                        
                    # Reshape for multi-device processing
                    dev_batch_size = active_seq_padded.shape[0] // num_devices
                    sharded_seq = active_seq_padded.reshape(
                        num_devices, dev_batch_size, active_seq_padded.shape[-1]
                    )
                    
                    # Convert to JAX array with proper sharding
                    sharded_seq = jnp.array(sharded_seq)
                else:
                    # Add dummy device dimension for single device
                    sharded_seq = jnp.array(active_seq)[None, ...]
                
                # Run parallel generation step
                next_tokens, sharded_rng = parallel_generate_step(
                    model, sharded_seq, sharded_rng, config
                )
                
                # Gather results from all devices
                if num_devices > 1:
                    # Reshape back to flat batch
                    next_tokens = next_tokens.reshape(-1)[:batch_size]
                else:
                    next_tokens = next_tokens[0]
                
                # Update output with next tokens
                full_output[:, pos] = next_tokens
                
                # Check for early stopping
                if _check_early_stopping(
                    tokenizer, next_tokens, pos - max_prompt_len, config
                ):
                    break
                
                # Report progress
                if callback and pos % 10 == 0:
                    callback(pos - max_prompt_len, config.max_new_tokens)
        
        # Process results
        for j in range(batch_size):
            output_ids = full_output[j, max_prompt_len:].tolist()
            # Remove padding and end tokens
            try:
                eos_idx = output_ids.index(tokenizer.eos_id)
                output_ids = output_ids[:eos_idx]
            except ValueError:
                # No EOS found, keep all tokens
                pass
                
            # Filter out any pad tokens
            output_ids = [token for token in output_ids if token != tokenizer.pad_id]
            
            # Decode text
            generated_text = tokenizer.decode(output_ids)
            all_generated.append(generated_text)
    
    # Collect results
    result = {'generated_texts': all_generated}
    if all_metrics:
        result['error_metrics'] = all_metrics
        
    return result

def _check_early_stopping(
    tokenizer: VishwamAITokenizer,
    next_tokens: jnp.ndarray,
    current_length: int,
    config: GenerationConfig
) -> bool:
    """Check early stopping conditions for generation."""
    if not config.early_stopping:
        return False
    
    min_length_reached = current_length >= config.max_new_tokens // 2
    if not min_length_reached:
        return False
    
    # Check if EOS tokens have been generated
    eos_generated = (next_tokens == tokenizer.eos_id).any()
    
    return eos_generated

def _check_dualpipe_stopping(
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
    parser.add_argument("--use-tot", action="store_true")
    parser.add_argument("--tot-search", type=str, default="beam", 
                        choices=["beam", "dfs", "bfs", "mcts"])
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
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
            use_dualpipe=args.use_dualpipe,
            use_tot=args.use_tot,
            tot_search_strategy=args.tot_search,
            chunk_size=args.chunk_size
        )
        
        # Initialize model and tokenizer
        model = VishwamAIModel(config)
        model.load_weights(args.model_path)
        tokenizer = VishwamAITokenizer.from_pretrained(args.tokenizer_path)
        
        # Initialize ToT model if enabled
        tot_model = None
        if args.use_tot:
            from .tot import TreeOfThoughts
            tot_model = TreeOfThoughts(
                transformer=model,
                tokenizer=tokenizer,
                max_thoughts=5,
                max_depth=3,
                beam_width=5,
                use_tpu=True
            )
            
            logger.info(f"Tree of Thoughts initialized with {args.tot_search} search strategy")
        
        # Set TPU-optimized random seed
        rng = jax.random.PRNGKey(args.seed)
        if jax.device_count() > 1:
            rng = jax.random.split(rng, jax.device_count())
        
        # Process inputs
        if args.input_file:
            with open(args.input_file) as f:
                if args.input_file.endswith('.json'):
                    prompts = json.load(f)
                else:
                    prompts = [line.strip() for line in f if line.strip()]
        else:
            prompts = []
            print("Enter prompts (type 'quit' to finish):")
            while True:
                line = input("> ")
                if line.lower() == 'quit':
                    break
                prompts.append(line)
        
        def progress_callback(current, total):
            logger.info(f"Generation progress: {current}/{total}")
        
        # Generate text with TPU optimization
        results = generate(
            model,
            tokenizer,
            prompts,
            gen_config,
            progress_callback,
            rng=rng,
            tot_model=tot_model
        )
        
        # Save or display results
        if args.output_file:
            with open(args.output_file, 'w') as f:
                if args.output_file.endswith('.json'):
                    json.dump(results, f, indent=2)
                else:
                    for text in results['generated_texts']:
                        f.write(text + '\n')
            logger.info(f"Results saved to {args.output_file}")
        else:
            for i, (prompt, generated) in enumerate(zip(prompts, results['generated_texts'])):
                print(f"\nPrompt {i+1}: {prompt}")
                print(f"Generated: {generated}")
                print("-" * 40)
        
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    main()
