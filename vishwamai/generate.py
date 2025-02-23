import os
import json
import logging
from typing import List, Optional, Dict, Any
from tqdm import tqdm

import jax
import jax.numpy as jnp
from transformers import AutoTokenizer

from .model import VishwamAIModel, ModelConfig
from .tokenizer import VishwamAITokenizer
from .error_correction import compute_error_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenerationConfig:
    def __init__(
        self,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.2,
        do_sample: bool = True,
        num_beams: int = 1,
        early_stopping: bool = True,
        batch_size: int = 4
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.do_sample = do_sample
        self.num_beams = num_beams
        self.early_stopping = early_stopping
        self.batch_size = batch_size

def top_k_top_p_filtering(logits: jnp.ndarray, top_k: int = 0, top_p: float = 1.0) -> jnp.ndarray:
    """Filter logits using top-k and nucleus (top-p) sampling"""
    if top_k > 0:
        indices_to_remove = logits < jax.lax.top_k(logits, min(top_k, logits.shape[-1]))[0][..., -1, None]
        logits = jnp.where(indices_to_remove, float('-inf'), logits)
    
    if top_p < 1.0:
        sorted_logits = jnp.sort(logits, axis=-1)[:, ::-1]
        cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits, axis=-1), axis=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove = jnp.roll(sorted_indices_to_remove, 1, axis=-1)
        sorted_indices_to_remove = sorted_indices_to_remove.at[:, 0].set(False)
        indices_to_remove = jnp.zeros_like(logits, dtype=bool).at[
            jnp.arange(logits.shape[0])[:, None],
            jnp.argsort(logits, axis=-1)[:, ::-1]
        ].set(sorted_indices_to_remove)
        logits = jnp.where(indices_to_remove, float('-inf'), logits)
    
    return logits

def sample_tokens(
    logits: jnp.ndarray,
    config: GenerationConfig,
    prev_tokens: Optional[jnp.ndarray] = None
) -> jnp.ndarray:
    """Enhanced token sampling with various decoding strategies"""
    if prev_tokens is not None:
        # Apply repetition penalty
        score = jnp.where(
            logits < 0,
            logits * config.repetition_penalty,
            logits / config.repetition_penalty
        )
    else:
        score = logits
    
    # Apply temperature
    if config.temperature > 0:
        score = score / jnp.maximum(config.temperature, 1e-5)
    
    # Apply filtering
    filtered_score = top_k_top_p_filtering(score, config.top_k, config.top_p)
    
    if config.do_sample:
        probs = jax.nn.softmax(filtered_score, axis=-1)
        next_tokens = jax.random.categorical(jax.random.PRNGKey(0), probs)
    else:
        next_tokens = jnp.argmax(filtered_score, axis=-1)
    
    return next_tokens

def generate(
    model: VishwamAIModel,
    tokenizer: VishwamAITokenizer,
    prompts: List[str],
    config: Optional[GenerationConfig] = None,
    callback: Optional[callable] = None
) -> Dict[str, Any]:
    """Enhanced generation with better control and monitoring"""
    if config is None:
        config = GenerationConfig()
    
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
        
        # Generate tokens with progress tracking
        generated_tokens = []
        for pos in tqdm(range(max_prompt_len, max_prompt_len + config.max_new_tokens)):
            # Get model outputs
            outputs = model(tokens[:, :pos])
            next_token_logits = outputs['logits'][:, -1, :]
            
            # Apply error correction
            corrected_logits, error_gates = model.error_corrector(
                next_token_logits.reshape(-1, 1, next_token_logits.shape[-1])
            )
            
            # Sample next tokens
            next_tokens = sample_tokens(
                corrected_logits,
                config,
                tokens[:, :pos]
            )
            
            # Update tokens
            tokens = tokens.at[:, pos].set(next_tokens)
            generated_tokens.append(next_tokens)
            
            # Compute metrics
            metrics = compute_error_metrics(
                corrected_logits,
                next_token_logits
            )
            all_metrics.append(metrics)
            
            # Early stopping check
            if config.early_stopping and all(
                (next_tokens == tokenizer.eos_id).all()
                for next_tokens in generated_tokens[-5:]
            ):
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
    """Enhanced main function with better configuration and error handling"""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, required=True)
    parser.add_argument("--input-file", type=str)
    parser.add_argument("--output-file", type=str)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    try:
        # Load configurations
        with open(args.config) as f:
            config = ModelConfig(**json.load(f))
        
        gen_config = GenerationConfig(
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            batch_size=args.batch_size
        )
        
        # Initialize model and tokenizer
        model = VishwamAIModel(config)
        model.load_weights(args.model_path)
        tokenizer = VishwamAITokenizer.from_pretrained(args.tokenizer_path)
        
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
        
        results = generate(model, tokenizer, prompts, gen_config, progress_callback)
        
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
