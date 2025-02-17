import os
import json
import logging
import signal
from argparse import ArgumentParser
from typing import List, Optional, Dict, Any, Union
from dataclasses import dataclass
from contextlib import contextmanager

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, PreTrainedTokenizer
from safetensors.torch import load_model

from .model import Transformer, ModelArgs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_new_tokens: int = 200
    temperature: float = 0.2
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    batch_size: int = 1

def setup_distributed() -> tuple[int, int, int]:
    """
    Setup distributed training environment.
    
    Returns:
        Tuple of (world_size, rank, local_rank)
    """
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    
    if world_size > 1:
        try:
            dist.init_process_group("nccl")
            logger.info(f"Initialized distributed process group (rank {rank}/{world_size})")
        except Exception as e:
            logger.error(f"Failed to initialize distributed process group: {e}")
            raise
            
    return world_size, rank, local_rank

def sample_top_p_top_k(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 0.9,
    top_k: int = 50
) -> torch.Tensor:
    """
    Sample from logits using combined top-p and top-k sampling.
    
    Args:
        logits: Raw logits tensor
        temperature: Sampling temperature
        top_p: Nucleus sampling probability threshold
        top_k: Number of top tokens to consider
        
    Returns:
        Tensor of sampled token indices
    """
    # Apply temperature
    logits = logits / max(temperature, 1e-5)
    
    # Top-k filtering
    top_k = min(top_k, logits.size(-1))
    values, indices = torch.topk(logits, top_k)
    
    # Top-p filtering
    probs = torch.softmax(values, dim=-1)
    cumsum_probs = torch.cumsum(probs, dim=-1)
    mask = cumsum_probs <= top_p
    mask[..., 0] = True  # Keep at least one token
    
    # Sample from filtered distribution
    filtered_probs = probs * mask
    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
    sample_idx = torch.multinomial(filtered_probs, num_samples=1)
    
    return indices.gather(-1, sample_idx).squeeze(-1)

@torch.inference_mode()
def generate(
    model: Transformer,
    tokenizer: PreTrainedTokenizer,
    prompt_tokens: List[List[int]],
    gen_config: GenerationConfig,
    device: str = "cuda"
) -> List[List[int]]:
    """
    Generate text completions.
    
    Args:
        model: Transformer model
        tokenizer: Tokenizer instance
        prompt_tokens: List of token sequences
        gen_config: Generation configuration
        device: Device to use for generation
        
    Returns:
        List of generated token sequences
        
    Raises:
        ValueError: If prompt length exceeds model maximum sequence length
        RuntimeError: If generation fails
    """
    try:
        # Validate inputs
        prompt_lens = [len(t) for t in prompt_tokens]
        if max(prompt_lens) > model.max_seq_len:
            raise ValueError(
                f"Prompt length ({max(prompt_lens)}) exceeds model maximum "
                f"sequence length ({model.max_seq_len})"
            )
            
        # Initialize generation
        total_len = min(
            model.max_seq_len,
            gen_config.max_new_tokens + max(prompt_lens)
        )
        tokens = torch.full(
            (len(prompt_tokens), total_len),
            -1,
            dtype=torch.long,
            device=device
        )
        
        # Copy prompts to device
        for i, t in enumerate(prompt_tokens):
            tokens[i, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
            
        # Track generation state
        prev_pos = 0
        finished = torch.tensor([False] * len(prompt_tokens), device=device)
        prompt_mask = tokens != -1
        
        # Generate tokens
        for cur_pos in range(min(prompt_lens), total_len):
            # Get model predictions
            logits = model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            
            # Sample next tokens
            if gen_config.temperature > 0:
                next_token = sample_top_p_top_k(
                    logits,
                    gen_config.temperature,
                    gen_config.top_p,
                    gen_config.top_k
                )
            else:
                next_token = logits.argmax(dim=-1)
                
            # Handle prompt tokens
            next_token = torch.where(
                prompt_mask[:, cur_pos],
                tokens[:, cur_pos],
                next_token
            )
            tokens[:, cur_pos] = next_token
            
            # Check for completion
            finished |= torch.logical_and(
                ~prompt_mask[:, cur_pos],
                next_token == tokenizer.eos_token_id
            )
            prev_pos = cur_pos
            
            # Early stopping if all sequences finished
            if finished.all():
                break
                
        # Extract completions
        completion_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            toks = toks[prompt_lens[i]:prompt_lens[i] + gen_config.max_new_tokens]
            if tokenizer.eos_token_id in toks:
                toks = toks[:toks.index(tokenizer.eos_token_id)]
            completion_tokens.append(toks)
            
        return completion_tokens
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        raise RuntimeError("Text generation failed") from e

@contextmanager
def graceful_exit():
    """Context manager for graceful process termination."""
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    
    def signal_handler(signum, frame):
        logger.info("Received termination signal, cleaning up...")
        if dist.is_initialized():
            dist.destroy_process_group()
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)
        exit(0)
        
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        yield
    finally:
        signal.signal(signal.SIGINT, original_sigint)
        signal.signal(signal.SIGTERM, original_sigterm)

def load_model_and_tokenizer(
    ckpt_path: str,
    config_path: str,
    world_size: int,
    rank: int,
    device: str = "cuda"
) -> tuple[Transformer, PreTrainedTokenizer]:
    """
    Load model and tokenizer.
    
    Args:
        ckpt_path: Path to checkpoint directory
        config_path: Path to model config file
        world_size: Number of distributed processes
        rank: Current process rank
        device: Device to load model on
        
    Returns:
        Tuple of (model, tokenizer)
        
    Raises:
        FileNotFoundError: If checkpoint or config files not found
        RuntimeError: If model loading fails
    """
    try:
        # Load model config
        with open(config_path) as f:
            config = ModelArgs(**json.load(f))
            
        # Initialize model
        with torch.device(device):
            model = Transformer(config).to(device)
            
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
        
        # Load model weights
        model_path = os.path.join(
            ckpt_path,
            f"model{rank}-mp{world_size}.safetensors"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
            
        load_model(model, model_path)
        model.eval()
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"Failed to load model and tokenizer: {str(e)}")
        raise

def process_interactive_session(
    model: Transformer,
    tokenizer: PreTrainedTokenizer,
    gen_config: GenerationConfig,
    world_size: int,
    rank: int
) -> None:
    """
    Run interactive generation session.
    
    Args:
        model: Transformer model
        tokenizer: Tokenizer instance
        gen_config: Generation configuration
        world_size: Number of distributed processes
        rank: Current process rank
    """
    messages: List[Dict[str, str]] = []
    
    try:
        while True:
            # Get input prompt
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
                
            # Handle special commands
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                logger.info("Cleared conversation history")
                continue
                
            # Generate completion
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True
            )
            
            completion_tokens = generate(
                model,
                tokenizer,
                [prompt_tokens],
                gen_config
            )
            
            completion = tokenizer.decode(
                completion_tokens[0],
                skip_special_tokens=True
            )
            
            if rank == 0:
                print(completion)
                
            messages.append({
                "role": "assistant",
                "content": completion
            })
            
    except KeyboardInterrupt:
        logger.info("Interactive session terminated by user")
    except Exception as e:
        logger.error(f"Interactive session error: {str(e)}")
        raise

def process_batch_input(
    model: Transformer,
    tokenizer: PreTrainedTokenizer,
    input_file: str,
    gen_config: GenerationConfig,
    rank: int
) -> None:
    """
    Process batch input from file.
    
    Args:
        model: Transformer model
        tokenizer: Tokenizer instance
        input_file: Path to input file
        gen_config: Generation configuration
        rank: Current process rank
    """
    try:
        # Read prompts
        with open(input_file) as f:
            prompts = [line.strip() for line in f.readlines()]
            
        # Validate batch size
        if len(prompts) > gen_config.batch_size:
            raise ValueError(
                f"Number of prompts ({len(prompts)}) exceeds "
                f"maximum batch size ({gen_config.batch_size})"
            )
            
        # Process batches
        for i in range(0, len(prompts), gen_config.batch_size):
            batch = prompts[i:i + gen_config.batch_size]
            prompt_tokens = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True
                )
                for prompt in batch
            ]
            
            completion_tokens = generate(
                model,
                tokenizer,
                prompt_tokens,
                gen_config
            )
            
            if rank == 0:
                completions = tokenizer.batch_decode(
                    completion_tokens,
                    skip_special_tokens=True
                )
                for prompt, completion in zip(batch, completions):
                    print("Prompt:", prompt)
                    print("Completion:", completion)
                    print()
                    
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        raise

def main(
    ckpt_path: str,
    config_path: str,
    input_file: str = "",
    interactive: bool = True,
    gen_config: Optional[GenerationConfig] = None
) -> None:
    """
    Main entry point for text generation.
    
    Args:
        ckpt_path: Path to model checkpoint
        config_path: Path to model config
        input_file: Optional path to input file
        interactive: Whether to run in interactive mode
        gen_config: Optional generation configuration
    """
    with graceful_exit():
        try:
            # Setup distributed environment
            world_size, rank, local_rank = setup_distributed()
            torch.cuda.set_device(local_rank)
            
            # Configure PyTorch
            torch.set_default_dtype(torch.bfloat16)
            torch.set_num_threads(8)
            torch.manual_seed(965)
            
            # Load model and tokenizer
            model, tokenizer = load_model_and_tokenizer(
                ckpt_path,
                config_path,
                world_size,
                rank
            )
            
            # Use default config if none provided
            if gen_config is None:
                gen_config = GenerationConfig()
                
            # Run generation
            if interactive:
                process_interactive_session(
                    model,
                    tokenizer,
                    gen_config,
                    world_size,
                    rank
                )
            else:
                process_batch_input(
                    model,
                    tokenizer,
                    input_file,
                    gen_config,
                    rank
                )
                
        except Exception as e:
            logger.error(f"Fatal error: {str(e)}")
            raise
        finally:
            # Cleanup
            if world_size > 1 and dist.is_initialized():
                dist.destroy_process_group()

if __name__ == "__main__":
    parser = ArgumentParser(description="Text generation using Transformer model")
    parser.add_argument("--ckpt-path", type=str, required=True,
                      help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True,
                      help="Path to model configuration file")
    parser.add_argument("--input-file", type=str, default="",
                      help="Path to input file for batch processing")
    parser.add_argument("--interactive", action="store_true",
                      help="Run in interactive mode")
    parser.add_argument("--max-new-tokens", type=int, default=200,
                      help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.2,
                      help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9,
                      help="Nucleus sampling probability threshold")
    parser.add_argument("--top-k", type=int, default=50,
                      help="Top-k sampling threshold")
    parser.add_argument("--batch-size", type=int, default=1,
                      help="Batch size for processing")
    
    args = parser.parse_args()
    
    if not (args.input_file or args.interactive):
        parser.error("Either input-file or interactive mode must be specified")
        
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        batch_size=args.batch_size
    )
    
    main(
        args.ckpt_path,
        args.config,
        args.input_file,
        args.interactive,
        gen_config
    )
