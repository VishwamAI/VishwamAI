import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Tuple
from dataclasses import dataclass
from .base_layers import Linear
from .config import ModelArgs
from .tokenizer import VishwamAITokenizer, TokenizerConfig
from .Transformer import Transformer
from .MoE import MoE

def create_model(args: Union[dict, ModelArgs], device: Optional[torch.device] = None) -> Tuple[nn.Module, VishwamAITokenizer]:
    """
    Create a model instance based on configuration.
    Args:
        args: Either a dictionary with model configuration or a ModelArgs instance
        device: Optional device to place the model on
    Returns:
        Tuple of (model, tokenizer)
    """
    # Convert dict to ModelArgs if necessary
    if not isinstance(args, ModelArgs):
        # Convert dictionary configuration to ModelArgs
        if isinstance(args, dict):
            args = ModelArgs(
                max_batch_size=args.get('max_batch_size', 8),
                max_seq_len=args.get('max_seq_len', 32768),
                dtype=args.get('dtype', 'bf16'),
                vocab_size=args.get('vocab_size', 102400),
                dim=args.get('dim', 2048),
                n_layers=args.get('n_layers', 27),
                n_heads=args.get('n_heads', 16),
                n_dense_layers=0 if args.get('model_type') == 'moe' else args.get('n_layers', 27),
                inter_dim=args.get('inter_dim', args.get('dim', 2048) * 4),
                moe_inter_dim=args.get('moe_inter_dim', 1408),
                n_routed_experts=args.get('num_experts', 128) if args.get('model_type') == 'moe' else 0,
                n_shared_experts=args.get('num_shared_experts', 4),
                n_activated_experts=args.get('num_activated_experts', 8),
                n_expert_groups=args.get('num_expert_groups', 2),
                n_limited_groups=args.get('num_limited_groups', 1),
                score_func=args.get('score_func', 'softmax'),
                route_scale=args.get('route_scale', 1.0),
                gradient_checkpointing=args.get('gradient_checkpointing', True),
                use_alibi=args.get('use_alibi', True),
                use_rope_scaling=args.get('use_rope_scaling', True)
            )
        else:
            raise TypeError("args must be either a dictionary or ModelArgs instance")

    # Print configuration for debugging
    print(f"Creating model with configuration:")
    print(f"  dim: {args.dim}")
    print(f"  max_seq_len: {args.max_seq_len}")
    print(f"  n_layers: {args.n_layers}")
    print(f"  n_heads: {args.n_heads}")
    print(f"  n_routed_experts: {args.n_routed_experts}")

    # Determine if we're using MoE
    is_moe = args.n_routed_experts > 0

    # Create base transformer model
    model = Transformer(args=args, device=device)
    
    # Store args in model for later use
    model.args = args
    
    # Wrap with MoE if needed
    if is_moe:
        print("Creating MoE wrapper...")
        model = MoE(model=model, args=args)

    # Create tokenizer
    tokenizer = VishwamAITokenizer(TokenizerConfig(
        vocab_size=args.vocab_size,
        max_sentence_length=args.max_seq_len
    ))

    return model, tokenizer
