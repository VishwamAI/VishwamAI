import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
import logging
import torch.nn.utils.prune
import torch.nn.utils.prune
from transformers import PreTrainedModel, PretrainedConfig

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device='cpu').float() / dim))
    t = torch.arange(end, device='cpu')
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Ensure freqs_cis matches the sequence length
    seqlen = xq.size(1)
    freqs_cis = freqs_cis[:seqlen]
    if freqs_cis.size(0) != seqlen:
        raise ValueError(f"freqs_cis size {freqs_cis.size(0)} does not match sequence length {seqlen}")
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis[:, None, :]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (x[:, :, :, None, :]
            .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
            .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim))

@dataclass
class VishwamaiConfig(PretrainedConfig):
    model_type: str = "vishwamai"
    dim: int = 4096  # Model dimension
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = 102400
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 8192
    rope_theta: float = 10000.0
    qk_dim: int = 128
    v_dim: int = 128
    rope_scaling: Optional[float] = None

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

class AdvancedAttention(nn.Module):
    def __init__(self, config: VishwamaiConfig):
        super().__init__()
        self.config = config  # Add this line to store the config
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads or config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.dim // self.n_heads
        
        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        
        self.cache_k = torch.zeros((config.max_batch_size, config.max_seq_len, 
                                  self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((config.max_batch_size, config.max_seq_len,
                                  self.n_kv_heads, self.head_dim))

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        
        self.cache_k = self.cache_k.to(xq.device).type_as(xq)
        self.cache_v = self.cache_v.to(xq.device).type_as(xq)

        if start_pos + seqlen > self.config.max_seq_len:
            raise ValueError(f"Assigning xk with seqlen {seqlen} exceeds max_seq_len {self.config.max_seq_len}")

        self.cache_k[:bsz, start_pos:start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos:start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, :start_pos + seqlen]
        values = self.cache_v[:bsz, :start_pos + seqlen]

        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        xq = xq.transpose(1, 2)  # [bsz, n_heads, seqlen, head_dim]
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # Ensure mask has correct dimensions for attention scores
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seqlen, seqlen]
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [bsz, 1, seqlen, seqlen]
            
            # Ensure mask can be broadcast to scores shape
            target_shape = (bsz, self.n_heads, seqlen, scores.size(-1))
            if mask.size(-1) != scores.size(-1):  # Fix size mismatch
                pad_size = scores.size(-1) - mask.size(-1)
                if pad_size > 0:
                    mask = F.pad(mask, (0, pad_size), value=float('-inf'))
                # Expand head dimension only
                mask = mask.expand(-1, self.n_heads, -1, -1)
            
            scores = scores + mask.type_as(scores)
        
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, values)
        
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

class VishwamaiBlock(nn.Module):
    def __init__(self, config: VishwamaiConfig):
        super().__init__()
        self.attention = AdvancedAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.dim, 4 * config.dim),
            nn.GELU(),
            nn.Linear(4 * config.dim, config.dim)
        )
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

class VishwamaiV1(PreTrainedModel):
    config_class = VishwamaiConfig
    def __init__(self, config: VishwamaiConfig):
        super().__init__(config)
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        
        self.layers = nn.ModuleList()
        for _ in range(config.n_layers):
            self.layers.append(VishwamaiBlock(config))

        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis = precompute_freqs_cis(
            config.dim // config.n_heads,
            config.max_seq_len * 2,
            config.rope_theta
        )

        self.prune_ratio = 0.1  # Prune 10% of weights
        
        # Apply initial pruning
        self.apply_pruning()  # Apply pruning after initializing parameters
        # Ensure all parameters require gradients after pruning
        for param in self.parameters():
            param.requires_grad = True

        # Initialize query and key layers for attention
        self.query = nn.Linear(config.dim, config.dim)
        self.key = nn.Linear(config.dim, config.dim)
    
    def apply_pruning(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                torch.nn.utils.prune.l1_unstructured(module, name='weight', amount=self.prune_ratio)
                # Remove pruning re-parametrization to make it permanent
                torch.nn.utils.prune.remove(module, 'weight')

    def forward(self, tokens: torch.Tensor, start_pos: int = 0, attention_mask: Optional[torch.Tensor] = None):
        try:
            if tokens.size(0) == 0:
                raise ValueError("Batch size cannot be zero.")
            if tokens.dim() != 2:
                raise ValueError("tokens must be a 2D tensor [batch_size, sequence_length]")
            
            if (tokens >= self.config.vocab_size).any():
                raise ValueError("Token indices exceed vocabulary size.")
            
            bsz, seqlen = tokens.shape
            
            # Always truncate if sequence length exceeds max_seq_len
            if seqlen > self.config.max_seq_len:
                tokens = tokens[:, :self.config.max_seq_len]
                seqlen = self.config.max_seq_len
            
            # Additional check for start_pos + seqlen
            if start_pos + seqlen > self.config.max_seq_len:
                seqlen = self.config.max_seq_len - start_pos
                tokens = tokens[:, :seqlen]
            
            h = self.tok_embeddings(tokens)
            self.freqs_cis = self.freqs_cis.to(h.device)
            freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]

            # Create causal mask
            mask = torch.triu(
                torch.full((seqlen, seqlen), float('-inf'), device=h.device),
                diagonal=1
            ) if seqlen > 1 else None

            # Apply attention mask if provided
            if attention_mask is not None:
                attention_mask = attention_mask[:, :seqlen].unsqueeze(1)
                if mask is None:
                    mask = attention_mask
                else:
                    mask = mask + attention_mask
            
            for layer in self.layers:
                h = layer(h, start_pos, freqs_cis, mask)
            
            h = self.norm(h)
            return self.output(h)
            
        except Exception as e:
            logging.error(f"Real-world error in VishwamaiV1.forward: {e}")
            raise e

    def forward_hidden(self, tokens: torch.Tensor, start_pos: int = 0):
        if tokens.size(0) == 0:
            raise ValueError("Batch size cannot be zero.")
        if tokens.dim() != 2:
            raise ValueError("tokens must be a 2D tensor [batch_size, sequence_length]")
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([
                torch.zeros((seqlen, start_pos), device=tokens.device),
                mask
            ]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)
        h = self.norm(h)
        return h

    def robust_inference(self, tokens: torch.Tensor):
        if tokens.size(1) > self.config.max_seq_len:
            logging.warning("Sequence length exceeds max_seq_len, truncating.")
            tokens = tokens[:, :self.config.max_seq_len]
        return self.forward(tokens)

    def generate(self, input_ids: torch.Tensor,
                 max_length: int, domain_type: str = None,
                 do_sample: bool = True, temperature: float = 1.0,
                 top_k: int = 0, top_p: float = 1.0,
                 context_strategy: str = None,
                 monitor_resources: bool = False,
                 start_pos: int = 0):
        try:
            # Truncate input if it exceeds max_seq_len
            if (input_ids.size(1) > self.config.max_seq_len):
                input_ids = input_ids[:, :self.config.max_seq_len]
            
            # Initialize generated with truncated input
            generated = input_ids.clone()
            
            # Ensure max_length doesn't exceed model's max_seq_len
            max_length = min(max_length, self.config.max_seq_len)
            
            with torch.no_grad():
                for _ in range(max_length - generated.size(1)):
                    # Create causal mask for current sequence
                    seqlen = generated.size(1)
                    mask = torch.triu(
                        torch.full((seqlen, seqlen), float('-inf'), device=generated.device),
                        diagonal=1
                    )
                    
                    outputs = self.forward(generated, start_pos=start_pos, attention_mask=mask)
                    next_token_logits = outputs[:, -1, :] / temperature

                    # Apply sampling methods
                    if top_k > 0 or top_p < 1.0:
                        next_token_logits = self.adjust_logits_for_sampling(
                            next_token_logits, top_k=top_k, top_p=top_p
                        )

                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, 1) if do_sample else torch.argmax(probs, dim=-1).unsqueeze(-1)
                    
                    generated = torch.cat((generated, next_token), dim=1)
                    start_pos += 1
                    
                    if generated.size(1) >= self.config.max_seq_len:
                        break

            return generated

        except Exception as e:
            logging.error(f"Error in generate: {str(e)}")
            raise

    def generate_incremental(self, input_ids: torch.Tensor, max_length: int, 
                             domain_type: str = None, do_sample: bool = True,
                             temperature: float = 1.0, top_k: int = 0, 
                             top_p: float = 1.0, context_strategy: str = None,
                             return_generator: bool = False):
        self.eval()
        generated = input_ids
        eos_token = self.config.vocab_size - 1  # Assuming the last token is EOS
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                outputs = self.forward(generated)
                next_token_logits = outputs[:, -1, :] / temperature

                if top_k > 0:
                    top_k_values, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
                    mask = torch.full_like(next_token_logits, float('-inf'))
                    mask.scatter_(dim=-1, index=top_k_indices, src=top_k_values)
                    next_token_logits = mask

                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0

                    next_token_logits = next_token_logits.masked_fill(sorted_indices_to_remove, float('-inf'))

                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1) if do_sample else torch.argmax(probs, dim=-1).unsqueeze(-1)
                generated = torch.cat((generated, next_token), dim=1)

                # Update finished mask
                finished = finished | (next_token.squeeze(-1) == eos_token)
                if finished.all():
                    break

        return generated

    def load_generation_state(self, state):
        """
        Load the generation state for the model.

        Args:
            state (dict): The state dictionary to load.
        """
        # Ensure that the state is a dictionary
        if not isinstance(state, dict):
            raise TypeError("Generation state must be a dictionary.")
        
        # Load the state dictionary into the model
        self.load_state_dict(state)
        # Additional handling for generation state if necessary
        self.generation_state = state  # Example attribute

    def generate_with_safety(self, input_ids: torch.Tensor, **kwargs):
        """
        Generates text with safety checks applied to the output.
        
        Args:
            input_ids (torch.Tensor): The input token IDs.
            **kwargs: Additional parameters for generation.
        
        Returns:
            torch.Tensor: The generated token IDs after safety checks.
        """
        try:
            # Generate output using the existing generate method
            generated_ids = self.generate(input_ids, max_length=kwargs.get('max_length', 50), **kwargs)
            
            # Apply safety checks to the generated output
            # Example: Filter out unsafe tokens or modify the output accordingly
            safe_generated_ids = self.apply_safety_checks(generated_ids)
            
            return safe_generated_ids
        except Exception as e:
            logging.error(f"Error in generate_with_safety: {e}")
            raise e

    def apply_safety_checks(self, generated_ids: torch.Tensor) -> torch.Tensor:
        """
        Applies safety filters to the generated token IDs.
        
        Args:
            generated_ids (torch.Tensor): The generated token IDs.
        
        Returns:
            torch.Tensor: The token IDs after safety filtering.
        """
        # Implement safety mechanisms such as:
        # - Detecting and removing harmful content
        # - Ensuring outputs adhere to specific guidelines
        # For demonstration, we'll assume all outputs are safe
        # Replace this with actual safety logic as needed
        
        # Example placeholder for safety check
        # safe_ids = some_safety_filter_function(generated_ids)
        safe_ids = generated_ids  # No-op for now
        
        return safe_ids

    def save_pretrained(self, save_directory: str, **kwargs):
        super().save_pretrained(save_directory, **kwargs)
        # Add any additional saving mechanisms if necessary

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *model_args, **kwargs):
        config = VishwamaiConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
        model = super().from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)
        return model

    """
def init_model(config: VishwamaiConfig) -> VishwamaiV1:
    model = VishwamaiV1(config)
    # Initialize with proper scaling
    for name, param in model.named_parameters():
        if param.ndim == 2:
            torch.nn.init.normal_(param, mean=0.0, std=0.02)
        # Ensure all parameters require gradients
        for param in model.parameters():
            param.requires_grad = True
    return model

# Add the following line at the end of the file to create an alias
TransformerBlock = VishwamaiBlock
    Pushes the model and tokenizer to Hugging Face Hub.
    
    Args:
        model: The trained model instance.
        tokenizer: The tokenizer instance.
        model_name (str): The name to assign to the model on Hugging Face Hub.
        token (str): Hugging Face authentication token.
    """
    model.push_to_hub(model_name, use_auth_token=token)
    tokenizer.push_to_hub(model_name, use_auth_token=token)

# Add the following line at the end of the file to create an alias
TransformerBlock = VishwamaiBlock