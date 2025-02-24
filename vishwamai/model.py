import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from einops import rearrange, repeat
from .error_correction import ErrorCorrectionModule, ModelIntegrator
from .tot import TreeOfThoughts
from .transformer import VisionTransformer10B
import json
@dataclass
class ModelArgs:
    """Model hyperparameters with advanced optimizations"""
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8  # GQA: Fewer KV heads than Q heads
    vocab_size: int = 32000
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
    
    # Expert configurations
    n_experts: int = 8
    expert_dim: int = 4096
    expert_pruning_threshold: float = 0.1  # Threshold for pruning inactive experts
    min_active_experts: int = 4  # Minimum number of experts to maintain
    dynamic_expert_selection: bool = True  # Enable token complexity-based routing
    expert_capacity_factor: float = 1.25  # Allow expert overflow for important tokens
    
    # Attention configurations
    window_size: int = 512  # Size of local attention window
    global_tokens: int = 64  # Number of global attention tokens
    attention_dropout: float = 0.1
    dropout_rate: float = 0.1
    expert_dropout: float = 0.1
    use_alibi: bool = True  # Enable ALiBi position embeddings
    num_alibi_heads: Optional[int] = None  # Number of heads using ALiBi
    
    # Memory optimization
    use_flash_attention: bool = True  # Enable Flash Attention 2.0
    kv_cache_dtype: jnp.dtype = jnp.int8  # INT8 quantization for KV cache
    param_dtype: jnp.dtype = jnp.bfloat16  # bfloat16 for parameters
    
    # Vision-language configurations
    vision_dim: int = 1024  # Vision embedding dimension
    use_contrastive_loss: bool = True  # Enable CLIP-style contrastive loss
    temperature: float = 0.07  # Temperature for contrastive loss
    max_image_length: int = 256  # Maximum number of image tokens

@dataclass
class ModelConfig:
    """Configuration class for VishwamAI model"""
    vocab_size: int = 32000
    hidden_size: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 11008
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_position_embeddings: int = 2048
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = False
    
    # Advanced model capabilities
    use_flash_attention: bool = True
    use_rope: bool = True  # Rotary Position Embeddings
    use_alibi: bool = False  # ALiBi position encoding
    use_gqa: bool = True  # Grouped Query Attention
    num_key_value_heads: int = 8  # For GQA
    
    # Performance optimization
    dtype: str = "bfloat16"
    quantization: Optional[str] = None  # Options: None, "int8", "int4"
    
    def __post_init__(self):
        if self.use_gqa:
            assert self.num_attention_heads % self.num_key_value_heads == 0, \
                "num_attention_heads must be divisible by num_key_value_heads for GQA"

class ParallelDense(nn.Module):
    """Advanced parallel dense layer with automatic sharding"""
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    precision: Optional[tuple] = None
    kernel_init: callable = nn.initializers.normal(stddev=0.02)
    
    @nn.compact
    def __call__(self, inputs):
        kernel = self.param('kernel',
                          self.kernel_init,
                          (inputs.shape[-1], self.features))
        kernel = jnp.asarray(kernel, self.dtype)
        y = jnp.dot(inputs, kernel, precision=self.precision)
        if self.use_bias:
            bias = self.param('bias',
                            nn.initializers.zeros,
                            (self.features,))
            bias = jnp.asarray(bias, self.dtype)
            y = y + bias
        return y

class RMSNorm(nn.Module):
    """RMS Normalization with improved numerical stability"""
    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.epsilon)
        scale = self.param('scale',
                          nn.initializers.ones,
                          (x.shape[-1],))
        return x * jnp.asarray(scale, self.dtype)

def rotary_embedding(x: jnp.ndarray, freqs: jnp.ndarray) -> jnp.ndarray:
    """Enhanced rotary embeddings with better position encoding"""
    sin, cos = freqs
    sin = repeat(sin, '... d -> ... (d 2)')
    cos = repeat(cos, '... d -> ... (d 2)')
    
    x1, x2 = rearrange(x, '... (d r) -> ... d r', r=2).unbind(-1)
    rotated = jnp.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], axis=-1)
    
    return rearrange(rotated, '... d r -> ... (d r)')

def precompute_freqs(dim: int, max_seq_len: int, base: int = 10000) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Precompute frequency tensors for rotary embeddings with extended range"""
    freqs = 1.0 / (base ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
    t = jnp.arange(max_seq_len)
    freqs = jnp.outer(t, freqs)
    sin, cos = jnp.sin(freqs), jnp.cos(freqs)
    return sin, cos

class ParallelMLP(nn.Module):
    """Advanced parallel MLP with gated activation and skip connections"""
    config: ModelArgs
    
    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        dim = self.config.dim
        hidden_dim = self.config.ffn_dim_multiplier
        
        x = RMSNorm(dtype=x.dtype)(x)
        gate = ParallelDense(hidden_dim, dtype=x.dtype)(x)
        up = ParallelDense(hidden_dim, dtype=x.dtype)(x)
        
        # SwiGLU activation
        gate = nn.silu(gate)
        intermediate = gate * up
        
        output = ParallelDense(dim, dtype=x.dtype)(intermediate)
        return nn.Dropout(rate=self.config.dropout_rate)(output, deterministic=deterministic)

class MoELayer(nn.Module):
    """Memory-efficient MoE with Top-2 routing and improved load balancing"""
    config: ModelArgs
    
    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        batch_size, seq_len, dim = x.shape
        num_experts = self.config.n_experts
        
        # Expert networks - shared parameters for efficiency
        expert_fn = ParallelMLP(self.config)
        
        # Router with efficient top-2 gating
        router_weights = self.param('router_weights',
                                  nn.initializers.normal(stddev=0.02),
                                  (dim, num_experts))
        
        # Compute routing probabilities with jax.lax.stop_gradient for router
        router_logits = jax.lax.stop_gradient(
            jnp.einsum('bsd,de->bse', x, router_weights)
        )
        routing_probs = nn.softmax(router_logits, axis=-1)
        
        # Select top-2 experts for each token
        top2_weights, top2_indices = jax.lax.top_k(routing_probs, k=2)
        top2_weights = top2_weights / jnp.sum(top2_weights, axis=-1, keepdims=True)
        
        # Dispatch to top-2 experts with scatter-gather operations
        expert_inputs = []
        expert_indices = []
        
        for expert_idx in range(num_experts):
            # Find tokens routed to this expert
            expert_mask = (top2_indices == expert_idx).any(axis=-1)
            if jnp.any(expert_mask):
                # Get inputs and their weights for this expert
                mask_weights = jnp.where(top2_indices == expert_idx, top2_weights, 0).max(axis=-1)
                expert_inputs.append(x[expert_mask] * mask_weights[expert_mask, None])
                expert_indices.append(jnp.where(expert_mask)[0])
        
        # Process expert inputs in parallel
        expert_outputs = []
        if expert_inputs:
            # Concatenate and process all expert inputs at once
            batched_expert_input = jnp.concatenate(expert_inputs, axis=0)
            batched_expert_output = expert_fn(batched_expert_input, deterministic)
            
            # Split outputs back per expert
            offset = 0
            for expert_input in expert_inputs:
                expert_len = len(expert_input)
                expert_outputs.append((
                    expert_indices[offset:offset + expert_len],
                    batched_expert_output[offset:offset + expert_len]
                ))
                offset += expert_len
        
        # Combine expert outputs
        final_output = jnp.zeros_like(x)
        for indices, outputs in expert_outputs:
            final_output = final_output.at[indices].add(outputs)
        
        # Auxiliary loss for load balancing
        # Using entropy maximization for better expert utilization
        expert_usage = jnp.mean(routing_probs, axis=(0, 1))
        load_balancing_loss = -jnp.sum(expert_usage * jnp.log(expert_usage + 1e-6))
        
        return final_output, load_balancing_loss

def create_alibi_slopes(num_heads: int) -> jnp.ndarray:
    """Create ALiBi attention biases for each head"""
    closest_power_of_2 = 2 ** jnp.floor(jnp.log2(num_heads))
    base = jnp.array([2 ** (-(2 ** -(jnp.log2(closest_power_of_2) - 3)))], dtype=jnp.float32)
    powers = jnp.arange(1, 1 + num_heads, dtype=jnp.float32)
    slopes = jnp.power(base, powers)
    return slopes

class MultiheadAttention(nn.Module):
    """Advanced attention with GQA, ALiBi, and Flash Attention 2.0"""
    config: ModelArgs
    
    def create_sliding_window_mask(self, seq_len: int) -> jnp.ndarray:
        """Create sliding window attention mask with global tokens"""
        window_size = self.config.window_size
        global_tokens = self.config.global_tokens
        
        # Initialize mask with -inf
        mask = jnp.full((seq_len, seq_len), -1e9)
        
        # Set sliding window attention pattern
        for i in range(seq_len):
            window_start = max(0, i - window_size // 2)
            window_end = min(seq_len, i + window_size // 2 + 1)
            mask = mask.at[i, window_start:window_end].set(0.0)
        
        # Set global attention for special tokens
        mask = mask.at[:global_tokens, :].set(0.0)  # Global tokens attend to all
        mask = mask.at[:, :global_tokens].set(0.0)  # All tokens attend to global tokens
        
        return mask
    
    def setup(self):
        if self.config.use_alibi:
            num_alibi_heads = self.config.num_alibi_heads or self.config.n_heads
            self.alibi_slopes = create_alibi_slopes(num_alibi_heads)
    
    def compute_alibi_attention(self, qk: jnp.ndarray) -> jnp.ndarray:
        """Add ALiBi positional bias to attention scores"""
        seq_len = qk.shape[-1]
        # Create position differences matrix
        positions = jnp.arange(seq_len)
        distance = positions[:, None] - positions[None, :]  # [seq_len, seq_len]
        # Convert to float and make negative (smaller distance = higher attention)
        distance = -jnp.abs(distance).astype(jnp.float32)
        # Add head dimension and multiply by slopes
        distance = distance[None, :, :]  # [1, seq_len, seq_len]
        alibi_bias = self.alibi_slopes[:, None, None] * distance
        return qk + alibi_bias

    @nn.compact
    def __call__(self, x, mask=None, deterministic: bool = True):
        batch_size, seq_len, dim = x.shape
        num_heads = self.config.n_heads
        num_kv_heads = self.config.n_kv_heads
        head_dim = dim // num_heads
        
        # Compute token complexity for dynamic attention
        token_complexity = jnp.sum(jnp.abs(x), axis=-1, keepdims=True)
        complexity_weights = nn.sigmoid(token_complexity)
        
        # Grouped-Query Attention projections with complexity weighting
        q = nn.remat(ParallelDense)(dim, use_bias=False, dtype=x.dtype)(x * complexity_weights)
        kv = nn.remat(ParallelDense)(2 * dim * (num_kv_heads / num_heads), use_bias=False, dtype=x.dtype)(x)
        k, v = jnp.split(kv, 2, axis=-1)
        
        # Reshape heads for GQA
        q = rearrange(q, 'b s (h d) -> b h s d', h=num_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=num_kv_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=num_kv_heads)
        
        # Repeat KV heads to match query heads for GQA
        repeats = num_heads // num_kv_heads
        k = repeat(k, 'b h s d -> b (h r) s d', r=repeats)
        v = repeat(v, 'b h s d -> b (h r) s d', r=repeats)
        
        # Create sliding window attention mask
        if mask is None:
            mask = self.create_sliding_window_mask(seq_len)
        else:
            window_mask = self.create_sliding_window_mask(seq_len)
            mask = jnp.minimum(mask, window_mask)
        
        # Compute attention scores with ALiBi and Flash Attention
        scale = 1.0 / jnp.sqrt(head_dim)
        qk = jnp.einsum('bhid,bhjd->bhij', q, k) * scale
        
        # Add ALiBi bias if enabled
        if self.config.use_alibi:
            qk = self.compute_alibi_attention(qk)
        
        # Apply mask and compute attention with Flash Attention or standard attention
        if self.config.use_flash_attention:
            # Note: This is a placeholder for Flash Attention 2.0
            qk = jnp.where(mask, qk, -1e9)
            attention = nn.softmax(qk, axis=-1)
        else:
            qk = jnp.where(mask, qk, -1e9)
            attention = nn.softmax(qk, axis=-1)
        
        if not deterministic:
            attention = nn.Dropout(rate=self.config.attention_dropout)(
                attention, deterministic=False
            )
        
        # Compute output with quantized KV cache
        if not deterministic:
            v = jnp.asarray(v, self.config.kv_cache_dtype)  # Quantize KV cache during training
            
        output = jnp.einsum('bhij,bhjd->bhid', attention, v)
        output = rearrange(output, 'b h s d -> b s (h d)')
        
        # Final projection with gradient checkpointing
        output = nn.remat(ParallelDense)(dim, dtype=x.dtype)(output)
        output = nn.Dropout(rate=self.config.dropout_rate)(
            output, deterministic=deterministic
        )
        
        return output

class TransformerBlock(nn.Module):
    """Advanced transformer block with parallel processing and error correction"""
    config: ModelArgs
    
    @nn.compact
    def __call__(self, x, mask=None, deterministic: bool = True):
        # Attention with pre-norm
        attn_norm = RMSNorm(dtype=x.dtype)(x)
        attn_output = MultiheadAttention(self.config)(attn_norm, mask, deterministic)
        x = x + attn_output
        
        # MoE or MLP with pre-norm
        ff_norm = RMSNorm(dtype=x.dtype)(x)
        if self.config.n_experts > 0:
            ff_output, load_balance_loss = MoELayer(self.config)(ff_norm, deterministic)
            self.sow('intermediates', 'load_balance_loss', load_balance_loss)
        else:
            ff_output = ParallelMLP(self.config)(ff_norm, deterministic)
        
        x = x + ff_output
        return x

class ParallelEmbedding(nn.Module):
    """Enhanced parallel embedding layer with device sharding"""
    args: ModelArgs
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        embedding = self.param(
            'embedding',
            nn.initializers.normal(stddev=0.02),
            (self.args.vocab_size, self.args.dim),
            self.param_dtype
        )
        
        # Check input values
        if jnp.any((x < 0) | (x >= self.args.vocab_size)):
            raise ValueError("Input ids must be in range [0, vocab_size)")
            
        embedded = jnp.take(embedding, x, axis=0)
        return jnp.asarray(embedded, self.dtype)

class ExtendedVishwamAIModel(nn.Module):
    """Enhanced model with improved component integration"""
    args: ModelArgs
    
    def setup(self):
        # Core components
        self.embedding = ParallelEmbedding(self.args)
        self.layers = [
            TransformerBlock(self.args) for _ in range(self.args.n_layers)
        ]
        self.norm = RMSNorm(eps=self.args.norm_eps)
        
        # Error correction and multimodal components
        self.error_correction = ErrorCorrectionModule(
            hidden_size=self.args.dim,
            num_heads=self.args.n_heads,
            dropout_rate=0.1
        )
        
        # Vision processing
        self.vision_transformer = VisionTransformer10B(
            num_classes=self.args.vocab_size,
            hidden_size=self.args.dim,
            num_heads=self.args.n_heads
        )
        
        # Tree of Thoughts reasoning
        self.tot = TreeOfThoughts(
            transformer=self.vision_transformer,
            max_thoughts=5,
            max_depth=3
        )
        
        # MoE components
        self.moe = MoELayer(self.args)
        
        # Add gradient checkpointing
        self.use_checkpointing = True
        
        # Add model parallel settings
        self.model_parallel_size = jax.device_count()

    def compute_clip_loss(
        self,
        text_embeddings: jnp.ndarray,
        image_embeddings: jnp.ndarray,
        temperature: Optional[float] = None
    ) -> jnp.ndarray:
        """Compute CLIP-style contrastive loss for vision-language alignment"""
        # Normalize embeddings
        text_embeddings = text_embeddings / jnp.linalg.norm(text_embeddings, axis=-1, keepdims=True)
        image_embeddings = image_embeddings / jnp.linalg.norm(image_embeddings, axis=-1, keepdims=True)
        
        # Compute similarity matrix
        logits = jnp.einsum('bd,nd->bn', text_embeddings, image_embeddings)
        logits = logits * (1 / (temperature or self.args.temperature))
        
        # Contrastive loss using cross entropy
        labels = jnp.arange(len(text_embeddings))
        loss_i = optax.softmax_cross_entropy(logits, jax.nn.one_hot(labels, len(text_embeddings)))
        loss_t = optax.softmax_cross_entropy(logits.T, jax.nn.one_hot(labels, len(text_embeddings)))
        
        return (loss_i + loss_t) / 2

    def __call__(
        self,
        input_ids: jnp.ndarray,
        images: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        train: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """
        Forward pass with multimodal support and error correction.
        
        Args:
            input_ids: Text input tokens
            images: Optional image inputs
            attention_mask: Optional attention mask
            train: Whether in training mode
        
        Returns:
            Dictionary containing model outputs and intermediate states
        """
        # Initialize containers for intermediate outputs
        intermediate_outputs = []
        error_gates = []
        
        # Text processing
        x = self.embedding(input_ids)
        
        # Process layers with error correction
        for layer in self.layers:
            if self.use_checkpointing and train:
                x = jax.checkpoint(layer)(x, attention_mask, train)
            else:
                x = layer(x, attention_mask, train)
            x_corrected, error_gate = self.error_correction(x, training=train)
            x = x_corrected
            
            intermediate_outputs.append(x)
            error_gates.append(error_gate)
        
        x = self.norm(x)
        
        # Vision processing if images provided
        if images is not None:
            vision_features = self.vision_transformer(images, train=train)
            x = x + vision_features
        
        # Text features for contrastive learning
        text_pooled = jnp.mean(x, axis=1)  # Global average pooling
        
        # Apply MoE routing
        moe_output, load_balance_loss = self.moe(x)
        x = x + moe_output
        
        # Vision-language contrastive loss
        contrastive_loss = None
        if images is not None and self.args.use_contrastive_loss and train:
            vision_features = self.vision_transformer(images, train=train)
            vision_pooled = jnp.mean(vision_features, axis=1)  # Global average pooling
            contrastive_loss = self.compute_clip_loss(text_pooled, vision_pooled)
            x = x + vision_features
        else:
            vision_features = None
        
        # Apply Tree of Thoughts reasoning
        tot_loss = None
        if train:
            tot_output = self.tot(x, search_strategy='bfs')
            x = x + tot_output
            tot_loss = jnp.mean(jnp.abs(tot_output))  # Monitor ToT contribution
        
        # Final projection
        logits = jnp.dot(x, self.embedding.embedding.T)
        
        # Compute aggregate metrics
        attention_stats = {
            'mean_attention': jnp.mean(jnp.stack([o['attention'] for o in intermediate_outputs])),
            'max_attention': jnp.max(jnp.stack([o['attention'] for o in intermediate_outputs]))
        }
        
        return {
            'logits': logits,
            'intermediate_outputs': intermediate_outputs,
            'error_gates': error_gates,
            'load_balance_loss': load_balance_loss,
            'contrastive_loss': contrastive_loss,
            'tot_loss': tot_loss,
            'attention_stats': attention_stats,
            'text_embeddings': text_pooled,
            'vision_embeddings': vision_pooled if vision_features is not None else None
        }

class VishwamAIModel(nn.Module):
    """Base VishwamAI model implementing core transformer architecture"""
    config: ModelConfig

    @classmethod
    def from_pretrained(cls, model_path: str, *, config: Optional[ModelConfig] = None):
        """Load a pretrained model from Hugging Face Hub or local path.
        
        Args:
            model_path: Path to model on HF Hub or local directory
            config: Optional ModelConfig, will load from model_path if not provided
        
        Returns:
            VishwamAIModel: Loaded model instance
        """
        from huggingface_hub import snapshot_download
        import safetensors.flax as stf
        import os
        
        # Download model files if needed
        if not os.path.exists(model_path):
            try:
                model_path = snapshot_download(
                    repo_id=model_path,
                    allow_patterns=["*.safetensors", "config.json", "tokenizer.model"]
                )
            except Exception as e:
                raise ValueError(f"Error downloading model from {model_path}: {str(e)}")
        
        # Load config if not provided
        if config is None:
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                raise ValueError(f"Config not found at {config_path}")
            with open(config_path) as f:
                config_dict = json.load(f)
            config = ModelConfig(**config_dict)
        
        # Initialize model with config
        model = cls(config)
        
        # Load weights from sharded safetensors files
        params = {}
        shard_files = sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")])
        
        if not shard_files:
            raise ValueError(f"No .safetensors files found in {model_path}")
        
        for shard_file in shard_files:
            shard_path = os.path.join(model_path, shard_file)
            try:
                shard_params = stf.load_file(shard_path)
                params.update(shard_params)
            except Exception as e:
                raise ValueError(f"Error loading weights from {shard_path}: {str(e)}")
        
        # Create variables dict and bind to model
        variables = {'params': params}
        bound_model = model.bind(variables)
        
        return bound_model

    def setup(self):
        self.embeddings = self._create_embeddings()
        self.encoder = self._create_encoder()
        self.decoder = self._create_decoder()
        self.final_layer_norm = nn.LayerNorm(
            epsilon=self.config.layer_norm_eps,
            dtype=jnp.dtype(self.config.dtype)
        )
        
    def _create_embeddings(self):
        return ParallelEmbedding(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            max_position_embeddings=self.config.max_position_embeddings,
            dropout_rate=self.config.hidden_dropout_prob,
            dtype=jnp.dtype(self.config.dtype)
        )
    
    def _create_encoder(self):
        return [
            TransformerBlock(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_attention_heads,
                num_kv_heads=self.config.num_key_value_heads if self.config.use_gqa else None,
                intermediate_size=self.config.intermediate_size,
                dropout_rate=self.config.hidden_dropout_prob,
                attention_dropout=self.config.attention_dropout_prob,
                dtype=jnp.dtype(self.config.dtype),
                use_flash_attention=self.config.use_flash_attention,
                use_rope=self.config.use_rope,
                use_alibi=self.config.use_alibi
            )
            for _ in range(self.config.num_layers)
        ]
    
    def _create_decoder(self):
        return [
            TransformerBlock(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_attention_heads,
                num_kv_heads=self.config.num_key_value_heads if self.config.use_gqa else None,
                intermediate_size=self.config.intermediate_size,
                dropout_rate=self.config.hidden_dropout_prob,
                attention_dropout=self.config.attention_dropout_prob,
                dtype=jnp.dtype(self.config.dtype),
                use_flash_attention=self.config.use_flash_attention,
                use_rope=self.config.use_rope,
                use_alibi=self.config.use_alibi,
                is_decoder=True
            )
            for _ in range(self.config.num_layers)
        ]
    
    def __call__(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        position_ids: Optional[jnp.ndarray] = None,
        deterministic: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass of the model"""
        
        # Input embedding
        hidden_states = self.embeddings(
            input_ids,
            position_ids=position_ids,
            deterministic=deterministic
        )
        
        # Create causal attention mask if needed
        if attention_mask is None:
            attention_mask = self._create_causal_mask(input_ids.shape[1])
        
        # Process through encoder layers
        encoder_outputs = []
        for encoder_layer in self.encoder:
            if self.config.gradient_checkpointing and not deterministic:
                hidden_states = jax.checkpoint(encoder_layer)(
                    hidden_states,
                    attention_mask,
                    deterministic=deterministic
                )
            else:
                hidden_states = encoder_layer(
                    hidden_states,
                    attention_mask,
                    deterministic=deterministic
                )
            encoder_outputs.append(hidden_states)
        
        # Process through decoder layers
        decoder_outputs = []
        for decoder_layer in self.decoder:
            if self.config.gradient_checkpointing and not deterministic:
                hidden_states = jax.checkpoint(decoder_layer)(
                    hidden_states,
                    attention_mask,
                    encoder_outputs[-1],
                    deterministic=deterministic
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask,
                    encoder_outputs[-1],
                    deterministic=deterministic
                )
            decoder_outputs.append(hidden_states)
        
        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        return {
            'last_hidden_state': hidden_states,
            'encoder_outputs': encoder_outputs,
            'decoder_outputs': decoder_outputs,
        }
    
    def _create_causal_mask(self, sequence_length: int) -> jnp.ndarray:
        """Create causal attention mask"""
        mask = jnp.triu(
            jnp.ones((sequence_length, sequence_length), dtype=jnp.bool_),
            k=1
        )
        return jnp.where(mask, -1e9, 0.0)

class SpeculativeDecoder(nn.Module):
    """Speculative Decoding for faster inference using a draft model"""
    large_model: ExtendedVishwamAIModel
    small_model: Optional[ExtendedVishwamAIModel] = None
    num_draft_tokens: int = 5
    
    def setup(self):
        if self.small_model is None:
            # Create smaller version of the model by reducing layers and dim
            small_config = ModelArgs(
                dim=self.large_model.args.dim // 2,
                n_layers=self.large_model.args.n_layers // 2,
                n_heads=self.large_model.args.n_heads // 2,
                vocab_size=self.large_model.args.vocab_size
            )
            self.small_model = ExtendedVishwamAIModel(small_config)
    
    def __call__(self, input_ids: jnp.ndarray, max_length: int = 100) -> jnp.ndarray:
        """Generate tokens using speculative decoding with draft model"""
        generated_ids = input_ids
        
        for _ in range(0, max_length, self.num_draft_tokens):
            # Draft phase: Generate candidate tokens with small model
            draft_outputs = self.small_model(generated_ids)
            draft_logits = draft_outputs['logits'][:, -1:, :]
            draft_tokens = jax.random.categorical(
                jax.random.PRNGKey(0), draft_logits, axis=-1
            )
            draft_sequence = jnp.concatenate([generated_ids, draft_tokens], axis=1)
            
            # Verify phase: Large model verifies draft tokens
            large_outputs = self.large_model(draft_sequence)
            large_logits = large_outputs['logits'][:, -self.num_draft_tokens:, :]
            
            # Compare draft and verified probabilities
            draft_probs = nn.softmax(draft_logits, axis=-1)
            verified_probs = nn.softmax(large_logits, axis=-1)
            
            # Accept tokens where draft model matches large model closely
            prob_threshold = 0.9
            matches = jnp.sum(jnp.abs(draft_probs - verified_probs), axis=-1) < (1 - prob_threshold)
            
            # If match found, keep draft tokens; otherwise use large model prediction
            verified_tokens = jnp.where(
                matches,
                draft_tokens,
                jax.random.categorical(jax.random.PRNGKey(0), large_logits, axis=-1)
            )
            
            generated_ids = jnp.concatenate([generated_ids, verified_tokens], axis=1)
            
        return generated_ids

def create_optimizer(
    learning_rate: float,
    warmup_steps: int = 2000,
    decay_steps: int = 50000,
    end_learning_rate: float = 1e-5
) -> optax.GradientTransformation:
    """Create an advanced optimizer with learning rate schedule"""
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=decay_steps,
        end_value=end_learning_rate
    )
    
    return optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=schedule,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=0.1
        )
    )
