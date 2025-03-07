import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import List, Dict, Tuple, Optional, Callable, Any
from functools import partial
import logging
import numpy as np

from .model import MoELayer, ModelConfig, ParallelDense
from .transformer import VishwamAIModel
from .tot import TreeOfThoughts, Thought
from .error_correction import ErrorCorrectionModule, MixtureDensityNetwork as ErrorMDN

logger = logging.getLogger(__name__)

class MixtureDensityNetwork(nn.Module):
    """
    TPU-optimized Mixture of Depths (MoD) implementation that varies network depth 
    dynamically based on input complexity.
    """
    hidden_size: int
    num_mixtures: int = 5
    dropout_rate: float = 0.1
    use_bfloat16: bool = True
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> Tuple[jnp.ndarray, jnp.ndarray]:
        # Convert to optimal dtype for TPU
        dtype = jnp.bfloat16 if self.use_bfloat16 else jnp.float32
        x = x.astype(dtype)
        
        # Compute input complexity features with TPU optimization
        complexity_features = nn.Dense(self.hidden_size // 2, dtype=dtype)(x)
        complexity_features = nn.relu(complexity_features)
        complexity_features = nn.Dense(self.num_mixtures, dtype=dtype)(complexity_features)
        
        # Compute mixture weights with TPU-optimized softmax
        mixture_weights = nn.softmax(complexity_features, axis=-1)
        
        # TPU-optimized path creation with JAX's vmap for parallel processing
        def create_path(depth_index, input_tensor):
            depth = depth_index + 1
            path = input_tensor
            
            # Use scan for better TPU performance with sequential operations
            def process_layer(layer_state, _):
                path = layer_state
                path = nn.Dense(self.hidden_size, dtype=dtype)(path)
                path = nn.relu(path)
                if not deterministic:
                    path = nn.Dropout(rate=self.dropout_rate)(path, deterministic=False)
                return path, None
            
            # Apply depth-specific layers with scan
            path, _ = jax.lax.scan(process_layer, path, None, length=depth)
            path = nn.Dense(self.hidden_size, dtype=dtype)(path)
            return path
        
        # Create paths for different depths in parallel
        depth_indices = jnp.arange(self.num_mixtures)
        paths = jax.vmap(create_path, in_axes=(0, None))(depth_indices, x)
        
        # Apply mixture weights with TPU optimization
        weighted_sum = jnp.sum(paths * mixture_weights[..., None], axis=0)
        
        return weighted_sum, mixture_weights

class ToTIntegrationLayer(nn.Module):
    """
    TPU-optimized integration layer that connects Tree of Thoughts with
    error correction and other model components like MoE and MoD.
    """
    config: Any
    use_error_correction: bool = True
    use_tot: bool = True
    use_bfloat16: bool = True
    
    def setup(self):
        # Extract hidden size consistently
        hidden_size = self.config.hidden_size if hasattr(self.config, 'hidden_size') else self.config.dim
        
        # Create MoE layer for thought processing
        self.thought_moe = MoELayer(self.config)
        
        # Create MoD layer for dynamic depth processing
        self.thought_mod = MixtureDensityNetwork(
            hidden_size=hidden_size,
            use_bfloat16=self.use_bfloat16
        )
        
        # Create error correction module if enabled
        if self.use_error_correction:
            self.error_module = ErrorCorrectionModule(
                hidden_dim=hidden_size,
                use_bfloat16=self.use_bfloat16
            )
        
        # Create gating mechanisms with TPU-optimized initialization
        self.tot_gate = nn.Dense(hidden_size, dtype=jnp.bfloat16 if self.use_bfloat16 else jnp.float32)
        self.model_gate = nn.Dense(hidden_size, dtype=jnp.bfloat16 if self.use_bfloat16 else jnp.float32)
        self.error_gate = nn.Dense(hidden_size, dtype=jnp.bfloat16 if self.use_bfloat16 else jnp.float32)
        
        # Fusion layer with TPU optimization
        self.fusion = nn.Dense(hidden_size, dtype=jnp.bfloat16 if self.use_bfloat16 else jnp.float32)
        self.layer_norm = nn.LayerNorm(epsilon=1e-5)
    
    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self, 
        model_features: jnp.ndarray, 
        tot_features: Optional[jnp.ndarray] = None, 
        deterministic: bool = True,
        error_indicator: Optional[jnp.ndarray] = None
    ):
        """
        TPU-optimized integration of model features with Tree of Thoughts and error correction.
        
        Args:
            model_features: Features from the base model
            tot_features: Optional features from the Tree of Thoughts
            deterministic: Whether to use deterministic mode
            error_indicator: Optional tensor indicating error probabilities
            
        Returns:
            Integrated features and additional information
        """
        # Convert to optimal dtype for TPU
        dtype = jnp.bfloat16 if self.use_bfloat16 else jnp.float32
        model_features = model_features.astype(dtype)
        
        # Process with error correction if enabled
        error_corrected_features = None
        if self.use_error_correction:
            error_outputs = self.error_module(
                hidden_states=model_features,
                deterministic=deterministic
            )
            error_corrected_features = error_outputs["corrected_states"]
            error_probs = error_outputs["error_probs"]
        else:
            error_corrected_features = model_features
            error_probs = jnp.zeros(model_features.shape[:-1] + (1,), dtype=dtype)
        
        # Process with ToT if enabled and features provided
        tot_output = None
        mixture_weights = None
        load_balance_loss = None
        
        if self.use_tot and tot_features is not None:
            tot_features = tot_features.astype(dtype)
            
            # Process thoughts using MoE with TPU optimization
            tot_moe_output, load_balance_loss = self.thought_moe(
                tot_features.reshape(-1, tot_features.shape[-1]), 
                deterministic
            )
            tot_moe_output = tot_moe_output.reshape(tot_features.shape)
            
            # Process thoughts using MoD with TPU optimization
            tot_mod_output, mixture_weights = self.thought_mod(tot_features, deterministic)
            
            # Combine MoE and MoD outputs
            tot_output = tot_moe_output + tot_mod_output
        else:
            # If no ToT, just use model features
            tot_output = model_features
        
        # Create adaptive gates with TPU optimization
        if tot_output is not None:
            tot_importance = jax.nn.sigmoid(self.tot_gate(tot_output))
        else:
            tot_importance = jnp.zeros_like(model_features[..., :1])
            
        model_importance = jax.nn.sigmoid(self.model_gate(model_features))
        error_importance = jax.nn.sigmoid(self.error_gate(error_corrected_features))
        
        # Use error information to adjust importance if available
        if error_indicator is not None:
            error_importance = error_importance * error_indicator
        elif error_probs is not None:
            error_importance = error_importance * error_probs
        
        # Normalize importance scores with TPU optimization
        total_importance = tot_importance + model_importance + error_importance + 1e-6
        tot_weight = tot_importance / total_importance
        model_weight = model_importance / total_importance
        error_weight = error_importance / total_importance
        
        # Weighted combination with TPU optimization
        if tot_output is not None:
            combined = (
                tot_weight * tot_output + 
                model_weight * model_features + 
                error_weight * error_corrected_features
            )
        else:
            combined = model_weight * model_features + error_weight * error_corrected_features
        
        # Apply fusion layer and normalization
        fused = self.fusion(combined)
        output = self.layer_norm(fused)
        
        # Collect integration information
        integration_info = {
            'tot_weight': tot_weight,
            'model_weight': model_weight,
            'error_weight': error_weight,
            'mixture_weights': mixture_weights,
            'load_balance_loss': load_balance_loss,
            'error_probs': error_probs
        }
        
        return output, integration_info

class MultiLevelToTAttention(nn.Module):
    """
    TPU-optimized Multi-Level Attention for Tree of Thoughts integration that 
    connects thought processes across different abstraction levels.
    """
    hidden_size: int
    num_heads: int = 8
    dropout_rate: float = 0.1
    use_bfloat16: bool = True
    
    @nn.compact
    def __call__(
        self, 
        model_features: jnp.ndarray, 
        tot_features: Optional[jnp.ndarray] = None,
        thoughts: Optional[List[jnp.ndarray]] = None,
        deterministic: bool = True
    ):
        """
        TPU-optimized multi-level attention between model features and thoughts.
        
        Args:
            model_features: Features from the base model
            tot_features: Features from the Tree of Thoughts (optional)
            thoughts: List of thought representations at different levels (optional)
            deterministic: Whether to use deterministic mode
            
        Returns:
            Enhanced features with thought-based attention
        """
        # Convert to optimal dtype for TPU
        dtype = jnp.bfloat16 if self.use_bfloat16 else jnp.float32
        model_features = model_features.astype(dtype)
        
        # If no thoughts provided, just return model features
        if (thoughts is None or len(thoughts) == 0) and tot_features is None:
            return model_features, None
        
        # Process ToT features directly if no individual thoughts
        if thoughts is None or len(thoughts) == 0:
            if tot_features is not None:
                tot_features = tot_features.astype(dtype)
                
                # Single attention layer for tot_features with TPU optimization
                attention = nn.MultiHeadDotProductAttention(
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                    dtype=dtype
                )(
                    model_features, tot_features, deterministic=deterministic
                )
                
                # Simple residual connection and normalization
                output = model_features + attention
                output = nn.LayerNorm(dtype=dtype)(output)
                return output, jnp.ones((model_features.shape[0], 1), dtype=dtype)
            
            return model_features, None
        
        # Process each thought level with separate attention
        thought_outputs = []
        for i, thought in enumerate(thoughts):
            thought = thought.astype(dtype)
            
            # Cross-attention between model features and thought
            attention = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                dropout_rate=self.dropout_rate,
                dtype=dtype
            )(
                model_features, thought.reshape(1, -1, thought.shape[-1]), deterministic=deterministic
            )
            
            thought_outputs.append(attention)
        
        # Compute importance weights for each thought level with TPU optimization
        scores = []
        for output in thought_outputs:
            # Calculate feature relevance score
            score = nn.Dense(1, dtype=dtype)(output)
            scores.append(score)
        
        # Normalize weights using JAX's numerically stable softmax
        scores_concat = jnp.concatenate(scores, axis=-1)
        attention_weights = jax.nn.softmax(scores_concat, axis=-1)
        
        # Combine thought outputs based on weights with TPU optimization
        processed_outputs = [
            nn.Dense(self.hidden_size, dtype=dtype)(output) 
            for output in thought_outputs
        ]
        
        # Stack and combine with efficient TPU operations
        stacked_outputs = jnp.stack(processed_outputs, axis=1)
        weighted_combination = jnp.sum(
            stacked_outputs * attention_weights[..., None], 
            axis=1
        )
        
        # Combine with original features through residual connection
        output = model_features + weighted_combination
        output = nn.LayerNorm(dtype=dtype)(output)
        
        return output, attention_weights

class ToTModelIntegrator:
    """
    TPU-optimized helper class to integrate TreeOfThoughts with model components.
    """
    def __init__(
        self, 
        model: VishwamAIModel,
        tot_model: TreeOfThoughts,
        config: Any,
        use_error_correction: bool = True,
        use_dualpipe: bool = True,
        use_bfloat16: bool = True
    ):
        """
        Initialize the integrator with TPU optimizations.
        
        Args:
            model: Base VishwamAI model
            tot_model: Tree of Thoughts model
            config: Configuration
            use_error_correction: Whether to use error correction
            use_dualpipe: Whether to use dualpipe processing
            use_bfloat16: Whether to use bfloat16 for TPU optimization
        """
        self.model = model
        self.tot_model = tot_model
        self.config = config
        self.use_error_correction = use_error_correction
        self.use_dualpipe = use_dualpipe
        self.use_bfloat16 = use_bfloat16
        
        # Extract hidden size consistently
        hidden_size = config.hidden_size if hasattr(config, 'hidden_size') else config.dim
        
        # Initialize components with TPU optimization options
        self.integration_layer = ToTIntegrationLayer(
            config, 
            use_error_correction=use_error_correction,
            use_bfloat16=use_bfloat16
        )
        
        self.mla = MultiLevelToTAttention(
            hidden_size=hidden_size,
            use_bfloat16=use_bfloat16
        )
        
        # Set up TPU mesh sharding if multiple devices
        self._setup_tpu_sharding()
    
    def _setup_tpu_sharding(self):
        """Configure TPU sharding for optimal performance."""
        try:
            # Get available TPU devices
            devices = jax.devices("tpu")
            num_devices = len(devices)
            
            if num_devices > 0:
                # Create device mesh for sharding
                device_mesh = jax.sharding.Mesh(
                    np.array(devices).reshape(-1),
                    ('batch',)
                )
                self.device_mesh = device_mesh
                logger.info(f"ToT integrator configured with {num_devices} TPU devices")
            else:
                self.device_mesh = None
                logger.warning("No TPU devices found for integrator, falling back to CPU")
        except Exception as e:
            self.device_mesh = None
            logger.error(f"Failed to initialize TPU for integrator: {str(e)}")
    
    @partial(jax.jit, static_argnums=(0, 3))
    def process_thought_tree(self, thought: Thought, max_depth: int = 5):
        """
        TPU-optimized extraction of features from thought tree.
        
        Args:
            thought: Root thought from ToT
            max_depth: Maximum depth to process
            
        Returns:
            List of thought features at different depths
        """
        dtype = jnp.bfloat16 if self.use_bfloat16 else jnp.float32
        thought_features = []
        
        def extract_thought_path(current_thought, depth=0):
            if depth >= max_depth or current_thought is None:
                return []
                
            features = current_thought.embeddings.astype(dtype)
            thought_features.append(features)
            
            if current_thought.children:
                # Find best child by score
                best_child = None
                best_score = float('-inf')
                
                for child in current_thought.children:
                    if child.score > best_score:
                        best_score = child.score
                        best_child = child
                        
                if best_child:
                    return extract_thought_path(best_child, depth + 1)
            return thought_features
            
        return extract_thought_path(thought)
    
    def create_dualpipe_processing(self, inputs):
        """
        Create dualpipe processing for TPU-optimized parallel computation.
        
        Args:
            inputs: Input tensor
            
        Returns:
            Forward and backward processing tensors
        """
        if not self.use_dualpipe:
            return inputs, None
            
        # Split input for dualpipe processing
        batch_size = inputs.shape[0]
        split_point = batch_size // 2
        
        forward_inputs = inputs[:split_point]
        backward_inputs = inputs[split_point:] if split_point < batch_size else None
        
        return forward_inputs, backward_inputs
    
    @partial(jax.jit, static_argnums=(0, 4, 5))
    def generate_with_tot(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        rng_key: Optional[jnp.ndarray] = None,
        search_strategy: str = "beam",
        chunk_size: int = 32
    ):
        """
        TPU-optimized generation with Tree of Thoughts integration.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            rng_key: Random key for ToT search
            search_strategy: ToT search strategy
            chunk_size: Chunk size for memory-efficient processing
            
        Returns:
            Enhanced model outputs
        """
        logger.info(f"Generating with ToT using {search_strategy} search")
        dtype = jnp.bfloat16 if self.use_bfloat16 else jnp.float32
        
        # Initialize random key if not provided
        if rng_key is None:
            rng_key = jax.random.PRNGKey(0)
        
        # Process in chunks to optimize memory
        chunked_outputs = []
        for i in range(0, input_ids.shape[0], chunk_size):
            chunk_ids = input_ids[i:i+chunk_size]
            chunk_mask = attention_mask[i:i+chunk_size] if attention_mask is not None else None
            
            # Use dualpipe if enabled
            if self.use_dualpipe:
                forward_ids, backward_ids = self.create_dualpipe_processing(chunk_ids)
                forward_mask, backward_mask = (
                    self.create_dualpipe_processing(chunk_mask)
                    if chunk_mask is not None else (None, None)
                )
                
                # Forward processing
                forward_outputs = self._process_chunk(
                    forward_ids, 
                    forward_mask, 
                    rng_key,
                    search_strategy
                )
                
                # Backward processing (if we have backward inputs)
                if backward_ids is not None:
                    rng_key, backward_key = jax.random.split(rng_key)
                    backward_outputs = self._process_chunk(
                        backward_ids,
                        backward_mask,
                        backward_key,
                        search_strategy
                    )
                    
                    # Combine forward and backward outputs
                    chunk_output = self._combine_dualpipe_outputs(
                        forward_outputs,
                        backward_outputs
                    )
                else:
                    chunk_output = forward_outputs
            else:
                # Standard processing
                chunk_output = self._process_chunk(
                    chunk_ids,
                    chunk_mask,
                    rng_key,
                    search_strategy
                )
                
            chunked_outputs.append(chunk_output)
            
        # Combine chunk results
        model_outputs = jax.tree_map(
            lambda *xs: jnp.concatenate(xs, axis=0) if xs[0] is not None else None,
            *chunked_outputs
        )
        
        return model_outputs
    
    def _combine_dualpipe_outputs(self, forward_outputs, backward_outputs):
        """Combine outputs from dualpipe processing."""
        # Combine using tree_map for nested structures
        combined = jax.tree_map(
            lambda f, b: jnp.concatenate([f, b], axis=0) if f is not None and b is not None else f or b,
            forward_outputs,
            backward_outputs
        )
        return combined
    
    def _process_chunk(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray],
        rng_key: jnp.ndarray,
        search_strategy: str
    ):
        """
        Process a single chunk with TPU optimization.
        
        Args:
            input_ids: Chunk of input token IDs
            attention_mask: Chunk of attention mask
            rng_key: Random key
            search_strategy: ToT search strategy
            
        Returns:
            Processed model outputs for this chunk
        """
        # Get base model features
        model_outputs = self.model(
            input_ids, 
            attention_mask, 
            output_hidden_states=True
        )
        model_features = model_outputs['hidden_states'][-1]
        
        # Split random key for different components
        rng_key, tot_key, mla_key, integration_key = jax.random.split(rng_key, 4)
        
        # Generate thoughts using ToT
        thoughts = self.tot_model(model_features, tot_key, search_strategy=search_strategy)
        
        # Extract thought features
        if thoughts is not None:
            thought_features = self.process_thought_tree(thoughts)
            tot_features = jnp.stack(thought_features) if thought_features else None
        else:
            thought_features = []
            tot_features = None
        
        # Apply multi-level attention
        enhanced_features, attn_weights = self.mla(
            model_features,
            tot_features,
            thought_features,
            deterministic=True
        )
        
        # Get error indicators if available
        error_indicator = model_outputs.get('error_probs')
        
        # Integrate model and ToT features with error correction
        integrated_features, integration_info = self.integration_layer(
            model_features,
            enhanced_features,
            deterministic=True,
            error_indicator=error_indicator
        )
        
        # Update model outputs with integrated features
        updated_outputs = dict(model_outputs)
        updated_outputs['integrated_features'] = integrated_features
        updated_outputs['tot_thoughts'] = thoughts
        updated_outputs['tot_attention_weights'] = attn_weights
        updated_outputs['integration_info'] = integration_info
        
        # Add integrated features to logits computation if model has a language modeling head
        if 'logits' in model_outputs:
            if hasattr(self.model, 'lm_head'):
                lm_head = self.model.lm_head
                updated_outputs['tot_logits'] = lm_head(integrated_features)
        
        return updated_outputs
