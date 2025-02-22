"""
VishwamaiModel implementation
"""
import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from ..config import ModelConfig, PrecisionConfig, PrecisionMode
from .attention import MultiHeadAttention
from .embeddings import TokenEmbedding, PositionalEmbedding
from .feed_forward import FeedForward
from .layer_norm import RMSNorm

def get_model_dtype(precision_config: PrecisionConfig) -> torch.dtype:
    """Get model dtype based on precision config"""
    mode_to_dtype = {
        PrecisionMode.FP16: torch.float16,
        PrecisionMode.BF16: torch.bfloat16,
        PrecisionMode.FP32: torch.float32,
        PrecisionMode.FP64: torch.float64,
        PrecisionMode.TF32: torch.float32  # TF32 is handled by CUDA internally
    }
    return mode_to_dtype[precision_config.mode]

class VishwamaiModel(nn.Module):
    """
    Vishwamai model implementation
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Set model precision
        self.dtype = get_model_dtype(config.precision)
        
        # Embeddings
        self.token_embedding = TokenEmbedding(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            dtype=self.dtype
        )
        
        self.position_embedding = PositionalEmbedding(
            max_position_embeddings=config.max_position_embeddings,
            hidden_size=config.hidden_size,
            position_embedding_type=config.position_embedding_type,
            dtype=self.dtype
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerLayer(config, dtype=self.dtype)
            for _ in range(config.num_layers)
        ])
        
        # Tree planner if enabled
        if config.tree_planner["enabled"]:
            from ..tree_planner import TreePlanner
            self.tree_planner = TreePlanner(
                hidden_size=config.tree_planner["tree_hidden_size"],
                num_layers=config.tree_planner["num_tree_layers"],
                max_depth=config.tree_planner["max_tree_depth"],
                dtype=self.dtype
            )
        
        # Query generator for information retrieval if enabled
        if config.information_retrieval["enabled"]:
            self.query_generator = QueryGenerator(
                hidden_size=config.hidden_size,
                dtype=self.dtype
            )
        
        # Output layer norm
        self.layer_norm = RMSNorm(config.hidden_size, dtype=self.dtype)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Forward pass"""
        # Get embeddings
        hidden_states = self.token_embedding(input_ids)
        position_embeddings = self.position_embedding(input_ids)
        hidden_states = hidden_states + position_embeddings
        
        # Apply transformer layers
        if self.config.gradient_checkpointing:
            layer_outputs = []
            for layer in self.layers:
                if torch.is_grad_enabled():
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(layer),
                        hidden_states,
                        attention_mask
                    )
                else:
                    hidden_states = layer(hidden_states, attention_mask)
                layer_outputs.append(hidden_states)
        else:
            layer_outputs = []
            for layer in self.layers:
                hidden_states = layer(hidden_states, attention_mask)
                layer_outputs.append(hidden_states)
                
        # Apply tree planning if enabled
        if hasattr(self, 'tree_planner'):
            tree_structure = self.tree_planner(hidden_states)
        else:
            tree_structure = None
            
        # Generate search queries if enabled
        if hasattr(self, 'query_generator'):
            queries = self.query_generator(hidden_states)
        else:
            queries = None
        
        # Apply final layer norm
        hidden_states = self.layer_norm(hidden_states)
        
        if not return_dict:
            return hidden_states
            
        outputs = {
            "hidden_states": hidden_states,
            "layer_outputs": layer_outputs
        }
        
        if tree_structure is not None:
            outputs["tree_structure"] = tree_structure
            
        if queries is not None:
            outputs["generated_queries"] = queries
            
        return outputs
        
    def generate_search_queries(self, input_texts: list) -> list:
        """Generate search queries for input texts"""
        if not hasattr(self, 'query_generator'):
            raise RuntimeError("Information retrieval is not enabled in the model config")
            
        # TODO: Implement tokenization and query generation
        raise NotImplementedError
        
class TransformerLayer(nn.Module):
    """Transformer layer implementation"""
    def __init__(self, config: ModelConfig, dtype: torch.dtype):
        super().__init__()
        self.attention = MultiHeadAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads,
            dropout=config.attention_dropout_prob,
            use_flash_attention=config.use_flash_attention,
            dtype=dtype
        )
        self.feed_forward = FeedForward(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            dropout=config.hidden_dropout_prob,
            dtype=dtype
        )
        self.layer_norm1 = RMSNorm(config.hidden_size, dtype=dtype)
        self.layer_norm2 = RMSNorm(config.hidden_size, dtype=dtype)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass"""
        # Self attention
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # Feed forward
        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states
        
class QueryGenerator(nn.Module):
    """Search query generator"""
    def __init__(self, hidden_size: int, dtype: torch.dtype):
        super().__init__()
        self.query_projection = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Generate search queries from hidden states"""
        return self.query_projection(hidden_states)
