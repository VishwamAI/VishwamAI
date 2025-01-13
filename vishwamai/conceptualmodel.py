import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

from .architecture import VishwamaiV1
from .training import VishwamaiTrainer

@dataclass
class ConceptualModelConfig:
    concept_dim: int = 512           # Dimension of concept embeddings
    n_concepts: int = 1000          # Number of learnable concepts
    concept_dropout: float = 0.1    # Dropout rate for concept integration
    use_concept_attention: bool = True
    concept_layer_norm: bool = True
    num_concept_layers: int = 2

class ConceptualLayer(nn.Module):
    """
    Handles concept integration and reasoning within the model.
    """
    def __init__(self, config: ConceptualModelConfig, base_dim: int):
        super().__init__()
        self.config = config
        self.base_dim = base_dim
        
        # Concept embeddings
        self.concept_embeddings = nn.Parameter(
            torch.randn(config.n_concepts, config.concept_dim)
        )
        
        # Concept attention for integration
        if config.use_concept_attention:
            self.concept_query = nn.Linear(base_dim, config.concept_dim, bias=False)
            self.concept_key = nn.Linear(config.concept_dim, config.concept_dim)
            self.concept_value = nn.Linear(config.concept_dim, config.concept_dim)
        
        # Integration layers - Fixed dimensions
        self.concept_projection = nn.Linear(config.concept_dim, base_dim)
        # Updated gate dimensions to match concatenated input
        self.concept_gate = nn.Linear(base_dim + base_dim, base_dim, bias=False)
        
        if config.concept_layer_norm:
            self.concept_norm = nn.LayerNorm(base_dim)
        
        self.dropout = nn.Dropout(config.concept_dropout)

    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = hidden_states.shape
        
        if self.config.use_concept_attention:
            # Compute attention scores between input and concepts
            queries = self.concept_query(hidden_states)  # [batch, seq_len, concept_dim]
            keys = self.concept_key(self.concept_embeddings)  # [n_concepts, concept_dim]
            values = self.concept_value(self.concept_embeddings)  # [n_concepts, concept_dim]
            
            # Compute attention scores
            attention_scores = torch.matmul(queries, keys.t()) / math.sqrt(self.config.concept_dim)  # [batch, seq_len, n_concepts]
            
            # Apply attention mask if provided
            if attention_mask is not None:
                # Ensure attention_mask is of shape [batch, seq_len]
                if attention_mask.shape != (batch_size, seq_len):
                    raise ValueError(f"attention_mask shape {attention_mask.shape} does not match tokens shape ({batch_size}, {seq_len})")
                
                mask = attention_mask.unsqueeze(-1)  # [batch, seq_len, 1]
                
                attention_scores = attention_scores.masked_fill(
                    mask == 0,  # [batch, seq_len, 1]
                    float('-inf')
                )
            
            attention_probs = F.softmax(attention_scores, dim=-1)  # [batch, seq_len, n_concepts]
            attention_probs = self.dropout(attention_probs)
            
            # Get concept-aware representations
            concept_states = torch.matmul(attention_probs, values)  # [batch, seq_len, concept_dim]
        else:
            # Simple concept integration without attention
            concept_states = self.concept_embeddings.mean(0, keepdim=True)  # [1, concept_dim]
            concept_states = concept_states.expand(batch_size, seq_len, -1)  # [batch, seq_len, concept_dim]
            concept_states = self.dropout(concept_states)
        
        # Project concept states to hidden dimension
        concept_features = self.concept_projection(concept_states)  # [batch, seq_len, base_dim]
        
        # Compute gating values
        gate_input = torch.cat([hidden_states, concept_features], dim=-1)  # [batch, seq_len, base_dim * 2]
        gate_values = torch.sigmoid(self.concept_gate(gate_input))  # [batch, seq_len, base_dim]
        
        # Combine original and concept features
        output = hidden_states + gate_values * concept_features
        
        if self.config.concept_layer_norm:
            output = self.concept_norm(output)
        
        return output, attention_probs if self.config.use_concept_attention else None
    
class ConceptualReasoningModule(nn.Module):
    """
    Multi-layer conceptual reasoning module.
    """
    def __init__(self, config: ConceptualModelConfig, base_dim: int):
        super().__init__()
        self.layers = nn.ModuleList([
            ConceptualLayer(config, base_dim)
            for _ in range(config.num_concept_layers)
        ])

    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        attention_probs_list = []
        
        for layer in self.layers:
            hidden_states, attention_probs = layer(
                hidden_states,
                attention_mask=attention_mask
            )
            if attention_probs is not None:
                attention_probs_list.append(attention_probs)
        
        return hidden_states, attention_probs_list

class ConceptAwareVishwamai(nn.Module):
    """
    Enhanced Vishwamai model with conceptual reasoning capabilities.
    """
    def __init__(self, base_config, conceptual_config: ConceptualModelConfig):
        super().__init__()
        self.config = base_config
        # Base Vishwamai components
        # Base Vishwamai components
        self.base_model = VishwamaiV1(base_config)
        
        # Conceptual components
        self.concept_module = ConceptualReasoningModule(
            conceptual_config,
            base_config.dim
        )
        
        # Output adaptation
        self.concept_output_adapter = nn.Linear(base_config.dim, base_config.dim)
        self.concept_layer_norm = nn.LayerNorm(base_config.dim)

    def forward(
        self,
        tokens: torch.Tensor,  # Changed from input_ids to tokens
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_concept_info: bool = False
    ):
        try:
            # Truncate tokens if they exceed max_seq_len
            if tokens.size(1) > self.config.max_seq_len:
                tokens = tokens[:, :self.config.max_seq_len]
            
            # Get base model outputs
            base_hidden_states = self.base_model.forward_hidden(
                tokens=tokens,
                start_pos=0
            )
            
            # Apply conceptual reasoning
            concept_enhanced_states, concept_attention_probs = self.concept_module(
                base_hidden_states,
                attention_mask=attention_mask
            )
            
            # Adapt and normalize outputs
            final_hidden_states = self.concept_output_adapter(concept_enhanced_states)
            final_hidden_states = self.concept_layer_norm(final_hidden_states)
            
            # Compute loss if labels provided
            loss = None
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    final_hidden_states.view(-1, final_hidden_states.size(-1)),
                    labels.view(-1)
                )
            
            if return_concept_info:
                outputs = {
                    'hidden_states': final_hidden_states,
                    'concept_attention_probs': concept_attention_probs
                }
                if labels is not None:
                    outputs['loss'] = loss
                return outputs
            
            outputs = {
                'hidden_states': final_hidden_states
            }
            if labels is not None:
                outputs['loss'] = loss
            
            return outputs
        except Exception as e:
            logging.error(f"Error in ConceptAwareVishwamai.forward: {e}")
            raise e

# Training extensions for conceptual features
class ConceptualTrainer(VishwamaiTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            labels=inputs['labels'],
            return_concept_info=True
        )
        
        # Combine base loss with concept-specific objectives
        loss = outputs['loss']
        
        # Add concept diversity loss
        if self.args.concept_diversity_weight > 0:
            concept_attention = outputs['concept_attention_probs']
            diversity_loss = self.compute_concept_diversity_loss(concept_attention)
            loss += self.args.concept_diversity_weight * diversity_loss
        
        return (loss, outputs) if return_outputs else loss
    
    def compute_concept_diversity_loss(self, concept_attention):
        """Encourages diverse concept usage"""
        # Average attention across batch and positions
        avg_attention = concept_attention.mean(dim=[0, 1])
        # Compute entropy
        entropy = -(avg_attention * torch.log(avg_attention + 1e-10)).sum()
        return -entropy  # Negative entropy to maximize diversity

def ensure_gpu_availability():
    if not torch.cuda.is_available():
        raise RuntimeError("GPU not available for advanced conceptual flow.")

def advanced_concept_flow(concepts: torch.Tensor):
    try:
        calculations = torch.log1p(concepts)  # Example advanced calc
        return calculations
    except Exception as e:
        raise ValueError(f"Error in advanced_concept_flow: {e}")