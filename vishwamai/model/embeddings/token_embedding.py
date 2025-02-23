"""Token embedding implementation with weight tying support."""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedding(nn.Module):
    """Token embedding layer with optional weight tying."""
    
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: Optional[int] = None,
        scale_embeddings: bool = True,
        embedding_scale: Optional[float] = None,
        dropout_prob: float = 0.1,
        max_norm: Optional[float] = None,
        norm_type: float = 2.0,
        tie_weights: bool = False,
    ):
        """Initialize token embedding.
        
        Args:
            num_embeddings: Size of the vocabulary
            embedding_dim: Dimension of embeddings
            padding_idx: Index used for padding token
            scale_embeddings: Whether to scale embeddings by sqrt(d_model)
            embedding_scale: Optional custom embedding scale factor
            dropout_prob: Dropout probability
            max_norm: Max norm for embedding normalization
            norm_type: Norm type for normalization (usually L2)
            tie_weights: Whether to enable weight tying with output layer
        """
        super().__init__()
        
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.scale_embeddings = scale_embeddings
        self.embedding_scale = embedding_scale or (embedding_dim ** 0.5 if scale_embeddings else 1.0)
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.tie_weights = tie_weights
        
        # Create embedding layer
        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim)))
        
        # Create dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.weight, mean=0.0, std=0.02)
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)
                
    def forward(
        self,
        input_ids: torch.LongTensor,
        positions: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """Convert input token IDs to embeddings.
        
        Args:
            input_ids: Token IDs of shape [batch_size, seq_len]
            positions: Optional position IDs for position-dependent embeddings
            
        Returns:
            Embedding tensor of shape [batch_size, seq_len, embedding_dim]
        """
        if self.max_norm is not None and self.training:
            # Apply max norm constraint during training
            with torch.no_grad():
                F.normalize(self.weight, p=self.norm_type, dim=-1, max_norm=self.max_norm)
                
        embeddings = F.embedding(
            input_ids,
            self.weight,
            padding_idx=self.padding_idx,
            max_norm=self.max_norm if not self.training else None,
            norm_type=self.norm_type,
        )
        
        # Scale embeddings if needed
        if self.scale_embeddings:
            embeddings = embeddings * self.embedding_scale
            
        # Apply position-dependent scaling if positions provided
        if positions is not None:
            embeddings = embeddings * positions.unsqueeze(-1)
            
        # Apply dropout
        embeddings = self.dropout(embeddings)
        
        return embeddings
    
    def tie_weights_with(self, output_layer: nn.Linear):
        """Tie embedding weights with output layer.
        
        Args:
            output_layer: Linear output layer to tie weights with
        """
        if not self.tie_weights:
            raise ValueError("Weight tying is disabled for this embedding layer")
            
        if output_layer.weight.shape != self.weight.shape:
            raise ValueError(
                f"Output layer weight shape {output_layer.weight.shape} "
                f"does not match embedding shape {self.weight.shape}"
            )
            
        # Share weights between embedding and output layer
        output_layer.weight = self.weight
        
    def add_embeddings(
        self,
        num_new_tokens: int,
        initialize: bool = True,
        copy_from: Optional[Union[int, list[int]]] = None
    ):
        """Add new token embeddings.
        
        Args:
            num_new_tokens: Number of new tokens to add
            initialize: Whether to initialize new embeddings
            copy_from: Optional token index or list of indices to copy embeddings from
        """
        if num_new_tokens <= 0:
            return
            
        # Create new embedding weights
        new_embeddings = nn.Parameter(
            torch.empty((num_new_tokens, self.embedding_dim), device=self.weight.device)
        )
        
        # Initialize or copy new embeddings
        if copy_from is not None:
            if isinstance(copy_from, int):
                new_embeddings.data.copy_(self.weight[copy_from].unsqueeze(0).expand(num_new_tokens, -1))
            else:
                if len(copy_from) != num_new_tokens:
                    raise ValueError(
                        f"Length of copy_from indices ({len(copy_from)}) "
                        f"must match num_new_tokens ({num_new_tokens})"
                    )
                new_embeddings.data.copy_(self.weight[copy_from])
        elif initialize:
            nn.init.normal_(new_embeddings, mean=0.0, std=0.02)
            
        # Concatenate new embeddings
        self.weight = nn.Parameter(torch.cat([self.weight, new_embeddings], dim=0))
        self.num_embeddings += num_new_tokens

    def resize_embeddings(self, new_num_tokens: int):
        """Resize embedding matrix to new vocabulary size.
        
        Args:
            new_num_tokens: New vocabulary size
        """
        old_num_tokens = self.num_embeddings
        if new_num_tokens == old_num_tokens:
            return
            
        # Create new embedding weights
        new_embeddings = nn.Parameter(
            torch.empty((new_num_tokens, self.embedding_dim), device=self.weight.device)
        )
        
        # Copy existing embeddings
        copy_size = min(old_num_tokens, new_num_tokens)
        new_embeddings.data[:copy_size].copy_(self.weight.data[:copy_size])
        
        # Initialize remaining embeddings if expanding
        if new_num_tokens > old_num_tokens:
            nn.init.normal_(
                new_embeddings.data[old_num_tokens:],
                mean=0.0,
                std=0.02
            )
            
        # Update embedding layer
        self.weight = new_embeddings
        self.num_embeddings = new_num_tokens
        
    def extra_repr(self) -> str:
        """Return extra representation string."""
        return (
            f"num_embeddings={self.num_embeddings}, "
            f"embedding_dim={self.embedding_dim}, "
            f"padding_idx={self.padding_idx}, "
            f"scale={self.embedding_scale}, "
            f"max_norm={self.max_norm}, "
            f"tie_weights={self.tie_weights}"
        )
