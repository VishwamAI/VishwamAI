"""
Data collator for batching samples
"""
from typing import Dict, List, Any, Optional
import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase

@dataclass
class DataCollator:
    """
    Data collator for Vishwamai model
    
    This collator handles batching samples together, including padding
    and attention mask creation.
    """
    tokenizer: PreTrainedTokenizerBase
    padding: bool = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"
    
    def __call__(
        self,
        features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate batch of samples
        
        Args:
            features: List of samples to batch
            
        Returns:
            Batch dictionary with padded tensors
        """
        # Extract all keys that contain tensor values
        tensor_keys = [
            k for k in features[0].keys()
            if isinstance(features[0][k], torch.Tensor)
        ]
        
        # Initialize batch dictionary
        batch = {
            k: [f[k] for f in features]
            for k in tensor_keys
        }
        
        # Add non-tensor metadata if present
        metadata_keys = [
            k for k in features[0].keys()
            if k not in tensor_keys
        ]
        if metadata_keys:
            batch.update({
                k: [f[k] for f in features]
                for k in metadata_keys
            })
        
        # Pad sequences
        if self.padding:
            for key in tensor_keys:
                batch[key] = self._pad_sequence(
                    batch[key],
                    key == "labels"
                )
                
        # Convert lists to tensors
        for key in tensor_keys:
            if isinstance(batch[key], list):
                if torch.is_tensor(batch[key][0]):
                    batch[key] = torch.stack(batch[key])
                else:
                    batch[key] = torch.tensor(batch[key])
                    
        return batch
        
    def _pad_sequence(
        self,
        sequences: List[torch.Tensor],
        is_labels: bool = False
    ) -> torch.Tensor:
        """
        Pad sequence tensors to same length
        
        Args:
            sequences: List of tensors to pad
            is_labels: Whether sequences are labels
            
        Returns:
            Padded tensor
        """
        # Get max length
        max_len = max(seq.size(0) for seq in sequences)
        
        # Adjust max length if needed
        if self.max_length is not None:
            max_len = min(max_len, self.max_length)
            
        if self.pad_to_multiple_of is not None:
            padding_multiple = self.pad_to_multiple_of
            max_len = (
                (max_len + padding_multiple - 1)
                // padding_multiple
                * padding_multiple
            )
            
        # Pad sequences
        padded_sequences = []
        for seq in sequences:
            # Truncate if needed
            if seq.size(0) > max_len:
                seq = seq[:max_len]
                
            # Pad with appropriate value
            pad_value = self.tokenizer.pad_token_id if not is_labels else -100
            padding = torch.full(
                (max_len - seq.size(0),),
                pad_value,
                dtype=seq.dtype,
                device=seq.device
            )
            padded_seq = torch.cat([seq, padding])
            padded_sequences.append(padded_seq)
            
        return torch.stack(padded_sequences)

@dataclass
class TreeDataCollator(DataCollator):
    """
    Data collator with additional tree structure handling
    """
    def __call__(
        self,
        features: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """
        Collate batch including tree structures
        
        Args:
            features: List of samples with tree data
            
        Returns:
            Batch dictionary with tree tensors
        """
        batch = super().__call__(features)
        
        # Process tree structures if present
        if "tree_structure" in features[0]:
            tree_structures = [f["tree_structure"] for f in features]
            
            # Convert tree adjacency matrices to tensor
            if isinstance(tree_structures[0], torch.Tensor):
                max_nodes = max(t.size(0) for t in tree_structures)
                padded_trees = []
                
                for tree in tree_structures:
                    if tree.size(0) < max_nodes:
                        padding = torch.zeros(
                            (max_nodes - tree.size(0), max_nodes),
                            dtype=tree.dtype,
                            device=tree.device
                        )
                        padded_tree = torch.cat([tree, padding], dim=0)
                        padding = torch.zeros(
                            (max_nodes, max_nodes - tree.size(0)),
                            dtype=tree.dtype,
                            device=tree.device
                        )
                        padded_tree = torch.cat([padded_tree, padding], dim=1)
                    else:
                        padded_tree = tree[:max_nodes, :max_nodes]
                    padded_trees.append(padded_tree)
                    
                batch["tree_structure"] = torch.stack(padded_trees)
                
        return batch
