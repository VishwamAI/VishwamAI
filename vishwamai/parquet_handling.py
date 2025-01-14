import pyarrow.parquet as pq
import pyarrow as pa
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Any
import numpy as np
from dataclasses import dataclass
from vishwamai.conceptual_tokenizer import ConceptualTokenizer, ConceptualTokenizerConfig
import os  # Added import

@dataclass
class ParquetConfig:
    chunk_size: int = 10000
    batch_size: int = 32
    num_workers: int = 4
    shuffle_buffer: int = 10000
    cache_size: int = 100000

class ParquetDataset(Dataset):
    def __init__(
        self, 
        parquet_path: str,
        config: ParquetConfig,
        tokenizer: Optional[ConceptualTokenizer] = None,  # Changed parameter
        text_column: str = "text",
        max_length: int = 2048
    ):
        # Check if the parquet file exists
        if not os.path.isfile(parquet_path):
            raise FileNotFoundError(f"Parquet file not found at path: {parquet_path}")
        
        self.parquet_file = pq.ParquetFile(parquet_path)
        self.tokenizer = tokenizer if tokenizer else ConceptualTokenizer(ConceptualTokenizerConfig())  # Updated initialization
        self.config = config
        self.text_column = text_column
        self.max_length = max_length
        self.length = self.parquet_file.metadata.num_rows
        
        # Initialize cache
        self._cache = {}
        self._cache_indices = []
    
    def _read_chunk(self, start_idx: int, end_idx: int) -> Dict[str, torch.Tensor]:
        """Read a chunk of data from Parquet file."""
        try:
            row_group = start_idx // self.config.chunk_size
            if row_group >= self.parquet_file.num_row_groups:
                raise IndexError(f"Row group {row_group} out of range.")
            chunk = self.parquet_file.read_row_group(row_group)
            texts = chunk[self.text_column].to_pandas()
            
            input_ids_list = []
            attention_mask_list = []
            
            for text in texts.tolist():
                # Handle empty or invalid text
                if not isinstance(text, str) or not text.strip():
                    text = "empty"
                
                # Encode and handle potential errors
                try:
                    encoded = self.tokenizer.encode(text, add_special_tokens=True)
                    if not isinstance(encoded, list):
                        encoded = encoded.tolist()
                except Exception as e:
                    print(f"Encoding error for text '{text}': {str(e)}")
                    encoded = [self.tokenizer.config.unk_id]
                
                # Padding and truncation
                if len(encoded) > self.max_length:
                    encoded = encoded[:self.max_length]
                else:
                    padding_length = self.max_length - len(encoded)
                    encoded = encoded + [self.tokenizer.config.pad_id] * padding_length
                
                # Create attention mask
                attention_mask = [1] * len(encoded)
                if len(attention_mask) < self.max_length:
                    attention_mask.extend([0] * (self.max_length - len(attention_mask)))
                
                input_ids_list.append(encoded)
                attention_mask_list.append(attention_mask)
            
            return {
                'input_ids': torch.tensor(input_ids_list, dtype=torch.long),
                'attention_mask': torch.tensor(attention_mask_list, dtype=torch.long)
            }
        except Exception as e:
            raise RuntimeError(f"Error reading chunk: {str(e)}")

    def _get_cached_item(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get item from cache or load from disk."""
        chunk_idx = idx // self.config.chunk_size
        
        if chunk_idx not in self._cache:
            # Load new chunk
            start_idx = chunk_idx * self.config.chunk_size
            end_idx = min(start_idx + self.config.chunk_size, self.length)
            
            chunk_data = self._read_chunk(start_idx, end_idx)
            
            # Update cache
            self._cache[chunk_idx] = chunk_data
            self._cache_indices.append(chunk_idx)
            
            # Remove oldest cache entry if cache is full
            if len(self._cache) > self.config.cache_size:
                oldest_idx = self._cache_indices.pop(0)
                del self._cache[oldest_idx]
        
        item_idx = idx % self.config.chunk_size
        return {
            'input_ids': self._cache[chunk_idx]['input_ids'][item_idx],
            'attention_mask': self._cache[chunk_idx]['attention_mask'][item_idx]
        }
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self._get_cached_item(idx)

def create_parquet_dataloader(
    dataset: ParquetDataset,
    config: ParquetConfig,
    distributed: bool = False
) -> DataLoader:
    """Create DataLoader with optional distributed training support."""
    if distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        sampler = None
        
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        sampler=sampler,
        shuffle=(sampler is None),
        pin_memory=True
    )

def save_to_parquet(
    texts: List[str],
    output_path: str,
    chunk_size: int = 10000,
    compression: str = 'snappy'
):
    """Save text data to Parquet format."""
    # Convert to Arrow Table
    table = pa.Table.from_arrays(
        [pa.array(texts)],
        names=['text']
    )
    
    # Write to Parquet with chunking
    pq.write_table(
        table,
        output_path,
        row_group_size=chunk_size,
        compression=compression
    )

def merge_parquet_files(input_paths: List[str], output_path: str, row_group_size: Optional[int] = None):
    """Merge multiple Parquet files into one."""
    tables = [pq.read_table(path) for path in input_paths]
    merged_table = pa.concat_tables(tables)
    pq.write_table(merged_table, output_path, row_group_size=row_group_size)

def update_parquet_model(parquet_path: str, new_texts: List[str], output_path: str) -> None:
    """
    Merges new text data into the existing parquet dataset and saves to output_path.
    """
    # Load existing parquet data
    table = pq.read_table(parquet_path)
    existing_texts = table["text"].to_pylist()
    
    # Merge old and new
    merged_texts = existing_texts + new_texts
    
    # Convert merged texts back to an Arrow table
    merged_table = pa.Table.from_arrays([pa.array(merged_texts)], names=['text'])
    
    # Write merged data to new parquet file
    pq.write_table(merged_table, output_path)