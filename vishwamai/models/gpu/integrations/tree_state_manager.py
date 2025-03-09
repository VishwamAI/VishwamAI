"""
Tree state management for Tree-of-Thoughts and distributed processing.
"""

import torch
import os
import smallpond
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass
import time
import json
from pathlib import Path

@dataclass
class TreeNode:
    """Node in thought tree"""
    node_id: str
    text: str
    parent_id: Optional[str] = None
    score: float = 0.0
    hidden_state: Optional[torch.Tensor] = None

class TreeStateManager:
    """Manages thought tree state with distributed storage"""
    
    def __init__(self,
                storage_dir: str,
                embed_dim: int,
                use_smallpond: bool = True):
        self.storage_dir = storage_dir
        self.embed_dim = embed_dim
        self.use_smallpond = use_smallpond
        
        os.makedirs(storage_dir, exist_ok=True)
        os.makedirs(os.path.join(storage_dir, 'trees'), exist_ok=True)
        os.makedirs(os.path.join(storage_dir, 'nodes'), exist_ok=True)
        os.makedirs(os.path.join(storage_dir, 'paths'), exist_ok=True)
        
        # Initialize smallpond
        if use_smallpond:
            try:
                self.sp_session = smallpond.init(
                    num_executors=torch.cuda.device_count(),
                    data_root=storage_dir,
                    bind_numa_node=True
                )
            except:
                self.sp_session = None
                self.use_smallpond = False
                
    def store_node(self,
                  node_id: str,
                  node_data: Dict[str, Any],
                  hidden_state: Optional[torch.Tensor] = None):
        """Store tree node with optional hidden state"""
        if not self.use_smallpond or self.sp_session is None:
            # Save locally
            node_path = os.path.join(
                self.storage_dir,
                'nodes',
                f'{node_id}.pt'
            )
            torch.save({
                'data': node_data,
                'hidden_state': hidden_state
            }, node_path)
            return
            
        # Convert data for smallpond
        if hidden_state is not None:
            hidden_state = hidden_state.detach().cpu().numpy()
            
        # Create DataFrame
        df = self.sp_session.create_dataframe({
            'node_id': [node_id],
            'data': [node_data],
            'hidden_state': [hidden_state]
        })
        
        # Save to parquet
        node_path = os.path.join(
            self.storage_dir,
            'nodes',
            f'{node_id}.parquet'
        )
        df.write_parquet(node_path)
        
    def load_node(self,
                 node_id: str,
                 load_hidden: bool = False) -> Tuple[Optional[Dict], Optional[torch.Tensor], bool]:
        """Load node data and optionally hidden state"""
        if not self.use_smallpond or self.sp_session is None:
            # Load locally
            node_path = os.path.join(
                self.storage_dir,
                'nodes',
                f'{node_id}.pt'
            )
            if os.path.exists(node_path):
                data = torch.load(node_path)
                return (
                    data['data'],
                    data['hidden_state'] if load_hidden else None,
                    True
                )
            return None, None, False
            
        # Load from distributed storage
        node_path = os.path.join(
            self.storage_dir,
            'nodes',
            f'{node_id}.parquet'
        )
        
        if not os.path.exists(node_path):
            return None, None, False
            
        df = self.sp_session.read_parquet(node_path)
        row = df.to_pandas().iloc[0]
        
        hidden_state = None
        if load_hidden and row['hidden_state'] is not None:
            hidden_state = torch.from_numpy(row['hidden_state'])
            
        return row['data'], hidden_state, True
        
    def store_tree_structure(self, tree_id: str, tree_data: Dict[str, Any]):
        """Store tree structure metadata"""
        if not self.use_smallpond or self.sp_session is None:
            # Save locally
            tree_path = os.path.join(
                self.storage_dir,
                'trees',
                f'{tree_id}.json'
            )
            with open(tree_path, 'w') as f:
                json.dump(tree_data, f)
            return
            
        # Create DataFrame
        df = self.sp_session.create_dataframe({
            'tree_id': [tree_id],
            'data': [tree_data]
        })
        
        # Save to parquet
        tree_path = os.path.join(
            self.storage_dir,
            'trees',
            f'{tree_id}.parquet'
        )
        df.write_parquet(tree_path)
        
    def load_tree_structure(self, tree_id: str) -> Optional[Dict[str, Any]]:
        """Load tree structure metadata"""
        if not self.use_smallpond or self.sp_session is None:
            # Load locally
            tree_path = os.path.join(
                self.storage_dir,
                'trees',
                f'{tree_id}.json'
            )
            if os.path.exists(tree_path):
                with open(tree_path, 'r') as f:
                    return json.load(f)
            return None
            
        # Load from distributed storage
        tree_path = os.path.join(
            self.storage_dir,
            'trees',
            f'{tree_id}.parquet'
        )
        
        if not os.path.exists(tree_path):
            return None
            
        df = self.sp_session.read_parquet(tree_path)
        return df.to_pandas().iloc[0]['data']
        
    def store_path(self,
                  tree_id: str,
                  path: List[str],
                  score: float):
        """Store successful thought path"""
        if not self.use_smallpond or self.sp_session is None:
            # Save locally
            path_dir = os.path.join(
                self.storage_dir,
                'paths',
                tree_id
            )
            os.makedirs(path_dir, exist_ok=True)
            
            path_id = f"path_{int(time.time())}"
            path_path = os.path.join(path_dir, f'{path_id}.json')
            
            with open(path_path, 'w') as f:
                json.dump({
                    'path': path,
                    'score': score
                }, f)
            return
            
        # Create DataFrame
        df = self.sp_session.create_dataframe({
            'tree_id': [tree_id],
            'path_id': [f"path_{int(time.time())}"],
            'path': [path],
            'score': [score]
        })
        
        # Save to parquet
        path_path = os.path.join(
            self.storage_dir,
            'paths',
            f'{tree_id}.parquet'
        )
        df.write_parquet(
            path_path,
            mode='append' if os.path.exists(path_path) else 'overwrite'
        )
        
    def get_cached_paths(self,
                       tree_id: str,
                       min_score: float = 0.0) -> List[Tuple[List[str], float]]:
        """Get previously successful paths with minimum score"""
        if not self.use_smallpond or self.sp_session is None:
            # Load locally
            path_dir = os.path.join(
                self.storage_dir,
                'paths',
                tree_id
            )
            
            if not os.path.exists(path_dir):
                return []
                
            paths = []
            for file in os.listdir(path_dir):
                if not file.endswith('.json'):
                    continue
                    
                with open(os.path.join(path_dir, file), 'r') as f:
                    data = json.load(f)
                    if data['score'] >= min_score:
                        paths.append((data['path'], data['score']))
                        
            return sorted(paths, key=lambda x: x[1], reverse=True)
            
        # Load from distributed storage
        path_path = os.path.join(
            self.storage_dir,
            'paths',
            f'{tree_id}.parquet'
        )
        
        if not os.path.exists(path_path):
            return []
            
        df = self.sp_session.read_parquet(path_path)
        data = df.to_pandas()
        
        paths = []
        for _, row in data.iterrows():
            if row['score'] >= min_score:
                paths.append((row['path'], row['score']))
                
        return sorted(paths, key=lambda x: x[1], reverse=True)
        
    def cleanup(self):
        """Cleanup resources"""
        if self.use_smallpond and self.sp_session:
            self.sp_session.shutdown()
