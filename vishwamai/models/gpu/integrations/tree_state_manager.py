"""
Tree state management and persistence using 3FS for Tree of Thoughts optimization.
"""

import torch
import os
from typing import Dict, List, Optional, Tuple, Any
import json
import time

class TreeStateManager:
    """Manages tree-structured state persistence using 3FS"""
    def __init__(
        self,
        storage_dir: str,
        embed_dim: int,
        max_nodes: int = 10000,
        cache_size_gb: float = 10
    ):
        self.storage_dir = storage_dir
        self.embed_dim = embed_dim
        self.max_nodes = max_nodes
        self.cache_size = int(cache_size_gb * 1024 * 1024 * 1024)
        
        # Initialize storage directories
        self.tree_dir = os.path.join(storage_dir, 'trees')
        self.node_dir = os.path.join(storage_dir, 'nodes')
        self.hidden_dir = os.path.join(storage_dir, 'hidden_states')
        
        os.makedirs(self.tree_dir, exist_ok=True)
        os.makedirs(self.node_dir, exist_ok=True)
        os.makedirs(self.hidden_dir, exist_ok=True)
        
        # Access tracking
        self.node_access_counts = {}
        self._step = 0
        
    def _get_tree_path(self, tree_id: str) -> str:
        """Get path for tree structure storage"""
        return os.path.join(self.tree_dir, f"{tree_id}_tree.json")
        
    def _get_node_path(self, node_id: str) -> str:
        """Get path for node data storage"""
        return os.path.join(self.node_dir, f"{node_id}_node.pt")
        
    def _get_hidden_path(self, node_id: str) -> str:
        """Get path for hidden state storage"""
        return os.path.join(self.hidden_dir, f"{node_id}_hidden.pt")
        
    def store_tree_structure(
        self,
        tree_id: str,
        tree_data: Dict[str, Any]
    ) -> None:
        """Store tree structure metadata"""
        path = self._get_tree_path(tree_id)
        tree_data['timestamp'] = time.time()
        with open(path, 'w') as f:
            json.dump(tree_data, f)
            
    def load_tree_structure(
        self,
        tree_id: str
    ) -> Optional[Dict[str, Any]]:
        """Load tree structure metadata"""
        path = self._get_tree_path(tree_id)
        if not os.path.exists(path):
            return None
            
        with open(path, 'r') as f:
            return json.load(f)
            
    def store_node(
        self,
        node_id: str,
        node_data: Dict[str, torch.Tensor],
        hidden_state: Optional[torch.Tensor] = None,
        metadata: Optional[Dict] = None
    ) -> None:
        """Store node data and hidden state"""
        # Store node data
        node_path = self._get_node_path(node_id)
        data = {
            'node_data': node_data,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        torch.save(data, node_path)
        
        # Store hidden state if provided
        if hidden_state is not None:
            hidden_path = self._get_hidden_path(node_id)
            torch.save(hidden_state, hidden_path)
            
        # Update access tracking
        self.node_access_counts[node_id] = self.node_access_counts.get(node_id, 0) + 1
        self._step += 1
        
    def load_node(
        self,
        node_id: str,
        load_hidden: bool = False
    ) -> Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor], Optional[Dict]]:
        """Load node data and optionally hidden state"""
        node_path = self._get_node_path(node_id)
        if not os.path.exists(node_path):
            return None, None, None
            
        data = torch.load(node_path)
        hidden_state = None
        
        if load_hidden:
            hidden_path = self._get_hidden_path(node_id)
            if os.path.exists(hidden_path):
                hidden_state = torch.load(hidden_path)
                
        # Update access tracking
        self.node_access_counts[node_id] = self.node_access_counts.get(node_id, 0) + 1
        self._step += 1
        
        return data['node_data'], hidden_state, data['metadata']
        
    def store_path(
        self,
        tree_id: str,
        path: List[str],
        path_score: float
    ) -> None:
        """Store a successful solution path"""
        path_dir = os.path.join(self.tree_dir, 'paths')
        os.makedirs(path_dir, exist_ok=True)
        
        path_data = {
            'path': path,
            'score': path_score,
            'timestamp': time.time()
        }
        
        path_file = os.path.join(path_dir, f"{tree_id}_path_{len(os.listdir(path_dir))}.json")
        with open(path_file, 'w') as f:
            json.dump(path_data, f)
            
    def get_cached_paths(
        self,
        tree_id: str,
        min_score: float = 0.0
    ) -> List[Tuple[List[str], float]]:
        """Get previously cached successful paths"""
        path_dir = os.path.join(self.tree_dir, 'paths')
        if not os.path.exists(path_dir):
            return []
            
        paths = []
        for filename in os.listdir(path_dir):
            if filename.startswith(f"{tree_id}_path_"):
                with open(os.path.join(path_dir, filename), 'r') as f:
                    path_data = json.load(f)
                    if path_data['score'] >= min_score:
                        paths.append((path_data['path'], path_data['score']))
                        
        return sorted(paths, key=lambda x: x[1], reverse=True)
        
    def clear_tree(self, tree_id: str) -> None:
        """Remove all data associated with a tree"""
        # Remove tree structure
        tree_path = self._get_tree_path(tree_id)
        if os.path.exists(tree_path):
            os.remove(tree_path)
            
        # Remove all associated nodes
        for node_id in self.node_access_counts.keys():
            if node_id.startswith(f"{tree_id}_"):
                node_path = self._get_node_path(node_id)
                hidden_path = self._get_hidden_path(node_id)
                
                if os.path.exists(node_path):
                    os.remove(node_path)
                if os.path.exists(hidden_path):
                    os.remove(hidden_path)
                    
        # Remove paths
        path_dir = os.path.join(self.tree_dir, 'paths')
        if os.path.exists(path_dir):
            for filename in os.listdir(path_dir):
                if filename.startswith(f"{tree_id}_path_"):
                    os.remove(os.path.join(path_dir, filename))
                    
        # Update tracking
        self.node_access_counts = {k: v for k, v in self.node_access_counts.items()
                                 if not k.startswith(f"{tree_id}_")}
                                 
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        num_trees = len(os.listdir(self.tree_dir))
        num_nodes = len(os.listdir(self.node_dir))
        total_accesses = sum(self.node_access_counts.values())
        
        # Calculate storage usage
        tree_size = sum(os.path.getsize(os.path.join(self.tree_dir, f)) 
                       for f in os.listdir(self.tree_dir))
        node_size = sum(os.path.getsize(os.path.join(self.node_dir, f))
                       for f in os.listdir(self.node_dir))
        hidden_size = sum(os.path.getsize(os.path.join(self.hidden_dir, f))
                         for f in os.listdir(self.hidden_dir))
                         
        return {
            'num_trees': num_trees,
            'num_nodes': num_nodes,
            'total_accesses': total_accesses,
            'storage_usage_bytes': tree_size + node_size + hidden_size,
            'cache_utilization': (tree_size + node_size + hidden_size) / self.cache_size
        }
        
    def cleanup_old_trees(self, max_age_hours: float = 24.0) -> None:
        """Remove trees older than specified age"""
        current_time = time.time()
        max_age_secs = max_age_hours * 3600
        
        # Check tree structures
        for tree_file in os.listdir(self.tree_dir):
            if tree_file.endswith('_tree.json'):
                tree_path = os.path.join(self.tree_dir, tree_file)
                with open(tree_path, 'r') as f:
                    tree_data = json.load(f)
                    if current_time - tree_data['timestamp'] > max_age_secs:
                        tree_id = tree_file.replace('_tree.json', '')
                        self.clear_tree(tree_id)
