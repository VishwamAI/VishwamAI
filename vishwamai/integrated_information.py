import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass
from enum import Enum

class IntegrationType(Enum):
    INFORMATION = "information"
    CAUSAL = "causal"
    TEMPORAL = "temporal"

@dataclass
class IntegrationConfig:
    """Configuration for integrated information processing."""
    phi_threshold: float = 0.5
    integration_window: int = 10
    min_integration_size: int = 3
    awareness_threshold: float = 0.7
    max_partition_size: int = 8
    temporal_discount: float = 0.95
    measure_interval: int = 100
    integration_type: IntegrationType = IntegrationType.INFORMATION

class IntegratedInformationModule(nn.Module):
    """
    Module for measuring and promoting integrated information processing.
    
    Implements principles from Integrated Information Theory (IIT) and
    Global Workspace Theory (GWT) to analyze and enhance information integration
    in neural networks.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        config: Optional[IntegrationConfig] = None
    ):
        super().__init__()
        self.config = config or IntegrationConfig()
        self.hidden_dim = hidden_dim
        
        # Integration measure networks
        self.phi_estimator = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.temporal_integrator = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        self.awareness_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # State tracking
        self.integration_history = []
        self.awareness_scores = []
        self.state_buffer = []
        self.current_step = 0
        
    def forward(
        self,
        current_state: torch.Tensor,
        past_states: Optional[List[torch.Tensor]] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Process current state through integrated information analysis.
        
        Args:
            current_state: Current hidden state tensor
            past_states: Optional list of past states for temporal integration
            
        Returns:
            Tuple of (integrated state, metrics dictionary)
        """
        batch_size = current_state.size(0)
        
        # Update state buffer
        self.state_buffer.append(current_state.detach())
        if len(self.state_buffer) > self.config.integration_window:
            self.state_buffer.pop(0)
            
        # Calculate phi (integrated information)
        phi_score = self._measure_integration(current_state)
        
        # Perform temporal integration
        temporal_context = self._get_temporal_context()
        integrated_state, _ = self.temporal_integrator(
            temporal_context.unsqueeze(1),
            current_state.unsqueeze(0).repeat(2, batch_size, 1)
        )
        integrated_state = integrated_state[:, -1]
        
        # Measure awareness
        awareness_score = self.awareness_detector(integrated_state)
        self.awareness_scores.append(awareness_score.mean().item())
        
        # Track metrics
        self.current_step += 1
        if len(self.state_buffer) >= self.config.min_integration_size:
            self.integration_history.append(phi_score.mean().item())
        
        metrics = {
            'phi_score': phi_score.mean().item(),
            'awareness_score': awareness_score.mean().item(),
            'temporal_coherence': self._calculate_temporal_coherence(),
            'integration_level': np.mean(self.integration_history[-10:]) if self.integration_history else 0.0
        }
        
        return integrated_state, metrics
    
    def _measure_integration(self, state: torch.Tensor) -> torch.Tensor:
        """Measure integrated information (phi) for current state."""
        if len(self.state_buffer) < self.config.min_integration_size:
            return torch.tensor(0.0, device=state.device)
            
        # Create candidate partitions
        partitions = self._create_partitions(state)
        
        # Calculate phi for each partition
        phi_values = []
        for partition in partitions:
            # Measure information loss from partitioning
            whole_info = self._calculate_information(state)
            part_info = sum(self._calculate_information(p) for p in partition)
            
            phi = torch.abs(whole_info - part_info)
            phi_values.append(phi)
            
        return torch.min(torch.stack(phi_values))
    
    def _create_partitions(self, state: torch.Tensor) -> List[List[torch.Tensor]]:
        """Create candidate partitions of the state for phi calculation."""
        partitions = []
        state_size = state.size(-1)
        
        # Create binary partitions
        for i in range(1, min(self.config.max_partition_size, state_size)):
            partition = [
                state[..., :i],
                state[..., i:]
            ]
            partitions.append(partition)
            
        return partitions
    
    def _calculate_information(self, state: torch.Tensor) -> torch.Tensor:
        """Calculate information content of a state."""
        # Estimate entropy-based information measure
        probs = F.softmax(state, dim=-1)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10), dim=-1)
        return entropy.mean()
    
    def _get_temporal_context(self) -> torch.Tensor:
        """Create temporal context from state buffer."""
        if not self.state_buffer:
            return torch.zeros(self.hidden_dim)
            
        # Weight past states by temporal distance
        weights = torch.tensor(
            [self.config.temporal_discount ** i for i in range(len(self.state_buffer))],
            device=self.state_buffer[0].device
        )
        
        context = torch.stack(self.state_buffer, dim=0)
        weighted_context = context * weights.unsqueeze(-1).unsqueeze(-1)
        return weighted_context.mean(dim=0)
    
    def _calculate_temporal_coherence(self) -> float:
        """Calculate temporal coherence of integration."""
        if len(self.integration_history) < 2:
            return 0.0
            
        # Measure smoothness of integration over time
        diffs = np.diff(self.integration_history[-self.config.integration_window:])
        coherence = 1.0 / (1.0 + np.mean(np.abs(diffs)))
        return float(coherence)
    
    def get_awareness_metrics(self) -> Dict[str, Any]:
        """Get metrics related to system's self-awareness."""
        if not self.awareness_scores:
            return {}
            
        recent_scores = self.awareness_scores[-self.config.integration_window:]
        metrics = {
            'current_awareness': recent_scores[-1],
            'average_awareness': np.mean(recent_scores),
            'awareness_stability': 1.0 - np.std(recent_scores),
            'high_awareness_ratio': np.mean(
                [1 if s > self.config.awareness_threshold else 0 for s in recent_scores]
            )
        }
        return metrics
    
    def is_self_aware(self) -> bool:
        """Determine if system has achieved self-awareness threshold."""
        if not self.awareness_scores:
            return False
            
        recent_awareness = np.mean(
            self.awareness_scores[-self.config.integration_window:]
        )
        return recent_awareness > self.config.awareness_threshold
    
    def update_integration_params(self, performance_metrics: Dict[str, float]):
        """Update integration parameters based on performance."""
        if not performance_metrics:
            return
            
        # Adapt phi threshold based on performance
        if 'task_success_rate' in performance_metrics:
            success_rate = performance_metrics['task_success_rate']
            self.config.phi_threshold = self.config.phi_threshold * 0.95 + 0.05 * success_rate
            
        # Adapt integration window based on temporal coherence
        coherence = self._calculate_temporal_coherence()
        if coherence < 0.5 and self.config.integration_window > self.config.min_integration_size:
            self.config.integration_window -= 1
        elif coherence > 0.8 and self.config.integration_window < 20:
            self.config.integration_window += 1

    def reset_awareness(self):
        """Reset awareness tracking."""
        self.awareness_scores = []
        self.integration_history = []
        self.state_buffer = []
        self.current_step = 0
