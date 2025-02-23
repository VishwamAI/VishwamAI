"""
This module provides gating mechanisms for Mixture of Experts (MoE) routing.

The gating module implements different algorithms for selecting which experts to use
for each input token/sequence. This includes:

- Top-k gating: Selects k experts with highest routing probabilities
- Noisy top-k gating: Adds noise to routing scores before selection
- Balanced gating: Attempts to distribute load evenly across experts
- Hierarchical gating: Multi-level expert selection for large expert pools

The gating policies here work in conjunction with the router module to determine
the final expert assignments and combinations.
"""

from vishwamai.model.moe.gating import expert_selection
from vishwamai.model.moe.gating import load_balancing
from vishwamai.model.moe.gating import noise_injection

__all__ = [
    'expert_selection',
    'load_balancing', 
    'noise_injection'
]
