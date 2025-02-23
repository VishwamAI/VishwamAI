"""
This module implements routing mechanisms for Mixture of Experts (MoE) models.

The router is responsible for:
1. Computing routing weights/scores for each input
2. Determining which experts should process each input
3. Managing capacity and load balancing across experts
4. Handling routing decisions in both training and inference

Key components:
- Token routing: Routes individual tokens to appropriate experts
- Sequence routing: Routes entire sequences or blocks to experts
- Capacity management: Handles overflow when expert capacity is exceeded
- Load balancing: Ensures even distribution of computation across experts
"""

from vishwamai.model.moe.router import token_router
from vishwamai.model.moe.router import sequence_router
from vishwamai.model.moe.router import capacity_manager
from vishwamai.model.moe.router import load_balancer
from vishwamai.model.moe.router import routing_cache

__all__ = [
    'token_router',
    'sequence_router',
    'capacity_manager',
    'load_balancer',
    'routing_cache'
]
