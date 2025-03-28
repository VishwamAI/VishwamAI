"""Expert parallelism load balancing utilities."""

import jax
import jax.numpy as jnp
from typing import Tuple, Dict, Any, NamedTuple,Optional
from functools import partial

class ExpertBalanceResult(NamedTuple):
    """Results from expert balancing."""
    physical_to_logical_map: jnp.ndarray  # [layers, num_replicas]
    logical_to_physical_map: jnp.ndarray  # [layers, num_logical_experts, max_replicas]
    expert_count: jnp.ndarray  # [layers, num_logical_experts]

@partial(jax.jit, static_argnums=(1,2))
def rebalance_experts(
    weight: jnp.ndarray,
    num_replicas: int,
    num_groups: int,
    num_nodes: Optional[int] = None,
    num_gpus: Optional[int] = None,
    use_hierarchical: bool = True
) -> ExpertBalanceResult:
    """
    Expert-parallelism load balancer that optimizes expert placement across devices.

    Args:
        weight: [layers, num_logical_experts], the load statistics for all logical experts
        num_replicas: number of physical experts (must be multiple of num_gpus if specified)
        num_groups: number of expert groups
        num_nodes: number of server nodes for hierarchical balancing
        num_gpus: number of GPUs (must be multiple of num_nodes if specified)
        use_hierarchical: whether to use hierarchical load balancing

    Returns: 
        ExpertBalanceResult containing:
        - physical_to_logical_map: [layers, num_replicas], expert index for each replica
        - logical_to_physical_map: [layers, num_logical_experts, X], replica indices for each expert
        - expert_count: [layers, num_logical_experts], number of physical replicas per logical expert
    """
    if not use_hierarchical or num_nodes is None or num_gpus is None:
        # Use simple replication without hierarchy
        return _rebalance_experts_simple(weight, num_replicas, num_groups, num_gpus or 1)
    else:
        # Use hierarchical load balancing
        return _rebalance_experts_hierarchical(weight, num_replicas, num_groups, num_nodes, num_gpus)

@partial(jax.jit, static_argnums=(1,2,3))
def _rebalance_experts_simple(
    weight: jnp.ndarray,
    num_replicas: int,
    num_groups: int,
    num_gpus: int
) -> ExpertBalanceResult:
    """Simple expert rebalancing without hierarchy."""
    
    # First replicate experts based on load
    phy2log, rank, logcnt = replicate_experts(weight, num_replicas)
    
    # Then pack replicas to GPUs for balanced load
    tokens_per_phy = jnp.take_along_axis(
        weight / logcnt[:, :, None], 
        phy2log[..., None], 
        axis=1
    ).squeeze(-1)
    
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus)
    experts_per_gpu = num_replicas // num_gpus
    
    # Create final physical mapping
    phy2pphy = pack_index * experts_per_gpu + rank_in_pack
    pphy2phy = jnp.argsort(phy2pphy, axis=1)
    
    # Get final mappings
    physical_to_logical = jnp.take_along_axis(phy2log, pphy2phy, axis=1)
    
    # Create logical to physical mapping
    maxlogcnt = jnp.max(logcnt)
    logical_to_physical = jnp.full(
        (weight.shape[0], weight.shape[1], maxlogcnt),
        -1, dtype=jnp.int32
    )
    
    def update_layer_mapping(carry, idx):
        layer_idx, phys_idx = idx
        log_expert = physical_to_logical[layer_idx, phys_idx]
        count = jnp.sum(carry[layer_idx, log_expert] >= 0)
        carry = carry.at[layer_idx, log_expert, count].set(phys_idx)
        return carry, None
    
    # Create indices for all physical experts
    layer_indices = jnp.arange(weight.shape[0])[:, None]
    phys_indices = jnp.arange(num_replicas)[None, :]
    indices = jnp.stack([
        jnp.broadcast_to(layer_indices, (weight.shape[0], num_replicas)),
        jnp.broadcast_to(phys_indices, (weight.shape[0], num_replicas))
    ], axis=-1)
    
    # Update logical to physical mapping
    logical_to_physical, _ = jax.lax.scan(
        update_layer_mapping,
        logical_to_physical,
        indices.reshape(-1, 2)
    )
    
    return ExpertBalanceResult(physical_to_logical, logical_to_physical, logcnt)

@partial(jax.jit, static_argnums=(1,2,3,4))
def _rebalance_experts_hierarchical(
    weight: jnp.ndarray,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int
) -> ExpertBalanceResult:
    """Hierarchical expert rebalancing across nodes and GPUs."""
    
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups 
    assert num_groups % num_nodes == 0
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    
    # Step 1: Pack expert groups to nodes
    tokens_per_group = jnp.sum(
        weight.reshape(num_layers, num_groups, group_size),
        axis=-1
    )
    group_pack_index, group_rank = balanced_packing(tokens_per_group, num_nodes)
    
    # Step 2: Create intermediate mappings between nodes
    groups_per_node = num_groups // num_nodes
    expanded_indices = jnp.arange(group_size)
    indices_offset = (group_pack_index * groups_per_node + group_rank) * group_size
    log2mlog = jax.vmap(
        lambda offset: jnp.broadcast_to(
            offset[:, None], 
            (num_groups, group_size)
        ) + expanded_indices
    )(indices_offset).reshape(num_layers, num_logical_experts)
    
    mlog2log = jnp.argsort(log2mlog, axis=1)
    
    # Step 3: Replicate experts within nodes
    tokens_per_mlog = jnp.take_along_axis(
        weight,
        mlog2log,
        axis=1
    ).reshape(num_layers * num_nodes, num_logical_experts // num_nodes)
    
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog,
        num_physical_experts // num_nodes
    )
    
    # Step 4: Pack physical experts to GPUs within each node
    tokens_per_phy = jnp.take_along_axis(
        tokens_per_mlog / mlogcnt[:, :, None],
        phy2mlog[..., None],
        axis=1
    ).squeeze(-1)
    
    pack_index, rank_in_pack = balanced_packing(
        tokens_per_phy,
        num_gpus // num_nodes
    )
    phy2pphy = pack_index * (num_physical_experts // num_gpus) + rank_in_pack
    
    # Create final mappings
    pphy2phy = jnp.argsort(phy2pphy, axis=1)
    pphy2mlog = jnp.take_along_axis(phy2mlog, pphy2phy, axis=1)
    
    # Adjust indices based on node
    pphy2mlog = pphy2mlog.reshape(num_layers, num_nodes, -1)
    offsets = jnp.arange(0, num_logical_experts, num_logical_experts // num_nodes)
    pphy2mlog = pphy2mlog + offsets[None, :, None]
    pphy2mlog = pphy2mlog.reshape(num_layers, -1)
    
    # Map to logical experts
    physical_to_logical = jnp.take_along_axis(mlog2log, pphy2mlog, axis=1)
    phyrank = jnp.take_along_axis(
        phyrank.reshape(num_layers, -1),
        pphy2phy,
        axis=1
    )
    expert_count = jnp.take_along_axis(
        mlogcnt.reshape(num_layers, -1),
        log2mlog,
        axis=1
    )
    
    # Create logical to physical mapping
    maxlogcnt = jnp.max(expert_count)
    logical_to_physical = jnp.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1, dtype=jnp.int32
    )
    
    def update_layer_mapping(carry, idx):
        layer_idx, phys_idx = idx
        log_expert = physical_to_logical[layer_idx, phys_idx]
        count = jnp.sum(carry[layer_idx, log_expert] >= 0)
        carry = carry.at[layer_idx, log_expert, count].set(phys_idx)
        return carry, None
    
    # Create indices for all physical experts
    layer_indices = jnp.arange(num_layers)[:, None]
    phys_indices = jnp.arange(num_physical_experts)[None, :]
    indices = jnp.stack([
        jnp.broadcast_to(layer_indices, (num_layers, num_physical_experts)),
        jnp.broadcast_to(phys_indices, (num_layers, num_physical_experts))
    ], axis=-1)
    
    # Update logical to physical mapping
    logical_to_physical, _ = jax.lax.scan(
        update_layer_mapping,
        logical_to_physical,
        indices.reshape(-1, 2)
    )
    
    return ExpertBalanceResult(physical_to_logical, logical_to_physical, expert_count)

@partial(jax.jit, static_argnums=(1,))
def replicate_experts(
    weight: jnp.ndarray,
    num_phy: int
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Replicate experts based on load balancing.

    Args:
        weight: [X, num_log], load statistics for logical experts
        num_phy: total number of experts after replication
    
    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], replica rank for each physical expert
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    
    # Initialize mappings
    phy2log = jnp.broadcast_to(jnp.arange(num_log), (n, num_log))
    rank = jnp.zeros((n, num_log), dtype=jnp.int32)
    logcnt = jnp.ones((n, num_log), dtype=jnp.int32)
    
    if num_redundant == 0:
        return phy2log, rank, logcnt
        
    def add_replica(state, i):
        phy2log, rank, logcnt = state
        
        # Find expert with highest load
        load = weight / logcnt
        redundant_indices = jnp.argmax(load, axis=-1)
        
        # Add replica
        phy2log = jax.vmap(lambda p, idx: jnp.concatenate([p, jnp.array([idx])]))(
            phy2log, redundant_indices
        )
        new_ranks = jnp.take_along_axis(logcnt, redundant_indices[:, None], axis=1)
        rank = jax.vmap(lambda r, new_r: jnp.concatenate([r, new_r]))(
            rank, new_ranks[:, 0]
        )
        logcnt = jnp.add(logcnt, jnp.eye(num_log)[redundant_indices])
        
        return (phy2log, rank, logcnt), None
    
    (phy2log, rank, logcnt), _ = jax.lax.scan(
        add_replica,
        (phy2log, rank, logcnt),
        jnp.arange(num_redundant)
    )
    
    return phy2log, rank, logcnt

@partial(jax.jit, static_argnums=(1,))
def balanced_packing(
    weight: jnp.ndarray,
    num_packs: int
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Pack weighted items into balanced packs.

    Args:
        weight: [X, n], weight of each item
        num_packs: number of packs
    
    Returns: 
        pack_index: [X, n], pack index for each item
        rank_in_pack: [X, n], rank within assigned pack
    """
    num_layers, num_items = weight.shape
    assert num_items % num_packs == 0
    items_per_pack = num_items // num_packs
    
    if items_per_pack == 1:
        pack_index = jnp.broadcast_to(
            jnp.arange(num_items),
            (num_layers, num_items)
        )
        rank_in_pack = jnp.zeros_like(pack_index)
        return pack_index, rank_in_pack
    
    # Sort items by weight
    indices = jnp.argsort(-weight, axis=-1)
    
    def process_layer(carry, x):
        layer_idx, indices_i = x
        
        # Initialize pack state
        pack_weights = jnp.zeros((num_packs,))
        pack_items = jnp.zeros((num_packs,), dtype=jnp.int32)
        pack_index = jnp.full((num_items,), -1, dtype=jnp.int32)
        rank_in_pack = jnp.full((num_items,), -1, dtype=jnp.int32)
        
        def assign_item(state, item_idx):
            pack_weights, pack_items, pack_index, rank_in_pack = state
            item = indices_i[item_idx]
            
            # Find pack with minimum weight and space
            available = pack_items < items_per_pack
            modified_weights = jnp.where(available, pack_weights, jnp.inf)
            pack = jnp.argmin(modified_weights)
            
            # Update pack
            pack_weights = pack_weights.at[pack].add(weight[layer_idx, item])
            pack_items = pack_items.at[pack].add(1)
            pack_index = pack_index.at[item].set(pack)
            rank_in_pack = rank_in_pack.at[item].set(pack_items[pack] - 1)
            
            return (pack_weights, pack_items, pack_index, rank_in_pack), None
        
        # Process all items
        (_, _, pack_index, rank_in_pack), _ = jax.lax.scan(
            assign_item,
            (pack_weights, pack_items, pack_index, rank_in_pack),
            jnp.arange(num_items)
        )
        
        # Update results
        carry = (
            carry[0].at[layer_idx].set(pack_index),
            carry[1].at[layer_idx].set(rank_in_pack)
        )
        return carry, None
    
    # Initialize result arrays
    pack_index = jnp.full((num_layers, num_items), -1, dtype=jnp.int32)
    rank_in_pack = jnp.full((num_layers, num_items), -1, dtype=jnp.int32)
    
    # Process all layers
    (pack_index, rank_in_pack), _ = jax.lax.scan(
        process_layer,
        (pack_index, rank_in_pack),
        (jnp.arange(num_layers), indices)
    )
    
    return pack_index, rank_in_pack
