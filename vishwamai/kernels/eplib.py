from typing import Tuple, Dict, Any

import jax
import jax.numpy as jnp
from flax import struct


def balanced_packing(weight: jnp.ndarray, num_packs: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly n/m objects and the weights of all packs
    are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs
    
    Returns: 
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = jnp.broadcast_to(jnp.arange(weight.shape[-1], dtype=jnp.int32), weight.shape)
        rank_in_pack = jnp.zeros_like(weight, dtype=jnp.int32)
        return pack_index, rank_in_pack

    # Sort indices by weight in descending order
    indices = jnp.argsort(-weight, axis=-1)
    
    # Initialize arrays with default values
    pack_index = jnp.full_like(weight, -1, dtype=jnp.int32)
    rank_in_pack = jnp.full_like(pack_index, -1)
    
    # This is trickier in JAX due to its functional nature
    # We'll use a scan to build up the pack assignments
    
    def process_layer(carry, x):
        i, indices_i = x
        pack_index_i = jnp.full((num_groups,), -1, dtype=jnp.int32)
        rank_in_pack_i = jnp.full((num_groups,), -1, dtype=jnp.int32)
        
        # Initialize pack state
        pack_weights = jnp.zeros((num_packs,), dtype=jnp.float32)
        pack_items = jnp.zeros((num_packs,), dtype=jnp.int32)
        
        def process_group(state, group_idx):
            pack_weights, pack_items, pack_index_i, rank_in_pack_i = state
            group = indices_i[group_idx]
            
            # Find the pack with minimum weight that has space
            available_mask = pack_items < groups_per_pack
            # Large value for unavailable packs to ensure they're not selected
            modified_weights = jnp.where(available_mask, pack_weights, jnp.inf)
            pack = jnp.argmin(modified_weights)
            
            # Update pack state
            pack_weights = pack_weights.at[pack].add(weight[i, group])
            pack_items = pack_items.at[pack].add(1)
            pack_index_i = pack_index_i.at[group].set(pack)
            rank_in_pack_i = rank_in_pack_i.at[group].set(pack_items[pack] - 1)
            
            return (pack_weights, pack_items, pack_index_i, rank_in_pack_i), None
        
        # Process all groups for this layer
        (_, _, pack_index_i, rank_in_pack_i), _ = jax.lax.scan(
            process_group, 
            (pack_weights, pack_items, pack_index_i, rank_in_pack_i), 
            jnp.arange(num_groups)
        )
        
        # Update the i-th row in our result arrays
        pack_index = carry[0].at[i].set(pack_index_i)
        rank_in_pack = carry[1].at[i].set(rank_in_pack_i)
        
        return (pack_index, rank_in_pack), None
    
    # Process all layers
    (pack_index, rank_in_pack), _ = jax.lax.scan(
        process_layer,
        (pack_index, rank_in_pack),
        (jnp.arange(num_layers), indices)
    )
    
    return pack_index, rank_in_pack


def replicate_experts(weight: jnp.ndarray, num_phy: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum load of all replicas is minimized.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication
    
    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    
    # Initialize
    phy2log = jnp.broadcast_to(jnp.arange(num_log, dtype=jnp.int32), (n, num_log))
    rank = jnp.zeros((n, num_log), dtype=jnp.int32)
    logcnt = jnp.ones((n, num_log), dtype=jnp.int32)
    
    if num_redundant == 0:
        # No replication needed
        return phy2log, rank, logcnt
    
    # For each redundant expert, add a replica to the logical expert with highest load
    def add_replica(state, i):
        phy2log, rank, logcnt = state
        
        # Compute load for each logical expert
        load = weight / logcnt
        
        # Find logical expert with highest load
        redundant_indices = jnp.argmax(load, axis=-1)
        
        # Add this expert to the mapping at position num_log + i
        phy2log = jax.vmap(lambda p, idx: jnp.concatenate([p, jnp.array([idx])]))( 
            phy2log, redundant_indices
        )
        
        # Update rank
        new_ranks = jnp.take_along_axis(logcnt, redundant_indices[:, None], axis=1)
        rank = jax.vmap(lambda r, new_r: jnp.concatenate([r, new_r]))(
            rank, new_ranks[:, 0]
        )
        
        # Update count
        logcnt = jnp.add(logcnt, jnp.eye(num_log, dtype=jnp.int32)[redundant_indices])
        
        return (phy2log, rank, logcnt), None
    
    # Add replicas one by one
    (phy2log, rank, logcnt), _ = jax.lax.scan(
        add_replica,
        (phy2log, rank, logcnt),
        jnp.arange(num_redundant)
    )
    
    return phy2log, rank, logcnt


def inverse(perm: jnp.ndarray) -> jnp.ndarray:
    """Compute the inverse permutation."""
    return jnp.argsort(perm, axis=1)


def rebalance_experts_hierarchical(weight: jnp.ndarray, num_physical_experts: int,
                                   num_groups: int, num_nodes: int, num_gpus: int):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns: 
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups 
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    # Step 1: pack groups to nodes
    tokens_per_group = jnp.sum(
        jnp.reshape(weight, (num_layers, num_groups, group_size)), 
        axis=-1
    )
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)
    
    # Create mapping from logical to intermediate logical experts
    expanded_indices = jnp.arange(group_size, dtype=jnp.int32)
    indices_offset = (group_pack_index * groups_per_node + group_rank_in_pack) * group_size
    log2mlog = jax.vmap(lambda offset: jnp.broadcast_to(offset[:, None], (num_groups, group_size)) + expanded_indices)(
        indices_offset
    ).reshape(num_layers, num_logical_experts)
    
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    # Reshape to [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = jnp.take_along_axis(weight, mlog2log, axis=1).reshape(
        num_layers * num_nodes, num_logical_experts // num_nodes
    )
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, 
        num_physical_experts // num_nodes
    )

    # Step 3: pack physical_experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    tokens_per_phy = jnp.take_along_axis(
        tokens_per_mlog / mlogcnt, 
        phy2mlog, 
        axis=1
    )
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy, num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    # Final mappings
    pphy2mlog = jnp.take_along_axis(phy2mlog, pphy2phy, axis=1)
    pphy2mlog = jnp.reshape(pphy2mlog, (num_layers, num_nodes, -1))
    
    # Adjust indices based on node
    offsets = jnp.arange(0, num_logical_experts, num_logical_experts // num_nodes)
    pphy2mlog = pphy2mlog + offsets[None, :, None]
    pphy2mlog = jnp.reshape(pphy2mlog, (num_layers, -1))
    
    # Map to logical experts
    pphy2log = jnp.take_along_axis(mlog2log, pphy2mlog, axis=1)
    pphyrank = jnp.take_along_axis(phyrank, pphy2phy, axis=1).reshape(num_layers, -1)
    logcnt = jnp.take_along_axis(
        jnp.reshape(mlogcnt, (num_layers, -1)),
        log2mlog,
        axis=1
    )
    
    return pphy2log, pphyrank, logcnt


def rebalance_experts(weight: jnp.ndarray, num_replicas: int, num_groups: int,
                     num_nodes: int, num_gpus: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Entry point for expert-parallelism load balancer.

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all logical experts
        num_replicas: number of physical experts, must be a multiple of `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns: 
        physical_to_logical_map: [layers, num_replicas], the expert index of each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica indices for each expert
        expert_count: [layers, num_logical_experts], number of physical replicas for each logical expert
    """
    num_layers, num_logical_experts = weight.shape
    
    if num_groups % num_nodes == 0:
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, 
            num_replicas, 
            num_groups, 
            num_nodes, 
            num_gpus
        )
    else:
        # use global load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, 
            num_replicas, 
            1, 
            1, 
            num_gpus
        )
    
    # Convert to logical-to-physical mapping
    maxlogcnt = jnp.max(logcnt)
    log2phy = jnp.full((num_layers, num_logical_experts, maxlogcnt), -1, dtype=jnp.int32)
    
    # Create indices for physical experts
    phy_indices = jnp.arange(num_replicas, dtype=jnp.int32)
    phy_indices = jnp.broadcast_to(phy_indices, (num_layers, num_replicas))
    
    # Compute scatter indices
    scatter_indices = phy2log * maxlogcnt + phyrank
    
    # Create log2phy mapping using scatter
    def update_log2phy_layer(log2phy_layer, layer_info):
        idx, phy_indices_layer, scatter_indices_layer = layer_info
        log2phy_flat = log2phy_layer.reshape(-1)
        scatter_idx = scatter_indices_layer
        updates = phy_indices_layer
        
        # Use scatter_add with one-hot encoding for assignment in pure functional way
        one_hot = jax.nn.one_hot(scatter_idx, log2phy_flat.shape[0], dtype=jnp.int32)
        updates_expanded = jnp.expand_dims(updates, 1)
        # Multiply creates sparse updates, then reduce_sum applies them
        updates_scattered = jnp.sum(one_hot * updates_expanded[:, None], axis=0)
        
        # Mask to only update non-negative indices
        mask = scatter_idx >= 0
        valid_updates = jnp.where(
            jnp.any(mask[:, None] & (one_hot > 0), axis=0),
            updates_scattered,
            log2phy_flat
        )
        
        return valid_updates.reshape(log2phy_layer.shape), None
    
    # Process all layers
    (log2phy, _), _ = jax.lax.scan(
        update_log2phy_layer,
        log2phy,
        (jnp.arange(num_layers), phy_indices, scatter_indices)
    )
    
    return phy2log, log2phy, logcnt