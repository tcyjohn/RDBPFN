"""Hierarchical Stochastic Block Model (HSBM) for bipartite FK connectivity.

Ported from ``plurel/plurel/bipartite.py``, simplified to return FK indices
directly instead of building a full networkx graph.
"""

import numpy as np


def assign_cluster_at_levels(num_nodes: int, hierarchy: list) -> np.ndarray:
    """Assign nodes to nested hierarchical blocks.

    Parameters
    ----------
    num_nodes : int
        Number of nodes on one side of the bipartite graph.
    hierarchy : list of int
        Number of clusters at each level, e.g. ``[2, 4]`` means 2 top-level
        blocks, each subdivided into 4 leaf blocks (8 leaf blocks total).

    Returns
    -------
    np.ndarray of shape ``(num_nodes, len(hierarchy))``
        ``cluster_at_levels[n, l]`` is the cluster index of node ``n`` at
        hierarchy level ``l``.
    """
    num_base_clusters = int(np.prod(hierarchy))
    nodes_per_cluster = int(np.ceil(num_nodes / num_base_clusters))
    base_cluster_offsets = [
        nodes_per_cluster * (c_idx + 1) for c_idx in range(num_base_clusters - 1)
    ]
    base_cluster_offsets.append(num_nodes)
    cluster_at_levels = np.zeros((num_nodes, len(hierarchy)), dtype=int)
    cluster_node_idx_start = 0
    for c_idx in range(num_base_clusters):
        for l_idx in range(len(hierarchy)):
            fac = np.prod(hierarchy[l_idx + 1:]) if l_idx + 1 < len(hierarchy) else 1
            cluster_node_idx_end = base_cluster_offsets[c_idx]
            cluster_at_levels[cluster_node_idx_start:cluster_node_idx_end, l_idx] = (
                c_idx // fac
            ) % hierarchy[l_idx]
        cluster_node_idx_start = cluster_node_idx_end
    return cluster_at_levels


def _get_probs_at_levels(hierarchy_a: list, hierarchy_b: list) -> list:
    """Build per-level probability matrices.

    Diagonal entries (within-block) are set to 0.9; off-diagonal entries
    (cross-block) are sampled from Uniform(0.001, 0.002).
    """
    assert len(hierarchy_a) == len(hierarchy_b), (
        "only equal-length hierarchies are supported"
    )
    probs_at_levels = []
    for l_idx in range(len(hierarchy_a)):
        shape = (hierarchy_a[l_idx], hierarchy_b[l_idx])
        probs = np.random.uniform(0.001, 0.002, size=shape)
        for i in range(max(shape)):
            probs[i % shape[0], i % shape[1]] = 0.9
        probs_at_levels.append(probs)
    return probs_at_levels


def compute_hsbm_fk_ids(
    size_a: int,
    size_b: int,
    hierarchy_a: list,
    hierarchy_b: list,
    seed: int | None = None,
) -> np.ndarray:
    """Compute parent FK indices via HSBM bipartite sampling.

    Each of the ``size_b`` child rows samples exactly one parent row from the
    ``size_a`` available parent rows.  The connection probability between a
    parent row ``a`` and child row ``b`` is the product of per-level block
    probabilities.

    Parameters
    ----------
    size_a : int
        Number of parent rows (A-side).
    size_b : int
        Number of child rows (B-side).
    hierarchy_a : list of int
        Cluster counts per level on the parent side.
    hierarchy_b : list of int
        Cluster counts per level on the child side.
    seed : int or None
        RNG seed for reproducibility (applied locally).

    Returns
    -------
    np.ndarray of shape ``(size_b,)``
        ``fk_ids[j]`` is the parent-row index (0-based) that child row ``j``
        connects to.
    """
    assert len(hierarchy_a) == len(hierarchy_b), (
        "only equal-length hierarchies are supported"
    )
    rng = np.random.RandomState(seed)

    cluster_a = assign_cluster_at_levels(num_nodes=size_a, hierarchy=hierarchy_a)
    cluster_b = assign_cluster_at_levels(num_nodes=size_b, hierarchy=hierarchy_b)
    probs_at_levels = _get_probs_at_levels(hierarchy_a, hierarchy_b)

    fk_ids = np.empty(size_b, dtype=np.int64)

    for b_idx in range(size_b):
        probs = np.ones(size_a, dtype=np.float64)
        for l_idx in range(len(probs_at_levels)):
            cb = cluster_b[b_idx, l_idx]
            probs *= probs_at_levels[l_idx][cluster_a[:, l_idx], cb]
        p_sum = probs.sum()
        if p_sum > 0:
            probs /= p_sum
        else:
            probs = None
        fk_ids[b_idx] = rng.choice(size_a, p=probs)

    return fk_ids
