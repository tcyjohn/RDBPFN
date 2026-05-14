"""Hierarchical Stochastic Block Model (HSBM) for bipartite FK connectivity.

Ported from ``plurel/plurel/bipartite.py``, simplified to return FK indices
directly instead of building a full networkx graph.

Joint multi-parent sampling added: for each child row, a shared latent cluster
path is first sampled, then each parent FK is drawn within the aligned parent
cluster, introducing tuple correlation via the shared latent variable.
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


def _get_probs_at_levels(
    hierarchy_a: list,
    hierarchy_b: list,
    rng: np.random.RandomState | None = None,
) -> list:
    """Build per-level probability matrices.

    Diagonal entries (within-block) are set to 0.9; off-diagonal entries
    (cross-block) are sampled from Uniform(0.001, 0.002).
    """
    if rng is None:
        rng = np.random.RandomState()
    assert len(hierarchy_a) == len(hierarchy_b), (
        "only equal-length hierarchies are supported"
    )
    probs_at_levels = []
    for l_idx in range(len(hierarchy_a)):
        shape = (hierarchy_a[l_idx], hierarchy_b[l_idx])
        probs = rng.uniform(0.001, 0.002, size=shape)
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
    # Deterministic child cluster assignments (preserved for backward compat)
    cluster_b = assign_cluster_at_levels(num_nodes=size_b, hierarchy=hierarchy_b)
    probs_at_levels = _get_probs_at_levels(hierarchy_a, hierarchy_b, rng=rng)

    fk_ids = _sample_fk_per_parent(
        size_a=size_a,
        size_b=size_b,
        cluster_a=cluster_a,
        cluster_b=cluster_b,
        probs_at_levels=probs_at_levels,
        rng=rng,
    )
    return fk_ids


def _sample_fk_per_parent(
    size_a: int,
    size_b: int,
    cluster_a: np.ndarray,
    cluster_b: np.ndarray,
    probs_at_levels: list,
    rng: np.random.RandomState,
) -> np.ndarray:
    """Core sampling loop: for each child row, sample one parent row via HSBM."""
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


def compute_hsbm_fk_ids_multi(
    parent_sizes: list[int],
    child_size: int,
    hierarchies_parent: list[list[int]],
    hierarchy_child: list[int],
    seed: int | None = None,
) -> np.ndarray:
    """Joint FK sampling for multiple parents via shared latent cluster path.

    For each child row ``j``, a shared cluster path ``c_b[j, :]`` is first
    sampled uniformly at random across child-side hierarchy levels. All parents
    then share this same cluster path and the same per-level probability
    matrices, ensuring that tuple-level FK correlation comes from the shared
    latent variable rather than from explicit cross-parent interaction terms.

    The child-side hierarchy is sampled at the *maximum* depth across all
    parents.  Each parent reads only the prefix of ``c_b[j, :]`` that matches
    its own number of levels, so parents with different depths still share the
    same latent variable.

    Parameters
    ----------
    parent_sizes : list of int
        Number of rows for each parent table, ``[n_1, ..., n_k]``.
    child_size : int
        Number of child rows.
    hierarchies_parent : list of list of int
        Per-parent hierarchy, ``[[h_1, ...], [h_2, ...], ...]``.
        Parents may have different numbers of levels.
    hierarchy_child : list of int
        Cluster counts per level on the child side (shared across parents).
        Must be at least as long as the longest ``hierarchies_parent`` entry.
        Each parent ``p`` reads only the first ``len(hierarchies_parent[p])``
        elements of this hierarchy.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray of shape ``(child_size, num_parents)``
        ``fk_ids[j, p]`` is the parent-row index (0-based) that child row ``j``
        connects to for parent ``p``.
    """
    rng = np.random.RandomState(seed)
    num_parents = len(parent_sizes)
    num_levels = len(hierarchy_child)  # max depth across all parents

    # 1. Parent-side cluster assignments (deterministic, per parent)
    cluster_per_parent = []
    for p in range(num_parents):
        ca = assign_cluster_at_levels(
            num_nodes=parent_sizes[p], hierarchy=hierarchies_parent[p],
        )
        cluster_per_parent.append(ca)

    # 2. Shared probability matrices, each parent reads its own prefix only
    shared_probs_per_parent = []
    for p in range(num_parents):
        nlv_p = len(hierarchies_parent[p])
        probs = _get_probs_at_levels(
            hierarchies_parent[p],
            hierarchy_child[:nlv_p],
            rng=rng,
        )
        shared_probs_per_parent.append(probs)

    # 3. Shared child cluster path: sampled at max depth, each parent reads prefix
    cluster_b = np.empty((child_size, num_levels), dtype=int)
    for l in range(num_levels):
        cluster_b[:, l] = rng.randint(0, hierarchy_child[l], size=child_size)

    # 4. Per-parent FK sampling — _sample_fk_per_parent only iterates over
    #    len(probs_at_levels_p) levels, naturally reading the correct prefix.
    fk_ids = np.empty((child_size, num_parents), dtype=np.int64)
    for p in range(num_parents):
        fk_ids[:, p] = _sample_fk_per_parent(
            size_a=parent_sizes[p],
            size_b=child_size,
            cluster_a=cluster_per_parent[p],
            cluster_b=cluster_b,
            probs_at_levels=shared_probs_per_parent[p],
            rng=rng,
        )

    return fk_ids
