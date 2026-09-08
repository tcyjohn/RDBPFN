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
    null_prob: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    null_prob : float
        Probability a child row gets FK=-1 (no parent connection).
    Returns
    -------
    fk_ids : np.ndarray of shape ``(size_b,)``
        ``fk_ids[j]`` is the parent-row index (0-based) that child row ``j``
        connects to, or -1 if no connection.
    cluster_b : np.ndarray of shape ``(size_b, len(hierarchy_b))``
        Cluster path per child row (hierarchical block assignment).
    cluster_a : np.ndarray of shape ``(size_a, len(hierarchy_a))``
        Cluster path per parent row. Used for homophily labels on entity tables.
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
        null_prob=null_prob,
    )
    return fk_ids, cluster_b, cluster_a


def _sample_fk_per_parent(
    size_a: int,
    size_b: int,
    cluster_a: np.ndarray,
    cluster_b: np.ndarray,
    probs_at_levels: list,
    rng: np.random.RandomState,
    null_prob: float = 0.0,
) -> np.ndarray:
    """Sample one parent row per child row via HSBM.

    Child rows sharing the same cluster path across all hierarchy levels have
    identical probability vectors over parent rows.  We group by cluster path,
    compute probs once per group, then sample all rows in the group in a single
    call to ``rng.choice(..., size=N)``.

    Parameters
    ----------
    null_prob : float
        Probability that a child row gets FK=-1 (no parent connection).
        Applied independently per row after HSBM sampling.
    """
    num_levels = len(probs_at_levels)
    fk_ids = np.empty(size_b, dtype=np.int64)

    # Encode each child row's cluster path as a unique integer key
    max_clusters = cluster_b.max(axis=0) + 1  # hierarchy sizes per level
    multipliers = np.cumprod([1] + list(max_clusters[:-1]))
    cluster_keys = (cluster_b * multipliers).sum(axis=1)  # (size_b,)

    unique_keys = np.unique(cluster_keys)

    for key in unique_keys:
        mask = cluster_keys == key
        indices = np.nonzero(mask)[0]
        n_in_cluster = len(indices)
        cluster_path = cluster_b[indices[0]]

        probs = np.ones(size_a, dtype=np.float64)
        for l in range(num_levels):
            probs *= probs_at_levels[l][cluster_a[:, l], cluster_path[l]]

        p_sum = probs.sum()
        if p_sum > 0:
            probs /= p_sum
        else:
            probs = None

        fk_ids[indices] = rng.choice(size_a, size=n_in_cluster, p=probs)

    if null_prob > 0:
        null_mask = rng.random(size_b) < null_prob
        fk_ids[null_mask] = -1

    return fk_ids


def compute_hsbm_fk_ids_with_propensity(
    size_a: int,
    size_b: int,
    hierarchy_a: list[int],
    hierarchy_b: list[int],
    parent_causal_output: np.ndarray,
    propensity_rho: float,
    propensity_beta: float,
    seed: int | None = None,
    null_prob: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute FK indices via HSBM + FK Propensity (S12).

    Within each HSBM block, sampling is biased toward parent rows whose
    feature-derived rank score is close to a child-specific target value.

    Parameters
    ----------
    size_a : int
        Number of parent rows.
    size_b : int
        Number of child rows.
    hierarchy_a, hierarchy_b : list of int
        Cluster counts per level (parent and child sides).
    parent_causal_output : np.ndarray of shape (size_a, D)
        Parent table CAUSAL_OUTPUT features for rank-score computation.
    propensity_rho : float
        Mixing weight; t(j) = rho*z(j) + sqrt(1-rho^2)*eps(j).
    propensity_beta : float
        Temperature for exp(-beta * |rank_score[a] - t(j)|).
    seed : int or None
        RNG seed.
    null_prob : float
        Probability a child row gets FK=-1 (no parent connection).

    Returns
    -------
    fk_ids : np.ndarray of shape (size_b,)
    cluster_b : np.ndarray of shape (size_b, len(hierarchy_b))
    cluster_a : np.ndarray of shape (size_a, len(hierarchy_a))
    """
    assert len(hierarchy_a) == len(hierarchy_b), (
        "only equal-length hierarchies are supported"
    )
    rng = np.random.RandomState(seed)

    # 1. Rank score from parent CAUSAL_OUTPUT
    D = parent_causal_output.shape[1]
    if D >= 2:
        n_cols = rng.randint(2, min(4, D + 1))
        selected = rng.choice(D, size=n_cols, replace=False)
        weights = rng.randn(n_cols)
        raw_score = parent_causal_output[:, selected] @ weights
    elif D == 1:
        raw_score = parent_causal_output[:, 0]
    else:
        raw_score = rng.randn(size_a)
    order = np.argsort(raw_score)
    rank_score = np.empty(size_a, dtype=np.float64)
    for i, idx in enumerate(order):
        rank_score[idx] = (i + 1) / size_a

    # 2. Cluster assignments
    cluster_a = assign_cluster_at_levels(num_nodes=size_a, hierarchy=hierarchy_a)
    cluster_b = assign_cluster_at_levels(num_nodes=size_b, hierarchy=hierarchy_b)
    probs_at_levels = _get_probs_at_levels(hierarchy_a, hierarchy_b, rng=rng)

    # 3. Child target values
    z = rng.uniform(0, 1, size=size_b)
    eps = rng.uniform(0, 1, size=size_b)
    t = propensity_rho * z + np.sqrt(max(1 - propensity_rho**2, 0)) * eps

    # 4. Sample with propensity bias within HSBM blocks
    num_levels = len(probs_at_levels)
    fk_ids = np.empty(size_b, dtype=np.int64)

    max_clusters = cluster_b.max(axis=0) + 1
    multipliers = np.cumprod([1] + list(max_clusters[:-1]))
    cluster_keys = (cluster_b * multipliers).sum(axis=1)
    unique_keys = np.unique(cluster_keys)

    for key in unique_keys:
        mask = cluster_keys == key
        indices = np.nonzero(mask)[0]
        cluster_path = cluster_b[indices[0]]

        base_probs = np.ones(size_a, dtype=np.float64)
        for l in range(num_levels):
            base_probs *= probs_at_levels[l][cluster_a[:, l], cluster_path[l]]
        p_sum = base_probs.sum()
        if p_sum > 0:
            base_probs /= p_sum
        else:
            base_probs = np.ones(size_a) / size_a

        for idx in indices:
            t_j = t[idx]
            distances = np.abs(rank_score - t_j)
            w = base_probs * np.exp(-propensity_beta * distances)
            w_sum = w.sum()
            if w_sum > 0:
                w /= w_sum
            else:
                w = None
            fk_ids[idx] = rng.choice(size_a, p=w)

    if null_prob > 0:
        null_mask = rng.random(size_b) < null_prob
        fk_ids[null_mask] = -1

    return fk_ids, cluster_b, cluster_a


def compute_hsbm_fk_ids_multi(
    parent_sizes: list[int],
    child_size: int,
    hierarchies_parent: list[list[int]],
    hierarchy_child: list[int],
    seed: int | None = None,
    null_probs: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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
    null_probs : list of float or None
        Per-parent null probability. Child rows with FK=-1 have no connection
        to that parent. Length must match ``num_parents``.
    Returns
    -------
    fk_ids : np.ndarray of shape ``(child_size, num_parents)``
        ``fk_ids[j, p]`` is the parent-row index (0-based) that child row ``j``
        connects to for parent ``p``, or -1 if no connection.
    cluster_b : np.ndarray of shape ``(child_size, len(hierarchy_child))``
        Shared child cluster path used for all parents.
    cluster_per_parent : list of np.ndarray
        Per-parent parent-side cluster assignments, each of shape
        ``(parent_sizes[p], len(hierarchies_parent[p]))``.
    """
    rng = np.random.RandomState(seed)
    num_parents = len(parent_sizes)
    num_levels = len(hierarchy_child)  # max depth across all parents

    if null_probs is None:
        null_probs = [0.0] * num_parents
    assert len(null_probs) == num_parents, (
        f"null_probs length {len(null_probs)} != num_parents {num_parents}"
    )

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
            null_prob=null_probs[p],
        )

    return fk_ids, cluster_b, cluster_per_parent


def compute_hsbm_fk_ids_multi_with_matching(
    parent_sizes: list[int],
    child_size: int,
    hierarchies_parent: list[list[int]],
    hierarchy_child: list[int],
    parent_causal_outputs: list[np.ndarray],
    matching_latent_dim: int,
    matching_temperature: float,
    seed: int | None = None,
    null_probs: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Joint FK sampling with shared matching latent across parents.

    All parent CAUSAL_OUTPUT features are per-column z-score normalized, then
    projected through a shared random matrix W into a common low-D latent
    space. Each child row samples a target latent, and within each HSBM block,
    FK sampling is biased via softmax(-||latent - target||^2 / temperature).

    Parameters
    ----------
    parent_sizes : list of int
        Number of rows per parent table.
    child_size : int
        Number of child rows.
    hierarchies_parent : list of list of int
        Per-parent hierarchy cluster counts.
    hierarchy_child : list of int
        Child-side hierarchy (max depth).
    parent_causal_outputs : list of np.ndarray
        List of (N_p, D_p) parent CAUSAL_OUTPUT arrays.
    matching_latent_dim : int
        Shared latent space dimension D_latent.
    matching_temperature : float
        Softmax temperature (smaller = stronger bias).
    seed : int or None
        RNG seed.
    null_probs : list of float or None
        Per-parent null probability. FK=-1 for null rows.

    Returns
    -------
    fk_ids : np.ndarray of shape (child_size, num_parents)
    cluster_b : np.ndarray of shape (child_size, len(hierarchy_child))
    cluster_per_parent : list of np.ndarray
        Per-parent parent-side cluster assignments.
    """
    rng = np.random.RandomState(seed)
    num_parents = len(parent_sizes)
    num_levels = len(hierarchy_child)

    if null_probs is None:
        null_probs = [0.0] * num_parents
    assert len(null_probs) == num_parents, (
        f"null_probs length {len(null_probs)} != num_parents {num_parents}"
    )

    # 1. Per-table z-score normalization (per-column)
    X_norm_list: list[np.ndarray] = []
    for p in range(num_parents):
        X = parent_causal_outputs[p].astype(np.float64)
        mean = X.mean(axis=0, keepdims=True)
        std = X.std(axis=0, keepdims=True)
        std[std < 1e-10] = 1.0
        X_norm_list.append((X - mean) / std)

    # 2. Shared projection matrix W: (D_common, D_latent)
    D_common = min(X.shape[1] for X in parent_causal_outputs)
    D_latent = min(matching_latent_dim, D_common)
    W = rng.randn(D_common, D_latent).astype(np.float64) / np.sqrt(D_common)

    # 3. Per-parent cluster assignments + probability matrices + matching latents
    cluster_per_parent = []
    probs_per_parent = []
    matching_latents = []
    for p in range(num_parents):
        ca = assign_cluster_at_levels(
            num_nodes=parent_sizes[p], hierarchy=hierarchies_parent[p],
        )
        cluster_per_parent.append(ca)
        nlv_p = len(hierarchies_parent[p])
        probs = _get_probs_at_levels(
            hierarchies_parent[p], hierarchy_child[:nlv_p], rng=rng,
        )
        probs_per_parent.append(probs)
        ml = X_norm_list[p][:, :D_common] @ W  # (N_p, D_latent)
        matching_latents.append(ml)

    # 4. Shared child cluster path (random) + target latents
    cluster_b = np.empty((child_size, num_levels), dtype=int)
    for l in range(num_levels):
        cluster_b[:, l] = rng.randint(0, hierarchy_child[l], size=child_size)
    target = rng.randn(child_size, D_latent).astype(np.float64)

    # 5. Per-parent FK sampling with matching bias
    fk_ids = np.empty((child_size, num_parents), dtype=np.int64)

    for p in range(num_parents):
        nlv_p = len(hierarchies_parent[p])
        size_a = parent_sizes[p]
        cluster_a_p = cluster_per_parent[p]
        probs_p = probs_per_parent[p]
        ml_p = matching_latents[p]

        # Batch by child cluster-path prefix
        cluster_b_prefix = cluster_b[:, :nlv_p]
        unique_paths: dict[tuple, list[int]] = {}
        for j in range(child_size):
            key = tuple(int(x) for x in cluster_b_prefix[j])
            unique_paths.setdefault(key, []).append(j)

        for cp, child_indices in unique_paths.items():
            # Base probs from HSBM block structure
            base_probs = np.ones(size_a, dtype=np.float64)
            for l in range(nlv_p):
                base_probs *= probs_p[l][cluster_a_p[:, l], cp[l]]
            threshold = 0.01
            candidates = np.nonzero(base_probs > threshold)[0]

            if len(candidates) == 0:
                fk_ids[child_indices, p] = rng.randint(
                    0, size_a, size=len(child_indices),
                )
                continue

            candidate_latents = ml_p[candidates]  # (n_c, D_latent)

            for idx in child_indices:
                t_j = target[idx]
                diffs = candidate_latents - t_j
                sq_dists = (diffs * diffs).sum(axis=1)
                scores = np.exp(-sq_dists / max(matching_temperature, 1e-8))
                s_sum = scores.sum()
                if s_sum > 0:
                    scores /= s_sum
                else:
                    scores = None
                fk_ids[idx, p] = candidates[rng.choice(len(candidates), p=scores)]

    for p in range(num_parents):
        if null_probs[p] > 0:
            null_mask = rng.random(child_size) < null_probs[p]
            fk_ids[null_mask, p] = -1

    return fk_ids, cluster_b, cluster_per_parent
