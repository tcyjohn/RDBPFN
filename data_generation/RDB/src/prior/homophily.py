"""Homophily-controlled label generation for relational ICL diversity.

Based on OPENRFM (arXiv:2606.04320) Appendix G.

Generates per-RDB labels whose FK-structure dependence is controlled by a scalar
homophily target h, making label-FK dependence a support-identifiable latent.
"""

from __future__ import annotations

import torch
import numpy as np


def _build_pseudo_blocks(
    features: torch.Tensor,
    n_parent: int = 2,
    n_child: int = 4,
    seed: int = 0,
) -> torch.Tensor:
    """Build pseudo-block hierarchy from feature clustering.

    Used when the target table has no FK parents (ultimate root entity).

    Args:
        features: (n_rows, n_feats) — feature columns from the target table.
        n_parent: Number of parent-level blocks.
        n_child: Number of child-level blocks (must be >= n_parent).
        seed: RNG seed.

    Returns:
        block_paths: (n_rows, 2) — [b_parent, b_child] per row.
    """
    rng = np.random.RandomState(seed)
    n_feats = features.shape[1]
    n_pool = min(5, n_feats)
    col_idx = rng.choice(n_feats, size=n_pool, replace=False)
    pool = features[:, col_idx].float()

    # Random projection → scalar per row
    proj_weights = torch.from_numpy(rng.randn(n_pool).astype(np.float32))
    proj = pool @ proj_weights  # (n_rows,)

    # Quantile-based block assignment
    parent_quantiles = torch.linspace(0, 1, n_parent + 1)[1:-1]
    parent_thresholds = torch.quantile(proj, parent_quantiles)
    b_parent = torch.bucketize(proj, parent_thresholds)  # (n_rows,)

    child_quantiles = torch.linspace(0, 1, n_child + 1)[1:-1]
    child_thresholds = torch.quantile(proj, child_quantiles)
    b_child = torch.bucketize(proj, child_thresholds)  # (n_rows,)

    return torch.stack([b_parent, b_child], dim=-1)


def _build_hsbm_blocks(
    block_paths: torch.Tensor,
) -> torch.Tensor:
    """Extract 2-level block assignments from HSBM block paths.

    Args:
        block_paths: (n_rows, n_levels) — HSBM cluster assignments per row.

    Returns:
        (n_rows, 2) — [b_parent, b_child] per row.
    """
    n_levels = block_paths.shape[1]
    b_parent = block_paths[:, 0]
    if n_levels >= 2:
        b_child = block_paths[:, 1]
    else:
        b_child = (b_parent * 2) + (b_parent % 2)  # deterministic expansion
    return torch.stack([b_parent, b_child], dim=-1)


def get_block_assignments(
    target_table_name: str,
    rdb,
    hsbm_block_paths: dict[str, torch.Tensor] | None = None,
    hsbm_parent_blocks: dict[str, torch.Tensor] | None = None,
    seed: int = 0,
) -> torch.Tensor | None:
    """Get 2-level block assignments for homophily label generation.

    Priority:
    1. hsbm_block_paths[target_table] (cluster_b, entity as child)
    2. hsbm_parent_blocks[target_table] (cluster_a, entity as parent/root)
    3. pseudo-blocks from feature clustering (last resort)

    For entity tables with multi-row temporal snapshots, block assignment is
    per-entity (all snapshots of the same entity share the same block),
    broadcast to rows.

    Args:
        target_table_name: Name of the target table.
        rdb: The RDB object.
        hsbm_block_paths: Dict mapping table_name → block_paths tensor (cluster_b).
        hsbm_parent_blocks: Dict mapping table_name → parent_blocks tensor (cluster_a).
        seed: RNG seed for pseudo-block generation.

    Returns:
        (n_rows, 2) tensor [b_parent, b_child], or None if the table has no rows.
    """
    table = rdb.tables[target_table_name]
    n_rows = table.num_rows

    # Check entity metadata for per-entity broadcast
    gen = rdb.table_generators.get(target_table_name)
    is_entity = gen is not None and gen.is_entity_table
    num_entities = gen.num_entities if is_entity else n_rows
    snapshots = gen.snapshots_per_entity if is_entity else 1

    # Priority 1: hsbm_block_paths (cluster_b, child-side)
    if hsbm_block_paths and target_table_name in hsbm_block_paths:
        paths = hsbm_block_paths[target_table_name]
        result = _build_hsbm_blocks(paths)
        if is_entity and snapshots > 1:
            # cluster_b is per-row; subsample to entity-level then broadcast
            entity_blocks = result[::snapshots][:num_entities]
            result = entity_blocks.repeat_interleave(snapshots, dim=0)[:n_rows]
        return result

    # Priority 2: hsbm_parent_blocks (cluster_a, parent-side — root entities)
    if hsbm_parent_blocks and target_table_name in hsbm_parent_blocks:
        blocks = hsbm_parent_blocks[target_table_name]
        result = _build_hsbm_blocks(blocks)
        if is_entity and snapshots > 1:
            # cluster_a is per-row; subsample to entity-level then broadcast
            entity_blocks = result[::snapshots][:num_entities]
            result = entity_blocks.repeat_interleave(snapshots, dim=0)[:n_rows]
        return result

    # Priority 3: pseudo-blocks from feature clustering
    feature_cols = table.get_feature_columns(only_categorical=False)
    if not feature_cols:
        return None
    col_idx = [table.column_names.index(c) for c in feature_cols]
    features = table.data[:, col_idx]
    if is_entity and snapshots > 1:
        # Cluster at entity level by picking first snapshot per entity
        entity_features = features[::snapshots][:num_entities]
        entity_blocks = _build_pseudo_blocks(entity_features, seed=seed)
        return entity_blocks.repeat_interleave(snapshots, dim=0)[:n_rows]
    return _build_pseudo_blocks(features, seed=seed)


class HomophilyLabelGenerator:
    """Generate binary labels with controlled FK-structure dependence.

    Per-RDB homophily target h ∈ [-1,+1] controls the mixing between:
    - y_cluster: FK-block-structure-driven label
    - y_feature: feature-driven label (random projection)

    |h| → 0: label ≈ y_feature (FK structure irrelevant)
    |h| → 1: label ≈ y_cluster (FK structure dominant)
    h > 0: homophily (same parent block → same label)
    h < 0: heterophily (same parent, different child block → opposite label)

    Based on OPENRFM Appendix G.1, Eq. (22).
    """

    def __init__(
        self,
        h_target: float,
        block_assignments: torch.Tensor,
        features: torch.Tensor,
        seed: int = 0,
    ):
        assert -1.0 <= h_target <= 1.0
        assert h_target != 0, "h_target must be non-zero for well-defined y_cluster"
        assert block_assignments.shape == (features.shape[0], 2), (
            f"Expected ({features.shape[0]}, 2), got {block_assignments.shape}"
        )

        self.h_target = h_target
        self.b_parent = block_assignments[:, 0]
        self.b_child = block_assignments[:, 1]
        self.features = features
        self.rng = np.random.RandomState(seed)

    def _generate_y_cluster(self) -> torch.Tensor:
        """Generate cluster-driven binary labels from block assignments."""
        if self.h_target > 0:
            # Homophily: same parent block → same label
            y = self.b_parent % 2
        else:
            # Heterophily: XOR with child block flips labels within parent
            y = (self.b_parent % 2) ^ (self.b_child % 2)
        return y.int()

    def _generate_y_feature(self) -> torch.Tensor:
        """Generate feature-driven binary labels via random projection."""
        n_feats = self.features.shape[1]
        n_pool = min(5, n_feats)
        cols = self.rng.choice(n_feats, size=n_pool, replace=False)
        weights = torch.from_numpy(self.rng.randn(n_pool).astype(np.float32))
        logits = self.features[:, cols].float() @ weights
        return (torch.sigmoid(logits) > 0.5).int()

    def generate_label(self) -> tuple[torch.Tensor, dict]:
        """Generate binary labels with controlled FK-structure dependence.

        Returns:
            y: (n_rows,) int tensor with values in {0, 1}.
            meta: dict with diagnostic statistics.
        """
        y_cluster = self._generate_y_cluster()
        y_feature = self._generate_y_feature()

        abs_h = abs(self.h_target)
        # Per-row Bernoulli: pick y_cluster with prob |h|
        mask = torch.from_numpy(self.rng.random(len(y_cluster)) < abs_h)
        y = torch.where(mask, y_cluster, y_feature)

        # Diagnostic: actual homophily (fraction of FK-sibling pairs sharing label)
        actual_homophily = self._compute_actual_homophily(y)

        meta = {
            "h_target": self.h_target,
            "abs_h": abs_h,
            "cluster_ratio": float(mask.float().mean()),
            "actual_homophily": actual_homophily,
            "y_cluster_mean": float(y_cluster.float().mean()),
            "y_feature_mean": float(y_feature.float().mean()),
        }
        return y, meta

    def _compute_actual_homophily(self, y: torch.Tensor) -> float:
        """Compute realised homophily: fraction of same-parent-block pairs sharing label."""
        b = self.b_parent.numpy()
        y_np = y.numpy()
        n = len(y_np)
        if n < 2:
            return 0.0
        # Sample up to 5000 pairs to avoid O(n²)
        n_pairs = min(5000, n * (n - 1) // 2)
        rng = np.random.RandomState(42)
        idx = rng.choice(n, size=(n_pairs, 2), replace=True)
        same_block = b[idx[:, 0]] == b[idx[:, 1]]
        same_label = y_np[idx[:, 0]] == y_np[idx[:, 1]]
        if same_block.sum() == 0:
            return 0.0
        return float(same_label[same_block].mean())
