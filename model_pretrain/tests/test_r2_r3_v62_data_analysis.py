from __future__ import annotations

import numpy as np

from scripts.data_analysis.r2_r3_v62_analysis import (
    _schema_topology,
    effective_parent_coverage,
    exact_sign_flip_pvalue,
    holm_adjust,
    multi_parent_nmi,
    normalized_effective_rank,
    stable_seed,
)


def test_normalized_effective_rank_extremes() -> None:
    rng = np.random.default_rng(7)
    independent = rng.normal(size=(20_000, 4))
    latent = rng.normal(size=20_000)
    redundant = np.column_stack([latent, latent, latent, latent])

    assert normalized_effective_rank(independent) > 0.98
    assert np.isclose(normalized_effective_rank(redundant), 0.25, atol=1e-6)


def test_effective_parent_coverage_uniform_and_concentrated() -> None:
    uniform = np.repeat(np.arange(4), 10)
    concentrated = np.zeros(40)

    assert np.isclose(effective_parent_coverage(uniform, 4), 1.0)
    assert np.isclose(effective_parent_coverage(concentrated, 4), 0.25)


def test_multi_parent_nmi_detects_dependence_and_rejects_degenerate() -> None:
    left = np.tile(np.arange(4), 20)
    assert np.isclose(multi_parent_nmi(left, left), 1.0)
    assert np.isnan(multi_parent_nmi(np.zeros(20), np.zeros(20)))


def test_stable_seed_is_repeatable_and_keyed() -> None:
    assert stable_seed("same") == stable_seed("same")
    assert stable_seed("same") != stable_seed("different")


def test_exact_sign_flip_and_holm_adjustment() -> None:
    assert np.isclose(exact_sign_flip_pvalue(np.ones(4)), 0.125)
    adjusted = holm_adjust([0.01, 0.03, 0.2])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.2])


def test_schema_topology_counts_unique_edges_and_depth() -> None:
    metadata = {
        "tables": [
            {"name": "root", "columns": []},
            {
                "name": "middle",
                "columns": [
                    {"dtype": "foreign_key", "link_to": "root.id"},
                ],
            },
            {
                "name": "leaf",
                "columns": [
                    {"dtype": "foreign_key", "link_to": "middle.id"},
                    {"dtype": "foreign_key", "link_to": "root.id"},
                ],
            },
        ]
    }
    assert _schema_topology(metadata) == (3, 3, 2)
