from __future__ import annotations

import numpy as np

from src.eval_aligned import _downsample_aligned
from src.eval_utils import downsample_split


def test_downsample_aligned_preserves_row_identity() -> None:
    row_ids = np.arange(20)
    X = row_ids[:, None]
    y = row_ids + 100
    fk_values = row_ids + 200
    entity_ids = row_ids + 300
    parent_entity_ids = row_ids + 400

    result = _downsample_aligned(
        X,
        y,
        max_samples=7,
        seed=123,
        aligned_arrays=(fk_values, entity_ids, parent_entity_ids),
    )

    selected_ids = result.X[:, 0]
    np.testing.assert_array_equal(result.y, selected_ids + 100)
    np.testing.assert_array_equal(result.aligned_arrays[0], selected_ids + 200)
    np.testing.assert_array_equal(result.aligned_arrays[1], selected_ids + 300)
    np.testing.assert_array_equal(result.aligned_arrays[2], selected_ids + 400)


def test_downsample_aligned_preserves_none_auxiliary_arrays() -> None:
    X = np.arange(8)[:, None]
    y = np.arange(8)

    result = _downsample_aligned(
        X,
        y,
        max_samples=4,
        seed=7,
        aligned_arrays=(None,),
    )

    assert result.X.shape == (4, 1)
    assert result.y.shape == (4,)
    assert result.aligned_arrays == (None,)


def test_downsample_aligned_keeps_existing_xy_sampling_protocol() -> None:
    X = np.arange(60).reshape(20, 3)
    y = np.arange(20)

    expected_X, expected_y = downsample_split(X, y, max_samples=9, seed=42)
    result = _downsample_aligned(X, y, 9, 42, aligned_arrays=())

    np.testing.assert_array_equal(result.X, expected_X)
    np.testing.assert_array_equal(result.y, expected_y)


if __name__ == "__main__":
    test_downsample_aligned_preserves_row_identity()
    test_downsample_aligned_preserves_none_auxiliary_arrays()
    test_downsample_aligned_keeps_existing_xy_sampling_protocol()
