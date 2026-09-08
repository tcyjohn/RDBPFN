from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.eval_utils import (
    append_per_seed_results_to_csv,
    derive_per_seed_output_path,
    save_per_seed_results_to_csv,
)


def _result(seed: int, metric_value: float) -> dict:
    return {
        "dataset": "demo",
        "task": "classification",
        "seed": seed,
        "metric": "roc_auc",
        "metric_value": metric_value,
        "accuracy": 0.75,
        "balanced_acc": 0.70,
    }


def test_per_seed_results_are_saved_in_long_format() -> None:
    with TemporaryDirectory() as tmp_dir:
        aggregate_path = Path(tmp_dir) / "checkpoint.csv"
        detail_path = derive_per_seed_output_path(aggregate_path)
        save_per_seed_results_to_csv(
            {"checkpoints/run/model.pt": [_result(1, 0.8), _result(0, 0.7)]},
            detail_path,
        )

        saved = pd.read_csv(detail_path)
        assert detail_path.name == "checkpoint_per_seed.csv"
        assert saved["seed"].tolist() == [0, 1]
        assert saved["metric_value"].tolist() == [0.7, 0.8]


def test_append_replaces_matching_model_task_seed() -> None:
    with TemporaryDirectory() as tmp_dir:
        detail_path = Path(tmp_dir) / "results_per_seed.csv"
        save_per_seed_results_to_csv({"model-a": [_result(0, 0.5)]}, detail_path)
        append_per_seed_results_to_csv({"model-a": [_result(0, 0.9)]}, detail_path)

        saved = pd.read_csv(detail_path)
        assert len(saved) == 1
        assert saved.loc[0, "metric_value"] == 0.9


if __name__ == "__main__":
    test_per_seed_results_are_saved_in_long_format()
    test_append_replaces_matching_model_task_seed()
