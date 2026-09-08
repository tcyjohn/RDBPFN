"""Print entity density stats for each eval task."""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.dbinfer_bench_simplified.rdb_dataset import DBBRDBDataset
from src.dbinfer_bench_simplified.dataset_meta import DBBTaskType, DBBColumnDType


def main():
    dataset_dir = Path("rdb_datasets")
    for dpath in sorted(dataset_dir.iterdir()):
        if not dpath.is_dir():
            continue
        try:
            ds = DBBRDBDataset(dpath)
        except Exception:
            continue
        for task in ds.tasks:
            if task.metadata.task_type != DBBTaskType.classification:
                continue
            source = task.train_set
            if source is None:
                continue

            n_rows = len(next(iter(source.values())))

            # Entity IDs
            eid = None
            if "entity_id" in source:
                eid_raw = source["entity_id"].astype(np.float64)
                eid = np.nan_to_num(eid_raw, nan=-1).astype(np.int64)
            else:
                pk_cols = [col.name for col in task.metadata.columns
                           if col.dtype == DBBColumnDType.primary_key]
                if pk_cols and pk_cols[0] in source:
                    pk_raw = source[pk_cols[0]].astype(np.float64)
                    eid = np.nan_to_num(pk_raw, nan=-1).astype(np.int64)

            if eid is not None:
                valid = eid[eid >= 0]
                n_unique = len(np.unique(valid))
                n_valid = len(valid)
                entity_cov = n_valid / n_rows if n_rows > 0 else 0
                # If all rows are valid, entity pair ratio upper bound ≈ 1/n_unique
                pair_ratio_bound = 1.0 / n_unique if n_unique > 0 else 0
            else:
                n_unique = 0
                entity_cov = 0
                pair_ratio_bound = 0

            # FK columns
            fk_cols = [col.name for col in task.metadata.columns
                       if col.dtype == DBBColumnDType.foreign_key]
            n_fk_cols = len(fk_cols)

            # Entity snapshots: average rows per entity
            if eid is not None and n_unique > 0:
                _, counts = np.unique(valid, return_counts=True)
                avg_snap = counts.mean()
                max_snap = counts.max()
                min_snap = counts.min()
            else:
                avg_snap = max_snap = min_snap = 0

            print(f"{ds.dataset_name:<30s} {task.metadata.name:<30s} "
                  f"rows={n_rows:>6d} entities={n_unique:>5d} "
                  f"s/min={min_snap:>3.0f} s/avg={avg_snap:>5.1f} s/max={max_snap:>3.0f} "
                  f"eid_cov={entity_cov:.3f} pair_ub={pair_ratio_bound:.4f} "
                  f"fk_cols={n_fk_cols}")


if __name__ == "__main__":
    main()
