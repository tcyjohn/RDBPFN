"""Print task type and target column for each eval task."""
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
            n_rows = len(next(iter(source.values()))) if source else 0

            # Target info
            target = task.metadata.target_column

            # Task type
            task_type = str(task.metadata.task_type)

            # Get entity_id stats
            if "entity_id" in source:
                eid = source["entity_id"].astype(np.float64)
                eid = np.nan_to_num(eid, nan=-1).astype(np.int64)
                valid = eid[eid >= 0]
                n_entity = len(np.unique(valid))
                # Check if label is constant within each entity
                y = np.asarray(source[target])
                y_valid = y[eid >= 0]
                eid_valid = valid
                # For each entity, how many distinct labels?
                entity_labels = {}
                for i in range(len(eid_valid)):
                    e = eid_valid[i]
                    entity_labels.setdefault(e, set()).add(y_valid[i])
                const_label_ratio = sum(1 for v in entity_labels.values() if len(v) == 1) / len(entity_labels) if entity_labels else 0
            else:
                n_entity = 0
                const_label_ratio = -1

            print(f"{ds.dataset_name:<30s} {task.metadata.name:<30s} "
                  f"target={target:<20s} type={task_type:<20s} "
                  f"rows={n_rows:>7d} entities={n_entity:>6d} "
                  f"const_label={const_label_ratio:.3f}")

            # Print available column names for context
            cols = [c.name for c in task.metadata.columns[:10]]
            print(f"  cols: {cols}")


if __name__ == "__main__":
    main()
