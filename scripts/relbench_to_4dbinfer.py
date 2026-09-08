"""Convert raw RelBench dataset (relbench_cache/) to 4DBInfer format.

The 4DBInfer format is what tab2graph DFS preprocessing expects:
  - metadata.yaml (schema: tables, columns, relationships)
  - One .parquet file per table
  - Optionally, task CSV data

Usage:
    pixi run python scripts/relbench_to_4dbinfer.py \
        --input relbench_cache/rel-f1 \
        --output model_pretrain/rdb_datasets/rel-f1
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _infer_dtype(series: pd.Series) -> str:
    if pd.api.types.is_integer_dtype(series):
        # Check if it looks like a foreign key (suffixed with "Id" or small cardinality)
        return "category"
    elif pd.api.types.is_float_dtype(series):
        return "float"
    elif pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    else:
        return "category"


def _make_4dbinfer_metadata(
    db_dir: Path,
    fk_map: dict[str, dict[str, tuple[str, str]]],  # {table: {col: (parent_table, parent_col)}}
    time_col_map: dict[str, str | None],
    pkey_map: dict[str, str | None],
) -> dict:
    tables_meta = []
    for parquet_path in sorted(db_dir.glob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        table_name = parquet_path.stem
        columns = []

        for col_name in df.columns:
            dtype = _infer_dtype(df[col_name])

            col_meta: dict = {"dtype": dtype, "name": col_name}

            # Mark primary key
            if pkey_map.get(table_name) == col_name:
                col_meta["dtype"] = "primary_key"
                col_meta["capacity"] = int(df[col_name].max()) + 1000

            # Mark foreign key with link_to
            if table_name in fk_map and col_name in fk_map[table_name]:
                parent_table, parent_col = fk_map[table_name][col_name]
                col_meta["dtype"] = "foreign_key"
                col_meta["link_to"] = f"{parent_table}.{parent_col}"

            if col_meta["dtype"] == "category":
                col_meta["num_categories"] = int(df[col_name].nunique())

            columns.append(col_meta)

        table_meta: dict = {
            "name": table_name,
            "source": parquet_path.name,
            "format": "parquet",
            "columns": columns,
        }
        if time_col_map.get(table_name):
            table_meta["time_column"] = time_col_map[table_name]

        tables_meta.append(table_meta)

    return {
        "dataset_name": db_dir.parent.name,
        "tables": tables_meta,
        "tasks": [],
        "column_groups": None,
    }


def _infer_fk_pkey(db_dir: Path) -> tuple[dict, dict, dict]:
    """Infer foreign keys, primary keys, and time columns from column naming conventions."""
    # fk_map: {table: {col: (parent_table, parent_col)}}
    fk_map: dict[str, dict[str, tuple[str, str]]] = {}
    pkey_map: dict[str, str | None] = {}
    time_map: dict[str, str | None] = {}

    all_tables = {}
    for parquet_path in sorted(db_dir.glob("*.parquet")):
        df = pd.read_parquet(parquet_path)
        all_tables[parquet_path.stem] = df

    for table_name, df in all_tables.items():
        # Guess PK
        singular = table_name.rstrip("s")
        pk_candidates = [
            c for c in df.columns
            if c.lower() in (f"{table_name}id", f"{singular}id", f"{table_name}_id", f"{singular}_id", "id")
        ]
        if not pk_candidates:
            pk_candidates = [
                c for c in df.columns
                if c.lower().endswith("id") and df[c].nunique() == len(df)
            ]
        pkey_map[table_name] = pk_candidates[0] if pk_candidates else None

        # Guess FKs: columns matching another table's PK name
        fks = {}
        for col in df.columns:
            if col == pkey_map[table_name]:
                continue
            for other_table, other_df in all_tables.items():
                if other_table == table_name:
                    continue
                other_pk = pkey_map.get(other_table)
                if other_pk is None:
                    continue
                # FK column name should match other_table's PK column name
                if col.lower() == other_pk.lower():
                    fks[col] = (other_table, other_pk)
                    break
        fk_map[table_name] = fks

        # Guess time column
        time_candidates = [c for c in df.columns if c.lower() in ("date", "timestamp", "time", "datetime")]
        time_map[table_name] = time_candidates[0] if time_candidates else None

    return fk_map, pkey_map, time_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="RelBench dataset directory (db/ + tasks/)")
    parser.add_argument("--output", required=True, type=Path, help="Output 4DBInfer directory")
    args = parser.parse_args()

    db_dir = args.input / "db"
    if not db_dir.exists():
        raise FileNotFoundError(f"{db_dir} not found")

    args.output.mkdir(parents=True, exist_ok=True)

    # Copy parquet files
    import shutil
    for pq in db_dir.glob("*.parquet"):
        shutil.copy2(pq, args.output / pq.name)
    logger.info("Copied %d parquet files", len(list(db_dir.glob("*.parquet"))))

    # Infer schema
    fk_map, pkey_map, time_map = _infer_fk_pkey(db_dir)
    logger.info("Inferred: %d tables, FKs across %d tables", len(pkey_map), sum(1 for v in fk_map.values() if v))

    # Write metadata.yaml
    metadata = _make_4dbinfer_metadata(db_dir, fk_map, time_map, pkey_map)
    with open(args.output / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    logger.info("Wrote metadata.yaml")

    # Copy task CSV files if they exist
    tasks_dir = args.input / "tasks"
    if tasks_dir.exists():
        csv_dir = args.output / "csv_data"
        csv_dir.mkdir(exist_ok=True)
        for task_dir in sorted(tasks_dir.iterdir()):
            if task_dir.is_dir():
                for split_file in task_dir.glob("*.parquet"):
                    df = pd.read_parquet(split_file)
                    csv_name = f"{task_dir.name}_{split_file.stem}.csv"
                    df.to_csv(csv_dir / csv_name, index=False)
        logger.info("Copied %d task CSVs", len(list(csv_dir.glob("*.csv"))))

    logger.info("Done: %s", args.output)


if __name__ == "__main__":
    main()
