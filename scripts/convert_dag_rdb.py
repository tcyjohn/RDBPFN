"""
convert_dag_rdb.py（无 dbinfer_bench 依赖版本）
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml

from relbench.base import Database, Table


# ─────────────────────────────────────────────
#  路径配置
# ─────────────────────────────────────────────
_DEFAULT_SRC_ROOT = Path("/data/caijunyu/RDBPFN/data_generation/RDB_datasets/plurel_scale_complex_raw")
_DEFAULT_DST_ROOT = Path(os.environ["HOME"]) / "scratch" / "relbench"
_DEFAULT_DST_PREFIX = "dag_rdb_complex_"
_DEFAULT_NUM_DATASETS = 1024


# ─────────────────────────────────────────────
#  读单张表（npz 或 parquet）
# ─────────────────────────────────────────────
def load_table_file(path: Path, fmt: str = "numpy") -> Dict[str, np.ndarray]:
    use_parquet = (fmt == "parquet") or (path.suffix == ".parquet")
    if use_parquet:
        df = pd.read_parquet(path)
        return {col: df[col].values for col in df.columns}
    else:
        npz = np.load(str(path), allow_pickle=True)
        return {name: npz[name] for name in npz.files}


# ─────────────────────────────────────────────
#  dtype 转换：numpy array → pandas Series
# ─────────────────────────────────────────────
def cast_column(arr: np.ndarray, dtype_str: str, col_name: str) -> pd.Series:
    s = pd.Series(arr, name=col_name)
    if dtype_str == "float":
        return s.astype("float32")
    elif dtype_str == "datetime":
        if np.issubdtype(arr.dtype, np.integer):
            return pd.to_datetime(s, unit="s")
        return pd.to_datetime(s)
    elif dtype_str == "timestamp":
        return s.astype("int64")
    elif dtype_str in ("primary_key", "foreign_key"):
        try:
            return pd.array(s, dtype=pd.Int64Dtype())
        except Exception:
            return s
    else:  # category, text
        return s.astype(object)


# ─────────────────────────────────────────────
#  读 metadata.yaml，构建 RelBench Database
# ─────────────────────────────────────────────
def load_dbb_as_relbench_db(src_path: Path) -> tuple[Database, dict]:
    """
    返回 (Database, raw_meta_dict)
    raw_meta_dict 保留原始 metadata，供 task 落盘时使用
    """
    with open(src_path / "metadata.yaml") as f:
        meta = yaml.safe_load(f)

    table_dict: Dict[str, Table] = {}

    for tbl_schema in meta["tables"]:
        tbl_name = tbl_schema["name"]
        src_file = src_path / tbl_schema["source"]
        fmt      = tbl_schema["format"]  # "numpy" or "parquet"

        # 解析列 schema
        col_dtype_map = {col["name"]: col["dtype"] for col in tbl_schema["columns"]}
        
        # 解析 pkey / fkey / time_col
        pkey_col = next(
            (col["name"] for col in tbl_schema["columns"] if col["dtype"] == "primary_key"),
            None
        )
        fkey_col_to_pkey_table = {
            col["name"]: col["link_to"].split(".")[0]
            for col in tbl_schema["columns"]
            if col["dtype"] == "foreign_key"
        }
        time_col = tbl_schema.get("time_column")

        # 读文件
        raw = load_table_file(src_file, fmt)
        df  = pd.DataFrame({
            col_name: cast_column(arr, col_dtype_map.get(col_name, "text"), col_name)
            for col_name, arr in raw.items()
        })

        table_dict[tbl_name] = Table(
            df=df,
            fkey_col_to_pkey_table=fkey_col_to_pkey_table,
            pkey_col=pkey_col,
            time_col=time_col,
        )

    return Database(table_dict), meta


# ─────────────────────────────────────────────
#  落盘：db 表
# ─────────────────────────────────────────────
def save_db_tables(db: Database, dst_db_dir: Path):
    dst_db_dir.mkdir(parents=True, exist_ok=True)
    for tbl_name, tbl in db.table_dict.items():
        out = dst_db_dir / f"{tbl_name}.parquet"
        tbl.save(out)
        print(f"  [db] {tbl_name}: {len(tbl.df)} rows -> {out}")


# ─────────────────────────────────────────────
#  落盘：task 表
# ─────────────────────────────────────────────
def save_task_tables(src_path: Path, meta: dict, dst_tasks_dir: Path, pkey_maps: dict[str, dict]):
    """Write task splits as narrow parquets (RelBench ``EntityTask`` shape).

    Columns: entity id (FK to ``target_table``), optional timestamp, then target.
    Metadata: ``fkey_col_to_pkey_table`` is only ``{entity_col: target_table}``;
    ``pkey_col`` is null; ``time_col`` is set only if a time column is written.
    """
    dst_tasks_dir.mkdir(parents=True, exist_ok=True)

    for task_meta in meta.get("tasks", []):
        task_name = task_meta["name"]
        task_dir = dst_tasks_dir / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        time_col_meta = task_meta.get("time_column")
        target_table = task_meta.get("target_table", "") or ""
        target_column = task_meta.get("target_column")
        if not target_column:
            print(f"  [SKIP task/{task_name}] missing target_column in metadata.yaml")
            continue

        entity_col = next(
            (
                col["name"]
                for col in task_meta.get("columns", [])
                if col.get("dtype") == "primary_key"
            ),
            None,
        )
        if not entity_col or not target_table:
            print(
                f"  [SKIP task/{task_name}] need primary_key column and target_table "
                f"(entity_col={entity_col!r}, target_table={target_table!r})"
            )
            continue

        col_dtype_map = {col["name"]: col["dtype"] for col in task_meta.get("columns", [])}
        col_dtype_map[entity_col] = "foreign_key"

        source_template = task_meta["source"]

        for split in ["train", "validation", "test"]:
            src_file = src_path / source_template.format(split=split)
            if not src_file.exists():
                print(f"  [SKIP] {src_file} 不存在")
                continue

            raw = load_table_file(src_file)
            df = pd.DataFrame(
                {
                    col_name: cast_column(
                        arr, col_dtype_map.get(col_name, "text"), col_name
                    )
                    for col_name, arr in raw.items()
                }
            )

            if target_column not in df.columns:
                print(
                    f"  [SKIP task/{task_name}] {split}: "
                    f"no column {target_column!r} in wide table"
                )
                continue
            if entity_col not in df.columns:
                print(
                    f"  [SKIP task/{task_name}] {split}: "
                    f"no entity column {entity_col!r}"
                )
                continue

            if target_table in pkey_maps:
                df[entity_col] = (
                    df[entity_col]
                    .map(pkey_maps[target_table])
                    .astype(pd.Int64Dtype())
                )

            narrow_cols: list[str] = [entity_col]
            effective_time_col: Optional[str] = None
            if (
                time_col_meta
                and isinstance(time_col_meta, str)
                and time_col_meta in df.columns
            ):
                narrow_cols.append(time_col_meta)
                effective_time_col = time_col_meta

            narrow_cols.append(target_column)
            df_narrow = df[narrow_cols].copy()

            tbl = Table(
                df=df_narrow,
                fkey_col_to_pkey_table={entity_col: target_table},
                pkey_col=None,
                time_col=effective_time_col,
            )

            out_name = "val" if split == "validation" else split
            out_path = task_dir / f"{out_name}.parquet"
            tbl.save(out_path)
            print(
                f"  [task/{task_name}] {split}: narrow {len(df_narrow)} rows "
                f"cols={narrow_cols} -> {out_path}"
            )


# ─────────────────────────────────────────────
#  主转换函数
# ─────────────────────────────────────────────
def _src_latest_mtime(src_path: Path) -> float:
    """最新源修改时间：metadata.yaml + 所有源 parquet 里的最大 mtime。"""
    candidates = []
    meta = src_path / "metadata.yaml"
    if meta.exists():
        candidates.append(meta.stat().st_mtime)
    for p in src_path.glob("*.parquet"):
        candidates.append(p.stat().st_mtime)
    return max(candidates) if candidates else 0.0


def _dst_oldest_mtime(dst_path: Path) -> float:
    """已转换 dst 的最老 parquet 修改时间，缺失返回 -inf（表示一定要重转）。"""
    db_dir = dst_path / "db"
    if not db_dir.exists():
        return float("-inf")
    mtimes = [p.stat().st_mtime for p in db_dir.glob("*.parquet")]
    return min(mtimes) if mtimes else float("-inf")


def convert_one(idx: int, src_root: Path, dst_root: Path, dst_prefix: str, force: bool = False):
    src_path = src_root / f"dag_rdb_{idx}"
    dst_path = dst_root / f"{dst_prefix}{idx}"

    if not src_path.exists():
        print(f"[SKIP] {src_path} 不存在")
        return

    src_mtime = _src_latest_mtime(src_path)
    dst_mtime = _dst_oldest_mtime(dst_path)
    dst_db_exists = (dst_path / "db").exists()
    stale = dst_db_exists and src_mtime > dst_mtime

    if dst_db_exists and not stale and not force:
        print(f"[SKIP] {dst_prefix}{idx} 已是最新，跳过")
        return

    if dst_db_exists and (stale or force):
        print(
            f"[REBUILD] {dst_prefix}{idx}: "
            f"src_mtime={src_mtime:.0f} > dst_mtime={dst_mtime:.0f} 或 --force，重转"
        )
        for sub in ("db", "tasks"):
            stale_dir = dst_path / sub
            if stale_dir.exists():
                shutil.rmtree(stale_dir)

    print(f"\n{'='*50}")
    print(f"转换 dag_rdb_{idx} -> {dst_prefix}{idx}")
    print(f"  src: {src_path}")
    print(f"  dst: {dst_path}")
    print(f"{'='*50}")

    db, meta = load_dbb_as_relbench_db(src_path)
    pkey_maps: dict[str, dict] = {}
    for tbl_name, tbl in db.table_dict.items():
        if tbl.pkey_col is not None:
            pkey_maps[tbl_name] = {
                v: i for i, v in enumerate(tbl.df[tbl.pkey_col])
            }
    db.reindex_pkeys_and_fkeys()   # 字符串 ID → 0-indexed 整数，Rust 脚本必须
    save_db_tables(db, dst_path / "db")
    save_task_tables(src_path, meta, dst_path / "tasks", pkey_maps)

    print(f"[OK] {dst_prefix}{idx} 完成")


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--idx",   type=int)
    group.add_argument("--all",   action="store_true")
    group.add_argument("--range", type=int, nargs=2, metavar=("START", "END"))
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略 src/dst mtime 比较，强制重新转换（会先删除已有 dst/db 与 dst/tasks）",
    )
    parser.add_argument(
        "--src-root",
        type=Path,
        default=_DEFAULT_SRC_ROOT,
        help=f"Source root directory for dag_rdb_<idx> inputs (default: {_DEFAULT_SRC_ROOT})",
    )
    parser.add_argument(
        "--dst-root",
        type=Path,
        default=_DEFAULT_DST_ROOT,
        help=f"Destination root for relbench-format outputs (default: {_DEFAULT_DST_ROOT})",
    )
    parser.add_argument(
        "--dst-prefix",
        type=str,
        default=_DEFAULT_DST_PREFIX,
        help=f"Destination directory name prefix (default: '{_DEFAULT_DST_PREFIX}')",
    )
    parser.add_argument(
        "--num-datasets",
        type=int,
        default=_DEFAULT_NUM_DATASETS,
        help=f"Number of datasets when using --all (default: {_DEFAULT_NUM_DATASETS})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.idx is not None:
        convert_one(args.idx, args.src_root, args.dst_root, args.dst_prefix, force=args.force)
    elif args.all:
        for i in range(args.num_datasets):
            try:
                convert_one(i, args.src_root, args.dst_root, args.dst_prefix, force=args.force)
            except Exception as e:
                print(f"[ERROR] dag_rdb_{i}: {e}")
    elif args.range:
        for i in range(args.range[0], args.range[1]):
            try:
                convert_one(i, args.src_root, args.dst_root, args.dst_prefix, force=args.force)
            except Exception as e:
                print(f"[ERROR] dag_rdb_{i}: {e}")