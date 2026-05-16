#!/usr/bin/env python3
"""CLI：快速查看 Parquet 的 schema、行数与前几行（适用于 pixi 环境）."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import pyarrow as pa
import pandas as pd
import pyarrow.parquet as pq


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="读取并展示 Parquet：schema、行数、可选前几行。"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default="/data/caijunyu/relbench_cache/dag_rdb_complex_0/db/table_0.parquet",
        help="Parquet 文件路径",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=10,
        metavar="N",
        help="打印前 N 行（默认 10；设为 0 则不打印样本行）",
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="只打印 Arrow schema，不读全表",
    )
    parser.add_argument(
        "--json-schema",
        action="store_true",
        help="以 JSON 打印列名与 Arrow 类型（便于脚本解析）",
    )
    return parser.parse_args()


def _arrow_schema_dict(schema: pa.Schema) -> list[dict[str, str]]:
    """将 pyarrow Schema 转为可 JSON 序列化的简单结构."""
    out: list[dict[str, str]] = []
    for i in range(len(schema)):
        field = schema[i]
        out.append({"name": field.name, "type": str(field.type)})
    return out


def main() -> None:
    args = _parse_args()
    path = args.path.expanduser().resolve()
    if not path.is_file():
        print(f"错误：不是文件或不存在: {path}", file=sys.stderr)
        sys.exit(1)

    pf = pq.ParquetFile(path)
    meta = pf.metadata
    print(f"文件: {path}")
    print(f"行数: {meta.num_rows}  |  行组数: {meta.num_row_groups}")

    schema = pf.schema_arrow
    if args.json_schema:
        print(json.dumps(_arrow_schema_dict(schema), indent=2, ensure_ascii=False))
    else:
        print("Schema:")
        print(schema)

    if args.schema_only:
        return

    if args.head <= 0:
        return

    df = pd.read_parquet(path)
    print(f"\npandas dtypes:\n{df.dtypes}\n")
    print(f"前 {args.head} 行:")
    # 宽表时 to_string 比默认 print 更易读
    with pd.option_context("display.max_columns", 50, "display.width", 200):
        print(df.head(args.head).to_string())


if __name__ == "__main__":
    main()
