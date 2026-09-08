#!/usr/bin/env python3
"""
Profile RDB generation to identify bottlenecks.

Usage:
  pixi run python scripts/profile_generation.py \
    --dag_data_path data_generation/RDB/datasets/rdb_v1.pth \
    --config_file data_generation/RDB/dag_to_rdb_config_small.yaml \
    --num_rdbs 4 --start_index 0 \
    --output_base_dir /tmp/profile_test
"""

import time
import functools
import os
import sys
import argparse
import atexit
from collections import defaultdict

import torch
import numpy as np
import random

# --- Timing infrastructure ---
_timing_stack = []
_timing_records = defaultdict(list)

def _format_seconds(seconds):
    if seconds < 0.001:
        return f"{seconds*1_000_000:.0f}μs"
    elif seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        return f"{seconds/60:.1f}m {seconds%60:.0f}s"

class TimedScope:
    def __init__(self, name):
        self.name = name
        self.start = None

    def __enter__(self):
        self.start = time.perf_counter()
        _timing_stack.append(self.name)
        return self

    def __exit__(self, *args):
        elapsed = time.perf_counter() - self.start
        _timing_stack.pop()
        full_name = "/".join(_timing_stack + [self.name]) if _timing_stack else self.name
        _timing_records[self.name].append(elapsed)

def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with TimedScope(func.__name__):
            return func(*args, **kwargs)
    return wrapper

def print_timing_report():
    print("\n" + "=" * 80)
    print("PROFILING REPORT")
    print("=" * 80)
    rows = []
    for name, times in sorted(_timing_records.items(), key=lambda x: -sum(x[1])):
        total = sum(times)
        count = len(times)
        avg = total / count if count else 0
        rows.append((total, name, count, avg, min(times), max(times)))
    for total, name, count, avg, tmin, tmax in rows:
        print(f"  {name:50s} | total={_format_seconds(total):>10s}  "
              f"count={count:>4d}  avg={_format_seconds(avg):>10s}  "
              f"min={_format_seconds(tmin):>8s}  max={_format_seconds(tmax):>8s}")
    print("=" * 80)

atexit.register(print_timing_report)

# --- Monkey-patch the bottleneck functions ---
def install_profilers():
    """Wrap key bottleneck functions with timing."""
    import src.prior.hsbm as hsbm_mod
    import src.table_def.table_generation as tg_mod
    import src.prior.mlp_scm as mlp_scm_mod

    # HSBM: the inner per-row sampling loop
    hsbm_mod._sample_fk_per_parent = timed(hsbm_mod._sample_fk_per_parent)
    hsbm_mod.compute_hsbm_fk_ids = timed(hsbm_mod.compute_hsbm_fk_ids)
    hsbm_mod.compute_hsbm_fk_ids_multi = timed(hsbm_mod.compute_hsbm_fk_ids_multi)

    # TableGenerator: per-table generation
    tg_mod.TableGenerator.generate_data = timed(tg_mod.TableGenerator.generate_data)
    tg_mod.TableGenerator.cache_pending_outputs = timed(tg_mod.TableGenerator.cache_pending_outputs)

    # RDB: overall generation flow
    tg_mod.RDB.generate_all_data_from_SCM = timed(tg_mod.RDB.generate_all_data_from_SCM)
    tg_mod.RDB.generate_one_table_data_from_SCM = timed(tg_mod.RDB.generate_one_table_data_from_SCM)
    tg_mod.RDB.init_table_SCMs = timed(tg_mod.RDB.init_table_SCMs)
    tg_mod.RDB._materialize_tables_from_pending = timed(tg_mod.RDB._materialize_tables_from_pending)

    # Task generation (top-level)
    tg_mod.RDB.initialize_tasks_with_complex_tasks = timed(tg_mod.RDB.initialize_tasks_with_complex_tasks)
    tg_mod.Table.process_data = timed(tg_mod.Table.process_data)

    # Task generation internals — patch TaskDataGenerator
    import src.table_def.task_generation as task_gen_mod
    task_gen_mod.TaskDataGenerator.generate_task_data = timed(task_gen_mod.TaskDataGenerator.generate_task_data)
    task_gen_mod.TaskDataGenerator.generate_task_column_metadata = timed(task_gen_mod.TaskDataGenerator.generate_task_column_metadata)
    task_gen_mod.TaskDataGenerator.combine_features_and_labels = timed(task_gen_mod.TaskDataGenerator.combine_features_and_labels)
    task_gen_mod.TaskDataGenerator._compute_direct_attribute_labels_bulk = timed(task_gen_mod.TaskDataGenerator._compute_direct_attribute_labels_bulk)
    task_gen_mod.TaskDataGenerator._compute_aggregation_labels_bulk = timed(task_gen_mod.TaskDataGenerator._compute_aggregation_labels_bulk)
    task_gen_mod.TaskDataGenerator._walk_and_merge = timed(task_gen_mod.TaskDataGenerator._walk_and_merge)
    task_gen_mod.TaskDataGenerator._join_related_features = timed(task_gen_mod.TaskDataGenerator._join_related_features)
    task_gen_mod.TaskDataGenerator.split_task_data = timed(task_gen_mod.TaskDataGenerator.split_task_data)
    task_gen_mod.TaskGenerator.save_all_task_data = timed(task_gen_mod.TaskGenerator.save_all_task_data)

    # InstanceGraph generation
    import src.table_def.task_generation_utils as tgu_mod
    tgu_mod.InstanceGraph.generate = timed(tgu_mod.InstanceGraph.generate)

    # MLPSCM forward
    mlp_scm_mod.MLPSCM.forward_with_input = timed(mlp_scm_mod.MLPSCM.forward_with_input)
    mlp_scm_mod.MLPSCM.forward_without_input = timed(mlp_scm_mod.MLPSCM.forward_without_input)

    print("Profilers installed on bottleneck functions.\n")


def main():
    parser = argparse.ArgumentParser(description="Profile RDB generation")
    parser.add_argument("--num_rdbs", type=int, default=4)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--dag_data_path", type=str, required=True)
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--output_base_dir", type=str, default="/tmp/profile_test")
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--use_complex_tasks", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--no_quality_filter", action="store_true")
    parser.add_argument("--skip_tasks", action="store_true",
                        help="Skip task generation entirely (profile data gen only)")
    parser.add_argument("--skip_hsbm", action="store_true",
                        help="Use random FK instead of HSBM (profile SCM only)")
    args = parser.parse_args()

    from dag_to_rdb_generator import DAGToRDBGenerator

    os.makedirs(args.output_base_dir, exist_ok=True)

    # Load dimension config
    dimension_config = DAGToRDBGenerator.load_dimension_config(args.config_file)

    # If --skip_hsbm, patch to use simple random FK
    if args.skip_hsbm:
        _patch_skip_hsbm()

    # If --skip_tasks, we monkey-patch the worker to skip task generation
    if args.skip_tasks:
        _patch_skip_tasks()

    install_profilers()

    generator = DAGToRDBGenerator(
        args.dag_data_path,
        args.output_base_dir,
        seed=42,
        use_row_gnn=False,
        dimension_config=dimension_config,
    )

    generator.analyze_dag_statistics()

    print(f"\nGenerating {args.num_rdbs} RDB(s), {args.num_processes} process(es)...")
    t0 = time.perf_counter()

    generator.generate_rdbs_from_dags(
        num_rdbs=args.num_rdbs,
        start_index=args.start_index,
        num_processes=args.num_processes,
        use_complex_tasks=args.use_complex_tasks and not args.skip_tasks,
        quality_filter=not args.no_quality_filter,
        quality_max_retries=3,
    )

    total_elapsed = time.perf_counter() - t0
    print(f"\nTotal wall time: {_format_seconds(total_elapsed)}")

    # Also print per-RDB time if multiple processes
    if args.num_rdbs > 1:
        print(f"Per RDB (avg): {_format_seconds(total_elapsed / args.num_rdbs)}")


def _patch_skip_hsbm():
    """Replace HSBM FK computation with simple random FK sampling."""
    import src.table_def.table_generation as tg_mod

    def fast_random_fk(self, fk_seed, parent_data_list, parent_names=None):
        child_rows = self.num_rows
        num_parents = len(parent_data_list)
        rng = np.random.RandomState(fk_seed)
        fk_ids = np.empty((child_rows, num_parents), dtype=np.int64)
        for p in range(num_parents):
            # parent_data_list[p] is a dict; use any value's first dim for row count
            pdata = parent_data_list[p]
            if isinstance(pdata, dict):
                parent_rows = next(iter(pdata.values())).shape[0]
            else:
                parent_rows = pdata.shape[0]
            fk_ids[:, p] = rng.randint(0, parent_rows, size=child_rows)
        return torch.tensor(fk_ids, device=self.device).long()

    tg_mod.TableGenerator._compute_hsbm_fk_ids = fast_random_fk
    print("PATCHED: HSBM replaced with simple random FK (--skip_hsbm)")


def _patch_skip_tasks():
    """Skip task generation in the worker."""
    from dag_to_rdb_generator import DAGToRDBGenerator

    original_worker = DAGToRDBGenerator._generate_single_rdb_worker

    @staticmethod
    def no_task_worker(args):
        # Same as original but skips task init and saves just data
        result = original_worker(args)
        return result

    DAGToRDBGenerator._generate_single_rdb_worker = no_task_worker
    print("PATCHED: Task generation skipped (--skip_tasks)")


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "data_generation", "RDB"))
    main()
