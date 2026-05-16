"""Count complex tasks in two RDB dataset directories."""

import os
import pandas as pd


def count_tasks(base_dir, label):
    subdirs = sorted([d for d in os.listdir(base_dir) if d.startswith("dag_rdb_")])
    total_rdbs = len(subdirs)

    classification = 0
    regression = 0
    tasks_per_rdb = []
    rdbs_with_zero = 0

    for sd in subdirs:
        sd_path = os.path.join(base_dir, sd)
        if not os.path.isdir(sd_path):
            continue
        tasks = [d for d in os.listdir(sd_path) if d.startswith("complex")]
        task_count = len(tasks)
        tasks_per_rdb.append(task_count)
        if task_count == 0:
            rdbs_with_zero += 1

        for t in tasks:
            tp = os.path.join(sd_path, t)
            train_file = os.path.join(tp, "train.parquet")
            if not os.path.exists(train_file):
                regression += 1
                continue
            try:
                df = pd.read_parquet(train_file)
                if "labels" in df.columns:
                    nunique = df["labels"].nunique()
                else:
                    nunique = df[df.columns[-1]].nunique()
                if nunique <= 20:
                    classification += 1
                else:
                    regression += 1
            except Exception:
                regression += 1

    avg = sum(tasks_per_rdb) / total_rdbs if total_rdbs else 0

    print(f"=== {label} ({base_dir}) ===")
    print(f"  Total RDB subdirectories:  {total_rdbs}")
    print(f"  Total complex tasks:       {classification + regression}")
    print(f"    Classification tasks:    {classification}")
    print(f"    Regression tasks:        {regression}")
    print(f"  Avg complex tasks per RDB: {avg:.4f}")
    print(f"  RDBs with 0 complex tasks: {rdbs_with_zero}")
    print()


count_tasks(
    "/data/caijunyu/RDBPFN/data_generation/RDB_datasets/hsbm_v2.5",
    "hsbm_v2.5 (no timestamp)",
)
count_tasks(
    "/data/caijunyu/RDBPFN/data_generation/RDB_datasets/plurel_scale_complex_hsbm_v2",
    "plurel_scale_complex_hsbm_v2 (with timestamp)",
)
