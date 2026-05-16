"""Download eval relational datasets from HuggingFace."""
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "yamboo/RDB_PFN"
LOCAL_BASE = Path(__file__).resolve().parent.parent / "model_pretrain"

# All datasets from the config
ALL_DATASETS = [
    "amazon-dfs-2", "avs-dfs-2", "diginetica-dfs-2",
    "outbrain-small-dfs-2", "retailrocket-dfs-2", "stackexchange-dfs-2",
    "rel-amazon-dfs-2", "rel-avito-dfs-2", "rel-event-dfs-2",
    "rel-f1-dfs-2", "rel-hm-dfs-2", "rel-stack-dfs-2", "rel-trial-dfs-2",
]

missing = [d for d in (sys.argv[1:] if len(sys.argv) > 1 else ALL_DATASETS)
           if not (LOCAL_BASE / "rdb_datasets" / d).exists()]

if not missing:
    print("All datasets already present.")
    sys.exit(0)

print(f"Downloading {len(missing)} datasets: {missing}")

for ds in missing:
    print(f"\n=== Downloading {ds} ===")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=f"model_pretrain/rdb_datasets/{ds}/*",
        local_dir=str(LOCAL_BASE),
        local_dir_use_symlinks=False,
        max_workers=8,
    )
    print(f"Done: {ds}")

print(f"\nAll {len(missing)} datasets downloaded.")
