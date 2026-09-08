"""Download benchmark datasets from yamboo/RDB_PFN on HuggingFace.

Downloads:
  1. Flat single-table CSVs → model_pretrain/datasets/clf_real/
  2. Pre-computed subsample splits → model_pretrain/datasets/clf_real_subsamples/
  3. Small relational benchmarks (rel-f1, rel-event) → model_pretrain/rdb_datasets/
"""

import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = "yamboo/RDB_PFN"
BASE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main"
PROXY = "http://127.0.0.1:7897"

os.environ["HTTP_PROXY"] = PROXY
os.environ["HTTPS_PROXY"] = PROXY

ROOT = Path(__file__).resolve().parent.parent
MAX_RETRIES = 3


def download(repo_path: str, local_path: Path) -> bool:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        print(f"  SKIP {local_path.name} (exists)")
        return False

    url = f"{BASE_URL}/{repo_path}"
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "RDBPFN-download/1.0")
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            local_path.write_bytes(data)
            size_mb = len(data) / (1024 * 1024)
            print(f"  OK   {local_path.name} ({size_mb:.1f}MB)")
            return True
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"  RETRY {local_path.name} (attempt {attempt}/{MAX_RETRIES}): {e}")
                time.sleep(2)
            else:
                print(f"  FAIL {local_path.name}: {e}", file=sys.stderr)
                return False


def main():
    t0 = time.time()
    total = 0

    # --- Category 1: Flat single-table CSVs (no duplicates) ---
    csv_names = [
        "Bioresponse.csv", "Diabetes130US.csv", "Higgs.csv",
        "MagicTelescope.csv", "MiniBooNE.csv", "albert.csv",
        "bank-marketing.csv", "california.csv", "compas-two-years.csv",
        "covertype.csv", "credit.csv", "default-of-credit-card-clients.csv",
        "electricity.csv", "eye_movements.csv", "heloc.csv",
        "house_16H.csv", "jannis.csv", "pol.csv", "road-safety.csv",
    ]
    print(f"=== Category 1: Flat CSVs ({len(csv_names)} files) ===")
    for name in csv_names:
        repo_path = f"model_pretrain/datasets/clf/{name}"
        local_path = ROOT / "model_pretrain/datasets/clf_real" / name
        if download(repo_path, local_path):
            total += 1

    # --- Category 2: Pre-computed subsample splits (no duplicates) ---
    npz_names = [
        "Bioresponse_split.npz", "Diabetes130US_split.npz", "Higgs_split.npz",
        "MagicTelescope_split.npz", "MiniBooNE_split.npz", "albert_split.npz",
        "bank-marketing_split.npz", "california_split.npz",
        "compas-two-years_split.npz", "covertype_split.npz", "credit_split.npz",
        "default-of-credit-card-clients_split.npz", "electricity_split.npz",
        "eye_movements_split.npz", "heloc_split.npz", "house_16H_split.npz",
        "jannis_split.npz", "pol_split.npz", "road-safety_split.npz",
    ]
    print(f"\n=== Category 2: Subsample splits ({len(npz_names)} files) ===")
    for name in npz_names:
        repo_path = f"model_pretrain/datasets/clf_subsamples/{name}"
        base = name.replace("_split.npz", "")
        local_path = ROOT / "model_pretrain/datasets/clf_real_subsamples" / name
        if download(repo_path, local_path):
            total += 1

    # --- Category 3: Small relational benchmarks ---
    rel_files = [
        # rel-f1-dfs-2: 3 tasks × 3 splits + metadata = 10 files
        "model_pretrain/rdb_datasets/rel-f1-dfs-2/metadata.yaml",
        "model_pretrain/rdb_datasets/rel-f1-dfs-2/driver-dnf/train.npz",
        "model_pretrain/rdb_datasets/rel-f1-dfs-2/driver-dnf/validation.npz",
        "model_pretrain/rdb_datasets/rel-f1-dfs-2/driver-dnf/test.npz",
        "model_pretrain/rdb_datasets/rel-f1-dfs-2/driver-position/train.npz",
        "model_pretrain/rdb_datasets/rel-f1-dfs-2/driver-position/validation.npz",
        "model_pretrain/rdb_datasets/rel-f1-dfs-2/driver-position/test.npz",
        "model_pretrain/rdb_datasets/rel-f1-dfs-2/driver-top3/train.npz",
        "model_pretrain/rdb_datasets/rel-f1-dfs-2/driver-top3/validation.npz",
        "model_pretrain/rdb_datasets/rel-f1-dfs-2/driver-top3/test.npz",
        # rel-event-dfs-2: 3 tasks × 3 splits + metadata = 10 files
        "model_pretrain/rdb_datasets/rel-event-dfs-2/metadata.yaml",
        "model_pretrain/rdb_datasets/rel-event-dfs-2/user-attendance/train.npz",
        "model_pretrain/rdb_datasets/rel-event-dfs-2/user-attendance/validation.npz",
        "model_pretrain/rdb_datasets/rel-event-dfs-2/user-attendance/test.npz",
        "model_pretrain/rdb_datasets/rel-event-dfs-2/user-ignore/train.npz",
        "model_pretrain/rdb_datasets/rel-event-dfs-2/user-ignore/validation.npz",
        "model_pretrain/rdb_datasets/rel-event-dfs-2/user-ignore/test.npz",
        "model_pretrain/rdb_datasets/rel-event-dfs-2/user-repeat/train.npz",
        "model_pretrain/rdb_datasets/rel-event-dfs-2/user-repeat/validation.npz",
        "model_pretrain/rdb_datasets/rel-event-dfs-2/user-repeat/test.npz",
    ]
    print(f"\n=== Category 3: Relational benchmarks ({len(rel_files)} files) ===")
    for repo_path in rel_files:
        local_path = ROOT / repo_path
        if download(repo_path, local_path):
            total += 1

    elapsed = time.time() - t0
    print(f"\n=== Done: {total} files downloaded in {elapsed:.0f}s ===")


if __name__ == "__main__":
    main()
