"""Task quality checker for complex tasks.

Stage A (ultra-cheap, rules-only, no sklearn):
  a1. label_nunique == 1
  a2. pos_ratio < 0.05 or > 0.95
  a3. n < 64 or min(n_pos, n_neg) < 8
  a4. zero feature_ columns or all constant
  a5. structural independence (schema-based, best-effort)
  a6. extreme child/parent size ratio (> 100:1 or < 1:100)

Stage B (model-based):
  b1. Lightweight ExtraTrees OOF AUC
  b2. Bootstrap CI for gray-zone tasks (0.50 < OOF AUC < 0.60)
  b3. Leakage check on FK/ID/timestamp columns

Reference: TabLeak (https://arxiv.org/html/2602.11139v1)

Usage from pipeline (in-memory, no disk I/O):
    from .task_quality import diagnose_dataframe
    result = diagnose_dataframe(df, schema_data=...)

Usage from CLI:
    from .task_quality import diagnose_parquet_path
    result = diagnose_parquet_path("/path/to/train.parquet")
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── constants ──────────────────────────────────────────────────────────────
AUC_HARD_REJECT = 0.52       # OOF AUC <= this -> reject directly
AUC_HARD_ACCEPT = 0.58       # OOF AUC >= this -> accept, no bootstrap
AUC_BASELINE = 0.55           # bootstrap CI upper bound must reach this
LEAKAGE_SINGLE_AUC = 0.90     # single-col AUC threshold (suspicious)
LEAKAGE_DROP = 0.10           # full-model AUC must drop by this much
CHILD_PARENT_RATIO_MAX = 100  # reject if child/parent row ratio exceeds this


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Return feature columns (focal + joined from related tables), non-constant."""
    cols = []
    for c in df.columns:
        if c == "labels" or c == "timestamp":
            continue
        if c.endswith("_id"):
            continue
        if "feature_" in c:
            cols.append(c)
    if not cols:
        return []
    valid = df[cols].nunique() > 1
    return [c for c, ok in zip(cols, valid) if ok]


def _count_fk_cols(df: pd.DataFrame) -> int:
    """Count columns that look like foreign keys (name ends with _id)."""
    return sum(1 for c in df.columns
               if c.endswith("_id") and c != "labels" and "feature_" not in c)


def _etoof(X: np.ndarray, y: np.ndarray, n_splits: int = 3,
           n_estimators: int = 64, max_depth: int = 12,
           random_state: int = 42) -> Tuple[np.ndarray, float]:
    """ExtraTrees OOF predictions + AUC. Returns (oof_preds, auc)."""
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    n = len(y)
    if n < 2 * n_splits:
        n_splits = max(2, n // 2)
    min_class = int(min(np.bincount(y)))
    if min_class < n_splits:
        n_splits = max(2, min_class)
    oof = np.full(n, np.nan, dtype=np.float64)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for tr, te in cv.split(X, y):
        m = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                 random_state=random_state, n_jobs=1)
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    valid = ~np.isnan(oof)
    if valid.sum() < 2 or len(np.unique(y[valid])) < 2:
        return oof, 0.5
    return oof, float(roc_auc_score(y[valid], oof[valid]))


def _bootstrap_auc_from_oof(oof_preds: np.ndarray, y: np.ndarray,
                             n_bootstrap: int = 100,
                             random_state: int = 42) -> Tuple[float, float, float]:
    """Bootstrap AUC CI by resampling OOF predictions (no retraining)."""
    from sklearn.metrics import roc_auc_score

    valid = ~np.isnan(oof_preds)
    p = oof_preds[valid]
    t = y[valid]
    if len(np.unique(t)) < 2:
        return 0.5, 0.5, 0.5

    rng = np.random.RandomState(random_state)
    n = len(p)
    aucs = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        if len(np.unique(t[idx])) < 2:
            aucs[i] = 0.5
        else:
            aucs[i] = float(roc_auc_score(t[idx], p[idx]))
    return (float(np.mean(aucs)),
            float(np.percentile(aucs, 2.5)),
            float(np.percentile(aucs, 97.5)))


def _check_leakage(df: pd.DataFrame, y: np.ndarray, feature_cols: List[str],
                   full_auc: float) -> Tuple[bool, str]:
    """Check FK/ID/timestamp columns for label leakage."""
    id_cols = [c for c in df.columns
               if (c.endswith("_id") and c != "labels" and "feature_" not in c)
               or c == "timestamp"]
    if not id_cols:
        return False, ""

    for col in id_cols:
        if col not in df.columns:
            continue
        ser = df[col].copy()
        if pd.api.types.is_datetime64_any_dtype(ser):
            ser = ser.astype("int64")
        ser = pd.to_numeric(ser, errors="coerce")
        if ser.nunique() < 2:
            continue
        sX = ser.fillna(ser.median()).values.reshape(-1, 1).astype(np.float64)
        _, single_auc = _etoof(sX, y, n_splits=5, n_estimators=30, max_depth=6)
        if single_auc < LEAKAGE_SINGLE_AUC:
            continue

        wo_cols = [c for c in feature_cols if c != col]
        if len(wo_cols) < 1:
            continue
        wo_X = df[wo_cols].copy()
        wo_X = wo_X.replace([np.inf, -np.inf], np.nan)
        wo_X = wo_X.fillna(wo_X.median())
        wo_val = wo_X.values.astype(np.float64)
        _, wo_auc = _etoof(wo_val, y, n_splits=5, n_estimators=30, max_depth=6)
        drop = full_auc - wo_auc
        if drop > LEAKAGE_DROP:
            return True, (f"b3_leakage:{col} single={single_auc:.3f} "
                          f"full={full_auc:.3f} wo={wo_auc:.3f} drop={drop:.3f}")

    return False, ""


# ═════════════════════════════════════════════════════════════════════════════
# Stage A: rule-based pre-screening
# ═════════════════════════════════════════════════════════════════════════════

def stage_a_check_df(df: pd.DataFrame,
                     schema_data: Optional[Dict] = None,
                     focal_table: Optional[str] = None,
                     target_table: Optional[str] = None,
                     task_name: Optional[str] = None) -> dict:
    """Run all Stage A checks on a DataFrame.

    Args:
        df: Task dataframe (must have ``labels`` column).
        schema_data: Optional schema dict with ``table_generation_schemas`` and
                     ``task_generation_schemas`` keys (for a5/a6 checks).
        focal_table: Focal table name (for a5/a6 context).
        target_table: Target table name (for a5/a6 context).
        task_name: Task name (for a5/a6 context).

    Returns:
        dict with keys: a_pass, a_failures, n_samples, n_pos, n_neg,
        pos_ratio, n_features, n_fk_cols.
    """
    result = {
        "a_pass": False,
        "a_failures": [],
        "n_samples": 0, "n_pos": 0, "n_neg": 0, "pos_ratio": float("nan"),
        "n_features": 0, "n_fk_cols": 0,
    }

    if df is None or df.empty or "labels" not in df.columns:
        result["a_failures"].append("load_error")
        return result

    y = df["labels"].values
    n = len(df)
    n_unique = len(np.unique(y))
    result["n_samples"] = n

    # a1: label all-same
    if n_unique < 2:
        result["a_failures"].append("a1_label_unique_1")
        result["pos_ratio"] = float(np.mean(y))
        return result

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    pos_ratio = n_pos / n
    result["n_pos"] = n_pos
    result["n_neg"] = n_neg
    result["pos_ratio"] = pos_ratio

    # a2: extreme label imbalance
    if pos_ratio < 0.05 or pos_ratio > 0.95:
        result["a_failures"].append(f"a2_pos_ratio_{pos_ratio:.3f}")

    # a3: too few samples
    if n < 64:
        result["a_failures"].append(f"a3_n_{n}_lt_64")
    if min(n_pos, n_neg) < 8:
        result["a_failures"].append(f"a3_min_class_{min(n_pos, n_neg)}_lt_8")

    # a4: no valid features
    raw_features = [c for c in df.columns if "feature_" in c]
    valid_features = _get_feature_cols(df)
    result["n_features"] = len(valid_features)
    if len(raw_features) == 0:
        result["a_failures"].append("a4_zero_feature_cols")
    elif len(valid_features) == 0:
        result["a_failures"].append("a4_all_constant")

    # a5: structural independence (schema-based, best-effort)
    if schema_data is not None and focal_table and target_table:
        tables = schema_data.get("table_generation_schemas", {})
        if isinstance(tables, dict):
            if target_table not in tables or focal_table not in tables:
                result["a_failures"].append("a5_table_not_in_schema")

    # a6: extreme child/parent size ratio
    result["n_fk_cols"] = _count_fk_cols(df)
    if schema_data is not None and result["n_fk_cols"] > 0:
        tables_dict = schema_data.get("table_generation_schemas", {})
        task_schemas = schema_data.get("task_generation_schemas", [])
        if isinstance(tables_dict, dict) and task_schemas:
            row_counts = {}
            for tname, info in tables_dict.items():
                scm = info.get("scm_params", {})
                if "seq_len" in scm:
                    row_counts[tname] = scm["seq_len"]
            child_rows = n
            for ts in task_schemas:
                if ts.get("task_name") != task_name:
                    continue
                parent_tables_list = ts.get("primary_table_generation_schema", {}).get(
                    "parent_tables", [])
                for pt in parent_tables_list:
                    parent_rows = row_counts.get(pt)
                    if parent_rows and child_rows > 0:
                        ratio = child_rows / parent_rows
                        if ratio > CHILD_PARENT_RATIO_MAX or ratio < (1.0 / CHILD_PARENT_RATIO_MAX):
                            result["a_failures"].append(f"a6_size_ratio_{ratio:.1f}")
                            break

    result["a_pass"] = len(result["a_failures"]) == 0
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Stage B: model-based checks
# ═════════════════════════════════════════════════════════════════════════════

def stage_b_check_df(df: pd.DataFrame) -> dict:
    """Stage B: shared OOF fit -> bootstrap CI on preds, leakage check."""
    result = {
        "b_pass": False,
        "b_failures": [],
        "et_auc": float("nan"),
        "et_ci_lower": float("nan"),
        "et_ci_upper": float("nan"),
        "was_gray_zone": False,
    }

    if df is None:
        result["b_failures"].append("b_load_error")
        return result

    y = df["labels"].values
    feature_cols = _get_feature_cols(df)
    if not feature_cols:
        result["b_failures"].append("b_no_features")
        return result

    X = df[feature_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    X_val = X.values.astype(np.float64)

    # shared OOF fit (once per task)
    try:
        oof_preds, oof_auc = _etoof(X_val, y, n_splits=3, n_estimators=64, max_depth=12)
    except Exception as e:
        result["b_failures"].append(f"b1_oof_failed:{e}")
        return result

    # b1+b2: three-zone AUC check
    if oof_auc <= AUC_HARD_REJECT:
        result["et_auc"] = oof_auc
        result["b_failures"].append(f"b2_unlearnable: AUC={oof_auc:.4f} <= {AUC_HARD_REJECT}")
    elif oof_auc >= AUC_HARD_ACCEPT:
        result["et_auc"] = oof_auc  # clear pass
    else:
        result["was_gray_zone"] = True
        try:
            mean_auc, ci_l, ci_u = _bootstrap_auc_from_oof(
                oof_preds, y, n_bootstrap=100)
            result["et_auc"] = mean_auc
            result["et_ci_lower"] = ci_l
            result["et_ci_upper"] = ci_u
            if ci_u < AUC_BASELINE:
                result["b_failures"].append(
                    f"b2_unlearnable: AUC={mean_auc:.4f} CI=[{ci_l:.4f},{ci_u:.4f}]")
        except Exception as e:
            result["b_failures"].append(f"b2_bootstrap_failed:{e}")

    # b3: leakage check
    is_leak, leak_reason = _check_leakage(df, y, feature_cols, oof_auc)
    if is_leak:
        result["b_failures"].append(leak_reason)

    result["b_pass"] = len(result["b_failures"]) == 0
    return result


# ═════════════════════════════════════════════════════════════════════════════
# Main entry points
# ═════════════════════════════════════════════════════════════════════════════

def diagnose_dataframe(df: pd.DataFrame,
                       schema_data: Optional[Dict] = None,
                       focal_table: Optional[str] = None,
                       target_table: Optional[str] = None,
                       task_name: Optional[str] = None) -> dict:
    """Two-stage diagnosis on an in-memory DataFrame (no disk I/O).

    Args:
        df: Task dataframe with ``labels`` column.
        schema_data: Optional schema dict for a5/a6 checks.
        focal_table, target_table, task_name: Context for a5/a6 checks.

    Returns:
        dict with keys: passed, stage, et_auc, et_ci_lower, et_ci_upper,
        all_failures, n_samples, n_pos, n_neg, pos_ratio, n_features.
    """
    a = stage_a_check_df(df, schema_data, focal_table, target_table, task_name)
    if not a["a_pass"]:
        return {
            **a,
            "passed": False,
            "stage": "A",
            "et_auc": float("nan"),
            "et_ci_lower": float("nan"),
            "et_ci_upper": float("nan"),
            "all_failures": a["a_failures"],
        }
    b = stage_b_check_df(df)
    return {
        **a,
        "passed": b["b_pass"],
        "stage": "B",
        "et_auc": b["et_auc"],
        "et_ci_lower": b["et_ci_lower"],
        "et_ci_upper": b["et_ci_upper"],
        "all_failures": a["a_failures"] + b["b_failures"],
    }


def diagnose_parquet_path(path: str) -> dict:
    """Two-stage diagnosis on a train.parquet file path.

    Loads the parquet, extracts schema from the parent directory if available.
    """
    df = pd.read_parquet(path)
    task_dir = os.path.dirname(path)
    task_name = os.path.basename(task_dir)

    # Best-effort schema loading
    schema_data = None
    rdb_dir = os.path.dirname(task_dir)
    for schema_name in ("generation_schemas.yaml", "metadata.yaml"):
        schema_path = os.path.join(rdb_dir, schema_name)
        if os.path.exists(schema_path):
            try:
                import yaml
                with open(schema_path) as f:
                    schema_data = yaml.safe_load(f)
                break
            except Exception:
                pass

    # Extract focal/target table from schema
    focal_table: Optional[str] = None
    target_table: Optional[str] = None
    if schema_data is not None:
        task_schemas = schema_data.get("task_generation_schemas", [])
        for ts in task_schemas:
            if ts.get("task_name") == task_name:
                focal_table = ts.get("primary_table_name")
                tns = ts.get("target_node_set", None)
                if tns and isinstance(tns, dict):
                    target_table = tns.get("table_name")
                break

    result = diagnose_dataframe(df, schema_data, focal_table, target_table, task_name)
    result["path"] = path
    return result
