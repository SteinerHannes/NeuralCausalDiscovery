import math
from typing import Iterable, List, Optional, Sequence, Tuple, Dict

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

INTERVENTION_REQUIRED_COLUMNS = (
    "agreement_rate_total",
    "model_agreement_rate",
)

def collect_outcome_std(dag_data: np.ndarray) -> np.ndarray:
    if dag_data.size == 0:
        return np.array([])
    std = dag_data.std(axis=0, ddof=0)
    std[std == 0] = 1.0
    return std


def instantiate_dataset(cfg: DictConfig, subset: str, fold: int):
    return hydra.utils.instantiate(cfg.data, subset=subset, fold=fold)


def add_standardized_effects(df: pd.DataFrame, outcome_std_by_fold: Dict[int, np.ndarray]) -> None:
    if df is None or df.empty:
        return
    if "outcome" not in df.columns or "fold" not in df.columns:
        return

    std_values = []
    for _, row in df.iterrows():
        fold = row.get("fold")
        std_arr = outcome_std_by_fold.get(int(fold)) if fold is not None else None
        if std_arr is None or std_arr.size == 0:
            std_values.append(np.nan)
            continue
        outcome_label = row.get("outcome")
        try:
            idx = int(str(outcome_label).replace("X", ""))
        except (TypeError, ValueError):
            std_values.append(np.nan)
            continue
        if idx < 0 or idx >= std_arr.size:
            std_values.append(np.nan)
            continue
        std_val = float(std_arr[idx])
        if std_val == 0.0:
            std_val = 1.0
        std_values.append(std_val)

    std_vals = np.asarray(std_values, dtype=float)
    for col in ["ate_true", "ate_est", "model_ate"]:
        if col in df.columns:
            df[f"{col}_std"] = df[col] / std_vals


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    if not p_values:
        return []
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])
    adjusted_sorted = [0.0] * m
    for rank, (_, p_val) in enumerate(indexed):
        adjusted_sorted[rank] = min(1.0, float(p_val) * (m - rank))
    for rank in range(1, m):
        adjusted_sorted[rank] = max(adjusted_sorted[rank], adjusted_sorted[rank - 1])

    adjusted = [1.0] * m
    for rank, (original_idx, _) in enumerate(indexed):
        adjusted[original_idx] = float(adjusted_sorted[rank])
    return adjusted


def benjamini_hochberg(p_values: Sequence[float]) -> List[float]:
    if not p_values:
        return []
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])
    adjusted_sorted = [0.0] * m
    for rank, (_, p_val) in enumerate(indexed, start=1):
        adjusted_sorted[rank - 1] = min(1.0, float(p_val) * m / rank)
    for rank in range(m - 2, -1, -1):
        adjusted_sorted[rank] = min(adjusted_sorted[rank], adjusted_sorted[rank + 1])

    adjusted = [1.0] * m
    for rank, (original_idx, _) in enumerate(indexed):
        adjusted[original_idx] = float(adjusted_sorted[rank])
    return adjusted


def apply_p_adjustment(
    rows: List[dict],
    p_key: str,
    out_key: str,
    method: str,
    group_keys: Sequence[str],
) -> None:
    method_normalized = str(method).lower()
    if method_normalized in {"none", "raw"}:
        for row in rows:
            p_val = row.get(p_key)
            row[out_key] = None if p_val is None or _is_nan(p_val) else float(p_val)
        return

    if method_normalized in {"holm", "holm_bonferroni"}:
        adjust_fn = holm_bonferroni
    elif method_normalized in {"bh", "fdr_bh", "benjamini_hochberg"}:
        adjust_fn = benjamini_hochberg
    else:
        raise ValueError(
            f"Unknown p-value adjustment method '{method}'. "
            "Supported: none/raw, holm/holm_bonferroni, bh/fdr_bh/benjamini_hochberg."
        )

    grouped_indices = {}
    for idx, row in enumerate(rows):
        p_val = row.get(p_key)
        if p_val is None or _is_nan(p_val):
            row[out_key] = None
            continue
        key = tuple(row.get(group_key) for group_key in group_keys)
        grouped_indices.setdefault(key, []).append(idx)

    for indices in grouped_indices.values():
        p_vals = [float(rows[idx][p_key]) for idx in indices]
        adjusted = adjust_fn(p_vals)
        for idx, adj in zip(indices, adjusted):
            rows[idx][out_key] = float(adj)


def entries_match_on_keys(meta_a: dict, meta_b: dict, match_keys: Sequence[str]) -> Tuple[bool, List[str]]:
    mismatches = []
    for key in match_keys:
        if meta_a.get(key) != meta_b.get(key):
            mismatches.append(key)
    return (len(mismatches) == 0), mismatches


def paired_common_folds(
    series_a: pd.Series,
    series_b: pd.Series,
    meta_a: dict,
    meta_b: dict,
    match_keys: Sequence[str],
) -> Tuple[pd.Index, List[str]]:
    matches, mismatches = entries_match_on_keys(meta_a, meta_b, match_keys)
    if not matches:
        return pd.Index([], dtype=int), mismatches
    return series_a.index.intersection(series_b.index), []


def normalize_intervention_summary_payload(
    payload: list,
    source_path: str,
    strict: bool = True,
) -> Tuple[list, bool]:
    if not isinstance(payload, list):
        raise ValueError(
            f"Invalid intervention summary format in {source_path}: expected a list of rows."
        )

    normalized = []

    for row in payload:
        if not isinstance(row, dict):
            raise ValueError(
                f"Invalid intervention summary row in {source_path}: expected object entries."
            )
        normalized.append(dict(row))

    non_global_rows = []
    for row in normalized:
        fold_value = row.get("fold")
        if fold_value in ("global", None):
            continue
        non_global_rows.append(row)

    if non_global_rows:
        missing_columns = []
        for column in INTERVENTION_REQUIRED_COLUMNS:
            if not any(column in row for row in non_global_rows):
                missing_columns.append(column)

        if missing_columns and strict:
            missing_text = ", ".join(missing_columns)
            raise ValueError(
                "Intervention summary schema is missing confirmed-effect agreement columns "
                f"({missing_text}) in {source_path}. "
                "Re-run interventions with the current confirmed-effect schema."
            )

        for row in normalized:
            for column in INTERVENTION_REQUIRED_COLUMNS:
                row.setdefault(column, None)

    return normalized, False


def serialize_stats_value(value):
    if value is None:
        return None
    if isinstance(value, (np.floating,)):
        if math.isnan(float(value)):
            return None
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def to_serializable_record(row: dict) -> dict:
    return {key: serialize_stats_value(value) for key, value in row.items()}


def _is_nan(value) -> bool:
    try:
        return bool(np.isnan(value))
    except TypeError:
        return False
