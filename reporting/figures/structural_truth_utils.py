from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import pandas as pd

from reporting.figures.common import (
    ALIGNMENT_TARGET_LABELS,
    ALIGNMENT_TARGET_ORDER,
    DATASET_ORDER,
    MODEL_ORDER,
    _alignment_target_label,
    _alignment_target_variant,
    _dataset_label,
    _deduplicate_invariant_rows,
    _metric_label,
    _model_label,
)


PRIMARY_STRUCTURAL_METRICS = ["edge_f1", "endpoint_f1"]
SECONDARY_STRUCTURAL_METRICS = ["edge_precision", "edge_recall", "shd_strict", "shd_partial"]
STRUCTURAL_SOURCE_LABELS = {
    "data_truth": "Data vs truth",
    "model_truth": "Model vs truth",
}


def _record_allowed(record: dict, predicate: Optional[Callable[[dict], bool]]) -> bool:
    return True if predicate is None else bool(predicate(record))


def _fold_rows_from_frame(
    frame: Optional[pd.DataFrame],
    meta: dict,
    metrics: Sequence[str],
    source: str,
) -> list[dict]:
    if frame is None or frame.empty:
        return []

    rows = []
    for fold, metric_row in frame.iterrows():
        try:
            fold_value = int(fold)
        except (TypeError, ValueError):
            continue
        for metric in metrics:
            if metric not in metric_row:
                continue
            value = metric_row.get(metric)
            if value is None or pd.isna(value):
                continue
            rows.append(
                {
                    "dataset": meta.get("dataset"),
                    "dataset_label": _dataset_label(meta.get("dataset")),
                    "model": meta.get("model"),
                    "model_label": _model_label(meta.get("model")),
                    "fold": fold_value,
                    "metric": metric,
                    "metric_label": _metric_label(metric),
                    "source": source,
                    "source_label": STRUCTURAL_SOURCE_LABELS[source],
                    "value": float(value),
                    "alignment_enabled": bool(meta.get("alignment_enabled", False)),
                    "alignment_target_mode": meta.get("alignment_target_mode"),
                    "alignment_variant": _alignment_target_variant(meta),
                    "alignment_variant_label": _alignment_target_label(_alignment_target_variant(meta)),
                    "extraction_method": meta.get("extraction_method"),
                    "sample_size": meta.get("sample_size"),
                    "baseline_sample_size": meta.get("baseline_sample_size"),
                    "ci_test": meta.get("ci_test"),
                    "ci_threshold": meta.get("ci_threshold"),
                    "num_nodes": meta.get("num_nodes"),
                    "benchmark_family": meta.get("benchmark_family"),
                    "topology": meta.get("topology"),
                    "noise_distribution": meta.get("noise_distribution"),
                    "mechanism_family": meta.get("mechanism_family"),
                    "record_name": meta.get("record_name"),
                }
            )
    return rows


def build_structural_truth_frame(
    records: Sequence[dict],
    metrics: Sequence[str] = PRIMARY_STRUCTURAL_METRICS,
    predicate: Optional[Callable[[dict], bool]] = None,
    include_sources: Sequence[str] = ("data_truth", "model_truth"),
    dedupe_data: bool = False,
) -> pd.DataFrame:
    rows = []
    include_sources = tuple(include_sources)
    for record in records:
        if not _record_allowed(record, predicate):
            continue
        meta = dict(record.get("metadata", {}))
        meta["record_name"] = record.get("name")
        if "model_truth" in include_sources:
            rows.extend(_fold_rows_from_frame(record.get("performance_df"), meta, metrics, source="model_truth"))
        if "data_truth" in include_sources:
            rows.extend(_fold_rows_from_frame(record.get("baseline_df"), meta, metrics, source="data_truth"))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    if dedupe_data:
        df = _deduplicate_invariant_rows(
            df,
            subset=["dataset", "fold", "metric", "source"],
            source_col="source",
            invariant_sources=["data_truth"],
        )
    return df


def build_structural_truth_pairs(
    records: Sequence[dict],
    metrics: Sequence[str] = PRIMARY_STRUCTURAL_METRICS,
    predicate: Optional[Callable[[dict], bool]] = None,
) -> pd.DataFrame:
    rows = []
    for record in records:
        if not _record_allowed(record, predicate):
            continue
        meta = dict(record.get("metadata", {}))
        performance_df = record.get("performance_df")
        baseline_df = record.get("baseline_df")
        if performance_df is None or baseline_df is None or performance_df.empty or baseline_df.empty:
            continue
        common_folds = baseline_df.index.intersection(performance_df.index)
        for fold in common_folds:
            try:
                fold_value = int(fold)
            except (TypeError, ValueError):
                continue
            for metric in metrics:
                if metric not in performance_df.columns or metric not in baseline_df.columns:
                    continue
                model_value = performance_df.loc[fold, metric]
                data_value = baseline_df.loc[fold, metric]
                if model_value is None or data_value is None or pd.isna(model_value) or pd.isna(data_value):
                    continue
                delta = float(model_value) - float(data_value)
                rows.append(
                    {
                        "dataset": meta.get("dataset"),
                        "dataset_label": _dataset_label(meta.get("dataset")),
                        "model": meta.get("model"),
                        "model_label": _model_label(meta.get("model")),
                        "fold": fold_value,
                        "metric": metric,
                        "metric_label": _metric_label(metric),
                        "data_truth": float(data_value),
                        "model_truth": float(model_value),
                        "delta": delta,
                        "alignment_enabled": bool(meta.get("alignment_enabled", False)),
                        "alignment_target_mode": meta.get("alignment_target_mode"),
                        "alignment_variant": _alignment_target_variant(meta),
                        "alignment_variant_label": _alignment_target_label(_alignment_target_variant(meta)),
                        "extraction_method": meta.get("extraction_method"),
                        "sample_size": meta.get("sample_size"),
                        "baseline_sample_size": meta.get("baseline_sample_size"),
                        "ci_test": meta.get("ci_test"),
                        "ci_threshold": meta.get("ci_threshold"),
                        "num_nodes": meta.get("num_nodes"),
                        "benchmark_family": meta.get("benchmark_family"),
                        "topology": meta.get("topology"),
                        "noise_distribution": meta.get("noise_distribution"),
                        "mechanism_family": meta.get("mechanism_family"),
                        "record_name": record.get("name"),
                    }
                )
    return pd.DataFrame(rows)


def dataset_order_present(df: pd.DataFrame, dataset_col: str = "dataset") -> list[str]:
    present = set(df[dataset_col].dropna().astype(str).tolist())
    ordered = [dataset for dataset in DATASET_ORDER if dataset in present]
    if ordered:
        return ordered
    return sorted(present)


def model_order_present(df: pd.DataFrame, model_col: str = "model") -> list[str]:
    present = set(df[model_col].dropna().astype(str).tolist())
    ordered = [model for model in MODEL_ORDER if model in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def alignment_variant_order_present(df: pd.DataFrame, variant_col: str = "alignment_variant") -> list[str]:
    present = set(df[variant_col].dropna().astype(str).tolist())
    ordered = [variant for variant in ALIGNMENT_TARGET_ORDER if variant in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


def shd_delta_direction_note(metric: str) -> str:
    if metric == "shd_strict":
        return "Negative values indicate that the model branch is closer to benchmark truth."
    return "Positive values indicate that the model branch is closer to benchmark truth."
