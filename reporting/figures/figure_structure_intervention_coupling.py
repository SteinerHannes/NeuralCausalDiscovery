from __future__ import annotations

from typing import List

import matplotlib.lines as mlines
import numpy as np
import pandas as pd

from reporting.figures.common import (
    DATASET_COLORS,
    DATASET_MARKERS,
    DATASET_ORDER,
    _bh_adjust_rows,
    _concise_stat_line,
    _dataset_label,
    _ensure_non_empty,
    _move_legend_bottom,
    _record_detection_summary,
    _save_figure,
    _spearman_stats,
    plt,
    set_reporting_theme,
)
from reporting.figures.structural_truth_utils import (
    build_structural_truth_pairs,
    dataset_order_present,
)


FIGURE_TITLE = "Delta-to-delta structure-to-intervention coupling"
STRUCTURE_METRICS = [
    ("edge_f1", "Δ Edge $F_1$ (model - data)"),
    ("endpoint_f1", "Δ Endpoint-mark $F_1$ (model - data)"),
    ("shd_strict", "Δ SHD (model - data)"),
]
INTERVENTION_METRICS = [
    ("f1_delta", "Δ Output confirmed intervention-effect $F_1$ (model - PAG)"),
    ("std_mae_delta", "Δ standardized MAE (model - DoWhy)"),
]


def _record_allowed(record: dict, dataset_ids: set[str]) -> bool:
    meta = record.get("metadata", {})
    if not bool(meta.get("alignment_enabled", False)):
        return False
    if str(meta.get("alignment_target_mode")) != "observable_state":
        return False
    if dataset_ids and str(meta.get("dataset")) not in dataset_ids:
        return False
    return True


def _fold_level_intervention_delta(record: dict) -> pd.DataFrame:
    dataset = record.get("metadata", {}).get("dataset")
    model = record.get("metadata", {}).get("model")
    record_name = record.get("name")

    rows = []
    detection_df = _record_detection_summary(record, role="output")
    if not detection_df.empty:
        for _, row in detection_df.iterrows():
            if pd.notna(row.get("model_f1")) and pd.notna(row.get("pag_f1")):
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "record_name": record_name,
                        "fold": int(row.get("fold")),
                        "intervention_metric": "f1_delta",
                        "intervention_label": "Δ Output confirmed intervention-effect $F_1$ (model - PAG)",
                        "intervention_delta": float(row.get("model_f1") - row.get("pag_f1")),
                    }
                )

    pair_df = record.get("pair_df")
    required_cols = {"fold", "outcome_role", "effect_present", "ate_true_std", "ate_est_std", "model_ate_std"}
    if pair_df is not None and not pair_df.empty and required_cols.issubset(pair_df.columns):
        effect_present = pair_df["effect_present"].fillna(False).astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})
        output_df = pair_df[
            (pair_df["outcome_role"] == "output")
            & effect_present
        ].copy()
        for fold, fold_df in output_df.groupby("fold"):
            valid = fold_df[["ate_true_std", "ate_est_std", "model_ate_std"]].dropna()
            if valid.empty:
                continue
            dowhy_mae = float(np.mean(np.abs(valid["ate_est_std"].to_numpy(dtype=float) - valid["ate_true_std"].to_numpy(dtype=float))))
            model_mae = float(np.mean(np.abs(valid["model_ate_std"].to_numpy(dtype=float) - valid["ate_true_std"].to_numpy(dtype=float))))
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "record_name": record_name,
                    "fold": int(fold),
                    "intervention_metric": "std_mae_delta",
                    "intervention_label": "Δ standardized MAE (model - DoWhy)",
                    "intervention_delta": float(model_mae - dowhy_mae),
                }
            )

    return pd.DataFrame(rows)


def _axis_limits(values: np.ndarray, bounded: bool) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-1.0, 1.0)
    if bounded:
        max_abs = min(1.0, float(np.max(np.abs(finite))) + 0.06)
        return (-max_abs, max_abs)
    max_abs = float(np.max(np.abs(finite)))
    span = max_abs + max(0.05, 0.12 * max_abs)
    return (-span, span)


def figure_structure_intervention_coupling(spec: dict, records: List[dict], output_dir: str) -> dict:
    set_reporting_theme(layout_profile="paper")
    title_generated = str(spec.get("title", FIGURE_TITLE))
    dataset_ids = set(spec.get("dataset_order", DATASET_ORDER))
    predicate = lambda record: _record_allowed(record, dataset_ids=dataset_ids)

    structural_df = build_structural_truth_pairs(records, predicate=predicate)
    structural_df = structural_df.dropna(subset=["delta"]).copy()
    intervention_rows = []
    for record in records:
        if not predicate(record):
            continue
        intervention_rows.append(_fold_level_intervention_delta(record))
    intervention_df = pd.concat(intervention_rows, ignore_index=True) if intervention_rows else pd.DataFrame()
    intervention_df = intervention_df.dropna(subset=["intervention_delta"]).copy()
    _ensure_non_empty(structural_df, spec["id"])
    _ensure_non_empty(intervention_df, spec["id"])

    rows = []
    for record in records:
        if not predicate(record):
            continue
        meta = record.get("metadata", {})
        dataset = meta.get("dataset")
        model = meta.get("model")
        record_name = record.get("name")
        struct_rows = structural_df[
            (structural_df["dataset"] == dataset)
            & (structural_df["model"] == model)
            & (structural_df["record_name"] == record_name)
        ].copy()
        intervention_rows_df = intervention_df[
            (intervention_df["dataset"] == dataset)
            & (intervention_df["model"] == model)
            & (intervention_df["record_name"] == record_name)
        ].copy()
        if struct_rows.empty or intervention_rows_df.empty:
            continue
        for fold in sorted(set(struct_rows["fold"]).intersection(set(intervention_rows_df["fold"]))):
            fold_struct = struct_rows[struct_rows["fold"] == fold]
            fold_int = intervention_rows_df[intervention_rows_df["fold"] == fold]
            for _, s_row in fold_struct.iterrows():
                for _, i_row in fold_int.iterrows():
                    rows.append(
                        {
                            "dataset": dataset,
                            "dataset_label": _dataset_label(dataset),
                            "model": model,
                            "fold": int(fold),
                            "structure_metric": s_row["metric"],
                            "structure_label": dict(STRUCTURE_METRICS)[s_row["metric"]],
                            "structure_delta": float(s_row["delta"]),
                            "intervention_metric": i_row["intervention_metric"],
                            "intervention_label": i_row["intervention_label"],
                            "intervention_delta": float(i_row["intervention_delta"]),
                        }
                    )

    df = pd.DataFrame(rows).dropna(subset=["structure_delta", "intervention_delta"])
    _ensure_non_empty(df, spec["id"])
    dataset_order = dataset_order_present(df)

    fig, axes = plt.subplots(
        len(STRUCTURE_METRICS),
        len(INTERVENTION_METRICS),
        figsize=(6.313, 7.7),
        dpi=170,
        sharex=False,
        sharey=False,
    )

    stats_rows = []
    panel_annotations = []
    panel_row_counts = {}
    for row_index, (structure_metric, structure_label) in enumerate(STRUCTURE_METRICS):
        for col_index, (intervention_metric, intervention_label) in enumerate(INTERVENTION_METRICS):
            ax = axes[row_index, col_index]
            panel_df = df[
                (df["structure_metric"] == structure_metric)
                & (df["intervention_metric"] == intervention_metric)
            ].copy()
            panel_row_counts[f"{structure_metric}__{intervention_metric}"] = int(len(panel_df))
            for dataset in dataset_order:
                dataset_df = panel_df[panel_df["dataset"] == dataset]
                if dataset_df.empty:
                    continue
                ax.scatter(
                    dataset_df["structure_delta"].to_numpy(dtype=float),
                    dataset_df["intervention_delta"].to_numpy(dtype=float),
                    color=DATASET_COLORS.get(dataset, "#6e6e6e"),
                    marker=DATASET_MARKERS.get(dataset, "o"),
                    alpha=0.72,
                    s=28,
                    linewidths=0.25,
                    edgecolors="#f1f1f1",
                )

            ax.axvline(0.0, color="#666666", linestyle="--", linewidth=0.9, zorder=0)
            ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.9, zorder=0)
            ax.set_title(f"{structure_label} | {intervention_label}")
            ax.set_xlabel(structure_label)
            ax.set_ylabel(intervention_label)
            ax.set_xlim(*_axis_limits(panel_df["structure_delta"].to_numpy(dtype=float), bounded=(structure_metric != "shd_strict")))
            ax.set_ylim(*_axis_limits(panel_df["intervention_delta"].to_numpy(dtype=float), bounded=(intervention_metric == "f1_delta")))
            ax.grid(alpha=0.28)

            rho, p_raw, n = _spearman_stats(
                panel_df["structure_delta"].to_numpy(dtype=float),
                panel_df["intervention_delta"].to_numpy(dtype=float),
            )
            stats_row = {
                "figure": spec["id"],
                "test": "spearman",
                "structure_metric": structure_metric,
                "intervention_metric": intervention_metric,
                "n": n,
                "rho": rho,
                "p_raw": p_raw,
            }
            stats_rows.append(stats_row)
            panel_annotations.append((ax, stats_row))

    _bh_adjust_rows(stats_rows)
    for ax, stats_row in panel_annotations:
        ax.text(
            0.98,
            0.02,
            _concise_stat_line(stats_row),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=7.4,
            bbox={"facecolor": "white", "edgecolor": "#b7b7b7", "alpha": 0.82, "boxstyle": "round,pad=0.2"},
        )
    legend_handles = [
        mlines.Line2D(
            [],
            [],
            color=DATASET_COLORS.get(dataset, "#6e6e6e"),
            marker=DATASET_MARKERS.get(dataset, "o"),
            linestyle="None",
            markersize=5.2,
            label=_dataset_label(dataset),
        )
        for dataset in dataset_order
    ]
    _move_legend_bottom(fig, title="Dataset", ncol=min(4, len(legend_handles)), extra_handles=legend_handles)

    caption_lines = [
        "Delta-to-delta coupling between structural recovery and intervention fidelity under observable-state alignment.",
        "Structural axes show the fold-matched truth-based delta `model vs truth - data vs truth`; intervention axes show fold-matched deltas for output confirmed intervention-effect $F_1$ (`model - PAG`) and standardized MAE (`model - DoWhy`).",
        "Positive $F_1$ deltas indicate better model-side intervention detection than the PAG branch. Negative standardized-MAE deltas indicate better model-side effect calibration than the DoWhy reference.",
        "Spearman correlations are reported per panel and Benjamini-Hochberg corrected within the figure family.",
    ]

    source_files = [path for record in records for path in record.get("source_files", [])]
    return _save_figure(
        fig,
        axes,
        spec["id"],
        output_dir,
        stats_rows,
        data_row_count=len(df),
        source_files=source_files,
        title_generated=title_generated,
        caption_lines=caption_lines,
        legend_mode="outside_bottom_shared",
        annotation_mode="dense_caption_only",
        layout_profile="paper",
        layout_rect=(0.0, 0.08, 1.0, 0.98),
        extra={
            "dataset_order": dataset_order,
            "panel_row_counts": panel_row_counts,
        },
    )
