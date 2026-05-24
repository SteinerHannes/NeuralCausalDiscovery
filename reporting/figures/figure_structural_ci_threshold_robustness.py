from __future__ import annotations

from typing import List

import matplotlib.lines as mlines
import numpy as np
import pandas as pd

from reporting.figures.common import (
    DATASET_COLORS,
    DATASET_MARKERS,
    DATASET_ORDER,
    STRUCTURAL_SOURCE_COLORS,
    STRUCTURAL_TRUTH_DELTA_AXIS_LABEL,
    THESIS_TEXT_WIDTH,
    _add_panel_letter,
    _apply_bounded_metric_y_axis,
    _apply_right_delta_y_axis,
    _bh_adjust_rows,
    _dataset_label,
    _deduplicate_invariant_rows,
    _ensure_non_empty,
    _metric_label,
    _move_legend_bottom,
    _save_figure,
    _symmetric_delta_limits,
    _spearman_stats,
    plt,
    set_reporting_theme,
)
from reporting.figures.structural_truth_utils import (
    PRIMARY_STRUCTURAL_METRICS,
    build_structural_truth_frame,
    build_structural_truth_pairs,
    dataset_order_present,
    shd_delta_direction_note,
)


FIGURE_TITLE = "CI-Threshold Robustness Against Ground Truth"
FIGURE_SIZE = (THESIS_TEXT_WIDTH, 5.4)
LAYOUT_RECT = (0.05, 0.07, 0.95, 1.0)
SOURCE_ORDER = ["data_truth", "model_truth"]
SOURCE_LABELS = {
    "data_truth": "Data-level PAG",
    "model_truth": "Model-level PAG",
}
DATASET_LINESTYLES = {
    "LinearSEMDataset": "-",
    "ElectricalCircuitData": "--",
}
PANEL_LETTERS = {
    ("edge_f1", "absolute"): "A",
    ("edge_f1", "delta"): "B",
    ("endpoint_f1", "absolute"): "C",
    ("endpoint_f1", "delta"): "D",
}


def _record_allowed(record: dict, model_id: str, dataset_ids: set[str]) -> bool:
    meta = record.get("metadata", {})
    if not bool(meta.get("alignment_enabled", False)):
        return False
    if str(meta.get("alignment_target_mode")) != "observable_state":
        return False
    if model_id and str(meta.get("model")) != model_id:
        return False
    if dataset_ids and str(meta.get("dataset")) not in dataset_ids:
        return False
    return True


def _dedupe_data_truth(df: pd.DataFrame) -> pd.DataFrame:
    return _deduplicate_invariant_rows(
        df,
        subset=["dataset", "ci_threshold", "fold", "metric", "source"],
        source_col="source",
        invariant_sources=["data_truth"],
    )


def _mean_ci(summary_df: pd.DataFrame) -> pd.DataFrame:
    grouped = summary_df.groupby("ci_threshold", observed=False)["value"].agg(["mean", "std", "count"]).reset_index()
    if grouped.empty:
        return grouped
    grouped["ci95"] = 1.96 * grouped["std"].fillna(0.0) / np.sqrt(grouped["count"].clip(lower=1))
    grouped["lower"] = grouped["mean"] - grouped["ci95"]
    grouped["upper"] = grouped["mean"] + grouped["ci95"]
    return grouped.sort_values("ci_threshold")


def figure_structural_ci_threshold_robustness(spec: dict, records: List[dict], output_dir: str) -> dict:
    set_reporting_theme(layout_profile="paper")
    title_generated = str(spec.get("title", FIGURE_TITLE))
    dataset_ids = set(spec.get("dataset_order", DATASET_ORDER))
    model_id = str(spec.get("model_id", "ShallowMLP"))

    predicate = lambda record: _record_allowed(record, model_id=model_id, dataset_ids=dataset_ids)
    truth_df = build_structural_truth_frame(
        records,
        metrics=PRIMARY_STRUCTURAL_METRICS,
        predicate=predicate,
        include_sources=SOURCE_ORDER,
    )
    truth_df = truth_df.dropna(subset=["ci_threshold", "value"]).copy()
    truth_df = _dedupe_data_truth(truth_df)

    delta_df = build_structural_truth_pairs(
        records,
        metrics=PRIMARY_STRUCTURAL_METRICS,
        predicate=predicate,
    )
    delta_df = delta_df.dropna(subset=["ci_threshold", "delta"]).copy()
    _ensure_non_empty(truth_df, spec["id"])
    _ensure_non_empty(delta_df, spec["id"])

    dataset_order = dataset_order_present(truth_df)
    dataset_labels = [_dataset_label(dataset) for dataset in dataset_order]
    x_values = np.array(sorted({float(value) for value in truth_df["ci_threshold"].tolist() if float(value) > 0.0}), dtype=float)
    if x_values.size == 0:
        raise ValueError(f"{spec['id']}: expected strictly positive CI thresholds.")
    log_values = np.log10(x_values)
    if log_values.size > 1:
        pad_log = max(0.03, 0.35 * float(np.min(np.diff(log_values))))
    else:
        pad_log = 0.08
    x_limits = (10 ** (float(np.min(log_values)) - pad_log), 10 ** (float(np.max(log_values)) + pad_log))
    x_labels = [np.format_float_positional(value, trim="-") for value in x_values]

    fig, axes = plt.subplots(
        len(PRIMARY_STRUCTURAL_METRICS),
        2,
        figsize=FIGURE_SIZE,
        sharex="col",
    )
    fig.suptitle(title_generated, y=0.995)
    shared_delta_ylim = _symmetric_delta_limits(delta_df["delta"].to_numpy(dtype=float))

    stats_rows = []
    panel_row_counts = {"absolute": {}, "delta": {}}
    for metric_index, metric in enumerate(PRIMARY_STRUCTURAL_METRICS):
        absolute_ax = axes[metric_index, 0]
        delta_ax = axes[metric_index, 1]
        metric_truth_df = truth_df[truth_df["metric"] == metric].copy()
        metric_delta_df = delta_df[delta_df["metric"] == metric].copy()
        panel_row_counts["absolute"][metric] = int(len(metric_truth_df))
        panel_row_counts["delta"][metric] = int(len(metric_delta_df))

        for dataset in dataset_order:
            dataset_label = _dataset_label(dataset)
            marker = DATASET_MARKERS.get(dataset, "o")
            linestyle = DATASET_LINESTYLES.get(dataset, "-")
            dataset_delta_df = metric_delta_df[metric_delta_df["dataset"] == dataset].copy()
            if not dataset_delta_df.empty:
                summary = _mean_ci(dataset_delta_df.rename(columns={"delta": "value"}))
                delta_ax.fill_between(
                    summary["ci_threshold"].to_numpy(dtype=float),
                    summary["lower"].to_numpy(dtype=float),
                    summary["upper"].to_numpy(dtype=float),
                    color=DATASET_COLORS.get(dataset, "#6e6e6e"),
                    alpha=0.14,
                    linewidth=0.0,
                    zorder=1,
                )
                delta_ax.plot(
                    summary["ci_threshold"].to_numpy(dtype=float),
                    summary["mean"].to_numpy(dtype=float),
                    color=DATASET_COLORS.get(dataset, "#6e6e6e"),
                    linestyle=linestyle,
                    marker=marker,
                    markersize=4.0,
                    linewidth=1.35,
                    zorder=3,
                )
            rho, p_raw, n = _spearman_stats(
                np.log10(dataset_delta_df["ci_threshold"].to_numpy(dtype=float)),
                dataset_delta_df["delta"].to_numpy(dtype=float),
            )
            stats_rows.append(
                {
                    "figure": spec["id"],
                    "test": "spearman",
                    "panel": "delta",
                    "metric": metric,
                    "dataset": dataset,
                    "source": "delta",
                    "n": n,
                    "rho": rho,
                    "p_raw": p_raw,
                }
            )

            for source in SOURCE_ORDER:
                subset = metric_truth_df[
                    (metric_truth_df["dataset"] == dataset)
                    & (metric_truth_df["source"] == source)
                ].copy()
                if subset.empty:
                    continue
                summary = _mean_ci(subset)
                color = STRUCTURAL_SOURCE_COLORS[source]
                absolute_ax.fill_between(
                    summary["ci_threshold"].to_numpy(dtype=float),
                    summary["lower"].to_numpy(dtype=float),
                    summary["upper"].to_numpy(dtype=float),
                    color=color,
                    alpha=0.09,
                    linewidth=0.0,
                    zorder=1,
                )
                absolute_ax.plot(
                    summary["ci_threshold"].to_numpy(dtype=float),
                    summary["mean"].to_numpy(dtype=float),
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=4.0,
                    linewidth=1.4,
                    zorder=3,
                )
                rho, p_raw, n = _spearman_stats(
                    np.log10(subset["ci_threshold"].to_numpy(dtype=float)),
                    subset["value"].to_numpy(dtype=float),
                )
                stats_rows.append(
                    {
                        "figure": spec["id"],
                        "test": "spearman",
                        "panel": "absolute",
                        "metric": metric,
                        "dataset": dataset,
                        "source": source,
                        "n": n,
                        "rho": rho,
                        "p_raw": p_raw,
                    }
                )

        absolute_ax.set_xscale("log")
        delta_ax.set_xscale("log")
        absolute_ax.set_title("Absolute truth score" if metric_index == 0 else "")
        delta_ax.set_title("Model-data delta" if metric_index == 0 else "")
        absolute_ax.set_xlim(*x_limits)
        delta_ax.set_xlim(*x_limits)
        _apply_bounded_metric_y_axis(absolute_ax)
        absolute_ax.set_ylabel(_metric_label(metric))
        delta_ax.set_ylabel(STRUCTURAL_TRUTH_DELTA_AXIS_LABEL)
        _apply_right_delta_y_axis(delta_ax, shared_delta_ylim)
        absolute_ax.grid(axis="y", alpha=0.3)
        delta_ax.grid(axis="y", alpha=0.3)
        delta_ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.9, zorder=0)
        _add_panel_letter(absolute_ax, PANEL_LETTERS[(metric, "absolute")])
        _add_panel_letter(delta_ax, PANEL_LETTERS[(metric, "delta")], x=0.0)

        if metric_index == len(PRIMARY_STRUCTURAL_METRICS) - 1:
            absolute_ax.set_xlabel("CI significance threshold")
            delta_ax.set_xlabel("CI significance threshold")
            absolute_ax.set_xticks(x_values, x_labels, rotation=35, ha="right")
            delta_ax.set_xticks(x_values, x_labels, rotation=35, ha="right")
        else:
            absolute_ax.set_xlabel("")
            delta_ax.set_xlabel("")
            absolute_ax.tick_params(labelbottom=False)
            delta_ax.tick_params(labelbottom=False)

    _bh_adjust_rows(stats_rows)

    legend_handles = [
        mlines.Line2D([], [], color=STRUCTURAL_SOURCE_COLORS[source], linewidth=1.8, label=SOURCE_LABELS[source])
        for source in SOURCE_ORDER
    ]
    for dataset in dataset_order:
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=DATASET_COLORS.get(dataset, "#6e6e6e"),
                linestyle=DATASET_LINESTYLES.get(dataset, "-"),
                marker=DATASET_MARKERS.get(dataset, "o"),
                linewidth=1.4,
                label=_dataset_label(dataset),
            )
        )
    _move_legend_bottom(fig, title="Truth branch and dataset", ncol=min(4, len(legend_handles)), extra_handles=legend_handles)

    caption_lines = [
        "Truth-anchored structural robustness across the observable-state CI-threshold sweep.",
        f"Left panels show absolute data-level and model-level trends against benchmark truth; right panels show the matched fold-level delta, model minus data, for {', '.join(_metric_label(metric) for metric in PRIMARY_STRUCTURAL_METRICS)}.",
        f"Datasets represented: {', '.join(dataset_labels)}. Trend statistics use Spearman correlation over log-scaled thresholds and are Benjamini-Hochberg corrected within the figure family.",
        shd_delta_direction_note("shd_strict"),
    ]

    source_files = [path for record in records for path in record.get("source_files", [])]
    return _save_figure(
        fig,
        axes,
        spec["id"],
        output_dir,
        stats_rows,
        data_row_count=int(len(truth_df) + len(delta_df)),
        source_files=source_files,
        title_generated=title_generated,
        caption_lines=caption_lines,
        legend_mode="outside_bottom_shared",
        annotation_mode="dense_caption_only",
        layout_profile="paper",
        layout_rect=LAYOUT_RECT,
        extra={
            "dataset_order": dataset_order,
            "model_id": model_id,
            "panel_row_counts": panel_row_counts,
        },
    )
