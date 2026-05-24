from __future__ import annotations

from typing import List

import matplotlib.lines as mlines
import numpy as np
import pandas as pd

from reporting.figures.common import (
    DATASET_COLORS,
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
    shd_delta_direction_note,
)


FIGURE_TITLE = "Node-Count Robustness Against Ground Truth"
FIGURE_SIZE = (THESIS_TEXT_WIDTH, 5.4)
LAYOUT_RECT = (0.05, 0.07, 0.95, 1.0)
SOURCE_ORDER = ["data_truth", "model_truth"]
SOURCE_LABELS = {
    "data_truth": "Data-level PAG",
    "model_truth": "Model-level PAG",
}
PANEL_LETTERS = {
    ("edge_f1", "absolute"): "A",
    ("edge_f1", "delta"): "B",
    ("endpoint_f1", "absolute"): "C",
    ("endpoint_f1", "delta"): "D",
}
FAMILY_LINESTYLES = ["-", "--", ":", "-."]
FAMILY_MARKERS = ["o", "s", "D", "^", "P", "X"]


def _record_allowed(record: dict, model_id: str | None) -> bool:
    meta = record.get("metadata", {})
    if not bool(meta.get("alignment_enabled", False)):
        return False
    if str(meta.get("alignment_target_mode")) != "observable_state":
        return False
    if model_id and str(meta.get("model")) != model_id:
        return False
    return True


def _family_label(row: pd.Series) -> str:
    benchmark_family = str(row.get("benchmark_family") or "").strip()
    if benchmark_family:
        return _dataset_label(benchmark_family)
    dataset = str(row.get("dataset") or "").strip()
    return _dataset_label(dataset) if dataset else "Unknown family"


def _dedupe_data_truth(df: pd.DataFrame) -> pd.DataFrame:
    return _deduplicate_invariant_rows(
        df,
        subset=["family_label", "num_nodes", "fold", "metric", "source"],
        source_col="source",
        invariant_sources=["data_truth"],
    )


def _mean_ci(summary_df: pd.DataFrame, x_col: str) -> pd.DataFrame:
    grouped = summary_df.groupby(x_col, observed=False)["value"].agg(["mean", "std", "count"]).reset_index()
    if grouped.empty:
        return grouped
    grouped["ci95"] = 1.96 * grouped["std"].fillna(0.0) / np.sqrt(grouped["count"].clip(lower=1))
    grouped["lower"] = grouped["mean"] - grouped["ci95"]
    grouped["upper"] = grouped["mean"] + grouped["ci95"]
    return grouped.sort_values(x_col)


def figure_structural_node_count_robustness(spec: dict, records: List[dict], output_dir: str) -> dict:
    set_reporting_theme(layout_profile="paper")
    title_generated = str(spec.get("title", FIGURE_TITLE))
    model_id = spec.get("model_id")
    predicate = lambda record: _record_allowed(record, model_id=model_id)

    truth_df = build_structural_truth_frame(
        records,
        metrics=PRIMARY_STRUCTURAL_METRICS,
        predicate=predicate,
        include_sources=SOURCE_ORDER,
    )
    truth_df = truth_df.dropna(subset=["num_nodes", "value"]).copy()
    truth_df["family_label"] = truth_df.apply(_family_label, axis=1)
    truth_df = _dedupe_data_truth(truth_df)

    delta_df = build_structural_truth_pairs(
        records,
        metrics=PRIMARY_STRUCTURAL_METRICS,
        predicate=predicate,
    )
    delta_df = delta_df.dropna(subset=["num_nodes", "delta"]).copy()
    delta_df["family_label"] = delta_df.apply(_family_label, axis=1)
    _ensure_non_empty(truth_df, spec["id"])
    _ensure_non_empty(delta_df, spec["id"])

    family_order = sorted(truth_df["family_label"].dropna().astype(str).unique().tolist())
    family_dataset = (
        truth_df[["family_label", "dataset"]]
        .dropna()
        .drop_duplicates(subset=["family_label"])
        .set_index("family_label")["dataset"]
        .to_dict()
    )
    family_colors = {
        family_label: DATASET_COLORS.get(str(family_dataset.get(family_label)), plt.get_cmap("tab10")(idx % 10))
        for idx, family_label in enumerate(family_order)
    }
    x_values = np.array(sorted({float(value) for value in truth_df["num_nodes"].tolist()}), dtype=float)
    if x_values.size == 0:
        raise ValueError(f"{spec['id']}: expected node-count values.")
    pad = max(0.35, 0.35 * (float(np.min(np.diff(x_values))) if x_values.size > 1 else 1.0))
    x_limits = (float(np.min(x_values)) - pad, float(np.max(x_values)) + pad)

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

        for family_index, family_label in enumerate(family_order):
            marker = FAMILY_MARKERS[family_index % len(FAMILY_MARKERS)]
            linestyle = FAMILY_LINESTYLES[family_index % len(FAMILY_LINESTYLES)]
            family_delta_df = metric_delta_df[metric_delta_df["family_label"] == family_label].copy()
            if not family_delta_df.empty:
                summary = _mean_ci(family_delta_df.rename(columns={"delta": "value"}), x_col="num_nodes")
                color = family_colors[family_label]
                delta_ax.fill_between(
                    summary["num_nodes"].to_numpy(dtype=float),
                    summary["lower"].to_numpy(dtype=float),
                    summary["upper"].to_numpy(dtype=float),
                    color=color,
                    alpha=0.14,
                    linewidth=0.0,
                    zorder=1,
                )
                delta_ax.plot(
                    summary["num_nodes"].to_numpy(dtype=float),
                    summary["mean"].to_numpy(dtype=float),
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=4.0,
                    linewidth=1.35,
                    zorder=3,
                )
                rho, p_raw, n = _spearman_stats(
                    family_delta_df["num_nodes"].to_numpy(dtype=float),
                    family_delta_df["delta"].to_numpy(dtype=float),
                )
                stats_rows.append(
                    {
                        "figure": spec["id"],
                        "test": "spearman",
                        "panel": "delta",
                        "metric": metric,
                        "family_label": family_label,
                        "source": "delta",
                        "n": n,
                        "rho": rho,
                        "p_raw": p_raw,
                    }
                )

            for source in SOURCE_ORDER:
                subset = metric_truth_df[
                    (metric_truth_df["family_label"] == family_label)
                    & (metric_truth_df["source"] == source)
                ].copy()
                if subset.empty:
                    continue
                summary = _mean_ci(subset, x_col="num_nodes")
                color = STRUCTURAL_SOURCE_COLORS[source]
                absolute_ax.fill_between(
                    summary["num_nodes"].to_numpy(dtype=float),
                    summary["lower"].to_numpy(dtype=float),
                    summary["upper"].to_numpy(dtype=float),
                    color=color,
                    alpha=0.09,
                    linewidth=0.0,
                    zorder=1,
                )
                absolute_ax.plot(
                    summary["num_nodes"].to_numpy(dtype=float),
                    summary["mean"].to_numpy(dtype=float),
                    color=color,
                    linestyle=linestyle,
                    marker=marker,
                    markersize=4.0,
                    linewidth=1.4,
                    zorder=3,
                )
                rho, p_raw, n = _spearman_stats(
                    subset["num_nodes"].to_numpy(dtype=float),
                    subset["value"].to_numpy(dtype=float),
                )
                stats_rows.append(
                    {
                        "figure": spec["id"],
                        "test": "spearman",
                        "panel": "absolute",
                        "metric": metric,
                        "family_label": family_label,
                        "source": source,
                        "n": n,
                        "rho": rho,
                        "p_raw": p_raw,
                    }
                )

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
        delta_ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.9)
        _add_panel_letter(absolute_ax, PANEL_LETTERS[(metric, "absolute")])
        _add_panel_letter(delta_ax, PANEL_LETTERS[(metric, "delta")], x=0.0)

        if metric_index == len(PRIMARY_STRUCTURAL_METRICS) - 1:
            absolute_ax.set_xlabel("Node count")
            delta_ax.set_xlabel("Node count")
            absolute_ax.set_xticks(x_values, [str(int(value)) for value in x_values])
            delta_ax.set_xticks(x_values, [str(int(value)) for value in x_values])
        else:
            absolute_ax.set_xlabel("")
            delta_ax.set_xlabel("")
            absolute_ax.tick_params(labelbottom=False)
            delta_ax.tick_params(labelbottom=False)

    _bh_adjust_rows(stats_rows)

    legend_handles = [
        mlines.Line2D([], [], color=STRUCTURAL_SOURCE_COLORS[source], linewidth=1.6, marker="o", label=SOURCE_LABELS[source])
        for source in SOURCE_ORDER
    ]
    for family_index, family_label in enumerate(family_order):
        legend_handles.append(
            mlines.Line2D(
                [],
                [],
                color=family_colors[family_label],
                linestyle=FAMILY_LINESTYLES[family_index % len(FAMILY_LINESTYLES)],
                marker=FAMILY_MARKERS[family_index % len(FAMILY_MARKERS)],
                linewidth=1.4,
                label=family_label,
            )
        )
    _move_legend_bottom(fig, title="Truth branch and benchmark family", ncol=min(4, len(legend_handles)), extra_handles=legend_handles)

    caption_lines = [
        "Truth-anchored node-count robustness across expanded benchmark families under observable-state alignment.",
        f"Left panels show absolute data-level and model-level scores against benchmark truth; right panels show the matched truth-based delta, model minus data, for {', '.join(_metric_label(metric) for metric in PRIMARY_STRUCTURAL_METRICS)}.",
        f"Benchmark families represented: {', '.join(family_order)}. Spearman trend tests are reported per family-series combination and Benjamini-Hochberg corrected within the figure family.",
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
            "family_order": family_order,
            "model_id": model_id,
            "panel_row_counts": panel_row_counts,
        },
    )
