from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from reporting.figures.common import (
    CI_TEST_MODE_COLORS,
    DATASET_ORDER,
    STRUCTURAL_SOURCE_LABEL_COLORS,
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
    _wilcoxon_paired_details,
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


FIGURE_TITLE = "CI-Test Mismatch Sensitivity Against Ground Truth"
FIGURE_SIZE = (THESIS_TEXT_WIDTH, 5.6)
LAYOUT_RECT = (0.05, 0.07, 0.95, 1.0)
MODE_ORDER = [("matched", "Matched CI test"), ("mismatch", "Mismatched CI test")]
SOURCE_ORDER = [("data_truth", "Data-level PAG"), ("model_truth", "Model-level PAG")]
PANEL_LETTERS = {
    ("edge_f1", "absolute"): "A",
    ("edge_f1", "delta"): "B",
    ("endpoint_f1", "absolute"): "C",
    ("endpoint_f1", "delta"): "D",
}


def _record_allowed(record: dict, dataset_ids: set[str]) -> bool:
    meta = record.get("metadata", {})
    if not bool(meta.get("alignment_enabled", False)):
        return False
    if str(meta.get("alignment_target_mode")) != "observable_state":
        return False
    if dataset_ids and str(meta.get("dataset")) not in dataset_ids:
        return False
    return True


def _dedupe_mode_frame(df: pd.DataFrame) -> pd.DataFrame:
    return _deduplicate_invariant_rows(
        df,
        subset=["mode", "dataset", "fold", "metric", "source", "sample_size", "ci_threshold", "num_nodes", "benchmark_family"],
        source_col="source",
        invariant_sources=["data_truth"],
    )


def _pair_key_columns(df: pd.DataFrame, include_source: bool) -> list[str]:
    candidate_cols = [
        "dataset",
        "model",
        "metric",
        "fold",
        "sample_size",
        "baseline_sample_size",
        "ci_threshold",
        "num_nodes",
        "benchmark_family",
        "topology",
        "noise_distribution",
        "mechanism_family",
    ]
    if include_source:
        candidate_cols.insert(3, "source")
    return [column for column in candidate_cols if column in df.columns]


def _merge_modes(df: pd.DataFrame, value_col: str, include_source: bool) -> pd.DataFrame:
    key_cols = _pair_key_columns(df, include_source=include_source)
    matched_df = (
        df[df["mode"] == "matched"][key_cols + [value_col]]
        .drop_duplicates(subset=key_cols)
        .rename(columns={value_col: "matched"})
    )
    mismatch_df = (
        df[df["mode"] == "mismatch"][key_cols + [value_col]]
        .drop_duplicates(subset=key_cols)
        .rename(columns={value_col: "mismatch"})
    )
    if matched_df.empty or mismatch_df.empty:
        return pd.DataFrame(columns=key_cols + ["matched", "mismatch"])
    return matched_df.merge(mismatch_df, on=key_cols, how="inner")


def figure_structural_ci_mismatch_sensitivity(
    spec: dict,
    proper_records: List[dict],
    mismatch_records: List[dict],
    output_dir: str,
) -> dict:
    set_reporting_theme(layout_profile="paper")
    title_generated = str(spec.get("title", FIGURE_TITLE))
    dataset_ids = set(spec.get("dataset_order", DATASET_ORDER))
    predicate = lambda record: _record_allowed(record, dataset_ids=dataset_ids)

    mode_truth_frames = []
    mode_delta_frames = []
    for mode, records in [("matched", proper_records), ("mismatch", mismatch_records)]:
        truth_df = build_structural_truth_frame(
            records,
            metrics=PRIMARY_STRUCTURAL_METRICS,
            predicate=predicate,
            include_sources=[source for source, _ in SOURCE_ORDER],
        )
        truth_df = truth_df.dropna(subset=["value"]).copy()
        truth_df["mode"] = mode
        truth_df["mode_label"] = dict(MODE_ORDER)[mode]
        mode_truth_frames.append(truth_df)

        delta_df = build_structural_truth_pairs(
            records,
            metrics=PRIMARY_STRUCTURAL_METRICS,
            predicate=predicate,
        )
        delta_df = delta_df.dropna(subset=["delta"]).copy()
        delta_df["mode"] = mode
        delta_df["mode_label"] = dict(MODE_ORDER)[mode]
        mode_delta_frames.append(delta_df)

    truth_df = pd.concat(mode_truth_frames, ignore_index=True) if mode_truth_frames else pd.DataFrame()
    delta_df = pd.concat(mode_delta_frames, ignore_index=True) if mode_delta_frames else pd.DataFrame()
    truth_df = _dedupe_mode_frame(truth_df)
    _ensure_non_empty(truth_df, spec["id"])
    _ensure_non_empty(delta_df, spec["id"])

    dataset_order = dataset_order_present(truth_df)
    dataset_labels = [_dataset_label(dataset) for dataset in dataset_order]
    source_label_map = dict(SOURCE_ORDER)

    fig, axes = plt.subplots(
        len(PRIMARY_STRUCTURAL_METRICS),
        2,
        figsize=FIGURE_SIZE,
        sharex=False,
        sharey=False,
    )
    fig.suptitle(title_generated, y=0.995)
    shared_delta_ylim = _symmetric_delta_limits(delta_df["delta"].to_numpy(dtype=float))

    stats_rows = []
    panel_row_counts = {"absolute": {}, "delta": {}}
    for metric_index, metric in enumerate(PRIMARY_STRUCTURAL_METRICS):
        absolute_ax = axes[metric_index, 0]
        delta_ax = axes[metric_index, 1]
        metric_truth_df = truth_df[truth_df["metric"] == metric].copy()
        metric_truth_df["source_label"] = metric_truth_df["source"].map(source_label_map)
        metric_delta_df = delta_df[delta_df["metric"] == metric].copy()
        panel_row_counts["absolute"][metric] = int(len(metric_truth_df))
        panel_row_counts["delta"][metric] = int(len(metric_delta_df))

        sns.boxplot(
            data=metric_truth_df,
            x="mode_label",
            y="value",
            hue="source_label",
            order=[label for _, label in MODE_ORDER],
            hue_order=[label for _, label in SOURCE_ORDER],
            palette=STRUCTURAL_SOURCE_LABEL_COLORS,
            showfliers=False,
            linewidth=0.9,
            width=0.62,
            ax=absolute_ax,
        )
        sns.stripplot(
            data=metric_truth_df,
            x="mode_label",
            y="value",
            hue="source_label",
            order=[label for _, label in MODE_ORDER],
            hue_order=[label for _, label in SOURCE_ORDER],
            palette=STRUCTURAL_SOURCE_LABEL_COLORS,
            dodge=True,
            jitter=0.14,
            alpha=0.52,
            size=3.4,
            linewidth=0.25,
            edgecolor="#ffffff",
            ax=absolute_ax,
        )
        if absolute_ax.legend_ is not None:
            absolute_ax.legend_.remove()
        absolute_ax.set_title("Absolute truth score" if metric_index == 0 else "")
        absolute_ax.set_xlabel("")
        absolute_ax.set_ylabel(_metric_label(metric))
        _apply_bounded_metric_y_axis(absolute_ax)
        absolute_ax.grid(axis="y", alpha=0.3)
        _add_panel_letter(absolute_ax, PANEL_LETTERS[(metric, "absolute")])

        sns.boxplot(
            data=metric_delta_df,
            x="mode_label",
            y="delta",
            hue="mode_label",
            order=[label for _, label in MODE_ORDER],
            hue_order=[label for _, label in MODE_ORDER],
            palette=CI_TEST_MODE_COLORS,
            showfliers=False,
            linewidth=0.9,
            width=0.58,
            dodge=False,
            ax=delta_ax,
        )
        sns.stripplot(
            data=metric_delta_df,
            x="mode_label",
            y="delta",
            hue="mode_label",
            order=[label for _, label in MODE_ORDER],
            hue_order=[label for _, label in MODE_ORDER],
            palette=CI_TEST_MODE_COLORS,
            dodge=False,
            jitter=0.14,
            alpha=0.55,
            size=3.4,
            linewidth=0.25,
            edgecolor="#ffffff",
            ax=delta_ax,
        )
        if delta_ax.legend_ is not None:
            delta_ax.legend_.remove()
        delta_ax.axhline(0.0, color="#555555", linestyle="--", linewidth=0.9)
        delta_ax.set_title("Model-data delta" if metric_index == 0 else "")
        delta_ax.set_xlabel("")
        delta_ax.set_ylabel(STRUCTURAL_TRUTH_DELTA_AXIS_LABEL)
        _apply_right_delta_y_axis(delta_ax, shared_delta_ylim)
        delta_ax.grid(axis="y", alpha=0.3)
        _add_panel_letter(delta_ax, PANEL_LETTERS[(metric, "delta")], x=0.0)

        if metric_index == len(PRIMARY_STRUCTURAL_METRICS) - 1:
            absolute_ax.set_xlabel("CI-test configuration")
            delta_ax.set_xlabel("CI-test configuration")
        else:
            absolute_ax.tick_params(labelbottom=False)
            delta_ax.tick_params(labelbottom=False)

        paired_abs = _merge_modes(metric_truth_df, value_col="value", include_source=True)
        for dataset in dataset_order:
            for source, _ in SOURCE_ORDER:
                subset = paired_abs[
                    (paired_abs["dataset"] == dataset)
                    & (paired_abs["source"] == source)
                ]
                if subset.empty:
                    continue
                details = _wilcoxon_paired_details(
                    subset["matched"].to_numpy(dtype=float),
                    subset["mismatch"].to_numpy(dtype=float),
                )
                stats_rows.append(
                    {
                        "figure": spec["id"],
                        "panel": "absolute",
                        "metric": metric,
                        "dataset": dataset,
                        "source": source,
                        "test": "paired_wilcoxon",
                        "n": details["n"],
                        "stat": details["stat"],
                        "p_raw": details["p_raw"],
                        "delta_mean": details["delta_mean"],
                        "delta_median": details["delta_median"],
                        "rank_biserial": details["rank_biserial"],
                    }
                )

        paired_delta = _merge_modes(metric_delta_df, value_col="delta", include_source=False)
        for dataset in dataset_order:
            subset = paired_delta[paired_delta["dataset"] == dataset]
            if subset.empty:
                continue
            details = _wilcoxon_paired_details(
                subset["matched"].to_numpy(dtype=float),
                subset["mismatch"].to_numpy(dtype=float),
            )
            stats_rows.append(
                {
                    "figure": spec["id"],
                    "panel": "delta",
                    "metric": metric,
                    "dataset": dataset,
                    "source": "delta",
                    "test": "paired_wilcoxon",
                    "n": details["n"],
                    "stat": details["stat"],
                    "p_raw": details["p_raw"],
                    "delta_mean": details["delta_mean"],
                    "delta_median": details["delta_median"],
                    "rank_biserial": details["rank_biserial"],
                }
            )

    _bh_adjust_rows(stats_rows)
    legend_handles = [
        Patch(facecolor=STRUCTURAL_SOURCE_LABEL_COLORS[label], edgecolor="black", label=label)
        for _, label in SOURCE_ORDER
    ]
    _move_legend_bottom(axes[0, 0].figure, title="Truth branch", ncol=2, extra_handles=legend_handles)

    caption_lines = [
        "Appendix diagnostic comparing matched and deliberately mismatched CI-test configurations under the truth-anchored structural evaluation.",
        f"Left panels compare absolute data-level and model-level values against benchmark truth; right panels compare the matched truth-based structural delta, model minus data, for {', '.join(_metric_label(metric) for metric in PRIMARY_STRUCTURAL_METRICS)}.",
        f"Datasets represented: {', '.join(dataset_labels)}. Paired Wilcoxon tests compare matched and mismatched runs on fold-matched records and are Benjamini-Hochberg corrected within the figure family.",
        shd_delta_direction_note("shd_strict"),
    ]

    source_files = [path for record in proper_records + mismatch_records for path in record.get("source_files", [])]
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
            "panel_row_counts": panel_row_counts,
        },
    )
