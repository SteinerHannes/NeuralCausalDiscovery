import textwrap
from typing import List, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd
import seaborn as sns

from reporting.figures.common import (
    STRUCTURAL_TRUTH_DELTA_AXIS_LABEL,
    THESIS_TEXT_WIDTH,
    _apply_metric_limit,
    _bh_adjust_rows,
    _dataset_label,
    _ensure_non_empty,
    _friedman_details,
    _method_label,
    _metric_label,
    _save_figure,
    _wilcoxon_paired_details,
    method_palette,
    set_reporting_theme,
)
from reporting.figures.structural_truth_utils import (
    PRIMARY_STRUCTURAL_METRICS,
    build_structural_truth_frame,
    build_structural_truth_pairs,
    dataset_order_present,
)


FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_METRICS = list(PRIMARY_STRUCTURAL_METRICS)
FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_METHODS = ["activations", "pre_activations"]
FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_TITLE = (
    "Extraction-Mode Sensitivity"
)
FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_PANEL_LETTERS = {
    ("edge_f1", "absolute"): "A",
    ("edge_f1", "delta"): "B",
    ("endpoint_f1", "absolute"): "C",
    ("endpoint_f1", "delta"): "D",
}
REFERENCE_METHOD = "activations"
DELTA_TICK_STEP = 0.25


def _truth_predicate(record: dict) -> bool:
    meta = record.get("metadata", {})
    method = meta.get("extraction_method")
    return (
        bool(meta.get("alignment_enabled", False))
        and meta.get("alignment_target_mode") == "observable_state"
        and method in FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_METHODS
    )


def _resolve_dataset_order(spec: dict, df: pd.DataFrame) -> list[str]:
    if "dataset_order" not in spec:
        return dataset_order_present(df)
    present = set(df["dataset"].dropna().astype(str).tolist())
    order = [str(dataset) for dataset in spec.get("dataset_order", [])]
    missing = [dataset for dataset in order if dataset not in present]
    if missing:
        raise ValueError(f"{spec['id']}: requested datasets not present: {', '.join(missing)}.")
    return order


def _median_iqr(values: Sequence[float]) -> tuple[float, float]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size == 0:
        return float("nan"), float("nan")
    q1, median, q3 = np.quantile(numeric, [0.25, 0.5, 0.75])
    return float(median), float(q3 - q1)


def _delta_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-DELTA_TICK_STEP, DELTA_TICK_STEP)
    max_abs = float(np.max(np.abs(finite)))
    step_count = max(1, int(np.floor(max_abs / DELTA_TICK_STEP)) + 1)
    limit = float(step_count * DELTA_TICK_STEP)
    limit = min(1.0, limit)
    return (-limit, limit)


def _omnibus_stats(block: pd.DataFrame, methods: Sequence[str]) -> dict:
    if len(methods) < 3:
        complete = block.dropna(subset=list(methods))
        return {
            "stat": 0.0,
            "p_raw": 1.0,
            "n": int(complete.shape[0]),
            "degenerate": True,
            "reason": "not_defined_for_two_methods",
        }
    return _friedman_details(*(block[method].to_numpy(dtype=float) for method in methods))


def figure_structural_extraction_internal_truth(spec: dict, records: List[dict], output_dir: str) -> dict:
    set_reporting_theme(layout_profile="paper")
    title = str(spec.get("title", FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_TITLE))

    abs_df = build_structural_truth_frame(
        records=records,
        metrics=FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_METRICS,
        predicate=_truth_predicate,
        include_sources=("model_truth",),
    )
    pair_df = build_structural_truth_pairs(
        records=records,
        metrics=FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_METRICS,
        predicate=_truth_predicate,
    )
    _ensure_non_empty(abs_df, spec["id"])
    _ensure_non_empty(pair_df, spec["id"])
    abs_df["method_label"] = abs_df["extraction_method"].map(_method_label)
    pair_df["method_label"] = pair_df["extraction_method"].map(_method_label)

    dataset_order = _resolve_dataset_order(spec, abs_df)
    dataset_label_order = [_dataset_label(dataset) for dataset in dataset_order]
    dataset_tick_label_order = [
        "\n".join(textwrap.wrap(label, width=12, break_long_words=False))
        for label in dataset_label_order
    ]
    method_label_order = [_method_label(method) for method in FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_METHODS]
    palette = method_palette(FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_METHODS)

    for dataset in dataset_order:
        for metric in FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_METRICS:
            abs_subset = abs_df[(abs_df["dataset"] == dataset) & (abs_df["metric"] == metric)]
            pair_subset = pair_df[(pair_df["dataset"] == dataset) & (pair_df["metric"] == metric)]
            if abs_subset.empty or pair_subset.empty:
                raise ValueError(f"{spec['id']}: missing rows for dataset={dataset}, metric={metric}.")
            block = abs_subset.pivot_table(
                index=["model", "fold"],
                columns="extraction_method",
                values="value",
                aggfunc="first",
            )
            missing = [method for method in FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_METHODS if method not in block.columns]
            if missing:
                raise ValueError(
                    f"{spec['id']}: missing extraction methods for dataset={dataset}, metric={metric}: {', '.join(missing)}."
                )
            incomplete = block[FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_METHODS].isna().any(axis=1)
            if incomplete.any():
                examples = ", ".join(f"{model}/fold{fold}" for model, fold in block.index[incomplete].tolist()[:5])
                raise ValueError(
                    f"{spec['id']}: incomplete method coverage for dataset={dataset}, metric={metric}: {examples}."
                )

    fig, axes = plt.subplots(
        nrows=len(FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_METRICS),
        ncols=2,
        figsize=(THESIS_TEXT_WIDTH, 5.6),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    shared_delta_ylim = _delta_limits(pair_df["delta"].to_numpy(dtype=float))

    absolute_stats_rows = []
    contrast_rows = []
    table_rows = []
    for row_index, metric in enumerate(FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_METRICS):
        abs_metric_df = abs_df[abs_df["metric"] == metric].copy()
        pair_metric_df = pair_df[pair_df["metric"] == metric].copy()
        abs_ax = axes[row_index, 0]
        delta_ax = axes[row_index, 1]

        sns.boxplot(
            data=abs_metric_df,
            x="dataset_label",
            y="value",
            hue="method_label",
            order=dataset_label_order,
            hue_order=method_label_order,
            palette=palette,
            showfliers=False,
            width=0.72,
            linewidth=0.9,
            dodge=True,
            ax=abs_ax,
        )
        sns.stripplot(
            data=abs_metric_df,
            x="dataset_label",
            y="value",
            hue="method_label",
            order=dataset_label_order,
            hue_order=method_label_order,
            palette=palette,
            dodge=True,
            jitter=0.10,
            alpha=0.62,
            linewidth=0.2,
            edgecolor="#111111",
            size=4.6,
            ax=abs_ax,
        )
        if abs_ax.legend_ is not None:
            abs_ax.legend_.remove()
        _apply_metric_limit(abs_ax, metric, abs_metric_df["value"].to_numpy(dtype=float), axis="y")
        abs_ax.set_box_aspect(1)
        abs_ax.set_xticks(range(len(dataset_tick_label_order)))
        abs_ax.set_xticklabels(dataset_tick_label_order)
        abs_ax.set_title("")
        abs_ax.set_ylabel(_metric_label(metric))
        abs_ax.grid(axis="y", alpha=0.32)
        abs_ax.text(
            -0.12,
            1.03,
            FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_PANEL_LETTERS[(metric, "absolute")],
            transform=abs_ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        if row_index < len(FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_METRICS) - 1:
            abs_ax.set_xlabel("")
            abs_ax.tick_params(labelbottom=False)
        else:
            abs_ax.set_xlabel("Dataset")

        sns.boxplot(
            data=pair_metric_df,
            x="dataset_label",
            y="delta",
            hue="method_label",
            order=dataset_label_order,
            hue_order=method_label_order,
            palette=palette,
            showfliers=False,
            width=0.72,
            linewidth=0.9,
            dodge=True,
            ax=delta_ax,
        )
        sns.stripplot(
            data=pair_metric_df,
            x="dataset_label",
            y="delta",
            hue="method_label",
            order=dataset_label_order,
            hue_order=method_label_order,
            palette=palette,
            dodge=True,
            jitter=0.10,
            alpha=0.62,
            linewidth=0.2,
            edgecolor="#111111",
            size=4.6,
            ax=delta_ax,
        )
        if delta_ax.legend_ is not None:
            delta_ax.legend_.remove()
        delta_ax.axhline(0.0, color="#5e5e5e", linestyle=(0, (4, 2)), linewidth=1.0, alpha=0.95)
        delta_ax.set_ylim(*shared_delta_ylim)
        delta_ax.set_box_aspect(1)
        delta_ax.set_xticks(range(len(dataset_tick_label_order)))
        delta_ax.set_xticklabels(dataset_tick_label_order)
        delta_ax.set_title("")
        delta_ax.set_ylabel(STRUCTURAL_TRUTH_DELTA_AXIS_LABEL)
        delta_ax.yaxis.set_label_position("right")
        delta_ax.yaxis.tick_right()
        delta_ax.tick_params(axis="y", labelleft=False, left=False, right=False, length=0)
        delta_ax.yaxis.set_major_locator(MultipleLocator(DELTA_TICK_STEP))
        delta_ax.set_yticks(
            np.arange(
                shared_delta_ylim[0],
                shared_delta_ylim[1] + (DELTA_TICK_STEP * 0.5),
                DELTA_TICK_STEP,
            )
        )
        delta_ax.grid(axis="y", alpha=0.32)
        delta_ax.text(
            0.0,
            1.03,
            FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_PANEL_LETTERS[(metric, "delta")],
            transform=delta_ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        if row_index < len(FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_METRICS) - 1:
            delta_ax.set_xlabel("")
            delta_ax.tick_params(labelbottom=False)
        else:
            delta_ax.set_xlabel("Dataset")

        for dataset in dataset_order:
            dataset_label = _dataset_label(dataset)
            abs_subset = abs_metric_df[abs_metric_df["dataset"] == dataset]
            block = abs_subset.pivot_table(
                index=["model", "fold"],
                columns="extraction_method",
                values="value",
                aggfunc="first",
            )[FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_METHODS]
            block = block.dropna(subset=FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_METHODS)
            friedman = _omnibus_stats(block, FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_METHODS)
            absolute_stats_rows.append(
                {
                    "figure": spec["id"],
                    "test": "friedman",
                    "dataset": dataset,
                    "metric": metric,
                    "panel": "absolute",
                    "comparison": "all_internal_methods",
                    "n": int(friedman["n"]),
                    "stat": float(friedman["stat"]),
                    "p_raw": float(friedman["p_raw"]),
                    "degenerate": bool(friedman["degenerate"]),
                    "note": friedman["reason"],
                }
            )
            for method in FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_METHODS:
                median, iqr = _median_iqr(block[method].to_numpy(dtype=float))
                pair_subset = pair_metric_df[
                    (pair_metric_df["dataset"] == dataset) & (pair_metric_df["extraction_method"] == method)
                ]
                delta_median, delta_iqr = _median_iqr(pair_subset["delta"].to_numpy(dtype=float))
                table_rows.append(
                    {
                        "dataset": dataset_label,
                        "metric": _metric_label(metric),
                        "method": _method_label(method),
                        "n": int(block.shape[0]),
                        "model_truth_median": median,
                        "model_truth_iqr": iqr,
                        "delta_median": delta_median,
                        "delta_iqr": delta_iqr,
                        "friedman_q_bh": float("nan"),
                        "contrast_to_activations_q_bh": float("nan"),
                    }
                )
            for method in FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_METHODS:
                if method == REFERENCE_METHOD:
                    continue
                details = _wilcoxon_paired_details(
                    block[REFERENCE_METHOD].to_numpy(dtype=float),
                    block[method].to_numpy(dtype=float),
                )
                contrast_rows.append(
                    {
                        "figure": spec["id"],
                        "test": "paired_wilcoxon",
                        "dataset": dataset,
                        "metric": metric,
                        "panel": "absolute",
                        "comparison": f"{method}_vs_{REFERENCE_METHOD}",
                        "n": int(details["n"]),
                        "stat": float(details["stat"]),
                        "p_raw": float(details["p_raw"]),
                        "delta_mean": details["delta_mean"],
                        "delta_median": details["delta_median"],
                        "rank_biserial": details["rank_biserial"],
                    }
                )

    _bh_adjust_rows(absolute_stats_rows)
    _bh_adjust_rows(contrast_rows)

    table_index = {(row["dataset"], row["metric"], row["method"]): row for row in table_rows}
    for row in absolute_stats_rows:
        dataset_label = _dataset_label(row["dataset"])
        metric_label = _metric_label(row["metric"])
        for method in FIGURE_STRUCTURAL_EXTRACTION_INTERNAL_TRUTH_METHODS:
            table_index[(dataset_label, metric_label, _method_label(method))]["friedman_q_bh"] = row.get("q_bh")
    for row in contrast_rows:
        method = row["comparison"].split("_vs_")[0]
        dataset_label = _dataset_label(row["dataset"])
        metric_label = _metric_label(row["metric"])
        table_index[(dataset_label, metric_label, _method_label(method))]["contrast_to_activations_q_bh"] = row.get(
            "q_bh"
        )

    handles = [Patch(facecolor=palette[label], edgecolor="black", label=label) for label in method_label_order]
    fig.legend(
        handles=handles,
        labels=method_label_order,
        title="Extraction Method",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(title, y=0.995)

    dataset_labels = [_dataset_label(dataset) for dataset in dataset_order]
    source_files = [path for record in records for path in record.get("source_files", [])]
    caption_lines = [
        (
            f"Only observable-state aligned runs are included. Panels A--D pair the absolute model-level PAG "
            f"score against the ground truth DAG (left column) with the matched difference to the data-level "
            f"truth score and matched delta, Δ = model - data (right column), across "
            f"{len(dataset_labels)} benchmark datasets "
            f"({', '.join(dataset_labels)}). The panels report {_metric_label('edge_f1')} and "
            f"{_metric_label('endpoint_f1')}."
        ),
        (
            "The figure isolates activations and pre-activations as the retained hidden-state readouts for the "
            "main extraction comparison. Boxes show medians and interquartile ranges, points show individual matched observations, "
            "and the right-hand delta axes share one symmetric scale around zero using the larger observed absolute "
            "deviation across both delta panels rounded out to the next 0.25 tick."
        ),
        (
            "Within each dataset-metric block, a paired Wilcoxon signed-rank contrast compares pre-activations "
            "against activations on the absolute truth-based model score. Benjamini-Hochberg correction is applied within the figure."
        ),
    ]

    return _save_figure(
        fig=fig,
        axes=axes.flatten().tolist(),
        figure_id=spec["id"],
        output_dir=output_dir,
        stats_rows=absolute_stats_rows + contrast_rows,
        data_row_count=int(len(abs_df) + len(pair_df)),
        source_files=source_files,
        title_generated=title,
        caption_lines=caption_lines,
        legend_mode="figure_bottom",
        annotation_mode="none",
        layout_profile="paper",
        layout_rect=(0.05, 0.07, 0.95, 1.0),
        table_rows=list(table_index.values()),
    )
