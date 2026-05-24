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
    _is_observable_alignment_record,
    _metric_label,
    _method_label,
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


FIGURE_STRUCTURAL_EXTRACTION_MODES_TRUTH_TITLE = (
    "Additional Extraction Modes Under Observable-State Alignment"
)
FIGURE_STRUCTURAL_EXTRACTION_MODES = ["activations", "pre_activations", "inputs_outputs", "weights"]
FIGURE_STRUCTURAL_EXTRACTION_MODES_TRUTH_PANEL_LETTERS = {
    ("edge_f1", "absolute"): "A",
    ("edge_f1", "delta"): "B",
    ("endpoint_f1", "absolute"): "C",
    ("endpoint_f1", "delta"): "D",
}
REFERENCE_METHOD = "activations"
DELTA_TICK_STEP = 0.25


def _extraction_predicate(record: dict) -> bool:
    if not _is_observable_alignment_record(record):
        return False
    meta = record.get("metadata", {})
    return (
        str(meta.get("model")) == "ShallowMLP"
        and str(meta.get("extraction_method")) in FIGURE_STRUCTURAL_EXTRACTION_MODES
    )


def _resolve_dataset_order(df: pd.DataFrame, spec: dict) -> list[str]:
    if "dataset_order" not in spec:
        return dataset_order_present(df)
    present = set(df["dataset"].dropna().astype(str).tolist())
    order = [str(dataset) for dataset in spec.get("dataset_order", [])]
    missing = [dataset for dataset in order if dataset not in present]
    if missing:
        raise ValueError(f"{spec['id']}: requested datasets not present: {', '.join(missing)}.")
    return order


def _resolve_method_order(df: pd.DataFrame) -> list[str]:
    present = set(df["method"].dropna().astype(str))
    ordered = [method for method in FIGURE_STRUCTURAL_EXTRACTION_MODES if method in present]
    extras = sorted(present - set(ordered))
    return ordered + extras


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


def _summary_rows(
    abs_df: pd.DataFrame,
    delta_df: pd.DataFrame,
    dataset_order: Sequence[str],
    method_order: Sequence[str],
) -> list[dict]:
    rows: list[dict] = []
    for dataset in dataset_order:
        for metric in PRIMARY_STRUCTURAL_METRICS:
            abs_subset = abs_df[(abs_df["dataset"] == dataset) & (abs_df["metric"] == metric)]
            delta_subset = delta_df[(delta_df["dataset"] == dataset) & (delta_df["metric"] == metric)]
            for view_name, subset, value_col in (
                ("absolute", abs_subset, "value"),
                ("delta", delta_subset, "delta"),
            ):
                row = {
                    "dataset": _dataset_label(dataset),
                    "metric": metric,
                    "view": view_name,
                    "n": int(subset["fold"].nunique()) if not subset.empty else 0,
                }
                for method in method_order:
                    method_subset = subset[subset["method"] == method]
                    values = method_subset[value_col].dropna().to_numpy(dtype=float)
                    row[f"{method}_median"] = float(np.median(values)) if values.size else None
                    if values.size:
                        q1, q3 = np.quantile(values, [0.25, 0.75])
                        row[f"{method}_iqr"] = float(q3 - q1)
                    else:
                        row[f"{method}_iqr"] = None
                rows.append(row)
    return rows


def figure_structural_extraction_modes_truth(spec: dict, records: List[dict], output_dir: str) -> dict:
    set_reporting_theme(layout_profile="paper")
    title = str(spec.get("title", FIGURE_STRUCTURAL_EXTRACTION_MODES_TRUTH_TITLE))

    abs_df = build_structural_truth_frame(
        records,
        metrics=PRIMARY_STRUCTURAL_METRICS,
        predicate=_extraction_predicate,
        include_sources=("model_truth",),
    )
    delta_df = build_structural_truth_pairs(
        records,
        metrics=PRIMARY_STRUCTURAL_METRICS,
        predicate=_extraction_predicate,
    )

    _ensure_non_empty(abs_df, spec["id"])
    _ensure_non_empty(delta_df, spec["id"])

    abs_df = abs_df.copy()
    abs_df["method"] = abs_df["extraction_method"].astype(str)
    abs_df["method_label"] = abs_df["method"].map(_method_label)
    abs_df["dataset_label"] = abs_df["dataset"].map(_dataset_label)

    delta_df = delta_df.copy()
    delta_df["method"] = delta_df["extraction_method"].astype(str)
    delta_df["method_label"] = delta_df["method"].map(_method_label)
    delta_df["dataset_label"] = delta_df["dataset"].map(_dataset_label)

    dataset_order = _resolve_dataset_order(abs_df, spec)
    method_order = _resolve_method_order(abs_df)
    dataset_label_order = [_dataset_label(dataset) for dataset in dataset_order]
    method_label_order = [_method_label(method) for method in method_order]
    palette = method_palette(method_order)

    for dataset in dataset_order:
        for metric in PRIMARY_STRUCTURAL_METRICS:
            abs_subset = abs_df[(abs_df["dataset"] == dataset) & (abs_df["metric"] == metric)]
            delta_subset = delta_df[(delta_df["dataset"] == dataset) & (delta_df["metric"] == metric)]
            if abs_subset.empty or delta_subset.empty:
                raise ValueError(f"{spec['id']}: missing rows for dataset={dataset}, metric={metric}.")
            block = abs_subset.pivot_table(
                index=["model", "fold"],
                columns="method",
                values="value",
                aggfunc="first",
            )
            missing_methods = [method for method in method_order if method not in block.columns]
            if missing_methods:
                raise ValueError(
                    f"{spec['id']}: missing methods for dataset={dataset}, metric={metric}: {', '.join(missing_methods)}."
                )
            incomplete = block[method_order].isna().any(axis=1)
            if incomplete.any():
                examples = ", ".join(f"{model}/fold{fold}" for model, fold in block.index[incomplete].tolist()[:5])
                raise ValueError(
                    f"{spec['id']}: incomplete method coverage for dataset={dataset}, metric={metric}: {examples}."
                )

    fig, axes = plt.subplots(
        nrows=len(PRIMARY_STRUCTURAL_METRICS),
        ncols=2,
        figsize=(THESIS_TEXT_WIDTH, 5.6),
        sharex=True,
        sharey=False,
        squeeze=False,
    )
    shared_delta_ylim = _delta_limits(delta_df["delta"].to_numpy(dtype=float))

    for row_index, metric in enumerate(PRIMARY_STRUCTURAL_METRICS):
        abs_metric_df = abs_df[abs_df["metric"] == metric].copy()
        delta_metric_df = delta_df[delta_df["metric"] == metric].copy()
        abs_ax = axes[row_index, 0]
        abs_panel = abs_df[abs_df["metric"] == metric]
        sns.boxplot(
            data=abs_panel,
            x="dataset_label",
            y="value",
            hue="method_label",
            order=dataset_label_order,
            hue_order=method_label_order,
            palette=palette,
            dodge=True,
            showfliers=False,
            width=0.72,
            linewidth=0.9,
            ax=abs_ax,
        )
        sns.stripplot(
            data=abs_panel,
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
        abs_ax.set_title("")
        abs_ax.set_ylabel(_metric_label(metric))
        abs_ax.grid(axis="y", alpha=0.32)
        abs_ax.text(
            -0.12,
            1.03,
            FIGURE_STRUCTURAL_EXTRACTION_MODES_TRUTH_PANEL_LETTERS[(metric, "absolute")],
            transform=abs_ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        if row_index < len(PRIMARY_STRUCTURAL_METRICS) - 1:
            abs_ax.set_xlabel("")
            abs_ax.tick_params(labelbottom=False)
        else:
            abs_ax.set_xlabel("Dataset")

        delta_ax = axes[row_index, 1]
        delta_panel = delta_metric_df
        sns.boxplot(
            data=delta_panel,
            x="dataset_label",
            y="delta",
            hue="method_label",
            order=dataset_label_order,
            hue_order=method_label_order,
            palette=palette,
            dodge=True,
            showfliers=False,
            width=0.72,
            linewidth=0.9,
            ax=delta_ax,
        )
        sns.stripplot(
            data=delta_panel,
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
            FIGURE_STRUCTURAL_EXTRACTION_MODES_TRUTH_PANEL_LETTERS[(metric, "delta")],
            transform=delta_ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        if row_index < len(PRIMARY_STRUCTURAL_METRICS) - 1:
            delta_ax.set_xlabel("")
            delta_ax.tick_params(labelbottom=False)
        else:
            delta_ax.set_xlabel("Dataset")

    handles = [Patch(facecolor=palette[label], edgecolor="black", label=label) for label in method_label_order]
    fig.legend(
        handles=handles,
        labels=method_label_order,
        title="Extraction Method",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(title, y=0.995)

    stats_rows = []
    candidate_methods = [method for method in method_order if method != REFERENCE_METHOD]
    for dataset in dataset_order:
        for metric in PRIMARY_STRUCTURAL_METRICS:
            abs_block = (
                abs_df[(abs_df["dataset"] == dataset) & (abs_df["metric"] == metric)]
                .pivot_table(index="fold", columns="method", values="value", aggfunc="first")
            )
            delta_block = (
                delta_df[(delta_df["dataset"] == dataset) & (delta_df["metric"] == metric)]
                .pivot_table(index="fold", columns="method", values="delta", aggfunc="first")
            )
            for view_name, block in (("absolute", abs_block), ("delta", delta_block)):
                if REFERENCE_METHOD not in block.columns:
                    continue
                for candidate in candidate_methods:
                    if candidate not in block.columns:
                        continue
                    pair = block[[REFERENCE_METHOD, candidate]].dropna()
                    if pair.empty:
                        continue
                    details = _wilcoxon_paired_details(
                        pair[REFERENCE_METHOD].to_numpy(dtype=float),
                        pair[candidate].to_numpy(dtype=float),
                    )
                    stats_rows.append(
                        {
                            "figure": spec["id"],
                            "test": "paired_wilcoxon",
                            "dataset": dataset,
                            "metric": metric,
                            "view": view_name,
                            "comparison": f"{candidate}_vs_{REFERENCE_METHOD}",
                            "n": details["n"],
                            "stat": details["stat"],
                            "p_raw": details["p_raw"],
                            "delta_mean": details["delta_mean"],
                            "delta_median": details["delta_median"],
                            "rank_biserial": details["rank_biserial"],
                        }
                    )

    _bh_adjust_rows(stats_rows)
    table_rows = _summary_rows(abs_df, delta_df, dataset_order, method_order)

    dataset_labels = [_dataset_label(dataset) for dataset in dataset_order]
    source_files = [path for record in records for path in record.get("source_files", [])]
    caption_lines = [
        (
            f"Only observable-state aligned Shallow MLP runs are included. Panels A--D pair the absolute "
            f"model-level PAG score against the ground truth DAG (left column) with the matched difference to "
            f"the data-level truth score and matched delta, Δ = model - data (right column), across "
            f"{len(dataset_labels)} "
            f"benchmark datasets ({', '.join(dataset_labels)}). The panels report {_metric_label('edge_f1')} "
            f"and {_metric_label('endpoint_f1')}."
        ),
        (
            "Extraction method is encoded by color to contrast the retained internal readouts against the "
            "boundary-level input/output mode and the parameter-derived weights mode. Boxes show medians and interquartile "
            "ranges, points show individual matched observations, and the right-hand delta axes share one "
            "symmetric scale around zero using the larger observed absolute deviation across both delta panels "
            "rounded out to the next 0.25 tick."
        ),
        (
            "Within each dataset-metric block, paired two-sided Wilcoxon signed-rank tests compare "
            "pre-activations, inputs/outputs, and weights against activations in both the absolute and delta "
            "views; Benjamini-Hochberg correction is applied across the figure family."
        ),
    ]

    return _save_figure(
        fig,
        axes,
        spec["id"],
        output_dir,
        stats_rows,
        data_row_count=len(abs_df) + len(delta_df),
        source_files=source_files,
        title_generated=title,
        caption_lines=caption_lines,
        legend_mode="figure_bottom",
        annotation_mode="none",
        layout_profile="paper",
        layout_rect=(0.05, 0.07, 0.95, 1.0),
        table_rows=table_rows,
    )
