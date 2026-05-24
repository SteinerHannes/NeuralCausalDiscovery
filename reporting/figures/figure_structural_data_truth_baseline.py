import textwrap
from typing import List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator

from reporting.figures.common import (
    DATASET_COLORS,
    _apply_metric_limit,
    _dataset_label,
    _ensure_non_empty,
    _metric_label,
    _save_figure,
    THESIS_TEXT_WIDTH,
    set_reporting_theme,
)
from reporting.figures.structural_truth_utils import (
    build_structural_truth_frame,
    dataset_order_present,
)


FIGURE_STRUCTURAL_DATA_TRUTH_BASELINE_METRICS = ["edge_f1", "endpoint_f1", "shd_strict", "shd_partial"]
FIGURE_STRUCTURAL_DATA_TRUTH_BASELINE_TITLE = (
    "Data-Level Causal Discovery"
)
FIGURE_STRUCTURAL_DATA_TRUTH_BASELINE_PANEL_LETTERS = {
    "edge_f1": "A",
    "endpoint_f1": "B",
    "shd_strict": "C",
    "shd_partial": "D",
}
FIGURE_STRUCTURAL_DATA_TRUTH_BASELINE_SHD_METRICS = ("shd_strict", "shd_partial")


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


def figure_structural_data_truth_baseline(spec: dict, records: List[dict], output_dir: str) -> dict:
    set_reporting_theme(layout_profile="paper")
    title = str(spec.get("title", FIGURE_STRUCTURAL_DATA_TRUTH_BASELINE_TITLE))

    df = build_structural_truth_frame(
        records=records,
        metrics=FIGURE_STRUCTURAL_DATA_TRUTH_BASELINE_METRICS,
        include_sources=("data_truth",),
        dedupe_data=False,
    )
    _ensure_non_empty(df, spec["id"])

    dataset_order = _resolve_dataset_order(spec, df)
    dataset_label_order = [_dataset_label(dataset) for dataset in dataset_order]
    dataset_tick_label_order = [
        "\n".join(textwrap.wrap(label, width=12, break_long_words=False))
        for label in dataset_label_order
    ]
    palette = {
        _dataset_label(dataset): DATASET_COLORS.get(dataset, "#7f8c8d")
        for dataset in dataset_order
    }

    for dataset in dataset_order:
        for metric in FIGURE_STRUCTURAL_DATA_TRUTH_BASELINE_METRICS:
            subset = df[(df["dataset"] == dataset) & (df["metric"] == metric)]
            if subset.empty:
                raise ValueError(f"{spec['id']}: missing rows for dataset={dataset}, metric={metric}.")

    fig, axes = plt.subplots(2, 2, figsize=(THESIS_TEXT_WIDTH, 5.2), sharex=True, sharey=False)
    axes_flat = axes.flatten()
    axes_list = axes_flat.tolist()
    shd_axes = {}

    for idx, (ax, metric) in enumerate(zip(axes_flat, FIGURE_STRUCTURAL_DATA_TRUTH_BASELINE_METRICS)):
        metric_df = df[df["metric"] == metric].copy()
        sns.boxplot(
            data=metric_df,
            x="dataset_label",
            y="value",
            hue="dataset_label",
            order=dataset_label_order,
            hue_order=dataset_label_order,
            palette=palette,
            dodge=False,
            showfliers=False,
            width=0.72,
            linewidth=0.9,
            ax=ax,
        )
        sns.stripplot(
            data=metric_df,
            x="dataset_label",
            y="value",
            hue="dataset_label",
            order=dataset_label_order,
            hue_order=dataset_label_order,
            palette=palette,
            dodge=False,
            jitter=0.12,
            alpha=0.72,
            linewidth=0.25,
            edgecolor="#111111",
            size=4.6,
            ax=ax,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()
        _apply_metric_limit(ax, metric, metric_df["value"].to_numpy(dtype=float), axis="y")
        ax.set_box_aspect(1)
        ax.set_xticks(range(len(dataset_tick_label_order)))
        ax.set_xticklabels(dataset_tick_label_order)
        if idx < 2:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Dataset")
        ax.set_ylabel(_metric_label(metric))
        ax.set_title("")
        ax.grid(axis="y", alpha=0.32)
        ax.text(
            -0.12,
            1.03,
            FIGURE_STRUCTURAL_DATA_TRUTH_BASELINE_PANEL_LETTERS[metric],
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        if metric in FIGURE_STRUCTURAL_DATA_TRUTH_BASELINE_SHD_METRICS:
            shd_axes[metric] = ax

    if len(shd_axes) == len(FIGURE_STRUCTURAL_DATA_TRUTH_BASELINE_SHD_METRICS):
        shared_top = max(ax.get_ylim()[1] for ax in shd_axes.values())
        for ax in shd_axes.values():
            ax.set_ylim(-0.5, shared_top + 0.5)
            ax.yaxis.set_major_locator(MultipleLocator(2))

    fig.suptitle(title, y=0.995)

    stats_rows = []
    table_rows = []
    for dataset in dataset_order:
        dataset_label = _dataset_label(dataset)
        for metric in FIGURE_STRUCTURAL_DATA_TRUTH_BASELINE_METRICS:
            subset = df[(df["dataset"] == dataset) & (df["metric"] == metric)].copy()
            values = subset["value"].to_numpy(dtype=float)
            median, iqr = _median_iqr(values)
            stats_row = {
                "figure": spec["id"],
                "test": "descriptive",
                "dataset": dataset,
                "metric": metric,
                "n": int(np.isfinite(values).sum()),
                "median": median,
                "iqr": iqr,
                "value_min": float(np.nanmin(values)),
                "value_max": float(np.nanmax(values)),
                "fold_min": int(subset["fold"].min()),
                "fold_max": int(subset["fold"].max()),
            }
            stats_rows.append(stats_row)
            table_rows.append(
                {
                    "dataset": dataset_label,
                    "metric": _metric_label(metric),
                    "n": stats_row["n"],
                    "median": median,
                    "iqr": iqr,
                    "value_min": stats_row["value_min"],
                    "value_max": stats_row["value_max"],
                    "fold_min": stats_row["fold_min"],
                    "fold_max": stats_row["fold_max"],
                }
            )

    dataset_labels = [_dataset_label(dataset) for dataset in dataset_order]
    source_files = [path for record in records for path in record.get("source_files", [])]
    caption_lines = [
        (
            f"Panels A--D summarise matched benchmark discovery runs from {len(dataset_labels)} datasets "
            f"({', '.join(dataset_labels)}) by scoring the data-level PAG directly against the ground truth DAG. "
            f"The panels report {_metric_label('edge_f1')}, {_metric_label('endpoint_f1')}, "
            f"{_metric_label('shd_strict')}, and {_metric_label('shd_partial')} under the same discovery "
            "configuration later used for the model-level branch."
        ),
        (
            "Boxes show medians and interquartile ranges, and points show individual fold-level observations. "
            "Higher values indicate better recovery for Edge $F_1$ and endpoint-mark $F_1$, whereas lower values "
            "indicate better recovery for strict SHD and partial SHD."
        ),
        (
            "The figure is descriptive only: it establishes the benchmark-side reference for later truth-based "
            "comparisons, and the exported table reports matched fold counts, medians, IQRs, and observed fold "
            "ranges for each dataset-metric pair."
        ),
    ]

    return _save_figure(
        fig=fig,
        axes=axes_list,
        figure_id=spec["id"],
        output_dir=output_dir,
        stats_rows=stats_rows,
        data_row_count=len(df),
        source_files=source_files,
        title_generated=title,
        caption_lines=caption_lines,
        legend_mode="none",
        annotation_mode="dense_caption_only",
        layout_profile="paper",
        layout_rect=(0.05, 0.00, 0.95, 1.0),
        table_rows=table_rows,
    )
