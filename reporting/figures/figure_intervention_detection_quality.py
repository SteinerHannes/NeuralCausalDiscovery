from typing import List

import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from reporting.figures.common import (
    DATASET_ORDER,
    THESIS_TEXT_WIDTH,
    _bh_adjust_rows,
    _dataset_label,
    _ensure_non_empty,
    _is_observable_alignment_record,
    _record_detection_summary,
    _save_figure,
    _wilcoxon_paired_details,
    plt,
    set_reporting_theme,
)


FIGURE_TITLE = "Confirmed intervention-effect detection quality"
METRIC_ORDER = [
    ("precision", "Precision", "A"),
    ("recall", "Recall", "B"),
    ("specificity", "Specificity", "C"),
    ("f1", "$F_1$ score", "D"),
]
SOURCE_ORDER = [
    ("pag", "Graph-based claims"),
    ("model", "Model-based claims"),
]
SOURCE_COLORS = {
    "Graph-based claims": "#E67E22",
    "Model-based claims": "#59A14F",
}


def figure_intervention_detection_quality(spec: dict, records: List[dict], output_dir: str) -> dict:
    set_reporting_theme(layout_profile="paper")
    dataset_order = list(spec.get("dataset_order", DATASET_ORDER))
    title = str(spec.get("title", FIGURE_TITLE))

    active_records = [record for record in records if _is_observable_alignment_record(record)]
    rows = []
    for record in active_records:
        detection_df = _record_detection_summary(record, role="output")
        if detection_df.empty:
            continue
        dataset = record.get("metadata", {}).get("dataset")
        for _, row in detection_df.iterrows():
            for source_key, source_label in SOURCE_ORDER:
                for metric, metric_label, _ in METRIC_ORDER:
                    value = row.get(f"{source_key}_{metric}")
                    if value is None or pd.isna(value):
                        continue
                    rows.append(
                        {
                            "dataset": dataset,
                            "dataset_label": _dataset_label(dataset),
                            "model": row.get("model"),
                            "fold": row.get("fold"),
                            "source": source_key,
                            "source_label": source_label,
                            "metric": metric,
                            "metric_label": metric_label,
                            "value": float(value),
                        }
                    )

    df = pd.DataFrame(rows).dropna(subset=["value"])
    _ensure_non_empty(df, spec["id"])

    dataset_labels = [_dataset_label(dataset) for dataset in dataset_order if dataset in set(df["dataset"].tolist())]
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(THESIS_TEXT_WIDTH, 5.6),
        dpi=170,
        sharex=True,
        sharey=True,
    )
    axes_flat = axes.flatten()

    for idx, (ax, (metric, metric_label, panel_letter)) in enumerate(zip(axes_flat, METRIC_ORDER)):
        panel_df = df[df["metric"] == metric]
        sns.boxplot(
            data=panel_df,
            x="dataset_label",
            y="value",
            hue="source_label",
            order=dataset_labels,
            hue_order=[label for _, label in SOURCE_ORDER],
            palette=SOURCE_COLORS,
            showfliers=False,
            width=0.72,
            linewidth=0.9,
            ax=ax,
        )
        sns.stripplot(
            data=panel_df,
            x="dataset_label",
            y="value",
            hue="source_label",
            order=dataset_labels,
            hue_order=[label for _, label in SOURCE_ORDER],
            palette=SOURCE_COLORS,
            dodge=True,
            jitter=0.11,
            alpha=0.62,
            size=4.8,
            edgecolor="#111111",
            linewidth=0.2,
            ax=ax,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()
        ax.set_ylim(-0.05, 1.05)
        if idx < 2:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Dataset")
        ax.set_ylabel(metric_label)
        ax.set_title("")
        ax.grid(axis="y", alpha=0.32)
        ax.text(
            -0.10,
            1.03,
            panel_letter,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    legend_handles = [
        Patch(facecolor=SOURCE_COLORS[label], edgecolor="black", label=label)
        for _, label in SOURCE_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        labels=[label for _, label in SOURCE_ORDER],
        title="Claim source",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=2,
        frameon=False,
    )
    fig.suptitle(title, y=0.995)

    stats_rows = []
    for metric, _, _ in METRIC_ORDER:
        for dataset in dataset_order:
            subset = df[(df["metric"] == metric) & (df["dataset"] == dataset)]
            if subset.empty:
                continue
            pair = subset.pivot_table(
                index=["dataset", "model", "fold"],
                columns="source",
                values="value",
                aggfunc="first",
            )
            if not {"pag", "model"}.issubset(pair.columns):
                continue
            pair = pair[["pag", "model"]].dropna()
            if pair.empty:
                continue
            details = _wilcoxon_paired_details(
                pair["pag"].to_numpy(dtype=float),
                pair["model"].to_numpy(dtype=float),
            )
            stats_rows.append(
                {
                    "figure": spec["id"],
                    "test": "paired_wilcoxon",
                    "metric": metric,
                    "dataset": dataset,
                    "comparison": "model_vs_pag",
                    "n": int(details["n"]),
                    "stat": details["stat"],
                    "p_raw": details["p_raw"],
                    "delta_mean": details["delta_mean"],
                    "delta_median": details["delta_median"],
                    "rank_biserial": details["rank_biserial"],
                }
            )
    _bh_adjust_rows(stats_rows)

    caption_lines = [
        "Panels A--D summarize confirmed intervention-effect precision, recall, specificity, and $F_1$ score under observable-state alignment, restricted to output outcomes.",
        f"Each panel compares graph-based claims derived from recovered PAGs with model-based claims derived from the predictive model across the selected datasets: {', '.join(dataset_labels)}.",
        "Precision and recall quantify positive-effect detection, specificity quantifies false-positive control on effect-absent cases, and $F_1$ score summarizes the precision-recall trade-off.",
        "Paired Wilcoxon signed-rank tests compare model-based and PAG-based detection on matched fold-model observations within each dataset-metric panel, with Benjamini-Hochberg correction across the figure family.",
        "The figure pools architecture runs to summarize detection quality, while architecture-specific robustness is handled separately in the intervention architecture figure.",
        "Unlike the removed delta figure, these panels show the absolute detection-quality levels that directly support the thesis claim about intervention fidelity.",
    ]

    source_files = [path for record in active_records for path in record.get("source_files", [])]
    return _save_figure(
        fig,
        axes,
        spec["id"],
        output_dir,
        stats_rows,
        data_row_count=len(df),
        source_files=source_files,
        title_generated=title,
        caption_lines=caption_lines,
        legend_mode="figure_bottom",
        annotation_mode="none",
        layout_profile="paper",
        layout_rect=(0.05, 0.07, 0.95, 1.0),
    )
