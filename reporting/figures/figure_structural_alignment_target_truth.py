from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator

from reporting.figures.common import (
    ALIGNMENT_TARGET_COLORS,
    ALIGNMENT_TARGET_LABELS,
    ALIGNMENT_TARGET_ORDER,
    DATASET_COLORS,
    THESIS_TEXT_WIDTH,
    _apply_metric_limit,
    _bh_adjust_rows,
    _dataset_label,
    _ensure_non_empty,
    _friedman_details,
    _metric_label,
    _save_figure,
    _wilcoxon_paired_details,
    set_reporting_theme,
)
from reporting.figures.structural_truth_utils import (
    build_structural_truth_frame,
    dataset_order_present,
)


FIGURE_STRUCTURAL_ALIGNMENT_TARGET_TRUTH_METRICS = ["edge_f1", "endpoint_f1", "shd_strict", "shd_partial"]
FIGURE_STRUCTURAL_ALIGNMENT_TARGET_TRUTH_TITLE = (
    "Model-Level Recovery by Alignment Target"
)
FIGURE_STRUCTURAL_ALIGNMENT_TARGET_TRUTH_PANEL_LETTERS = {
    "edge_f1": "A",
    "endpoint_f1": "B",
    "shd_strict": "C",
    "shd_partial": "D",
}
FIGURE_STRUCTURAL_ALIGNMENT_TARGET_TRUTH_SHD_METRICS = ("shd_strict", "shd_partial")
POSTHOC_COMPARISONS = [
    ("observable_state", "off"),
    ("sem_truth", "off"),
    ("sem_truth", "observable_state"),
]


def _resolve_dataset_order(spec: dict, df: pd.DataFrame) -> list[str]:
    if "dataset_order" not in spec:
        return dataset_order_present(df)
    present = set(df["dataset"].dropna().astype(str).tolist())
    order = [str(dataset) for dataset in spec.get("dataset_order", [])]
    missing = [dataset for dataset in order if dataset not in present]
    if missing:
        raise ValueError(f"{spec['id']}: requested datasets not present: {', '.join(missing)}.")
    return order


def _variant_label(variant: str) -> str:
    return ALIGNMENT_TARGET_LABELS.get(variant, variant)


def _median_iqr(values: Sequence[float]) -> Tuple[float, float]:
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    numeric = numeric[np.isfinite(numeric)]
    if numeric.size == 0:
        return float("nan"), float("nan")
    q1, median, q3 = np.quantile(numeric, [0.25, 0.5, 0.75])
    return float(median), float(q3 - q1)


def _numeric_row_value(row: Optional[dict], key: str) -> float:
    if not row:
        return float("nan")
    value = row.get(key)
    if value is None or pd.isna(value):
        return float("nan")
    return float(value)


def figure_structural_alignment_target_truth(spec: dict, records: List[dict], output_dir: str) -> dict:
    set_reporting_theme(layout_profile="paper")
    title = str(spec.get("title", FIGURE_STRUCTURAL_ALIGNMENT_TARGET_TRUTH_TITLE))

    df = build_structural_truth_frame(
        records=records,
        metrics=FIGURE_STRUCTURAL_ALIGNMENT_TARGET_TRUTH_METRICS,
        include_sources=("model_truth",),
    )
    _ensure_non_empty(df, spec["id"])
    dataset_order = _resolve_dataset_order(spec, df)
    dataset_label_order = [_dataset_label(dataset) for dataset in dataset_order]
    variant_label_order = [_variant_label(variant) for variant in ALIGNMENT_TARGET_ORDER]
    palette = {
        _variant_label(variant): ALIGNMENT_TARGET_COLORS[variant]
        for variant in ALIGNMENT_TARGET_ORDER
    }

    for dataset in dataset_order:
        for metric in FIGURE_STRUCTURAL_ALIGNMENT_TARGET_TRUTH_METRICS:
            subset = df[(df["dataset"] == dataset) & (df["metric"] == metric)]
            if subset.empty:
                raise ValueError(f"{spec['id']}: missing rows for dataset={dataset}, metric={metric}.")
            block = subset.pivot_table(
                index=["model", "fold"],
                columns="alignment_variant",
                values="value",
                aggfunc="first",
            )
            missing = [variant for variant in ALIGNMENT_TARGET_ORDER if variant not in block.columns]
            if missing:
                raise ValueError(
                    f"{spec['id']}: missing alignment variants for dataset={dataset}, metric={metric}: {', '.join(missing)}."
                )
            incomplete = block[ALIGNMENT_TARGET_ORDER].isna().any(axis=1)
            if incomplete.any():
                examples = ", ".join(
                    f"{model}/fold{fold}" for model, fold in block.index[incomplete].tolist()[:5]
                )
                raise ValueError(
                    f"{spec['id']}: incomplete variant coverage for dataset={dataset}, metric={metric}: {examples}."
                )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(THESIS_TEXT_WIDTH, 5.6),
        sharex=True,
        sharey=False,
    )
    axes_flat = axes.flatten()
    axes_list = axes_flat.tolist()
    shd_axes = {}

    for idx, (ax, metric) in enumerate(zip(axes_flat, FIGURE_STRUCTURAL_ALIGNMENT_TARGET_TRUTH_METRICS)):
        metric_df = df[df["metric"] == metric].copy()
        sns.boxplot(
            data=metric_df,
            x="dataset_label",
            y="value",
            hue="alignment_variant_label",
            order=dataset_label_order,
            hue_order=variant_label_order,
            palette=palette,
            showfliers=False,
            width=0.72,
            linewidth=0.9,
            dodge=True,
            ax=ax,
        )
        sns.stripplot(
            data=metric_df,
            x="dataset_label",
            y="value",
            hue="alignment_variant_label",
            order=dataset_label_order,
            hue_order=variant_label_order,
            palette=palette,
            dodge=True,
            jitter=0.11,
            alpha=0.62,
            linewidth=0.2,
            edgecolor="#111111",
            size=4.8,
            ax=ax,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()
        _apply_metric_limit(ax, metric, metric_df["value"].to_numpy(dtype=float), axis="y")
        if idx < 2:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Dataset")
        ax.set_ylabel(_metric_label(metric))
        ax.set_title("")
        ax.text(
            -0.10,
            1.03,
            FIGURE_STRUCTURAL_ALIGNMENT_TARGET_TRUTH_PANEL_LETTERS[metric],
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        if metric in FIGURE_STRUCTURAL_ALIGNMENT_TARGET_TRUTH_SHD_METRICS:
            shd_axes[metric] = ax

    if len(shd_axes) == len(FIGURE_STRUCTURAL_ALIGNMENT_TARGET_TRUTH_SHD_METRICS):
        shared_top = max(ax.get_ylim()[1] for ax in shd_axes.values())
        for ax in shd_axes.values():
            ax.set_ylim(-0.5, shared_top + 0.5)
            ax.yaxis.set_major_locator(MultipleLocator(2))

    handles = [Patch(facecolor=palette[label], edgecolor="black", label=label) for label in variant_label_order]
    fig.legend(
        handles=handles,
        labels=variant_label_order,
        title="Alignment mode",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=3,
        frameon=False,
    )
    fig.suptitle(title, y=0.995)

    friedman_rows = []
    posthoc_rows = []
    table_rows = []
    for dataset in dataset_order:
        dataset_label = _dataset_label(dataset)
        for metric in FIGURE_STRUCTURAL_ALIGNMENT_TARGET_TRUTH_METRICS:
            subset = df[(df["dataset"] == dataset) & (df["metric"] == metric)]
            block = subset.pivot_table(
                index=["model", "fold"],
                columns="alignment_variant",
                values="value",
                aggfunc="first",
            )[ALIGNMENT_TARGET_ORDER]
            block = block.dropna(subset=ALIGNMENT_TARGET_ORDER)
            friedman = _friedman_details(
                block["off"].to_numpy(),
                block["observable_state"].to_numpy(),
                block["sem_truth"].to_numpy(),
            )
            friedman_rows.append(
                {
                    "figure": spec["id"],
                    "test": "friedman",
                    "dataset": dataset,
                    "metric": metric,
                    "comparison": "off_vs_observable_state_vs_sem_truth",
                    "n": int(friedman["n"]),
                    "stat": float(friedman["stat"]),
                    "p_raw": float(friedman["p_raw"]),
                    "degenerate": bool(friedman["degenerate"]),
                    "note": friedman["reason"],
                }
            )

            contrast_lookup = {}
            for left, right in POSTHOC_COMPARISONS:
                details = _wilcoxon_paired_details(
                    block[right].to_numpy(dtype=float),
                    block[left].to_numpy(dtype=float),
                )
                row = {
                    "figure": spec["id"],
                    "test": "paired_wilcoxon",
                    "dataset": dataset,
                    "metric": metric,
                    "comparison": f"{left}_vs_{right}",
                    "n": int(details["n"]),
                    "stat": float(details["stat"]),
                    "p_raw": float(details["p_raw"]),
                    "delta_mean": details["delta_mean"],
                    "delta_median": details["delta_median"],
                    "rank_biserial": details["rank_biserial"],
                }
                posthoc_rows.append(row)
                contrast_lookup[(left, right)] = row

            row_out = {
                "dataset": dataset_label,
                "metric": _metric_label(metric),
                "n": int(block.shape[0]),
            }
            for variant in ALIGNMENT_TARGET_ORDER:
                median, iqr = _median_iqr(block[variant].to_numpy(dtype=float))
                prefix = {
                    "off": "no_alignment",
                    "observable_state": "observable_state",
                    "sem_truth": "sem_target",
                }[variant]
                row_out[f"{prefix}_median"] = median
                row_out[f"{prefix}_iqr"] = iqr
            row_out["friedman_p_raw"] = friedman["p_raw"]
            row_out["q_obs_vs_no"] = float("nan")
            row_out["q_sem_vs_no"] = float("nan")
            row_out["q_sem_vs_obs"] = float("nan")
            table_rows.append(row_out)

    _bh_adjust_rows(friedman_rows)
    _bh_adjust_rows(posthoc_rows)
    for table_row in table_rows:
        metric_key = next(
            metric
            for metric in FIGURE_STRUCTURAL_ALIGNMENT_TARGET_TRUTH_METRICS
            if _metric_label(metric) == table_row["metric"]
        )
        dataset_key = next(dataset for dataset in dataset_order if _dataset_label(dataset) == table_row["dataset"])
        friedman_row = next(
            row
            for row in friedman_rows
            if row["dataset"] == dataset_key and row["metric"] == metric_key
        )
        table_row["friedman_q_bh"] = _numeric_row_value(friedman_row, "q_bh")
        table_row["q_obs_vs_no"] = _numeric_row_value(
            next(
                row
                for row in posthoc_rows
                if row["dataset"] == dataset_key
                and row["metric"] == metric_key
                and row["comparison"] == "observable_state_vs_off"
            ),
            "q_bh",
        )
        table_row["q_sem_vs_no"] = _numeric_row_value(
            next(
                row
                for row in posthoc_rows
                if row["dataset"] == dataset_key
                and row["metric"] == metric_key
                and row["comparison"] == "sem_truth_vs_off"
            ),
            "q_bh",
        )
        table_row["q_sem_vs_obs"] = _numeric_row_value(
            next(
                row
                for row in posthoc_rows
                if row["dataset"] == dataset_key
                and row["metric"] == metric_key
                and row["comparison"] == "sem_truth_vs_observable_state"
            ),
            "q_bh",
        )

    stats_rows = friedman_rows + posthoc_rows
    dataset_labels = [_dataset_label(dataset) for dataset in dataset_order]
    source_files = [path for record in records for path in record.get("source_files", [])]
    caption_lines = [
        (
            f"Panels A--D show {_metric_label('edge_f1')}, {_metric_label('endpoint_f1')}, {_metric_label('shd_strict')}, "
            f"and {_metric_label('shd_partial')} for model-level \acp{{PAG}} scored directly against the ground truth DAG "
            f"across {len(dataset_labels)} benchmark datasets ({', '.join(dataset_labels)}), holding the discovery "
            "configuration fixed while varying the alignment target."
        ),
        (
            "Observable-state alignment is the primary realistic condition, whereas SEM-target alignment uses privileged "
            "benchmark information and is included only as an oracle comparator. The plotted response is always the "
            "truth-based model score; the figure does not use agreement with a data-level PAG as its primary y-value."
        ),
        (
            "Dataset labels retain the benchmark palette from the shared figure styling, while orange, blue, and gray "
            "denote no alignment, observable-state alignment, and SEM-target alignment. Boxes show medians and "
            "interquartile ranges, and points show individual matched observations."
        ),
        (
            "Each dataset-metric block is evaluated with a Friedman omnibus test across matched model-fold triplets, "
            "followed by paired Wilcoxon signed-rank contrasts for observable-state vs no alignment, SEM-target vs no "
            "alignment, and SEM-target vs observable-state. Benjamini-Hochberg correction is applied within the figure."
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
        legend_mode="figure_bottom",
        annotation_mode="none",
        layout_profile="paper",
        layout_rect=(0.05, 0.07, 0.95, 1.0),
        table_rows=table_rows,
    )
