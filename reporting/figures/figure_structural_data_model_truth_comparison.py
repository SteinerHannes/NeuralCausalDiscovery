from typing import Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from reporting.figures.common import (
    ALIGNMENT_TARGET_COLORS,
    ALIGNMENT_TARGET_ORDER,
    ALIGNMENT_TARGET_SHORT_LABELS,
    STRUCTURAL_TRUTH_DELTA_AXIS_LABEL,
    _alignment_target_label,
    _bh_adjust_rows,
    _dataset_label,
    _ensure_non_empty,
    _jitter_values,
    _metric_label,
    _save_figure,
    _spearman_stats,
    set_reporting_theme,
    THESIS_TEXT_WIDTH,
)
from reporting.figures.structural_truth_utils import (
    PRIMARY_STRUCTURAL_METRICS,
    build_structural_truth_pairs,
)


FIGURE_STRUCTURAL_DATA_MODEL_TRUTH_COMPARISON_METRICS = list(PRIMARY_STRUCTURAL_METRICS)
FIGURE_STRUCTURAL_DATA_MODEL_TRUTH_COMPARISON_TITLE = (
    "Model-Level Recovery Relative to the Data-Level Reference"
)
FIGURE_STRUCTURAL_DATA_MODEL_TRUTH_COMPARISON_PANEL_LETTERS = {
    "edge_f1": {"scatter": "A", "delta": "B"},
    "endpoint_f1": {"scatter": "C", "delta": "D"},
}
DISPLAY_LOW = -0.05
DISPLAY_HIGH = 1.05
DELTA_LIMITS = {
    "edge_f1": (-1.05, 1.05),
    "endpoint_f1": (-1.05, 1.05)
}
JITTER_WIDTH = {"edge_f1": 0.008, "endpoint_f1": 0.018}
SCATTER_ALPHA = 0.46
BOX_ALPHA = 0.58
STRIP_ALPHA = 0.48


def _delta_wilcoxon_stats(values: np.ndarray) -> dict:
    valid = values[np.isfinite(values)]
    n = int(valid.size)
    if n == 0:
        return {
            "n": 0,
            "stat": 0.0,
            "p_raw": 1.0,
            "delta_mean": None,
            "delta_median": None,
        }
    non_zero = valid[~np.isclose(valid, 0.0, atol=1e-12, rtol=1e-8)]
    if non_zero.size < 2:
        stat = 0.0
        p_val = 1.0
    else:
        method = "exact" if non_zero.size <= 20 else "approx"
        result = stats.wilcoxon(non_zero, zero_method="wilcox", method=method)
        stat = 0.0 if np.isnan(result.statistic) else float(result.statistic)
        p_val = 1.0 if np.isnan(result.pvalue) else float(result.pvalue)
    return {
        "n": n,
        "stat": stat,
        "p_raw": p_val,
        "delta_mean": float(np.mean(valid)),
        "delta_median": float(np.median(valid)),
    }


def _alignment_target_plot_label(value: str) -> str:
    return ALIGNMENT_TARGET_SHORT_LABELS.get(value, _alignment_target_label(value))


def _fmt_prob(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "NA"
    value_f = float(value)
    if value_f < 0.001:
        return "<0.001"
    return f"{value_f:.3f}"


def _fmt_stat(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.3f}"


def figure_structural_data_model_truth_comparison(
    spec: dict,
    records: List[dict],
    output_dir: str,
) -> dict:
    set_reporting_theme(layout_profile="paper")
    title = str(spec.get("title", FIGURE_STRUCTURAL_DATA_MODEL_TRUTH_COMPARISON_TITLE))

    df = build_structural_truth_pairs(records=records, metrics=FIGURE_STRUCTURAL_DATA_MODEL_TRUTH_COMPARISON_METRICS)
    _ensure_non_empty(df, spec["id"])
    df = df.copy()
    df["alignment_variant_label"] = df["alignment_variant"].map(
        lambda value: _alignment_target_plot_label(str(value))
    )

    for metric in FIGURE_STRUCTURAL_DATA_MODEL_TRUTH_COMPARISON_METRICS:
        metric_df = df[df["metric"] == metric]
        present = set(metric_df["alignment_variant"].dropna().astype(str).tolist())
        missing = [variant for variant in ALIGNMENT_TARGET_ORDER if variant not in present]
        if missing:
            raise ValueError(f"{spec['id']}: missing alignment variants for {metric}: {', '.join(missing)}.")

    variant_label_order = [_alignment_target_plot_label(variant) for variant in ALIGNMENT_TARGET_ORDER]
    palette = {
        _alignment_target_plot_label(variant): ALIGNMENT_TARGET_COLORS[variant]
        for variant in ALIGNMENT_TARGET_ORDER
    }

    plot_df = df.copy()
    plot_df["data_truth_plot"] = plot_df["data_truth"].to_numpy(dtype=float)
    plot_df["model_truth_plot"] = plot_df["model_truth"].to_numpy(dtype=float)
    for seed_offset, metric in enumerate(FIGURE_STRUCTURAL_DATA_MODEL_TRUTH_COMPARISON_METRICS):
        metric_mask = plot_df["metric"] == metric
        metric_index = plot_df.index[metric_mask]
        if metric_index.empty:
            continue
        width = JITTER_WIDTH.get(metric, 0.01)
        if metric == "shd_strict":
            low = high = None
        else:
            low = DISPLAY_LOW
            high = DISPLAY_HIGH
        plot_df.loc[metric_index, "data_truth_plot"] = _jitter_values(
            plot_df.loc[metric_index, "data_truth"].to_numpy(dtype=float),
            seed=510 + seed_offset,
            width=width,
            low=low,
            high=high,
        )
        plot_df.loc[metric_index, "model_truth_plot"] = _jitter_values(
            plot_df.loc[metric_index, "model_truth"].to_numpy(dtype=float),
            seed=610 + seed_offset,
            width=width,
            low=low,
            high=high,
        )

    fig, scatter_grid = plt.subplots(
        nrows=len(FIGURE_STRUCTURAL_DATA_MODEL_TRUTH_COMPARISON_METRICS),
        ncols=1,
        figsize=(THESIS_TEXT_WIDTH, 7.0),
        sharex=False,
        sharey=False,
        squeeze=False,
    )

    metric_axes: Dict[str, dict] = {}
    for row_index, metric in enumerate(FIGURE_STRUCTURAL_DATA_MODEL_TRUTH_COMPARISON_METRICS):
        scatter_ax = scatter_grid[row_index, 0]
        metric_plot_df = plot_df[plot_df["metric"] == metric]
        sns.scatterplot(
            data=metric_plot_df,
            x="data_truth_plot",
            y="model_truth_plot",
            hue="alignment_variant_label",
            hue_order=variant_label_order,
            palette=palette,
            s=34,
            alpha=SCATTER_ALPHA,
            edgecolor="#f3f3f3",
            linewidth=0.25,
            legend=False,
            ax=scatter_ax,
        )
        scatter_ax.set_title(_metric_label(metric))
        divider = make_axes_locatable(scatter_ax)
        delta_ax = divider.append_axes("right", size="62%", pad=0.13)
        metric_axes[metric] = {"scatter": scatter_ax, "delta": delta_ax}

    corr_rows = []
    delta_rows = []
    table_rows = []
    scatter_axes = []
    delta_axes = []
    for row_index, metric in enumerate(FIGURE_STRUCTURAL_DATA_MODEL_TRUTH_COMPARISON_METRICS):
        metric_df = df[df["metric"] == metric].copy()
        scatter_ax = metric_axes[metric]["scatter"]
        delta_ax = metric_axes[metric]["delta"]

        combined_vals = np.concatenate(
            [
                metric_df["data_truth"].to_numpy(dtype=float),
                metric_df["model_truth"].to_numpy(dtype=float),
            ]
        )
        low, high = DISPLAY_LOW, DISPLAY_HIGH
        scatter_ax.plot([low, high], [low, high], linestyle="--", color="#333333", linewidth=0.9, zorder=1)
        scatter_ax.set_xlim(low, high)
        scatter_ax.set_ylim(low, high)
        scatter_ax.set_aspect("equal", adjustable="box")
        scatter_ax.set_xlabel("Data-level")
        scatter_ax.set_ylabel("Model-level")
        scatter_ax.grid(alpha=0.3)

        for variant in ALIGNMENT_TARGET_ORDER:
            variant_df = metric_df[metric_df["alignment_variant"] == variant]
            rho, p_val, n = _spearman_stats(
                variant_df["data_truth"].to_numpy(dtype=float),
                variant_df["model_truth"].to_numpy(dtype=float),
            )
            row = {
                "figure": spec["id"],
                "test": "spearman",
                "panel": "scatter",
                "metric": metric,
                "variant": variant,
                "n": n,
                "rho": rho,
                "p_raw": p_val,
            }
            corr_rows.append(row)
            table_rows.append(
                {
                    "metric": _metric_label(metric),
                    "alignment_mode": _alignment_target_label(variant),
                    "n": n,
                    "rho": rho,
                    "rho_q_bh": float("nan"),
                    "delta_mean": float("nan"),
                    "delta_median": float("nan"),
                    "delta_q_bh": float("nan"),
                }
            )
        scatter_axes.append(scatter_ax)

        sns.boxplot(
            data=metric_df,
            x="alignment_variant_label",
            y="delta",
            order=variant_label_order,
            hue="alignment_variant_label",
            hue_order=variant_label_order,
            palette=palette,
            dodge=False,
            width=0.58,
            showfliers=False,
            linewidth=0.95,
            boxprops={"alpha": BOX_ALPHA},
            medianprops={"linewidth": 1.6, "color": "#1c1c1c"},
            whiskerprops={"linewidth": 1.0},
            capprops={"linewidth": 1.0},
            ax=delta_ax,
        )
        sns.stripplot(
            data=metric_df,
            x="alignment_variant_label",
            y="delta",
            order=variant_label_order,
            hue="alignment_variant_label",
            hue_order=variant_label_order,
            palette=palette,
            dodge=False,
            jitter=0.16,
            alpha=STRIP_ALPHA,
            size=3.9,
            edgecolor="#111111",
            linewidth=0.4,
            ax=delta_ax,
        )
        if delta_ax.legend_ is not None:
            delta_ax.legend_.remove()
        delta_ax.axhline(0.0, color="#5e5e5e", linestyle=(0, (4, 2)), linewidth=1.0, alpha=0.95)
        delta_ax.set_title("Delta")
        low, high = DELTA_LIMITS.get(metric, (-1.0, 1.0))
        delta_ax.set_ylim(low, high)
        delta_ax.set_xlabel("Alignment mode")
        delta_ax.set_ylabel(STRUCTURAL_TRUTH_DELTA_AXIS_LABEL)
        delta_ax.yaxis.set_label_position("right")
        delta_ax.yaxis.tick_right()
        delta_ax.tick_params(axis="y", length=0)
        delta_ax.set_xticks(" ")
        delta_ax.grid(axis="y", alpha=0.3)

        for variant in ALIGNMENT_TARGET_ORDER:
            delta_values = metric_df.loc[metric_df["alignment_variant"] == variant, "delta"].to_numpy(dtype=float)
            details = _delta_wilcoxon_stats(delta_values)
            delta_rows.append(
                {
                    "figure": spec["id"],
                    "test": "one_sample_wilcoxon",
                    "panel": "delta",
                    "metric": metric,
                    "variant": variant,
                    "n": details["n"],
                    "stat": details["stat"],
                    "p_raw": details["p_raw"],
                    "delta_mean": details["delta_mean"],
                    "delta_median": details["delta_median"],
                }
            )
        delta_axes.append(delta_ax)

    _bh_adjust_rows(corr_rows)
    _bh_adjust_rows(delta_rows)

    for metric in FIGURE_STRUCTURAL_DATA_MODEL_TRUTH_COMPARISON_METRICS:
        scatter_ax = metric_axes[metric]["scatter"]
        delta_ax = metric_axes[metric]["delta"]
        panel_letters = FIGURE_STRUCTURAL_DATA_MODEL_TRUTH_COMPARISON_PANEL_LETTERS[metric]
        scatter_ax.text(
            -0.10,
            1.03,
            panel_letters["scatter"],
            transform=scatter_ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        delta_ax.text(
            -0.0,
            1.03,
            panel_letters["delta"],
            transform=delta_ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    table_index: Dict[tuple[str, str], dict] = {
        (row["metric"], row["alignment_mode"]): row for row in table_rows
    }
    for row in corr_rows:
        table_index[(_metric_label(row["metric"]), _alignment_target_label(row["variant"]))]["rho_q_bh"] = row.get(
            "q_bh"
        )
    for row in delta_rows:
        table_row = table_index[(_metric_label(row["metric"]), _alignment_target_label(row["variant"]))]
        table_row["delta_mean"] = row.get("delta_mean")
        table_row["delta_median"] = row.get("delta_median")
        table_row["delta_q_bh"] = row.get("q_bh")

    legend_handles = [
        Line2D(
            [],
            [],
            linestyle="",
            marker="o",
            markersize=6.2,
            markerfacecolor=palette[_alignment_target_plot_label(variant)],
            markeredgecolor="white",
            markeredgewidth=0.75,
            alpha=0.78,
            label=_alignment_target_plot_label(variant),
        )
        for variant in ALIGNMENT_TARGET_ORDER
    ]
    legend_handles.append(Line2D([], [], color="#333333", linestyle="--", linewidth=1.0, label="Identity line"))
    fig.legend(
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(title, y=0.995)

    dataset_labels = sorted({_dataset_label(dataset) for dataset in df["dataset"].dropna().unique().tolist()})
    model_count = int(df["model"].dropna().nunique())
    observation_count = int(df[["dataset", "model", "fold", "alignment_variant"]].drop_duplicates().shape[0])
    source_files = [path for record in records for path in record.get("source_files", [])]
    caption_lines = [
        (
            f"Matched folds from {len(dataset_labels)} benchmark datasets ({', '.join(dataset_labels)}) and "
            f"{model_count} model architectures are compared by scoring both the data-level PAG and the model-level PAG "
            f"against the same ground truth DAG ({observation_count} matched observations across alignment modes). "
            "Panels A and C show paired truth-based scatter plots with an identity line; panels B and D show the "
            "matched delta, Δ = model - data."
        ),
        (
            f"Panels report {_metric_label('edge_f1')}, {_metric_label('endpoint_f1')}, and {_metric_label('shd_strict')}. "
            "Positive delta values indicate model-side gains for the $F_1$ metrics, whereas SHD is interpreted in the "
            "opposite direction: negative deltas indicate that the model branch is closer to benchmark truth."
        ),
        (
            "Alignment mode is encoded by color. Delta distributions are tested against zero with one-sample "
            "Wilcoxon signed-rank tests, with Benjamini-Hochberg correction applied across the delta-test family."
        ),
    ]

    return _save_figure(
        fig=fig,
        axes=[metric_axes[metric]["scatter"] for metric in FIGURE_STRUCTURAL_DATA_MODEL_TRUTH_COMPARISON_METRICS]
             + [metric_axes[metric]["delta"] for metric in FIGURE_STRUCTURAL_DATA_MODEL_TRUTH_COMPARISON_METRICS],
        figure_id=spec["id"],
        output_dir=output_dir,
        stats_rows=corr_rows + delta_rows,
        data_row_count=len(df),
        source_files=source_files,
        title_generated=title,
        caption_lines=caption_lines,
        legend_mode="outside_bottom_shared",
        annotation_mode="dense_caption_only",
        layout_profile="paper",
        layout_rect=(0.0, 0.04, 1.0, 1.0),
        table_rows=list(table_index.values()),
    )
