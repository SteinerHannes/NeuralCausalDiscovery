from typing import List, Optional

import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns

from reporting.figures.common import (
    DATASET_COLORS,
    THESIS_TEXT_WIDTH,
    _dataset_label,
    _ensure_non_empty,
    _is_observable_alignment_record,
    _move_legend_bottom,
    _save_figure,
    plt,
    set_reporting_theme,
)
from reporting.figures.structural_truth_utils import dataset_order_present


FIGURE_INTERVENTION_MODEL_PAG_CONSISTENCY_BENCHMARK_EXPANSION_TITLE = (
    "Model-PAG Consistency with Model Response Across Synthetic Benchmark Families"
)
PANEL_SPECS = [
    ("agreement_rate", "Agreement rate", "A"),
    ("pag_miss_rate", "PAG miss rate", "B"),
    ("pag_overclaim_rate", "PAG overclaim rate", "C"),
    ("effect_positive_rate", "Model-confirmed effect rate", "D"),
]
EXPORT_ONLY_METRICS = [
    ("f1", "$F_1$ score"),
]
DATASET_SHORT_LABELS = {
    "LinearSEMDataset": "A",
    "ElectricalCircuitData": "B",
    "ChainLinearSEM": "C",
    "ForkColliderLinearSEM": "D",
    "LayeredSparseLinearSEM": "E",
    "LayeredSparseLinearSEM_Laplace": "F",
    "LayeredSparseLinearSEM_StudentT3": "G",
    "ChainNonlinearSEM": "H",
    "ForkColliderNonlinearSEM": "I",
    "LayeredSparseNonlinearSEM": "J",
}


def _metric_label_lookup() -> dict[str, str]:
    labels = {name: label for name, label, _ in PANEL_SPECS}
    labels.update({name: label for name, label in EXPORT_ONLY_METRICS})
    return labels


def _float_or_none(value) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _safe_divide(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _resolve_dataset_order(spec: dict, df: pd.DataFrame) -> list[str]:
    if "dataset_order" not in spec:
        return dataset_order_present(df)
    present = set(df["dataset"].dropna().astype(str).tolist())
    order = [str(dataset) for dataset in spec.get("dataset_order", [])]
    missing = [dataset for dataset in order if dataset not in present]
    if missing:
        raise ValueError(f"{spec['id']}: requested datasets not present: {', '.join(missing)}.")
    return order


def _resolve_dataset_short_labels(spec: dict, dataset_order: list[str]) -> dict[str, str]:
    explicit = spec.get("dataset_short_labels")
    if explicit:
        short_labels = {str(dataset): str(label) for dataset, label in dict(explicit).items()}
    else:
        short_labels = dict(DATASET_SHORT_LABELS)
    missing = [dataset for dataset in dataset_order if dataset not in short_labels]
    if missing:
        raise ValueError(f"{spec['id']}: missing short labels for datasets: {', '.join(missing)}.")
    return {dataset: short_labels[dataset] for dataset in dataset_order}


def _consistency_rows_from_record(record: dict) -> list[dict]:
    pair_df = record.get("pair_df")
    if pair_df is None or pair_df.empty:
        return []
    required_cols = {"fold", "outcome_role", "pag_predicts_total", "model_effect_present"}
    if not required_cols.issubset(pair_df.columns):
        return []

    dataset = record.get("metadata", {}).get("dataset")
    model = record.get("metadata", {}).get("model")
    output_df = pair_df[pair_df["outcome_role"] == "output"].copy()
    if output_df.empty:
        return []

    valid_mask = output_df["pag_predicts_total"].notna() & output_df["model_effect_present"].notna()
    output_df = output_df.loc[valid_mask]
    if output_df.empty:
        return []

    rows = []
    for fold, fold_df in output_df.groupby("fold"):
        y_pred = fold_df["pag_predicts_total"].astype(bool)
        y_ref = fold_df["model_effect_present"].astype(bool)

        tp = int((y_pred & y_ref).sum())
        fp = int((y_pred & ~y_ref).sum())
        tn = int((~y_pred & ~y_ref).sum())
        fn = int((~y_pred & y_ref).sum())
        n_pairs = int(tp + fp + tn + fn)
        if n_pairs == 0:
            continue

        agreement_rate = float((tp + tn) / n_pairs)
        f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)
        pag_miss_rate = float(fn / n_pairs)
        pag_overclaim_rate = float(fp / n_pairs)
        effect_positive_rate = float((tp + fn) / n_pairs)

        for metric, value in [
            ("agreement_rate", agreement_rate),
            ("pag_miss_rate", pag_miss_rate),
            ("pag_overclaim_rate", pag_overclaim_rate),
            ("effect_positive_rate", effect_positive_rate),
            ("f1", f1),
        ]:
            if value is None or pd.isna(value):
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": _dataset_label(dataset),
                    "model": model,
                    "fold": int(fold),
                    "metric": metric,
                    "value": float(value),
                    "n_pairs": n_pairs,
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                }
            )
    return rows


def _summary_rows(
    df: pd.DataFrame,
    dataset_order: list[str],
    dataset_short_labels: dict[str, str],
) -> list[dict]:
    rows = []
    label_lookup = _metric_label_lookup()
    metric_order = [metric for metric, _, _ in PANEL_SPECS] + [
        metric for metric, _ in EXPORT_ONLY_METRICS
    ]

    for dataset in dataset_order:
        for metric in metric_order:
            subset = df[(df["dataset"] == dataset) & (df["metric"] == metric)].copy()
            if subset.empty:
                continue

            values = subset["value"].dropna()
            n_blocks = int(values.shape[0])
            if n_blocks == 0:
                continue

            rows.append(
                {
                    "scope": "dataset_metric",
                    "dataset": dataset,
                    "dataset_label": _dataset_label(dataset),
                    "dataset_short_label": dataset_short_labels[dataset],
                    "metric": metric,
                    "metric_label": label_lookup.get(metric, metric),
                    "n_blocks": n_blocks,
                    "mean": float(values.mean()),
                    "median": float(values.median()),
                    "std": _float_or_none(values.std(ddof=1)) if n_blocks > 1 else None,
                    "q25": float(values.quantile(0.25)),
                    "q75": float(values.quantile(0.75)),
                    "value_min": float(values.min()),
                    "value_max": float(values.max()),
                    "n_pairs_total": int(subset["n_pairs"].sum()),
                    "n_pairs_mean": float(subset["n_pairs"].mean()),
                    "n_pairs_median": float(subset["n_pairs"].median()),
                    "tp_total": int(subset["tp"].sum()),
                    "fp_total": int(subset["fp"].sum()),
                    "tn_total": int(subset["tn"].sum()),
                    "fn_total": int(subset["fn"].sum()),
                }
            )

    return rows


def figure_intervention_model_pag_consistency_benchmark_expansion(
    spec: dict,
    records: List[dict],
    output_dir: str,
) -> dict:
    set_reporting_theme(layout_profile="paper")
    title = str(spec.get("title", FIGURE_INTERVENTION_MODEL_PAG_CONSISTENCY_BENCHMARK_EXPANSION_TITLE))

    active_records = [record for record in records if _is_observable_alignment_record(record)]
    rows = []
    for record in active_records:
        rows.extend(_consistency_rows_from_record(record))

    df = pd.DataFrame(rows)
    _ensure_non_empty(df, spec["id"])

    plot_metrics = [metric for metric, _, _ in PANEL_SPECS]
    plot_df = df[df["metric"].isin(plot_metrics)].copy()
    _ensure_non_empty(plot_df, spec["id"])

    dataset_order = _resolve_dataset_order(spec, plot_df)
    dataset_short_labels = _resolve_dataset_short_labels(spec, dataset_order)
    dataset_short_label_order = [dataset_short_labels[dataset] for dataset in dataset_order]
    palette = {dataset_short_labels[dataset]: DATASET_COLORS.get(dataset, "#6e6e6e") for dataset in dataset_order}

    df = df.copy()
    plot_df = plot_df.copy()
    df["dataset_short_label"] = df["dataset"].map(dataset_short_labels)
    plot_df["dataset_short_label"] = plot_df["dataset"].map(dataset_short_labels)

    for dataset in dataset_order:
        for metric in plot_metrics:
            subset = plot_df[(plot_df["dataset"] == dataset) & (plot_df["metric"] == metric)]
            if subset.empty:
                raise ValueError(f"{spec['id']}: missing rows for dataset={dataset}, metric={metric}.")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(THESIS_TEXT_WIDTH, 6.8),
        sharex=True,
        sharey=False,
    )
    axes_flat = axes.flatten()

    for idx, (ax, (metric, metric_label, panel_letter)) in enumerate(zip(axes_flat, PANEL_SPECS)):
        panel_df = plot_df[plot_df["metric"] == metric].copy()
        sns.boxplot(
            data=panel_df,
            x="dataset_short_label",
            y="value",
            hue="dataset_short_label",
            order=dataset_short_label_order,
            hue_order=dataset_short_label_order,
            palette=palette,
            showfliers=False,
            width=0.56,
            linewidth=0.9,
            dodge=False,
            ax=ax,
        )
        sns.stripplot(
            data=panel_df,
            x="dataset_short_label",
            y="value",
            hue="dataset_short_label",
            order=dataset_short_label_order,
            hue_order=dataset_short_label_order,
            palette=palette,
            jitter=0.12,
            alpha=0.72,
            linewidth=0.2,
            edgecolor="#111111",
            size=4.8,
            dodge=False,
            ax=ax,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()
        ax.set_ylim(-0.05, 1.05)
        if idx < 2:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
        else:
            ax.set_xlabel("Dataset code")
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
        mpatches.Patch(
            facecolor=DATASET_COLORS.get(dataset, "#6e6e6e"),
            edgecolor="none",
            label=f"{dataset_short_labels[dataset]} = {_dataset_label(dataset)}",
        )
        for dataset in dataset_order
    ]
    _move_legend_bottom(fig, title="Dataset codes", ncol=2, extra_handles=legend_handles)

    fig.suptitle(title, y=0.995)

    table_rows = []
    label_lookup = _metric_label_lookup()
    for row in df.to_dict(orient="records"):
        metric = row["metric"]
        table_rows.append(
            {
                "figure": spec["id"],
                "dataset": row["dataset"],
                "dataset_label": row["dataset_label"],
                "dataset_short_label": row["dataset_short_label"],
                "model": row["model"],
                "fold": row["fold"],
                "metric": metric,
                "metric_label": label_lookup.get(metric, metric),
                "value": row["value"],
                "n_pairs": row["n_pairs"],
                "tp": row["tp"],
                "fp": row["fp"],
                "tn": row["tn"],
                "fn": row["fn"],
            }
        )

    dataset_summary = ", ".join(
        f"{dataset_short_labels[dataset]} = {_dataset_label(dataset)}"
        for dataset in dataset_order
    )
    caption_lines = [
        "Panels A--D summarize agreement rate, PAG miss rate, PAG overclaim rate, and model-confirmed effect rate for model-PAG consistency under observable-state alignment, restricted to output outcomes and pooled across architectures with fold-level points over box summaries.",
        "Each fold-model block compares the model-level PAG claim \\texttt{pag\\_predicts\\_total} against the same model's confirmed response label \\texttt{model\\_effect\\_present\\_and} on matched treatment-outcome pairs.",
        f"Dataset codes on the x-axis expand to: {dataset_summary}.",
        "The response-side label is the reference for this diagnostic consistency check; it is not benchmark causal ground truth.",
        "A PAG miss means the PAG predicts no total effect while the model response confirms one; a PAG overclaim means the PAG predicts a total effect while the model response does not confirm one. The model-confirmed effect rate reports how often the model-response reference is positive within the evaluated output pairs.",
        "High consistency therefore indicates behavioral alignment between the extracted PAG and the model's own intervention response pattern, but it does not by itself establish causal correctness with respect to the benchmark DAG. The exported table additionally reports $F_1$ score as a compact secondary summary.",
    ]

    source_files = [path for record in active_records for path in record.get("source_files", [])]
    panel_row_counts = {metric: int((plot_df["metric"] == metric).sum()) for metric in plot_metrics}
    stats_rows = _summary_rows(df, dataset_order, dataset_short_labels)
    return _save_figure(
        fig,
        axes,
        spec["id"],
        output_dir,
        stats_rows=stats_rows,
        data_row_count=len(plot_df),
        source_files=source_files,
        title_generated=title,
        caption_lines=caption_lines,
        legend_mode="outside_bottom_shared",
        annotation_mode="none",
        layout_profile="paper",
        layout_rect=(0.05, 0.16, 0.95, 1.0),
        extra={
            "panel_row_counts": panel_row_counts,
            "dataset_short_labels": dataset_short_labels,
        },
        table_rows=table_rows,
    )
