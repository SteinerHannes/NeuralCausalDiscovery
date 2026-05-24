import textwrap
from typing import List, Optional

import pandas as pd
import seaborn as sns

from reporting.figures.common import (
    DATASET_COLORS,
    THESIS_TEXT_WIDTH,
    _dataset_label,
    _ensure_non_empty,
    _is_observable_alignment_record,
    _save_figure,
    plt,
    set_reporting_theme,
)
from reporting.figures.structural_truth_utils import dataset_order_present


FIGURE_INTERVENTION_MODEL_OUTPUT_DETECTION_TITLE = (
    "Benchmark-to-Model Detection of Output Intervention Effects"
)
PANEL_SPECS = [
    ("precision", "Precision", "A"),
    ("recall", "Recall", "B"),
    ("specificity", "Specificity", "C"),
    ("f1", "$F_1$ score", "D"),
]
DETECTION_Y_TICKS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
SUMMARY_RATE_COLUMNS = {
    "effect_positive_rate_output": [
        "effect_positive_rate_and_output",
        "effect_positive_rate_output",
    ],
}
SUMMARY_METRIC_COLUMNS = {
    "precision": ["model_precision_and_output", "model_precision_output"],
    "recall": ["model_recall_and_output", "model_recall_output"],
    "specificity": ["model_specificity_and_output", "model_specificity_output"],
    "f1": ["model_f1_and_output", "model_f1_output"],
}
SUMMARY_CONFUSION_COLUMNS = {
    "tp": ["model_tp_and_output", "model_tp_output"],
    "fp": ["model_fp_and_output", "model_fp_output"],
    "tn": ["model_tn_and_output", "model_tn_output"],
    "fn": ["model_fn_and_output", "model_fn_output"],
}
PAIR_LABEL_COLUMNS = {
    "y_true": ["effect_present_and", "effect_present"],
    "y_pred": ["model_effect_present_and", "model_effect_present"],
}


def _safe_divide(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _truthy_mask(series: pd.Series) -> pd.Series:
    normalized = series.fillna(False).astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "1.0"})


def _resolve_dataset_order(spec: dict, df: pd.DataFrame) -> list[str]:
    if "dataset_order" not in spec:
        return dataset_order_present(df)
    present = set(df["dataset"].dropna().astype(str).tolist())
    order = [str(dataset) for dataset in spec.get("dataset_order", [])]
    missing = [dataset for dataset in order if dataset not in present]
    if missing:
        raise ValueError(f"{spec['id']}: requested datasets not present: {', '.join(missing)}.")
    return order


def _first_present_column(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    return None


def _coerce_int(value) -> Optional[int]:
    if value is None or pd.isna(value):
        return None
    return int(value)


def _coerce_float(value) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    return float(value)


def _rows_from_counts(
    *,
    dataset: str,
    model: str,
    fold: int,
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    effect_positive_rate_output: Optional[float],
    metric_values: dict[str, Optional[float]],
) -> list[dict]:
    n_pairs = int(tp + fp + tn + fn)
    if n_pairs == 0:
        return []

    effect_positive_rate = effect_positive_rate_output
    if effect_positive_rate is None:
        effect_positive_rate = _safe_divide(tp + fn, n_pairs)
    model_positive_rate_output = _safe_divide(tp + fp, n_pairs)

    rows = []
    for metric, metric_label, _ in PANEL_SPECS:
        value = metric_values.get(metric)
        if value is None or pd.isna(value):
            continue
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": _dataset_label(dataset),
                "model": model,
                "fold": int(fold),
                "metric": metric,
                "metric_label": metric_label,
                "value": float(value),
                "n_pairs": n_pairs,
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "effect_positive_rate_output": effect_positive_rate,
                "model_positive_rate_output": model_positive_rate_output,
            }
        )
    return rows


def _detection_rows_from_summary_record(record: dict) -> list[dict]:
    int_df = record.get("intervention_summary_df")
    if int_df is None or int_df.empty:
        return []

    dataset = record.get("metadata", {}).get("dataset")
    model = record.get("metadata", {}).get("model")
    effect_rate_col = _first_present_column(
        int_df,
        SUMMARY_RATE_COLUMNS["effect_positive_rate_output"],
    )
    confusion_cols = {
        key: _first_present_column(int_df, candidates)
        for key, candidates in SUMMARY_CONFUSION_COLUMNS.items()
    }
    if any(col is None for col in confusion_cols.values()):
        return []

    metric_cols = {
        key: _first_present_column(int_df, candidates)
        for key, candidates in SUMMARY_METRIC_COLUMNS.items()
    }

    rows = []
    for fold, row in int_df.iterrows():
        try:
            fold_value = int(fold)
        except (TypeError, ValueError):
            continue

        tp = _coerce_int(row.get(confusion_cols["tp"]))
        fp = _coerce_int(row.get(confusion_cols["fp"]))
        tn = _coerce_int(row.get(confusion_cols["tn"]))
        fn = _coerce_int(row.get(confusion_cols["fn"]))
        if None in {tp, fp, tn, fn}:
            continue

        metric_values = {
            "precision": _coerce_float(row.get(metric_cols["precision"])) if metric_cols["precision"] else None,
            "recall": _coerce_float(row.get(metric_cols["recall"])) if metric_cols["recall"] else None,
            "specificity": _coerce_float(row.get(metric_cols["specificity"])) if metric_cols["specificity"] else None,
            "f1": _coerce_float(row.get(metric_cols["f1"])) if metric_cols["f1"] else None,
        }
        if metric_values["precision"] is None:
            metric_values["precision"] = _safe_divide(tp, tp + fp)
        if metric_values["recall"] is None:
            metric_values["recall"] = _safe_divide(tp, tp + fn)
        if metric_values["specificity"] is None:
            metric_values["specificity"] = _safe_divide(tn, tn + fp)
        if metric_values["f1"] is None:
            metric_values["f1"] = _safe_divide(2 * tp, 2 * tp + fp + fn)

        rows.extend(
            _rows_from_counts(
                dataset=dataset,
                model=model,
                fold=fold_value,
                tp=tp,
                fp=fp,
                tn=tn,
                fn=fn,
                effect_positive_rate_output=_coerce_float(row.get(effect_rate_col)) if effect_rate_col else None,
                metric_values=metric_values,
            )
        )
    return rows


def _detection_rows_from_pair_record(record: dict) -> list[dict]:
    pair_df = record.get("pair_df")
    if pair_df is None or pair_df.empty:
        return []
    if "fold" not in pair_df.columns or "outcome_role" not in pair_df.columns:
        return []

    y_true_col = _first_present_column(pair_df, PAIR_LABEL_COLUMNS["y_true"])
    y_pred_col = _first_present_column(pair_df, PAIR_LABEL_COLUMNS["y_pred"])
    if y_true_col is None or y_pred_col is None:
        return []

    dataset = record.get("metadata", {}).get("dataset")
    model = record.get("metadata", {}).get("model")
    output_df = pair_df[pair_df["outcome_role"] == "output"].copy()
    if output_df.empty:
        return []

    valid_mask = output_df[y_true_col].notna() & output_df[y_pred_col].notna()
    output_df = output_df.loc[valid_mask]
    if output_df.empty:
        return []

    rows = []
    for fold, fold_df in output_df.groupby("fold"):
        y_true = _truthy_mask(fold_df[y_true_col])
        y_pred = _truthy_mask(fold_df[y_pred_col])

        tp = int((y_pred & y_true).sum())
        fp = int((y_pred & ~y_true).sum())
        tn = int((~y_pred & ~y_true).sum())
        fn = int((~y_pred & y_true).sum())

        rows.extend(
            _rows_from_counts(
                dataset=dataset,
                model=model,
                fold=int(fold),
                tp=tp,
                fp=fp,
                tn=tn,
                fn=fn,
                effect_positive_rate_output=_safe_divide(tp + fn, tp + fp + tn + fn),
                metric_values={
                    "precision": _safe_divide(tp, tp + fp),
                    "recall": _safe_divide(tp, tp + fn),
                    "specificity": _safe_divide(tn, tn + fp),
                    "f1": _safe_divide(2 * tp, 2 * tp + fp + fn),
                },
            )
        )
    return rows


def _detection_rows_from_record(record: dict) -> list[dict]:
    summary_rows = _detection_rows_from_summary_record(record)
    if summary_rows:
        return summary_rows
    return _detection_rows_from_pair_record(record)


def figure_intervention_model_output_detection(
    spec: dict,
    records: List[dict],
    output_dir: str,
) -> dict:
    set_reporting_theme(layout_profile="paper")
    title = str(spec.get("title", FIGURE_INTERVENTION_MODEL_OUTPUT_DETECTION_TITLE))

    active_records = [record for record in records if _is_observable_alignment_record(record)]
    rows = []
    for record in active_records:
        rows.extend(_detection_rows_from_record(record))

    df = pd.DataFrame(rows)
    _ensure_non_empty(df, spec["id"])

    plot_metrics = [metric for metric, _, _ in PANEL_SPECS]
    plot_df = df[df["metric"].isin(plot_metrics)].copy()
    _ensure_non_empty(plot_df, spec["id"])

    dataset_order = _resolve_dataset_order(spec, plot_df)
    dataset_label_order = [_dataset_label(dataset) for dataset in dataset_order]
    dataset_tick_label_order = [
        "\n".join(textwrap.wrap(label, width=12, break_long_words=False))
        for label in dataset_label_order
    ]
    palette = {
        _dataset_label(dataset): DATASET_COLORS.get(dataset, "#6e6e6e")
        for dataset in dataset_order
    }

    for dataset in dataset_order:
        for metric in plot_metrics:
            subset = plot_df[(plot_df["dataset"] == dataset) & (plot_df["metric"] == metric)]
            if subset.empty:
                raise ValueError(f"{spec['id']}: missing rows for dataset={dataset}, metric={metric}.")

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(THESIS_TEXT_WIDTH, 5.2),
        sharex=True,
        sharey=False,
    )
    axes_flat = axes.flatten()

    for idx, (ax, (metric, metric_label, panel_letter)) in enumerate(zip(axes_flat, PANEL_SPECS)):
        panel_df = plot_df[plot_df["metric"] == metric].copy()
        sns.boxplot(
            data=panel_df,
            x="dataset_label",
            y="value",
            hue="dataset_label",
            order=dataset_label_order,
            hue_order=dataset_label_order,
            palette=palette,
            showfliers=False,
            width=0.56,
            linewidth=0.9,
            dodge=False,
            ax=ax,
        )
        sns.stripplot(
            data=panel_df,
            x="dataset_label",
            y="value",
            hue="dataset_label",
            order=dataset_label_order,
            hue_order=dataset_label_order,
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
        ax.set_yticks(DETECTION_Y_TICKS)
        ax.set_box_aspect(1)
        ax.set_xticks(range(len(dataset_tick_label_order)))
        ax.set_xticklabels(dataset_tick_label_order)
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

    fig.suptitle(title, y=0.995)

    table_rows = []
    for row in df.to_dict(orient="records"):
        table_rows.append(
            {
                "figure": spec["id"],
                "dataset": row["dataset"],
                "dataset_label": row["dataset_label"],
                "model": row["model"],
                "fold": row["fold"],
                "metric": row["metric"],
                "metric_label": row["metric_label"],
                "value": row["value"],
                "effect_positive_rate_output": row["effect_positive_rate_output"],
                "model_positive_rate_output": row["model_positive_rate_output"],
                "n_pairs": row["n_pairs"],
                "tp": row["tp"],
                "fp": row["fp"],
                "tn": row["tn"],
                "fn": row["fn"],
            }
        )

    dataset_summary = ", ".join(dataset_label_order)
    caption_lines = [
        "Panels A--D summarize benchmark-to-model precision, recall, specificity, and $F_1$ score under observable-state alignment, restricted to output outcomes and pooled across architectures with fold-level points over box summaries.",
        f"Each fold-model block compares the model-response confirmed-effect label against the benchmark confirmed-effect label on matched treatment-outcome intervention pairs within the datasets {dataset_summary}.",
        "The exported table reports both the benchmark-positive rate and the model-positive rate for each block so that precision and recall can be interpreted in the context of effect prevalence and the model's claim frequency.",
        "High precision with lower recall should be read as: the trained model rarely overclaims benchmark-positive output effects, but it still misses a non-trivial share of true effects.",
        "Because the benchmark label is the reference, any gap visible here arises at the benchmark-to-model stage, before the later model-PAG consistency comparison is introduced.",
    ]

    source_files = [path for record in active_records for path in record.get("source_files", [])]
    panel_row_counts = {
        metric: int((plot_df["metric"] == metric).sum())
        for metric in plot_metrics
    }
    return _save_figure(
        fig,
        axes,
        spec["id"],
        output_dir,
        stats_rows=[],
        data_row_count=len(plot_df),
        source_files=source_files,
        title_generated=title,
        caption_lines=caption_lines,
        legend_mode="none",
        annotation_mode="none",
        layout_profile="paper",
        layout_rect=(0.05, 0.0, 0.95, 1.0),
        extra={"panel_row_counts": panel_row_counts},
        table_rows=table_rows,
    )
