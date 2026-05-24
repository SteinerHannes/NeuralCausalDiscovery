from typing import List, Optional

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

from reporting.figures.common import (
    DATASET_ORDER,
    THESIS_TEXT_WIDTH,
    _dataset_label,
    _ensure_non_empty,
    _model_label,
    _save_figure,
    plt,
    set_reporting_theme,
)


FIGURE_TRAINING_DYNAMICS_TITLE = "Training and validation loss dynamics"
DEFAULT_MODEL_ORDER = ["ShallowMLP", "DeepMLP", "BottleneckMLP", "GatedMLP", "ResidualMLP"]
TRAIN_COLOR = "#2f6ca3"
VAL_COLOR = "#c95d25"
CURVE_LINEWIDTH = 1.25
DEFAULT_Y_TICK_INTERVAL = 0.2
EPOCH_BUDGET_COLOR = "#5d5d5d"
EPOCH_BUDGET_LINESTYLE = (0, (2.0, 2.0))
MODEL_SHORT_LABELS = {
    "ShallowMLP": "Shallow",
    "DeepMLP": "Deep",
    "BottleneckMLP": "Bottleneck",
    "GatedMLP": "Gated",
    "ResidualMLP": "Residual",
    "TabularMLP": "Tabular",
}


def _short_model_label(model: str) -> str:
    return MODEL_SHORT_LABELS.get(model, _model_label(model))


def _mean_or_none(values: pd.Series):
    clean = pd.to_numeric(values, errors="coerce").dropna()
    return float(clean.mean()) if not clean.empty else None


def _std_or_none(values: pd.Series):
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.std(ddof=1)) if clean.shape[0] > 1 else 0.0


def _epoch_budget(summary_panel: pd.DataFrame, panel_df: pd.DataFrame) -> Optional[int]:
    for frame, columns in [
        (summary_panel, ["planned_epochs", "effective_epochs", "configured_epochs"]),
        (panel_df, ["planned_epochs", "effective_epochs", "configured_epochs"]),
    ]:
        if frame is None or frame.empty:
            continue
        for column in columns:
            if column not in frame.columns:
                continue
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if not values.empty:
                return int(values.max())
    return None


def _mean_ci_curve(panel_df: pd.DataFrame, metric: str, expected_folds: int) -> pd.DataFrame:
    grouped = panel_df.groupby("epoch")[metric].agg(["mean", "std", "count"]).reset_index()
    if expected_folds > 0:
        grouped = grouped[grouped["count"] >= expected_folds]
        if grouped.empty:
            grouped = panel_df.groupby("epoch")[metric].agg(["mean", "std", "count"]).reset_index()
    grouped["ci95"] = 1.96 * grouped["std"].fillna(0.0) / np.sqrt(grouped["count"].clip(lower=1))
    grouped["lower"] = grouped["mean"] - grouped["ci95"]
    grouped["upper"] = grouped["mean"] + grouped["ci95"]
    return grouped


def _summary_row(
    figure_id: str,
    dataset: str,
    model: str,
    panel_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict:
    idx = panel_df.groupby("fold")["val_loss"].idxmin()
    best_per_fold = panel_df.loc[idx, ["fold", "epoch", "val_loss"]].copy()
    best_per_fold = best_per_fold.sort_values("fold")
    final_per_fold = (
        panel_df.sort_values(["fold", "epoch"])
        .groupby("fold", as_index=False)
        .tail(1)[["fold", "epoch", "train_loss", "val_loss"]]
        .rename(
            columns={
                "epoch": "final_epoch",
                "train_loss": "final_train_loss",
                "val_loss": "final_val_loss",
            }
        )
    )

    fold_summary = best_per_fold.rename(
        columns={
            "epoch": "best_epoch",
            "val_loss": "best_val_loss",
        }
    ).merge(final_per_fold, on="fold", how="outer")

    if test_df is not None and not test_df.empty and "test_loss" in test_df.columns:
        fold_summary = fold_summary.merge(
            test_df[["fold", "test_loss"]].dropna(subset=["fold", "test_loss"]),
            on="fold",
            how="left",
        )
    else:
        fold_summary["test_loss"] = np.nan

    fold_summary["final_val_minus_train_loss"] = (
        fold_summary["final_val_loss"] - fold_summary["final_train_loss"]
    )
    fold_summary["test_minus_best_val_loss"] = fold_summary["test_loss"] - fold_summary["best_val_loss"]

    early_stop_rate = None
    if summary_df is not None and not summary_df.empty and "early_stopping_triggered" in summary_df.columns:
        early_vals = summary_df["early_stopping_triggered"].dropna()
        if not early_vals.empty:
            early_stop_rate = float(early_vals.astype(bool).astype(float).mean())

    return {
        "figure": figure_id,
        "test": "descriptive",
        "dataset": dataset,
        "model": model,
        "n": int(best_per_fold.shape[0]),
        "n_folds": int(best_per_fold["fold"].nunique()),
        "best_epoch_mean": _mean_or_none(fold_summary["best_epoch"]),
        "best_epoch_std": _std_or_none(fold_summary["best_epoch"]),
        "final_epoch_mean": _mean_or_none(fold_summary["final_epoch"]),
        "final_epoch_std": _std_or_none(fold_summary["final_epoch"]),
        "final_train_loss_mean": _mean_or_none(fold_summary["final_train_loss"]),
        "final_train_loss_std": _std_or_none(fold_summary["final_train_loss"]),
        "final_val_loss_mean": _mean_or_none(fold_summary["final_val_loss"]),
        "final_val_loss_std": _std_or_none(fold_summary["final_val_loss"]),
        "best_val_loss_mean": _mean_or_none(fold_summary["best_val_loss"]),
        "best_val_loss_std": _std_or_none(fold_summary["best_val_loss"]),
        "test_loss_mean": _mean_or_none(fold_summary["test_loss"]),
        "test_loss_std": _std_or_none(fold_summary["test_loss"]),
        "final_val_minus_train_loss_mean": _mean_or_none(fold_summary["final_val_minus_train_loss"]),
        "test_minus_best_val_loss_mean": _mean_or_none(fold_summary["test_minus_best_val_loss"]),
        "early_stop_rate": early_stop_rate,
    }


def _table_row(row: dict) -> dict:
    return {
        "dataset": _dataset_label(row["dataset"]),
        "model": _model_label(row["model"]),
        "n_folds": row["n_folds"],
        "best_epoch_mean": row["best_epoch_mean"],
        "final_epoch_mean": row["final_epoch_mean"],
        "final_train_loss_mean": row["final_train_loss_mean"],
        "best_val_loss_mean": row["best_val_loss_mean"],
        "test_loss_mean": row["test_loss_mean"],
        "final_val_minus_train_loss_mean": row["final_val_minus_train_loss_mean"],
        "test_minus_best_val_loss_mean": row["test_minus_best_val_loss_mean"],
        "early_stop_rate": row["early_stop_rate"],
    }


def figure_training_dynamics(spec: dict, records: List[dict], output_dir: str) -> dict:
    set_reporting_theme(layout_profile="paper")

    history_rows = []
    summary_rows = []
    test_score_rows = []
    source_files = []
    for record in records:
        history_df = record.get("training_history_df")
        if history_df is None or history_df.empty:
            continue

        meta = record.get("metadata", {})
        dataset = meta.get("dataset")
        model = meta.get("model")
        if not dataset or not model:
            continue

        frame = history_df.copy()
        for col in ["epoch", "train_loss", "val_loss", "fold"]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
        frame = frame.dropna(subset=["epoch", "train_loss", "val_loss", "fold"])
        if frame.empty:
            continue
        frame["epoch"] = frame["epoch"].astype(int)
        frame["fold"] = frame["fold"].astype(int)
        frame["dataset"] = dataset
        frame["model"] = model
        history_columns = ["dataset", "model", "fold", "epoch", "train_loss", "val_loss"]
        for col in ["planned_epochs", "effective_epochs", "configured_epochs"]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")
                history_columns.append(col)
        history_rows.append(frame[history_columns])

        training_summary_df = record.get("training_summary_df")
        if training_summary_df is not None and not training_summary_df.empty:
            summary_frame = training_summary_df.copy()
            summary_frame["fold"] = summary_frame.index.astype(int)
            summary_frame["dataset"] = dataset
            summary_frame["model"] = model
            summary_rows.append(summary_frame)

        test_scores_df = record.get("train_test_scores_df")
        if test_scores_df is not None and not test_scores_df.empty:
            test_frame = test_scores_df.copy()
            if "TestLoss" in test_frame.columns and "test_loss" not in test_frame.columns:
                test_frame["test_loss"] = pd.to_numeric(test_frame["TestLoss"], errors="coerce")
            elif "test_loss" in test_frame.columns:
                test_frame["test_loss"] = pd.to_numeric(test_frame["test_loss"], errors="coerce")
            if "fold" in test_frame.columns and "test_loss" in test_frame.columns:
                test_frame["fold"] = pd.to_numeric(test_frame["fold"], errors="coerce")
                test_frame = test_frame.dropna(subset=["fold", "test_loss"])
                if not test_frame.empty:
                    test_frame["fold"] = test_frame["fold"].astype(int)
                    test_frame["dataset"] = dataset
                    test_frame["model"] = model
                    test_score_rows.append(test_frame[["dataset", "model", "fold", "test_loss"]])

        source_files.extend(record.get("training_source_files", []))

    history = pd.concat(history_rows, ignore_index=True) if history_rows else pd.DataFrame()
    _ensure_non_empty(history, spec["id"])
    summaries = pd.concat(summary_rows, ignore_index=True) if summary_rows else pd.DataFrame()
    test_scores = pd.concat(test_score_rows, ignore_index=True) if test_score_rows else pd.DataFrame()

    dataset_order = [d for d in DATASET_ORDER if d in set(history["dataset"].unique())]
    if not dataset_order:
        dataset_order = sorted(history["dataset"].dropna().unique().tolist())
    available_models = set(history["model"].dropna().unique().tolist())
    requested_order = list(spec.get("model_order", DEFAULT_MODEL_ORDER))
    if not requested_order:
        requested_order = list(DEFAULT_MODEL_ORDER)
    model_order = []
    for model in requested_order + DEFAULT_MODEL_ORDER + sorted(available_models):
        if model in available_models and model not in model_order:
            model_order.append(model)

    figure_height = float(spec.get("height", max(5.8, 1.05 * len(model_order) + 1.0)))
    fig, axes = plt.subplots(
        len(model_order),
        len(dataset_order),
        figsize=(THESIS_TEXT_WIDTH, figure_height),
        dpi=170,
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    stats_rows = []
    table_rows = []
    global_epoch_max = 1
    global_lower = []
    global_upper = []
    no_data_axes = set()
    for row_idx, model in enumerate(model_order):
        for col_idx, dataset in enumerate(dataset_order):
            ax = axes[row_idx, col_idx]
            ax.tick_params(axis="both", labelsize=7)
            panel = history[(history["dataset"] == dataset) & (history["model"] == model)]
            if panel.empty:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                no_data_axes.add((row_idx, col_idx))
                if row_idx == 0:
                    ax.set_title(_dataset_label(dataset))
                continue

            expected_folds = int(panel["fold"].nunique())
            train_curve = _mean_ci_curve(panel, metric="train_loss", expected_folds=expected_folds)
            val_curve = _mean_ci_curve(panel, metric="val_loss", expected_folds=expected_folds)
            if train_curve.empty or val_curve.empty:
                ax.text(0.5, 0.5, "Insufficient fold overlap", ha="center", va="center", transform=ax.transAxes, fontsize=9)
                ax.set_xticks([])
                ax.set_yticks([])
                no_data_axes.add((row_idx, col_idx))
                if row_idx == 0:
                    ax.set_title(_dataset_label(dataset))
                continue

            summary_panel = pd.DataFrame()
            if not summaries.empty:
                summary_panel = summaries[(summaries["dataset"] == dataset) & (summaries["model"] == model)]
            test_panel = pd.DataFrame()
            if not test_scores.empty:
                test_panel = test_scores[(test_scores["dataset"] == dataset) & (test_scores["model"] == model)]
            panel_epoch_budget = _epoch_budget(summary_panel, panel)

            global_epoch_max = max(global_epoch_max, int(train_curve["epoch"].max()), int(val_curve["epoch"].max()))
            if panel_epoch_budget is not None:
                global_epoch_max = max(global_epoch_max, panel_epoch_budget)
            global_lower.extend(train_curve["lower"].tolist())
            global_lower.extend(val_curve["lower"].tolist())
            global_upper.extend(train_curve["upper"].tolist())
            global_upper.extend(val_curve["upper"].tolist())

            ax.plot(train_curve["epoch"], train_curve["mean"], color=TRAIN_COLOR, linewidth=CURVE_LINEWIDTH, label="Train loss")
            ax.fill_between(train_curve["epoch"], train_curve["lower"], train_curve["upper"], color=TRAIN_COLOR, alpha=0.18)
            ax.plot(val_curve["epoch"], val_curve["mean"], color=VAL_COLOR, linewidth=CURVE_LINEWIDTH, label="Val loss")
            ax.fill_between(val_curve["epoch"], val_curve["lower"], val_curve["upper"], color=VAL_COLOR, alpha=0.18)
            if panel_epoch_budget is not None:
                ax.axvline(
                    panel_epoch_budget,
                    color=EPOCH_BUDGET_COLOR,
                    linestyle=EPOCH_BUDGET_LINESTYLE,
                    linewidth=0.8,
                    alpha=0.85,
                    zorder=0,
                )
                ax.text(
                    panel_epoch_budget,
                    0.97,
                    f"max {panel_epoch_budget}",
                    transform=ax.get_xaxis_transform(),
                    rotation=90,
                    va="top",
                    ha="right",
                    fontsize=6.5,
                    color=EPOCH_BUDGET_COLOR,
                )

            ax.grid(axis="y", alpha=0.3)
            if row_idx == len(model_order) - 1:
                ax.set_xlabel("Epoch")
            else:
                ax.set_xlabel("")
            if col_idx == 0:
                if row_idx == len(model_order) // 2:
                    ax.set_ylabel("Loss")
                ax.text(
                    -0.26,
                    0.5,
                    _short_model_label(model),
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=8,
                )
            if row_idx == 0:
                ax.set_title(_dataset_label(dataset), fontsize=9)

            stats_row = _summary_row(spec["id"], dataset, model, panel, summary_panel, test_panel)
            stats_rows.append(stats_row)
            table_rows.append(_table_row(stats_row))

    if global_lower and global_upper:
        y_min = float(np.nanmin(global_lower))
        y_max = float(np.nanmax(global_upper))
        y_tick_interval = float(spec.get("y_tick_interval", DEFAULT_Y_TICK_INTERVAL))
        if np.isfinite(y_min) and np.isfinite(y_max):
            if np.isclose(y_min, y_max):
                pad = max(0.05 * abs(y_min), 1e-3)
                y_limits = (y_min - pad, y_max + pad)
            else:
                pad = 0.05 * (y_max - y_min)
                y_limits = (y_min - pad, y_max + pad)
            for row_idx, col_idx in np.ndindex(axes.shape):
                if (row_idx, col_idx) in no_data_axes:
                    continue
                ax = axes[row_idx, col_idx]
                ax.set_ylim(*y_limits)
                ax.set_xlim(1, global_epoch_max)
                if y_tick_interval > 0:
                    ax.yaxis.set_major_locator(MultipleLocator(y_tick_interval))
                    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

    handles = [
        Line2D([0], [0], color=TRAIN_COLOR, linewidth=CURVE_LINEWIDTH, label="Train loss"),
        Line2D([0], [0], color=VAL_COLOR, linewidth=CURVE_LINEWIDTH, label="Validation loss"),
        Line2D(
            [0],
            [0],
            color=EPOCH_BUDGET_COLOR,
            linestyle=EPOCH_BUDGET_LINESTYLE,
            linewidth=0.9,
            label="Planned max epoch",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=3, frameon=False)
    fig.suptitle(FIGURE_TRAINING_DYNAMICS_TITLE, fontsize=11)

    caption_lines = [
        "Fold-aggregated training and validation loss trajectories for the core trained \\acp{MLP}.",
        "Lines show means across folds and shaded regions indicate 95\\% confidence intervals at each epoch.",
        "Rows correspond to model families and columns correspond to datasets; the companion table reports final training, best-validation, and held-out test-loss summaries.",
        "Dotted vertical lines mark the planned maximum epoch budget from the model-specific training profile.",
        "Curves are restricted to epochs observed in all folds of each panel to avoid late-epoch survivorship bias.",
        "This appendix diagnostic is used to check optimization stability and potential under- or overfitting confounds.",
    ]

    return _save_figure(
        fig=fig,
        axes=axes,
        figure_id=spec["id"],
        output_dir=output_dir,
        stats_rows=stats_rows,
        data_row_count=len(history),
        source_files=sorted({path for path in source_files if path}),
        title_generated=FIGURE_TRAINING_DYNAMICS_TITLE,
        caption_lines=caption_lines,
        legend_mode="outside_bottom_shared",
        annotation_mode="dense_caption_only",
        layout_profile="paper",
        layout_rect=(0.0, 0.06, 1.0, 0.97),
        table_rows=table_rows,
    )
