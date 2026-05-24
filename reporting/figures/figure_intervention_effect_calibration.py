from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns

from reporting.figures.common import (
    DATASET_COLORS,
    DATASET_ORDER,
    _deduplicate_invariant_rows,
    _dataset_label,
    _ensure_non_empty,
    _is_observable_alignment_record,
    _model_label,
    _move_legend_bottom,
    _save_figure,
    plt,
    set_reporting_theme, THESIS_TEXT_WIDTH,
)


FIGURE_TITLE = "Effect Magnitude Calibration"
SOURCE_ORDER = [("dowhy", "DoWhy reference"), ("model", "Model")]
SOURCE_PANEL_LETTERS = {"dowhy": "A", "model": "B"}
AXIS_LIMIT_PADDING = 0.05
AXIS_TICK_STEP = 0.5
SHOW_DATASET_REGRESSION_LINES = True
DATASET_REGRESSION_LINE_ALPHA = 0.45


def _join_list(items: List[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _shared_limits(values: np.ndarray) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-1.0, 1.0)
    limit = float(np.max(np.abs(finite)))
    if np.isclose(limit, 0.0):
        limit = 1.0
    limit *= 1.0 + AXIS_LIMIT_PADDING
    return (-limit, limit)


def _shared_ticks(low: float, high: float) -> np.ndarray:
    start = np.ceil(low / AXIS_TICK_STEP) * AXIS_TICK_STEP
    end = np.floor(high / AXIS_TICK_STEP) * AXIS_TICK_STEP
    ticks = np.arange(start, end + AXIS_TICK_STEP * 0.5, AXIS_TICK_STEP, dtype=float)
    if ticks.size == 0:
        ticks = np.array([0.0], dtype=float)
    return ticks


def _ols_fit(x: np.ndarray, y: np.ndarray) -> Tuple[Optional[float], Optional[float]]:
    if x.size < 2 or np.isclose(np.var(x), 0.0):
        return None, None
    try:
        beta, alpha = np.polyfit(x, y, 1)
    except (np.linalg.LinAlgError, ValueError):
        return None, None
    if not (np.isfinite(beta) and np.isfinite(alpha)):
        return None, None
    return float(beta), float(alpha)


def _pearson_r(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size < 2 or np.isclose(np.var(x), 0.0) or np.isclose(np.var(y), 0.0):
        return None
    corr = float(np.corrcoef(x, y)[0, 1])
    return corr if np.isfinite(corr) else None


def _mean_error(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size == 0 or y.size == 0:
        return None
    error = float(np.mean(y - x))
    return error if np.isfinite(error) else None


def _diagnostic_summary(x_vals: np.ndarray, y_vals: np.ndarray) -> dict:
    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[mask]
    y_vals = y_vals[mask]
    beta, alpha = _ols_fit(x_vals, y_vals)
    return {
        "n": int(x_vals.size),
        "r": _pearson_r(x_vals, y_vals),
        "mae": float(np.mean(np.abs(y_vals - x_vals))) if x_vals.size > 0 else None,
        "bias": _mean_error(x_vals, y_vals),
        "beta": beta,
        "alpha": alpha,
    }


def _draw_dataset_regression_lines(
    ax,
    panel_df: pd.DataFrame,
    dataset_order: List[str],
    palette: dict,
) -> None:
    if not SHOW_DATASET_REGRESSION_LINES:
        return
    for dataset in dataset_order:
        dataset_df = panel_df[panel_df["dataset"] == dataset]
        if dataset_df.empty:
            continue
        x_vals = dataset_df["x"].to_numpy(dtype=float)
        y_vals = dataset_df["y"].to_numpy(dtype=float)
        mask = np.isfinite(x_vals) & np.isfinite(y_vals)
        x_vals = x_vals[mask]
        y_vals = y_vals[mask]
        beta, alpha = _ols_fit(x_vals, y_vals)
        if beta is None or alpha is None or x_vals.size < 2:
            continue
        x_line = np.array([float(np.min(x_vals)), float(np.max(x_vals))], dtype=float)
        if np.isclose(x_line[0], x_line[1]):
            continue
        y_line = beta * x_line + alpha
        ax.plot(
            x_line,
            y_line,
            color=palette.get(_dataset_label(dataset), "#6e6e6e"),
            linewidth=1.5,
            alpha=DATASET_REGRESSION_LINE_ALPHA,
            solid_capstyle="round",
            zorder=2,
        )


def _fmt(value: Optional[float], digits: int) -> str:
    if value is None or not np.isfinite(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _is_truthy(value) -> bool:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "1.0"}


def figure_intervention_effect_calibration(spec: dict, records: List[dict], output_dir: str) -> dict:
    set_reporting_theme(layout_profile="paper")
    dataset_order = list(spec.get("dataset_order", DATASET_ORDER))
    title_generated = str(spec.get("title", FIGURE_TITLE))
    effect_presence_col = "effect_present"

    active_records = [record for record in records if _is_observable_alignment_record(record)]
    rows = []
    for record in active_records:
        pair_df = record.get("pair_df")
        if pair_df is None or pair_df.empty or effect_presence_col not in pair_df.columns:
            continue
        dataset = record.get("metadata", {}).get("dataset")
        model = record.get("metadata", {}).get("model")
        for _, row in pair_df.iterrows():
            if row.get("outcome_role") != "output" or not _is_truthy(row.get(effect_presence_col)):
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": _dataset_label(dataset),
                    "model": model,
                    "fold": row.get("fold"),
                    "treatment": row.get("treatment"),
                    "outcome": row.get("outcome"),
                    "source": "dowhy",
                    "source_label": "DoWhy reference",
                    "x": row.get("ate_true_std"),
                    "y": row.get("ate_est_std"),
                }
            )
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": _dataset_label(dataset),
                    "model": model,
                    "fold": row.get("fold"),
                    "treatment": row.get("treatment"),
                    "outcome": row.get("outcome"),
                    "source": "model",
                    "source_label": "Model",
                    "x": row.get("ate_true_std"),
                    "y": row.get("model_ate_std"),
                }
            )

    df = pd.DataFrame(rows).dropna(subset=["x", "y"])
    _ensure_non_empty(df, spec["id"])
    df = _deduplicate_invariant_rows(
        df,
        subset=["dataset", "fold", "treatment", "outcome", "source"],
        source_col="source",
        invariant_sources=["dowhy"],
    )

    panel_values = df[["x", "y"]].to_numpy(dtype=float).ravel()
    low, high = _shared_limits(panel_values)
    shared_ticks = _shared_ticks(low, high)
    present_datasets = set(df["dataset"].dropna().astype(str).tolist())
    hue_order = [_dataset_label(dataset) for dataset in dataset_order if dataset in present_datasets]
    if not hue_order:
        hue_order = sorted(df["dataset_label"].dropna().astype(str).unique().tolist())
    palette = {
        _dataset_label(dataset): DATASET_COLORS.get(dataset, "#6e6e6e")
        for dataset in dataset_order
    }
    for dataset_label in df["dataset_label"].dropna().astype(str).unique().tolist():
        palette.setdefault(dataset_label, "#6e6e6e")

    fig, axes = plt.subplots(1, 2, figsize=(THESIS_TEXT_WIDTH, 3.6), sharex=True, sharey=True)
    stats_rows = []
    table_rows = []
    mae_by_source = {}
    for ax, (source_key, source_label) in zip(axes.flat, SOURCE_ORDER):
        panel_df = df[df["source"] == source_key]
        sns.scatterplot(
            data=panel_df,
            x="x",
            y="y",
            hue="dataset_label",
            hue_order=hue_order,
            palette=palette,
            s=28,
            alpha=0.72,
            edgecolor="#f1f1f1",
            linewidth=0.3,
            ax=ax,
        )
        if ax.legend_ is not None:
            ax.legend_.remove()
        ax.plot([low, high], [low, high], linestyle=(0, (6, 3)), color="#404040", linewidth=1.2)
        _draw_dataset_regression_lines(ax, panel_df, dataset_order, palette)
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        ax.set_xticks(shared_ticks)
        ax.set_yticks(shared_ticks)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(source_label)
        ax.set_xlabel("True standardized ATE")
        ax.set_ylabel("Estimated standardized ATE")
        ax.grid(alpha=0.28)
        ax.text(
            0.0 if source_key == "model" else -0.10,
            1.03,
            SOURCE_PANEL_LETTERS[source_key],
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="bottom",
        )
        overall_summary = _diagnostic_summary(
            panel_df["x"].to_numpy(dtype=float),
            panel_df["y"].to_numpy(dtype=float),
        )
        mae_by_source[source_key] = overall_summary["mae"]
        stats_rows.append(
            {
                "figure": spec["id"],
                "test": "diagnostic",
                "source": source_key,
                "dataset": "all",
                "dataset_label": "All datasets",
                **overall_summary,
            }
        )
        table_rows.append(
            {
                "figure": spec["id"],
                "scope": "overall",
                "dataset": "all",
                "dataset_label": "All datasets",
                "source": source_key,
                "source_label": source_label,
                **overall_summary,
            }
        )

        for dataset in dataset_order:
            dataset_panel_df = panel_df[panel_df["dataset"] == dataset]
            if dataset_panel_df.empty:
                continue
            dataset_summary = _diagnostic_summary(
                dataset_panel_df["x"].to_numpy(dtype=float),
                dataset_panel_df["y"].to_numpy(dtype=float),
            )
            stats_rows.append(
                {
                    "figure": spec["id"],
                    "test": "diagnostic",
                    "source": source_key,
                    "dataset": dataset,
                    "dataset_label": _dataset_label(dataset),
                    **dataset_summary,
                }
            )
            table_rows.append(
                {
                    "figure": spec["id"],
                    "scope": "dataset",
                    "dataset": dataset,
                    "dataset_label": _dataset_label(dataset),
                    "source": source_key,
                    "source_label": source_label,
                    **dataset_summary,
                }
            )

    dowhy_mae = mae_by_source.get("dowhy")
    model_mae = mae_by_source.get("model")
    if dowhy_mae is not None and model_mae is not None:
        stats_rows.append(
            {
                "figure": spec["id"],
                "test": "delta",
                "metric": "mae",
                "dataset": "all",
                "dataset_label": "All datasets",
                "source": "model_minus_dowhy",
                "delta_mae": float(model_mae - dowhy_mae),
            }
        )
        table_rows.append(
            {
                "figure": spec["id"],
                "scope": "overall",
                "dataset": "all",
                "dataset_label": "All datasets",
                "source": "model_minus_dowhy",
                "source_label": "Model minus DoWhy",
                "metric": "mae",
                "delta_mae": float(model_mae - dowhy_mae),
            }
        )
    for dataset in dataset_order:
        dowhy_dataset = next(
            (
                row for row in stats_rows
                if row.get("test") == "diagnostic"
                and row.get("source") == "dowhy"
                and row.get("dataset") == dataset
            ),
            None,
        )
        model_dataset = next(
            (
                row for row in stats_rows
                if row.get("test") == "diagnostic"
                and row.get("source") == "model"
                and row.get("dataset") == dataset
            ),
            None,
        )
        if dowhy_dataset is None or model_dataset is None:
            continue
        delta_mae = (
            float(model_dataset["mae"] - dowhy_dataset["mae"])
            if model_dataset.get("mae") is not None and dowhy_dataset.get("mae") is not None
            else None
        )
        stats_rows.append(
            {
                "figure": spec["id"],
                "test": "delta",
                "metric": "mae",
                "dataset": dataset,
                "dataset_label": _dataset_label(dataset),
                "source": "model_minus_dowhy",
                "delta_mae": delta_mae,
            }
        )
        table_rows.append(
            {
                "figure": spec["id"],
                "scope": "dataset",
                "dataset": dataset,
                "dataset_label": _dataset_label(dataset),
                "source": "model_minus_dowhy",
                "source_label": "Model minus DoWhy",
                "metric": "mae",
                "delta_mae": delta_mae,
            }
        )

    fig.suptitle(title_generated, y=1.02)
    _move_legend_bottom(fig, title="Dataset", ncol=min(5, max(len(hue_order), 1)))

    dataset_labels = [_dataset_label(dataset) for dataset in dataset_order if dataset in present_datasets]
    if not dataset_labels:
        dataset_labels = list(hue_order)
    models_present = sorted(
        {
            _model_label(record.get("metadata", {}).get("model"))
            for record in active_records
            if str(record.get("metadata", {}).get("model", "")).strip()
        }
    )
    if dataset_order == list(DATASET_ORDER):
        caption_lines = [
            "Panels A and B show continuous intervention-effect calibration under observable-state alignment, restricted to output intervention pairs with a benchmark-confirmed effect and standardized effect magnitudes.",
            "Panel A shows the DoWhy reference baseline against true standardized effects; Panel B shows model-derived standardized effects against the same truth.",
            "Identity-line proximity indicates stronger calibration. The figure itself stays descriptive, while per-dataset and pooled diagnostics are exported separately as tables.",
            "Observations near y=0 in Panel B indicate near-zero model responses under the intervention protocol.",
            "The exported diagnostics report Pearson r, standardized MAE, mean error, OLS calibration slope/intercept, and the model-minus-DoWhy delta in standardized MAE.",
            "This figure replaces the prior multi-panel intervention calibration grid so the continuous evidence has one clear thesis role: effect-magnitude fidelity.",
        ]
    else:
        caption_lines = [
            "Panels A and B show an expanded continuous-fidelity diagnostic under observable-state alignment, restricted to output intervention pairs with a benchmark-confirmed effect and standardized effect magnitudes.",
            f"Datasets shown: {_join_list(dataset_labels)}. Models represented: {_join_list(models_present)}.",
            "Panel A shows DoWhy-reference estimates against true standardized effects; Panel B shows model-derived standardized effects against the same truth.",
            "Identity-line proximity indicates stronger calibration. The figure itself stays descriptive, while per-dataset and pooled diagnostics are exported separately as tables.",
            "Observations near y=0 in Panel B indicate near-zero model responses under the intervention protocol.",
            "The exported diagnostics report Pearson r, standardized MAE, mean error, OLS calibration slope/intercept, and the model-minus-DoWhy delta in standardized MAE.",
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
        title_generated=title_generated,
        caption_lines=caption_lines,
        legend_mode="outside_bottom_shared",
        annotation_mode="none",
        layout_profile="paper",
        layout_rect=(0.05, 0.15, 0.95, 1.0),
        table_rows=table_rows,
    )
