from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import seaborn as sns

from reporting.figures.common import (
    DATASET_ORDER,
    THESIS_TEXT_WIDTH,
    _alignment_target_label,
    _dataset_label,
    _ensure_non_empty,
    _legend_identity_handle,
    _model_label,
    _move_legend_bottom,
    _save_figure,
    plt,
    set_reporting_theme,
)


FIGURE_TITLE = "Aligned Proxy Variables Compared with SEM Intervention Effects"
DEFAULT_ALIGNMENT_TARGET_ORDER = ["observable_state", "sem_truth"]
EFFECT_STATE_ORDER = ["Confirmed SEM effect", "No confirmed SEM effect"]
EFFECT_STATE_COLORS = {
    "Confirmed SEM effect": "#1f4e79",
    "No confirmed SEM effect": "#b8b8b8",
}
AXIS_LIMIT_PADDING = 0.05
AXIS_TICK_STEP = 5.0


def _join_list(items: Sequence[str]) -> str:
    entries = [str(item) for item in items if str(item).strip()]
    if not entries:
        return ""
    if len(entries) == 1:
        return entries[0]
    if len(entries) == 2:
        return f"{entries[0]} and {entries[1]}"
    return ", ".join(entries[:-1]) + f", and {entries[-1]}"


def _is_truthy(value) -> bool:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return False
    return str(value).strip().lower() in {"true", "1", "1.0"}


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
    tick_low = np.ceil(low / AXIS_TICK_STEP) * AXIS_TICK_STEP
    tick_high = np.floor(high / AXIS_TICK_STEP) * AXIS_TICK_STEP
    ticks = np.arange(tick_low, tick_high + (AXIS_TICK_STEP * 0.5), AXIS_TICK_STEP)
    return ticks if ticks.size else np.array([0.0])


def _pearson_r(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size < 2 or np.isclose(np.var(x), 0.0) or np.isclose(np.var(y), 0.0):
        return None
    corr = float(np.corrcoef(x, y)[0, 1])
    return corr if np.isfinite(corr) else None


def _sign_agreement_rate(x: np.ndarray, y: np.ndarray) -> Optional[float]:
    if x.size == 0 or y.size == 0:
        return None
    agreement = np.sign(x) == np.sign(y)
    if agreement.size == 0:
        return None
    return float(np.mean(agreement))


def _summary_rows(scope: str, dataset: str, alignment_target: str, summary: dict) -> dict:
    return {
        "scope": scope,
        "dataset": dataset,
        "dataset_label": _dataset_label(dataset) if dataset != "all" else "All datasets",
        "alignment_target": alignment_target,
        "alignment_target_label": (
            _alignment_target_label(alignment_target) if alignment_target != "all" else "All alignment targets"
        ),
        **summary,
    }


def _proxy_summary(df: pd.DataFrame) -> dict:
    x_vals = df["x"].to_numpy(dtype=float)
    y_vals = df["y"].to_numpy(dtype=float)
    mask = np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[mask]
    y_vals = y_vals[mask]
    effect_vals = df.loc[mask, "sem_effect_present"].to_numpy(dtype=bool) if "sem_effect_present" in df.columns else None

    if x_vals.size == 0:
        return {
            "n": 0,
            "mae": None,
            "median_abs_delta": None,
            "bias": None,
            "r": None,
            "sign_agreement_rate": None,
            "sem_effect_present_rate": None,
        }

    delta = y_vals - x_vals
    sem_effect_present_rate = float(np.mean(effect_vals)) if effect_vals is not None and effect_vals.size > 0 else None
    return {
        "n": int(x_vals.size),
        "mae": float(np.mean(np.abs(delta))),
        "median_abs_delta": float(np.median(np.abs(delta))),
        "bias": float(np.mean(delta)),
        "r": _pearson_r(x_vals, y_vals),
        "sign_agreement_rate": _sign_agreement_rate(x_vals, y_vals),
        "sem_effect_present_rate": sem_effect_present_rate,
    }


def _ordered_present(requested: Sequence[str], observed: Sequence[str]) -> List[str]:
    observed_set = {str(value) for value in observed if str(value).strip()}
    ordered = [str(value) for value in requested if str(value) in observed_set]
    extras = sorted(observed_set - set(ordered))
    return ordered + extras


def figure_intervention_aligned_proxy_calibration(spec: dict, records: List[dict], output_dir: str) -> dict:
    set_reporting_theme(layout_profile="paper")
    title_generated = str(spec.get("title", FIGURE_TITLE))

    active_records = [
        record
        for record in records
        if bool(record.get("metadata", {}).get("alignment_enabled", False))
        and record.get("aligned_proxy_pair_df") is not None
        and not record.get("aligned_proxy_pair_df").empty
    ]

    rows = []
    source_files = []
    for record in active_records:
        pair_df = record.get("aligned_proxy_pair_df")
        if pair_df is None or pair_df.empty:
            continue
        dataset = record.get("metadata", {}).get("dataset")
        model = record.get("metadata", {}).get("model")
        default_target = record.get("metadata", {}).get("alignment_target_mode")
        working_df = pair_df.copy()
        working_df["ate_sem_do"] = pd.to_numeric(working_df.get("ate_sem_do"), errors="coerce")
        working_df["ate_aligned_dowhy"] = pd.to_numeric(working_df.get("ate_aligned_dowhy"), errors="coerce")
        working_df["delta_aligned_minus_sem"] = pd.to_numeric(
            working_df.get("delta_aligned_minus_sem"), errors="coerce"
        )
        working_df["abs_delta"] = pd.to_numeric(working_df.get("abs_delta"), errors="coerce")

        for _, row in working_df.iterrows():
            target_mode = row.get("target_mode") or default_target
            sem_effect_present = _is_truthy(row.get("sem_effect_present"))
            rows.append(
                {
                    "dataset": dataset,
                    "dataset_label": _dataset_label(dataset),
                    "model": model,
                    "fold": row.get("fold"),
                    "alignment_target": target_mode,
                    "alignment_target_label": _alignment_target_label(target_mode),
                    "treatment": row.get("treatment"),
                    "outcome": row.get("outcome"),
                    "x": row.get("ate_sem_do"),
                    "y": row.get("ate_aligned_dowhy"),
                    "delta": row.get("delta_aligned_minus_sem"),
                    "abs_delta": row.get("abs_delta"),
                    "sem_effect_present": sem_effect_present,
                    "effect_label": "Confirmed SEM effect" if sem_effect_present else "No confirmed SEM effect",
                    "dowhy_error": row.get("dowhy_error"),
                }
            )
        source_files.extend(record.get("aligned_proxy_source_files", []))

    df = pd.DataFrame(rows)
    _ensure_non_empty(df, spec["id"])
    df = df.dropna(subset=["x", "y"]).copy()
    _ensure_non_empty(df, spec["id"])

    present_datasets = df["dataset"].dropna().astype(str).unique().tolist()
    dataset_order = _ordered_present(spec.get("dataset_order", DATASET_ORDER), present_datasets)
    _ensure_non_empty(pd.DataFrame({"dataset": dataset_order}), spec["id"])

    present_targets = df["alignment_target"].dropna().astype(str).unique().tolist()
    target_order = _ordered_present(spec.get("alignment_target_order", DEFAULT_ALIGNMENT_TARGET_ORDER), present_targets)
    _ensure_non_empty(pd.DataFrame({"alignment_target": target_order}), spec["id"])

    panel_values = df[["x", "y"]].to_numpy(dtype=float).ravel()
    low, high = _shared_limits(panel_values)
    shared_ticks = _shared_ticks(low, high)

    n_rows = len(dataset_order)
    n_cols = len(target_order)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(THESIS_TEXT_WIDTH, max(3.2, 2.85 * n_rows)),
        sharex=True,
        sharey=True,
    )
    axes = np.atleast_2d(axes)
    if axes.shape != (n_rows, n_cols):
        axes = axes.reshape(n_rows, n_cols)

    stats_rows = []
    table_rows = []
    panel_index = 0

    for row_idx, dataset in enumerate(dataset_order):
        for col_idx, alignment_target in enumerate(target_order):
            ax = axes[row_idx, col_idx]
            panel_df = df[
                (df["dataset"] == dataset)
                & (df["alignment_target"] == alignment_target)
            ].copy()

            if panel_df.empty:
                ax.set_axis_off()
                continue

            sns.scatterplot(
                data=panel_df,
                x="x",
                y="y",
                hue="effect_label",
                hue_order=EFFECT_STATE_ORDER,
                palette=EFFECT_STATE_COLORS,
                s=28,
                alpha=0.78,
                edgecolor="#f1f1f1",
                linewidth=0.3,
                ax=ax,
            )
            if ax.legend_ is not None:
                ax.legend_.remove()

            ax.plot([low, high], [low, high], linestyle=(0, (6, 3)), color="#404040", linewidth=1.2)
            ax.set_xlim(low, high)
            ax.set_ylim(low, high)
            ax.set_xticks(shared_ticks)
            ax.set_yticks(shared_ticks)
            ax.set_aspect("equal", adjustable="box")
            ax.grid(alpha=0.28)
            ax.set_title(f"{_dataset_label(dataset)}\n{_alignment_target_label(alignment_target)}")

            if row_idx == n_rows - 1:
                ax.set_xlabel("Benchmark SEM interventional ATE")
            else:
                ax.set_xlabel("")
                ax.tick_params(labelbottom=False)

            if col_idx == 0:
                ax.set_ylabel("Aligned-space DoWhy ATE")
            else:
                ax.set_ylabel("")

            panel_letter = chr(ord("A") + panel_index)
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
            panel_index += 1

            dataset_alignment_summary = _proxy_summary(panel_df)
            stats_rows.append(
                {
                    "figure": spec["id"],
                    "test": "diagnostic",
                    **_summary_rows("dataset_alignment", dataset, alignment_target, dataset_alignment_summary),
                }
            )
            table_rows.append(
                {
                    "figure": spec["id"],
                    **_summary_rows("dataset_alignment", dataset, alignment_target, dataset_alignment_summary),
                }
            )

    for alignment_target in target_order:
        target_df = df[df["alignment_target"] == alignment_target].copy()
        if target_df.empty:
            continue
        summary = _proxy_summary(target_df)
        stats_rows.append(
            {
                "figure": spec["id"],
                "test": "diagnostic",
                **_summary_rows("alignment_target_pooled", "all", alignment_target, summary),
            }
        )
        table_rows.append(
            {
                "figure": spec["id"],
                **_summary_rows("alignment_target_pooled", "all", alignment_target, summary),
            }
        )

    overall_summary = _proxy_summary(df)
    stats_rows.append(
        {
            "figure": spec["id"],
            "test": "diagnostic",
            **_summary_rows("overall", "all", "all", overall_summary),
        }
    )
    table_rows.append(
        {
            "figure": spec["id"],
            **_summary_rows("overall", "all", "all", overall_summary),
        }
    )

    fig.suptitle(title_generated, y=0.995)
    _move_legend_bottom(
        fig,
        title="Pair status",
        ncol=3,
        extra_handles=[_legend_identity_handle()],
        bbox_y=0.01,
    )

    models_present = sorted(
        {
            _model_label(record.get("metadata", {}).get("model"))
            for record in active_records
            if str(record.get("metadata", {}).get("model", "")).strip()
        }
    )
    dataset_labels = [_dataset_label(dataset) for dataset in dataset_order]
    alignment_labels = [_alignment_target_label(target) for target in target_order]
    caption_lines = [
        "Secondary proxy-diagnostic figure comparing aligned-space DoWhy estimates against benchmark SEM interventional ATEs across ordered treatment-outcome pairs.",
        f"Datasets shown: {_join_list(dataset_labels)}. Alignment targets shown: {_join_list(alignment_labels)}.",
        "Each panel uses the benchmark interventional ATE on the x-axis and the aligned-space DoWhy estimate on the y-axis; the dashed identity line marks perfect proxy calibration.",
        "Point color distinguishes whether the benchmark SEM reference indicates a confirmed intervention effect for the pair.",
        "The exported table reports pooled and per-panel diagnostics including MAE, median absolute deviation, bias, Pearson correlation, sign-agreement rate, and SEM-effect prevalence.",
    ]
    if models_present:
        caption_lines.insert(
            2,
            f"Models represented: {_join_list(models_present)}.",
        )

    return _save_figure(
        fig,
        axes,
        spec["id"],
        output_dir,
        stats_rows=stats_rows,
        data_row_count=len(df),
        source_files=source_files,
        title_generated=title_generated,
        caption_lines=caption_lines,
        legend_mode="outside_bottom_shared",
        annotation_mode="none",
        layout_profile="paper",
        layout_rect=(0.05, 0.07, 0.95, 1.0),
        table_rows=table_rows,
    )
