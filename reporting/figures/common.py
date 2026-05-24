import os
import warnings
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import Bbox
from scipy import stats

from experiment_utils.reporting_utils import benjamini_hochberg, serialize_stats_value
from metrics.plot_labels import metric_label


DATASET_LABELS = {
    "LinearSEMDataset": "Linear SEM",
    "ElectricalCircuitData": "Current Measurement",
    "templated_linear": "Layered Sparse Linear SEM",
    "templated_nonlinear_additive": "Layered Sparse Nonlinear SEM",
    "ChainLinearSEM": "Chain Linear SEM",
    "ForkColliderLinearSEM": "Fork-Collider Linear SEM",
    "LayeredSparseLinearSEM": "Layered Sparse Linear SEM",
    "LayeredSparseLinearSEM_Laplace": "Layered Sparse Linear SEM (Laplace)",
    "LayeredSparseLinearSEM_StudentT3": "Layered Sparse Linear SEM (Student-t)",
    "ChainNonlinearSEM": "Chain Nonlinear SEM",
    "ForkColliderNonlinearSEM": "Fork-Collider Nonlinear SEM",
    "LayeredSparseNonlinearSEM": "Layered Sparse Nonlinear SEM",
}

DATASET_COLORS = {
    "LinearSEMDataset": "#0f5a9c",
    "ElectricalCircuitData": "#cc503e",
    "ChainLinearSEM": "#2f4b7c",
    "ForkColliderLinearSEM": "#4f6d9c",
    "LayeredSparseLinearSEM": "#6d8bb7",
    "LayeredSparseLinearSEM_Laplace": "#8ca7cf",
    "LayeredSparseLinearSEM_StudentT3": "#a9c0e2",
    "ChainNonlinearSEM": "#9b3d12",
    "ForkColliderNonlinearSEM": "#c05a2a",
    "LayeredSparseNonlinearSEM": "#de7c45",
}

STRUCTURAL_DATA_TRUTH_COLOR = "#F2C300"
STRUCTURAL_MODEL_TRUTH_COLOR = "#2E7D32"

STRUCTURAL_SOURCE_COLORS = {
    "data_truth": STRUCTURAL_DATA_TRUTH_COLOR,
    "model_truth": STRUCTURAL_MODEL_TRUTH_COLOR,
}

STRUCTURAL_SOURCE_LABEL_COLORS = {
    "Data-level PAG": STRUCTURAL_DATA_TRUTH_COLOR,
    "Model-level PAG": STRUCTURAL_MODEL_TRUTH_COLOR,
}

PAG_MODEL_SOURCE_COLORS = {
    "PAG": STRUCTURAL_DATA_TRUTH_COLOR,
    "Model": STRUCTURAL_MODEL_TRUTH_COLOR,
}

CI_TEST_MODE_COLORS = {
    "Matched CI test": STRUCTURAL_DATA_TRUTH_COLOR,
    "Mismatched CI test": STRUCTURAL_MODEL_TRUTH_COLOR,
}

ALIGNMENT_TARGET_ORDER = ["off", "observable_state", "sem_truth"]

ALIGNMENT_TARGET_COLORS = {
    "off": "#ff7f0e",
    "observable_state": "#1f77b4",
    "sem_truth": "#8fa3b8",
}

ALIGNMENT_TARGET_LABELS = {
    "off": "No alignment",
    "observable_state": "Observable-state alignment",
    "sem_truth": "SEM-target alignment",
}

ALIGNMENT_TARGET_SHORT_LABELS = {
    "off": "None",
    "observable_state": "Observable-state",
    "sem_truth": "SEM-target",
}

DATASET_MARKERS = {
    "LinearSEMDataset": "o",
    "ElectricalCircuitData": "s",
    "ChainLinearSEM": "D",
    "ForkColliderLinearSEM": "^",
    "LayeredSparseLinearSEM": "P",
    "LayeredSparseLinearSEM_Laplace": "X",
    "LayeredSparseLinearSEM_StudentT3": "v",
    "ChainNonlinearSEM": "D",
    "ForkColliderNonlinearSEM": "^",
    "LayeredSparseNonlinearSEM": "P",
}

ALIGNMENT_MARKERS = {
    "on": "o",
    "off": "X",
}

ALIGNMENT_LINESTYLES = {
    "on": "-",
    "off": "--",
}

METHOD_LABELS = {
    "activations": "Activations",
    "pre_activations": "Pre-activations",
    "input_gradients": "Input gradients",
    "inputs_outputs": "Inputs/outputs",
    "weights": "Weights",
}

METHOD_COLORS = {
    "activations": "#1F4E79",
    "pre_activations": "#4F81BD",
    "input_gradients": "#D1492E",
    "inputs_outputs": "#8E8E8E",
    "weights": "#E67E22",
}

MODEL_LABELS = {
    "ShallowMLP": "Shallow MLP (2x32, ReLU)",
    "DeepMLP": "Deep MLP (4x64, GELU)",
    "BottleneckMLP": "Bottleneck MLP (128-16-128, SiLU)",
    "GatedMLP": "Gated MLP (4x64, SiLU)",
    "ResidualMLP": "Residual MLP (128, 3 blocks, GELU)",
    "TabularMLP": "Tabular MLP (256-128-64, SiLU)",
}

MODEL_ORDER = ["ShallowMLP", "DeepMLP", "BottleneckMLP", "GatedMLP", "ResidualMLP"]

MODEL_COLORS = {
    "ShallowMLP": "#46327e",
    "DeepMLP": "#365c8d",
    "BottleneckMLP": "#277f8e",
    "GatedMLP": "#1fa187",
    "ResidualMLP": "#4ac16d",
    "TabularMLP": "#a0da39",
}

METHOD_ORDER_ALL = [
    "activations",
    "pre_activations",
    "input_gradients",
    "inputs_outputs",
    "weights",
]

METHOD_ORDER_INTERNAL = ["activations", "pre_activations", "input_gradients"]
DATASET_ORDER = ["LinearSEMDataset", "ElectricalCircuitData"]
ALIGNMENT_ORDER = ["off", "on"]
THESIS_TEXT_WIDTH = 6.313
BOUNDED_METRIC_TICK_STEP = 0.2
DELTA_TICK_STEP = 0.25
STRUCTURAL_TRUTH_DELTA_AXIS_LABEL = r"$\Delta = \mathrm{model} - \mathrm{data}$"
DELTA_AXIS_LABEL = STRUCTURAL_TRUTH_DELTA_AXIS_LABEL
BOUNDED_METRICS = {
    "edge_precision",
    "edge_recall",
    "edge_f1",
    "endpoint_precision",
    "endpoint_recall",
    "endpoint_f1",
    "orientation_accuracy",
    "agreement_rate",
    "agreement_rate_total",
    "model_agreement_rate",
    "mean_ks_p",
    "model_mean_ks_p",
    "effect_positive_rate_output",
    "pag_precision_output",
    "pag_recall_output",
    "pag_specificity_output",
    "pag_f1_output",
    "precision",
    "recall",
    "specificity",
    "f1",
    "pag_miss_rate",
    "pag_overclaim_rate",
    "model_precision_output",
    "model_recall_output",
    "model_specificity_output",
    "model_f1_output",
}

NONNEGATIVE_METRICS = {
    "missing_edges",
    "extra_edges",
    "orientation_errors",
    "partial_orientations",
    "correct_orientations",
    "shd_strict",
    "shd_partial",
    "shd_edges_only",
}

SEMANTIC_LEGEND_LABELS = {
    "dataset",
    "dataset_label",
    "alignment",
    "alignment_label",
    "alignment_target_mode",
    "alignment_target_label",
    "metric",
    "metric_label",
    "mode",
    "mode_label",
    "source",
    "source_label",
    "scale",
    "scale_label",
    "panel",
    "criterion",
    "comparison",
    "method",
    "method_label",
    "model",
    "model_label",
}


def get_palette() -> List[str]:
    return list(DATASET_COLORS.values())

def set_reporting_theme(layout_profile: str = "paper") -> None:
    rc = {
        #"figure.facecolor": "white",
        #"axes.facecolor": "#faf8f4",
        #"axes.edgecolor": "#242424",
        #"grid.color": "#d0ccc4",
        #"grid.alpha": 0.45,
        #"axes.titlesize": 11,
        #"axes.labelsize": 10,
        #"legend.fontsize": 9,
        #"legend.title_fontsize": 9,
    }
    sns.set_theme(style="whitegrid", palette=get_palette(), rc=rc)
    if layout_profile == "poster":
        sns.set_context("poster", font_scale=0.95)
    else:
        sns.set_context("paper", font_scale=1.0)


def _alignment_status(value: bool) -> str:
    return "on" if bool(value) else "off"


def _normalized_alignment_target_mode(value: Optional[str]) -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        return "sem_truth"
    if text in {"sem_truth", "observable_state"}:
        return text
    return text


def _alignment_target_mode(meta: dict) -> Optional[str]:
    if not bool(meta.get("alignment_enabled", False)):
        return None
    return _normalized_alignment_target_mode(meta.get("alignment_target_mode"))


def _alignment_target_variant(meta: dict) -> str:
    if not bool(meta.get("alignment_enabled", False)):
        return "off"
    return _normalized_alignment_target_mode(meta.get("alignment_target_mode"))


def _alignment_target_label(value: Optional[str]) -> str:
    if value is None:
        return "N/A"
    return ALIGNMENT_TARGET_LABELS.get(value, value)


def _is_sem_alignment_record(record: dict) -> bool:
    meta = record.get("metadata", {})
    return bool(meta.get("alignment_enabled", False)) and _alignment_target_mode(meta) == "sem_truth"


def _is_observable_alignment_record(record: dict) -> bool:
    meta = record.get("metadata", {})
    return bool(meta.get("alignment_enabled", False)) and _alignment_target_mode(meta) == "observable_state"


def _dataset_label(dataset_id: str) -> str:
    return DATASET_LABELS.get(dataset_id, dataset_id)


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def method_palette(methods: Sequence[str], by_label: bool = True) -> dict:
    palette = {}
    for method in methods:
        key = _method_label(method) if by_label else method
        palette[key] = METHOD_COLORS.get(method, "#6e6e6e")
    return palette


def model_palette(models: Sequence[str], by_label: bool = True) -> dict:
    palette = {}
    for model in models:
        key = _model_label(model) if by_label else model
        palette[key] = MODEL_COLORS.get(model, "#6e6e6e")
    return palette

def _model_label(model_id: str) -> str:
    text = str(model_id).strip() if model_id is not None else ""
    if not text:
        return "N/A"
    return MODEL_LABELS.get(text, text)


def _metric_label(metric: str) -> str:
    return metric_label(metric)


def _ci_test_label(ci_test_id: Optional[str]) -> str:
    text = str(ci_test_id).strip() if ci_test_id is not None else ""
    if not text:
        return "N/A"
    mapping = {
        "CondIndepParCorr": "Partial correlation (ParCorr)",
        "CondIndepKCI": "Kernel CI (KCI)",
    }
    return mapping.get(text, text)


def _ensure_non_empty(df: pd.DataFrame, figure_id: str) -> None:
    if df.empty:
        raise ValueError(f"{figure_id}: no data available for plotting.")


def _deduplicate_invariant_rows(
    df: pd.DataFrame,
    subset: Sequence[str],
    source_col: str,
    invariant_sources: Sequence[str],
) -> pd.DataFrame:
    if df.empty or source_col not in df.columns:
        return df.copy()
    dedupe_keys = [key for key in subset if key in df.columns]
    if not dedupe_keys:
        return df.copy()
    invariant_mask = df[source_col].isin(set(invariant_sources))
    invariant_df = df.loc[invariant_mask].drop_duplicates(subset=dedupe_keys)
    varying_df = df.loc[~invariant_mask]
    return pd.concat([invariant_df, varying_df], ignore_index=True)


def _confusion_metrics_from_columns(df: pd.DataFrame, pred_col: str, true_col: str) -> Optional[dict]:
    if df.empty or pred_col not in df.columns or true_col not in df.columns:
        return None
    valid_mask = df[pred_col].notna() & df[true_col].notna()
    if not valid_mask.any():
        return None
    y_true = df.loc[valid_mask, true_col].astype(bool)
    y_pred = df.loc[valid_mask, pred_col].astype(bool)
    tp = int((y_pred & y_true).sum())
    fp = int((y_pred & ~y_true).sum())
    tn = int((~y_pred & ~y_true).sum())
    fn = int((~y_pred & y_true).sum())

    def _safe_divide(num: int, den: int) -> Optional[float]:
        if den == 0:
            return None
        return float(num / den)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": _safe_divide(tp, tp + fp),
        "recall": _safe_divide(tp, tp + fn),
        "specificity": _safe_divide(tn, tn + fp),
        "f1": _safe_divide(2 * tp, 2 * tp + fp + fn),
    }


def _record_detection_summary(record: dict, role: str = "output") -> pd.DataFrame:
    dataset = record.get("metadata", {}).get("dataset")
    model = record.get("metadata", {}).get("model")
    role_suffix = f"_{role}"
    required_cols = [
        f"effect_positive_rate{role_suffix}",
        f"pag_precision{role_suffix}",
        f"pag_recall{role_suffix}",
        f"pag_specificity{role_suffix}",
        f"pag_f1{role_suffix}",
        f"model_precision{role_suffix}",
        f"model_recall{role_suffix}",
        f"model_specificity{role_suffix}",
        f"model_f1{role_suffix}",
    ]
    int_df = record.get("intervention_summary_df")
    if int_df is not None and not int_df.empty and all(col in int_df.columns for col in required_cols):
        rows = []
        for fold, row in int_df.iterrows():
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "fold": int(fold),
                    "effect_positive_rate": row.get(f"effect_positive_rate{role_suffix}"),
                    "pag_precision": row.get(f"pag_precision{role_suffix}"),
                    "pag_recall": row.get(f"pag_recall{role_suffix}"),
                    "pag_specificity": row.get(f"pag_specificity{role_suffix}"),
                    "pag_f1": row.get(f"pag_f1{role_suffix}"),
                    "model_precision": row.get(f"model_precision{role_suffix}"),
                    "model_recall": row.get(f"model_recall{role_suffix}"),
                    "model_specificity": row.get(f"model_specificity{role_suffix}"),
                    "model_f1": row.get(f"model_f1{role_suffix}"),
                }
            )
        return pd.DataFrame(rows)

    pair_df = record.get("pair_df")
    if pair_df is None or pair_df.empty or "fold" not in pair_df.columns or "outcome_role" not in pair_df.columns:
        return pd.DataFrame()

    rows = []
    true_col = "effect_present"
    model_pred_col = "model_effect_present"
    role_df = pair_df[pair_df["outcome_role"] == role]
    for fold, fold_df in role_df.groupby("fold"):
        effect_rate = float(fold_df[true_col].mean()) if true_col in fold_df.columns and fold_df[true_col].notna().any() else None
        pag_conf = _confusion_metrics_from_columns(fold_df, "pag_predicts_total", true_col)
        model_conf = _confusion_metrics_from_columns(fold_df, model_pred_col, true_col)
        rows.append(
            {
                "dataset": dataset,
                "model": model,
                "fold": int(fold),
                "effect_positive_rate": effect_rate,
                "pag_precision": None if pag_conf is None else pag_conf.get("precision"),
                "pag_recall": None if pag_conf is None else pag_conf.get("recall"),
                "pag_specificity": None if pag_conf is None else pag_conf.get("specificity"),
                "pag_f1": None if pag_conf is None else pag_conf.get("f1"),
                "model_precision": None if model_conf is None else model_conf.get("precision"),
                "model_recall": None if model_conf is None else model_conf.get("recall"),
                "model_specificity": None if model_conf is None else model_conf.get("specificity"),
                "model_f1": None if model_conf is None else model_conf.get("f1"),
            }
        )
    return pd.DataFrame(rows)


def _paired_rank_biserial(diffs: np.ndarray) -> float:
    valid = diffs[~np.isnan(diffs)]
    valid = valid[~np.isclose(valid, 0.0, atol=1e-12, rtol=1e-8)]
    if valid.size == 0:
        return 0.0
    n_pos = int(np.sum(valid > 0))
    n_neg = int(np.sum(valid < 0))
    denom = n_pos + n_neg
    if denom == 0:
        return 0.0
    return float((n_pos - n_neg) / denom)


def _wilcoxon_paired_details(a: np.ndarray, b: np.ndarray) -> dict:
    mask = ~np.isnan(a) & ~np.isnan(b)
    a_vals = a[mask]
    b_vals = b[mask]
    n = int(a_vals.size)
    diffs = b_vals - a_vals if n > 0 else np.array([])
    delta_mean = float(np.mean(diffs)) if diffs.size > 0 else None
    delta_median = float(np.median(diffs)) if diffs.size > 0 else None
    rank_biserial = _paired_rank_biserial(diffs) if diffs.size > 0 else None

    non_zero = diffs[~np.isclose(diffs, 0.0, atol=1e-12, rtol=1e-8)]
    if non_zero.size < 2:
        return {
            "stat": 0.0,
            "p_raw": 1.0,
            "n": n,
            "delta_mean": delta_mean,
            "delta_median": delta_median,
            "rank_biserial": rank_biserial,
        }

    method = "exact" if non_zero.size <= 20 else "approx"
    result = stats.wilcoxon(non_zero, zero_method="wilcox", method=method)
    stat = 0.0 if np.isnan(result.statistic) else float(result.statistic)
    p_val = 1.0 if np.isnan(result.pvalue) else float(result.pvalue)
    return {
        "stat": stat,
        "p_raw": p_val,
        "n": n,
        "delta_mean": delta_mean,
        "delta_median": delta_median,
        "rank_biserial": rank_biserial,
    }


def _wilcoxon_paired(a: np.ndarray, b: np.ndarray) -> Tuple[float, float, int]:
    details = _wilcoxon_paired_details(a, b)
    return float(details["stat"]), float(details["p_raw"]), int(details["n"])


def _friedman_details(*samples: np.ndarray) -> dict:
    arrays = [np.asarray(sample, dtype=float) for sample in samples]
    if not arrays:
        return {
            "stat": 0.0,
            "p_raw": 1.0,
            "n": 0,
            "degenerate": True,
            "reason": "no_samples",
        }

    lengths = {arr.shape[0] for arr in arrays}
    if len(lengths) != 1:
        raise ValueError("Friedman test samples must have equal length.")

    matrix = np.column_stack(arrays)
    if matrix.size == 0:
        return {
            "stat": 0.0,
            "p_raw": 1.0,
            "n": 0,
            "degenerate": True,
            "reason": "empty_samples",
        }

    finite_mask = np.isfinite(matrix).all(axis=1)
    matrix = matrix[finite_mask]
    n = int(matrix.shape[0])
    if n < 2:
        return {
            "stat": 0.0,
            "p_raw": 1.0,
            "n": n,
            "degenerate": True,
            "reason": "insufficient_rows",
        }

    # SciPy's tie correction can divide by zero when every block is perfectly tied.
    if np.all(np.isclose(matrix, matrix[:, [0]], atol=1e-12, rtol=1e-8)):
        return {
            "stat": 0.0,
            "p_raw": 1.0,
            "n": n,
            "degenerate": True,
            "reason": "all_ties",
        }

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="invalid value encountered in scalar divide",
            category=RuntimeWarning,
        )
        result = stats.friedmanchisquare(*(matrix[:, idx] for idx in range(matrix.shape[1])))

    stat = float(result.statistic) if np.isfinite(result.statistic) else 0.0
    p_val = float(result.pvalue) if np.isfinite(result.pvalue) else 1.0
    degenerate = not (np.isfinite(result.statistic) and np.isfinite(result.pvalue))
    return {
        "stat": stat,
        "p_raw": p_val,
        "n": n,
        "degenerate": degenerate,
        "reason": "nonfinite_result" if degenerate else None,
    }


def _bh_adjust_rows(rows: List[dict], p_col: str = "p_raw", q_col: str = "q_bh") -> None:
    p_values = []
    row_indices = []
    for idx, row in enumerate(rows):
        p_val = row.get(p_col)
        if p_val is None or pd.isna(p_val):
            row[q_col] = None
            continue
        row_indices.append(idx)
        p_values.append(float(p_val))
    if not p_values:
        return
    adjusted = benjamini_hochberg(p_values)
    for idx, q_val in zip(row_indices, adjusted):
        rows[idx][q_col] = float(q_val)


def _spearman_stats(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_vals = x[mask]
    y_vals = y[mask]
    n = int(x_vals.size)
    if n < 3:
        return 0.0, 1.0, n
    corr, p_val = stats.spearmanr(x_vals, y_vals)
    corr = 0.0 if np.isnan(corr) else float(corr)
    p_val = 1.0 if np.isnan(p_val) else float(p_val)
    return corr, p_val, n


def _pearson_stats(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, int]:
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_vals = x[mask]
    y_vals = y[mask]
    n = int(x_vals.size)
    if n < 3:
        return 0.0, 1.0, n
    corr, p_val = stats.pearsonr(x_vals, y_vals)
    corr = 0.0 if np.isnan(corr) else float(corr)
    p_val = 1.0 if np.isnan(p_val) else float(p_val)
    return corr, p_val, n


def _safe_axis_limits(values: np.ndarray, default: Tuple[float, float]) -> Tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return default
    low = float(np.min(finite))
    high = float(np.max(finite))
    if np.isclose(low, high):
        pad = 0.1 if np.isclose(low, 0.0) else abs(low) * 0.1
        return (low - pad, high + pad)
    pad = 0.06 * (high - low)
    return (low - pad, high + pad)


def _apply_metric_limit(ax, metric: str, values: np.ndarray, axis: str = "y") -> None:
    if metric in BOUNDED_METRICS:
        low, high = -0.1, 1.1
    else:
        low, high = _safe_axis_limits(values, (0.0, 1.0))
    if str(metric).startswith("shd_") or metric in NONNEGATIVE_METRICS:
        low = min(-0.1, low)
    if axis == "x":
        ax.set_xlim(low, high)
    else:
        ax.set_ylim(low, high)


def _bounded_metric_limits() -> Tuple[float, float]:
    return (-0.05, 1.05)


def _apply_bounded_metric_y_axis(ax, tick_step: float = BOUNDED_METRIC_TICK_STEP) -> None:
    low, high = _bounded_metric_limits()
    ax.set_ylim(low, high)
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    ax.set_yticks(np.arange(0.0, 1.0 + (tick_step * 0.5), tick_step))


def _symmetric_delta_limits(
    values: np.ndarray,
    tick_step: float = DELTA_TICK_STEP,
    max_limit: float = 1.0,
) -> Tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return (-tick_step, tick_step)
    max_abs = float(np.max(np.abs(finite)))
    step_count = max(1, int(np.floor(max_abs / tick_step)) + 1)
    limit = float(step_count * tick_step)
    if max_limit is not None:
        limit = min(float(max_limit), limit)
    return (-limit, limit)


def _apply_right_delta_y_axis(
    ax,
    limits: Tuple[float, float],
    tick_step: float = DELTA_TICK_STEP,
) -> None:
    ax.set_ylim(*limits)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.tick_params(axis="y", labelleft=False, left=False, right=False, length=0)
    ax.yaxis.set_major_locator(MultipleLocator(tick_step))
    ax.set_yticks(np.arange(limits[0], limits[1] + (tick_step * 0.5), tick_step))


def _add_panel_letter(ax, label: str, x: float = -0.12, y: float = 1.03) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _jitter_values(
    values: np.ndarray,
    seed: int,
    width: float,
    low: Optional[float] = None,
    high: Optional[float] = None,
) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    if vals.size == 0 or width <= 0:
        return vals.copy()
    rng = np.random.default_rng(seed)
    jittered = vals + rng.uniform(-width, width, size=vals.size)
    if low is not None and high is not None:
        jittered = np.clip(jittered, low, high)
    return jittered


def _coverage_metrics(pair_df: Optional[pd.DataFrame]) -> Tuple[int, int, Optional[float]]:
    if pair_df is None or pair_df.empty:
        return 0, 0, None
    n_total = int(pair_df.shape[0])
    model_cols = [col for col in ["model_ate", "model_ks_p", "model_prediction_source"] if col in pair_df.columns]
    if not model_cols:
        return n_total, 0, 0.0
    model_mask = np.zeros(n_total, dtype=bool)
    for col in model_cols:
        col_vals = pair_df[col]
        if col == "model_prediction_source":
            model_mask |= col_vals.notna().to_numpy()
        else:
            numeric_vals = pd.to_numeric(col_vals, errors="coerce").to_numpy()
            model_mask |= np.isfinite(numeric_vals)
    n_model = int(np.sum(model_mask))
    coverage = float(n_model / n_total) if n_total > 0 else None
    return n_total, n_model, coverage


def _format_decimal(value: Optional[float], digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _concise_stat_line(row: dict) -> str:
    p_key = "q_bh" if row.get("q_bh") is not None else "p_raw"
    p_name = "q" if p_key == "q_bh" else "p"
    parts = []
    if row.get(p_key) is not None:
        parts.append(f"{p_name}={_format_decimal(row.get(p_key), 3)}")
    if row.get("n") is not None:
        parts.append(f"n={int(row['n'])}")
    if row.get("delta_mean") is not None:
        parts.append(f"Δ={_format_decimal(row['delta_mean'], 3)}")
    if row.get("rank_biserial") is not None:
        parts.append(f"r_rb={_format_decimal(row['rank_biserial'], 3)}")
    if row.get("rho") is not None:
        parts.append(f"ρ={_format_decimal(row['rho'], 3)}")
    if row.get("mae") is not None:
        parts.append(f"MAE={_format_decimal(row['mae'], 3)}")
    return ", ".join(parts)


def _write_stats_csv(path: str, rows: List[dict]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_table_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        pd.DataFrame().to_csv(path, index=False)
        return

    columns = list(rows[0].keys())
    seen = set(columns)
    for row in rows[1:]:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            columns.append(key)

    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)


def _write_text_lines(path: str, title_generated: str, lines_in: Sequence[str]) -> None:
    lines = [title_generated.strip()] if title_generated else []
    lines.extend([line.strip() for line in lines_in if str(line).strip()])
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def _ensure_terminal_punctuation(text: str) -> str:
    stripped = str(text).strip()
    if not stripped:
        return ""
    if stripped[-1] in ".!?:":
        return stripped
    return stripped + "."


def _write_notes(path: str, title_generated: str, notes_lines: Sequence[str]) -> None:
    _write_text_lines(path, title_generated=title_generated, lines_in=notes_lines)


def _to_manifest_rows(rows: List[dict]) -> List[dict]:
    out = []
    for row in rows:
        out.append({k: serialize_stats_value(v) for k, v in row.items()})
    return out


def _axes_to_list(axes) -> List:
    if isinstance(axes, np.ndarray):
        return [ax for ax in axes.ravel() if hasattr(ax, "get_xlim")]
    if isinstance(axes, list):
        return [ax for ax in axes if hasattr(ax, "get_xlim")]
    return [axes] if hasattr(axes, "get_xlim") else []


def _legend_identity_handle(label: str = "Identity line"):
    return mlines.Line2D([], [], color="#595959", linestyle="--", linewidth=1.2, label=label)


def _legend_threshold_handle(label: str, color: str = "#7d7d7d"):
    return mlines.Line2D([], [], color=color, linestyle=":", linewidth=1.2, label=label)


def _legend_region_patch(label: str, color: str, alpha: float = 0.45):
    return mpatches.Patch(facecolor=color, edgecolor=color, alpha=alpha, label=label)


def _is_semantic_legend_label(label: str) -> bool:
    raw_label = str(label).strip()
    normalized = raw_label.lower()
    if not normalized:
        return True
    # Keep human-facing labels like "Model" or "Dataset"; only suppress raw
    # semantic identifiers that leak through from plotting internals.
    if raw_label == normalized:
        if normalized in SEMANTIC_LEGEND_LABELS:
            return True
        return normalized.endswith("_label")
    return False


def _dedupe_handles_labels(handles, labels):
    out_handles = []
    out_labels = []
    seen = set()
    for handle, label in zip(handles, labels):
        if not label:
            continue
        if _is_semantic_legend_label(label):
            continue
        if label in seen:
            continue
        seen.add(label)
        out_handles.append(handle)
        out_labels.append(label)
    return out_handles, out_labels


def _move_legend_bottom(
    target,
    title: Optional[str] = None,
    ncol: int = 4,
    extra_handles: Optional[List] = None,
    bbox_y: float = -0.01,
) -> None:
    axes_obj = getattr(target, "axes", None)
    axes_list = _axes_to_list(axes_obj) if axes_obj is not None else []
    handles: List = []
    labels: List[str] = []
    for ax in axes_list:
        ax_handles, ax_labels = ax.get_legend_handles_labels()
        if ax_handles:
            handles, labels = _dedupe_handles_labels(ax_handles, ax_labels)
            break

    if (not handles) and getattr(target, "_legend", None) is not None:
        legend_obj = target._legend
        legend_handles = list(getattr(legend_obj, "legend_handles", []))
        legend_labels = [text.get_text() for text in getattr(legend_obj, "texts", [])]
        handles, labels = _dedupe_handles_labels(legend_handles, legend_labels)

    if extra_handles:
        extra_list = list(extra_handles)
        handles = list(handles) + extra_list
        labels = list(labels) + [str(h.get_label()) for h in extra_list]
        handles, labels = _dedupe_handles_labels(handles, labels)

    if not handles:
        return

    if getattr(target, "_legend", None) is not None:
        target._legend.remove()
    for ax in axes_list:
        if getattr(ax, "legend_", None) is not None:
            ax.legend_.remove()

    fig = getattr(target, "fig", None) or (axes_list[0].figure if axes_list else None)
    if fig is None:
        return
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, bbox_y),
        ncol=ncol,
        title=title,
        frameon=False,
    )


def _save_figure(
    fig: plt.Figure,
    axes,
    figure_id: str,
    output_dir: str,
    stats_rows: List[dict],
    data_row_count: int,
    source_files: Sequence[str],
    title_generated: str,
    caption_lines: Sequence[str],
    legend_mode: str,
    annotation_mode: str,
    layout_profile: str,
    layout_rect: Optional[Tuple[float, float, float, float]] = None,
    extra: Optional[dict] = None,
    notes_lines: Optional[Sequence[str]] = None,
    table_rows: Optional[List[dict]] = None,
    apply_tight_layout: bool = True,
    write_stats_csv: bool = True,
    target_width: Optional[float] = THESIS_TEXT_WIDTH,
) -> dict:
    if int(data_row_count) <= 0:
        raise ValueError(f"{figure_id}: plotting aborted because no rows were available.")

    os.makedirs(output_dir, exist_ok=True)
    png_path = os.path.join(output_dir, f"{figure_id}.png")
    pdf_path = os.path.join(output_dir, f"{figure_id}.pdf")
    stats_path = os.path.join(output_dir, f"{figure_id}_stats.csv")
    table_path = os.path.join(output_dir, f"{figure_id}_table.csv")
    notes_path = os.path.join(output_dir, f"{figure_id}_notes.txt")

    current_width, current_height = fig.get_size_inches()
    if target_width is not None and not np.isclose(current_width, target_width):
        fig.set_size_inches(target_width, current_height, forward=True)

    if apply_tight_layout:
        if layout_rect is None:
            fig.tight_layout()
        else:
            fig.tight_layout(rect=layout_rect)
    if target_width is None:
        png_save_kwargs = {"dpi": 500, "bbox_inches": "tight"}
        pdf_save_kwargs = {"dpi": 500, "bbox_inches": "tight"}
    else:
        fig.canvas.draw()
        tight_bbox = fig.get_tightbbox(fig.canvas.get_renderer())
        export_bbox = Bbox.union([fig.bbox_inches, tight_bbox])
        png_save_kwargs = {"dpi": 500, "bbox_inches": export_bbox, "pad_inches": 0}
        pdf_save_kwargs = {"dpi": 500, "bbox_inches": export_bbox, "pad_inches": 0}

    fig.savefig(png_path, **png_save_kwargs)
    fig.savefig(pdf_path, **pdf_save_kwargs)
    plt.close(fig)

    if write_stats_csv:
        serialized_rows = _to_manifest_rows(stats_rows)
        _write_stats_csv(stats_path, serialized_rows)
    else:
        stats_path = None
    if table_rows:
        _write_table_csv(table_path, table_rows)
    else:
        table_path = None
    if notes_lines:
        _write_notes(notes_path, title_generated=title_generated, notes_lines=notes_lines)
    else:
        notes_path = None

    axes_list = _axes_to_list(axes)
    axis_limits = []
    for idx, ax in enumerate(axes_list):
        axis_limits.append(
            {
                "axis": idx,
                "x_limits": tuple(float(v) for v in ax.get_xlim()),
                "y_limits": tuple(float(v) for v in ax.get_ylim()),
            }
        )

    manifest_entry = {
        "figure_id": figure_id,
        "png_path": png_path,
        "pdf_path": pdf_path,
        "stats_path": stats_path,
        "table_path": table_path,
        "notes_path": notes_path,
        "figure_path": png_path,
        "row_count": int(data_row_count),
        "matched_n": [row.get("n") for row in stats_rows if row.get("n") is not None],
        "test_used": sorted({str(row.get("test")) for row in stats_rows if row.get("test")}),
        "raw_p_values": [serialize_stats_value(row.get("p_raw")) for row in stats_rows],
        "adjusted_q_values": [serialize_stats_value(row.get("q_bh")) for row in stats_rows],
        "axis_limits": axis_limits,
        "source_files": sorted({path for path in source_files if path}),
        "title_generated": title_generated,
        "legend_mode": legend_mode,
        "annotation_mode": annotation_mode,
        "layout_profile": layout_profile,
    }
    if extra:
        manifest_entry.update(extra)
    return manifest_entry
