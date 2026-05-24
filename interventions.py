import os
import json
import sys
import warnings
import numpy as np
import pandas as pd
import hydra
import logging
import torch
from typing import Optional
from omegaconf import DictConfig
from scipy import stats
from dowhy import CausalModel
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

from graphical_models import DAG, PAG
from metrics.performance_measures import _get_pag_edge_type
from data.InterventionDataset import InterventionDataset
from data.utils import get_global_stats
from experiment_utils.alignment import (
    ALIGNMENT_TARGET_OBSERVABLE_STATE,
    ALIGNMENT_TARGET_SEM_TRUTH,
    apply_ridge_alignment,
    observable_state_columns,
)
from extraction import extract_data
from experiment_utils.logging_utils import fold_log_context, pbar_disabled, resolve_experiment_root

_ALIGNED_PROXY_RESULT_COLUMNS = [
    "fold",
    "target_mode",
    "treatment",
    "outcome",
    "low_value",
    "high_value",
    "ate_aligned_dowhy",
    "ate_sem_do",
    "delta_aligned_minus_sem",
    "abs_delta",
    "sem_ks_stat",
    "sem_ks_p",
    "sem_effect_by_ate",
    "sem_effect_by_ks",
    "sem_effect_present",
    "dowhy_error",
]


def _dag_to_dot(dag: DAG, node_names: list[str]) -> str:
    lines = ["digraph {"]
    for child in sorted(dag.nodes_set):
        for parent in sorted(dag.parents(child)):
            lines.append(f"  {node_names[parent]} -> {node_names[child]};")
    lines.append("}")
    return "\n".join(lines)


def _get_intervention_values(series: pd.Series, config: DictConfig) -> tuple[float, float]:
    mode = config.intervention.value_mode
    if mode == "quantile":
        low = series.quantile(config.intervention.low_q)
        high = series.quantile(config.intervention.high_q)
        return float(low), float(high)
    if mode == "mean_std":
        mean = series.mean()
        std = series.std()
        low = mean - config.intervention.std_mult * std
        high = mean + config.intervention.std_mult * std
        return float(low), float(high)
    if mode == "fixed":
        low = config.intervention.fixed_low
        high = config.intervention.fixed_high
        return float(low), float(high)
    raise ValueError(f"Unknown intervention value_mode: {mode}")


def _treatment_seeds(config: DictConfig, fold: int, treatment: int) -> tuple[int, int]:
    seed_seq = np.random.SeedSequence([int(config.intervention.seed), int(fold), int(treatment)])
    low_seq, high_seq = seed_seq.spawn(2)
    low_seed = int(low_seq.generate_state(1, dtype=np.uint64)[0])
    high_seed = int(high_seq.generate_state(1, dtype=np.uint64)[0])
    return low_seed, high_seed


def _load_alignment_mapping(mapping_path: str) -> Optional[dict]:
    if not os.path.exists(mapping_path):
        return None
    data = np.load(mapping_path, allow_pickle=True)
    feature_columns = data.get("feature_columns")
    if feature_columns is not None:
        feature_columns = [str(col) for col in feature_columns.tolist()]
    target_columns = data.get("target_columns")
    if target_columns is not None:
        target_columns = [str(col) for col in target_columns.tolist()]
    target_mode = data.get("target_mode")
    if target_mode is None:
        raise ValueError(f"Unsupported target mode: {target_mode}")
    fit_intercept_flag = data.get("fit_intercept")
    fit_intercept = bool(int(fit_intercept_flag[0])) if fit_intercept_flag is not None else True
    return {
        "weights": data["weights"],
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "target_mode": target_mode,
        "x_mean": data["x_mean"],
        "x_std": data["x_std"],
        "y_mean": data["y_mean"],
        "y_std": data["y_std"],
        "fit_intercept": fit_intercept,
    }


def _expected_alignment_target_mode(prediction_source: str) -> Optional[str]:
    if prediction_source == "aligned_sem":
        return ALIGNMENT_TARGET_SEM_TRUTH
    if prediction_source == "aligned_observable_state":
        return ALIGNMENT_TARGET_OBSERVABLE_STATE
    if prediction_source == "outputs":
        return None
    raise ValueError(f"Unknown model prediction_source: {prediction_source}")


def _collect_outcome_std(dag_data: np.ndarray) -> np.ndarray:
    if dag_data.size == 0:
        return np.array([], dtype=float)
    std = np.asarray(dag_data.std(axis=0, ddof=0), dtype=float)
    std[std == 0.0] = 1.0
    return std


def _validate_observable_state_mapping(
    mapping: dict,
    input_size: int,
    output_size: int,
    num_nodes: int,
) -> list[str]:
    if input_size + output_size != num_nodes:
        raise ValueError(
            "All-node observable-state scoring requires full node coverage in x+y: "
            f"input_size + output_size = {input_size + output_size}, num_nodes = {num_nodes}."
        )
    expected_columns = observable_state_columns(input_size, output_size)
    target_columns = mapping.get("target_columns") or []
    if not target_columns:
        raise ValueError(
            "Observable-state alignment mapping is missing target_columns; "
            "full-state scoring cannot be validated."
        )
    missing_columns = [col for col in expected_columns if col not in target_columns]
    if missing_columns:
        raise ValueError(
            "Observable-state all-node scoring requires canonical full-state target columns. "
            f"Missing columns: {missing_columns}."
        )
    return [str(col) for col in target_columns]


def _observable_state_target_column(node_index: int, input_size: int) -> str:
    if node_index < input_size:
        return f"input_{node_index}"
    return f"output_{node_index - input_size}"


def _outcome_role(node_index: int, input_size: int) -> str:
    return "input" if node_index < input_size else "output"


def _prepare_model_inputs(
    data: np.ndarray,
    input_size: int,
    device: torch.device,
    x_mean: torch.Tensor,
    x_std: torch.Tensor,
) -> torch.Tensor:
    x = torch.as_tensor(data[:, :input_size], dtype=torch.float32, device=device)
    return (x - x_mean) / x_std


def _prepare_model_targets(
    data: np.ndarray,
    input_size: int,
    output_size: int,
    device: torch.device,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> torch.Tensor:
    y = torch.as_tensor(data[:, input_size:input_size + output_size], dtype=torch.float32, device=device)
    return (y - y_mean) / y_std


def _predict_model_outputs(
    model: torch.nn.Module,
    x: torch.Tensor,
    batch_size: int,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            batch = x[start:start + batch_size]
            preds = model(batch)
            preds = preds * y_std + y_mean
            outputs.append(preds.detach().cpu())
    return torch.cat(outputs, dim=0).numpy()


def _predict_aligned_targets(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    batch_size: int,
    extraction_cfg: DictConfig,
    mapping: dict,
    progress: bool,
    progress_desc: Optional[str],
) -> np.ndarray:
    if mapping is None or mapping.get("feature_columns") is None:
        raise ValueError("Alignment mapping or feature columns missing for aligned prediction.")

    dataset = TensorDataset(x.detach().cpu(), y.detach().cpu())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    features_df = extract_data(
        model,
        loader,
        extraction_cfg,
        progress=progress,
        progress_desc=progress_desc,
    )

    missing_cols = [col for col in mapping["feature_columns"] if col not in features_df.columns]
    if missing_cols:
        raise ValueError(f"Alignment feature columns missing from extracted data: {missing_cols}")

    features = features_df[mapping["feature_columns"]].values
    features = (features - mapping["x_mean"]) / mapping["x_std"]

    y_pred = apply_ridge_alignment(features, mapping["weights"], fit_intercept=mapping["fit_intercept"])
    y_pred = y_pred * mapping["y_std"] + mapping["y_mean"]
    return y_pred


def _pag_predicts_direct_effect(pag: PAG, i: int, j: int) -> bool:
    edge_type = _get_pag_edge_type(pag, i, j)
    return edge_type in {"directed_ij", "partial_ij", "partial_ij_tail"}


def _pag_predicts_total_effect(pag: PAG, i: int, j: int) -> bool:
    if hasattr(pag, "is_possible_ancestor"):
        return bool(pag.is_possible_ancestor(i, j))
    return _pag_predicts_direct_effect(pag, i, j)


def _safe_divide(numerator: int, denominator: int) -> Optional[float]:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _compute_confusion_metrics(
    df: pd.DataFrame,
    pred_col: str = "pag_predicts_total",
    true_col: str = "effect_present",
) -> Optional[dict]:
    if df.empty or pred_col not in df or true_col not in df:
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

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    f1 = _safe_divide(2 * tp, 2 * tp + fp + fn)

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
    }


def _mean_bool_column(df: pd.DataFrame, col: str) -> Optional[float]:
    if col not in df or not df[col].notna().any():
        return None
    return float(df[col].mean())


def _mean_numeric_column(df: pd.DataFrame, col: str, absolute: bool = False) -> Optional[float]:
    if col not in df:
        return None
    series = pd.to_numeric(df[col], errors="coerce").dropna()
    if series.empty:
        return None
    if absolute:
        series = series.abs()
    return float(series.mean())


_CONFUSION_METRIC_KEYS = ("tp", "fp", "tn", "fn", "precision", "recall", "specificity", "f1")


def _flatten_confusion_metrics(summary_row: dict, prefix: str, role: str, confusion: Optional[dict]) -> None:
    suffix = f"_{role}"
    for metric in _CONFUSION_METRIC_KEYS:
        summary_row[f"{prefix}_{metric}{suffix}"] = None if confusion is None else confusion.get(metric)


def _add_role_detection_summary_metrics(summary_row: dict, fold_df: pd.DataFrame, role: str) -> None:
    role_df = fold_df[fold_df["outcome_role"] == role]
    true_col = "effect_present"
    summary_row[f"effect_positive_rate_{role}"] = _mean_bool_column(role_df, true_col)
    pag_confusion = _compute_confusion_metrics(
        role_df,
        pred_col="pag_predicts_total",
        true_col=true_col,
    )
    model_confusion = _compute_confusion_metrics(
        role_df,
        pred_col="model_effect_present",
        true_col=true_col,
    )
    _flatten_confusion_metrics(summary_row, "pag", role, pag_confusion)
    _flatten_confusion_metrics(summary_row, "model", role, model_confusion)


def _truthy_mask(series: pd.Series) -> pd.Series:
    normalized = series.fillna(False).astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "1.0"})


def _add_role_summary_metrics(summary_row: dict, fold_df: pd.DataFrame, role: str) -> None:
    role_df = fold_df[fold_df["outcome_role"] == role]
    summary_row[f"model_agreement_rate_{role}"] = _mean_bool_column(role_df, "model_agreement")
    summary_row[f"model_mean_ks_p_{role}"] = _mean_numeric_column(role_df, "model_ks_p")
    if {"model_ate_std", "ate_true_std"}.issubset(role_df.columns):
        std_error_all = (
            pd.to_numeric(role_df["model_ate_std"], errors="coerce")
            - pd.to_numeric(role_df["ate_true_std"], errors="coerce")
        ).abs()
        std_error_all = std_error_all.dropna()
        summary_row[f"std_mae_{role}_all"] = float(std_error_all.mean()) if not std_error_all.empty else None

        if "effect_present" in role_df.columns:
            positive_mask = _truthy_mask(role_df["effect_present"])
            criterion_df = role_df.loc[positive_mask]
            if criterion_df.empty:
                summary_row[f"std_mae_{role}"] = None
            else:
                std_error = (
                    pd.to_numeric(criterion_df["model_ate_std"], errors="coerce")
                    - pd.to_numeric(criterion_df["ate_true_std"], errors="coerce")
                ).abs().dropna()
                summary_row[f"std_mae_{role}"] = float(std_error.mean()) if not std_error.empty else None
        else:
            summary_row[f"std_mae_{role}"] = None
    else:
        summary_row[f"std_mae_{role}"] = None
        summary_row[f"std_mae_{role}_all"] = None
    ate_true_std = pd.to_numeric(role_df.get("ate_true_std"), errors="coerce") if "ate_true_std" in role_df else pd.Series(dtype=float)
    model_ate_std = pd.to_numeric(role_df.get("model_ate_std"), errors="coerce") if "model_ate_std" in role_df else pd.Series(dtype=float)
    summary_row[f"eta_data_{role}"] = float(ate_true_std.abs().mean()) if not ate_true_std.dropna().empty else None
    summary_row[f"eta_model_{role}"] = float(model_ate_std.abs().mean()) if not model_ate_std.dropna().empty else None
    summary_row[f"mean_ks_p_{role}"] = _mean_numeric_column(role_df, "ks_p")
    _add_role_detection_summary_metrics(summary_row, fold_df, role)

def _configure_dowhy_logging() -> None:
    logger = logging.getLogger("dowhy.causal_identifier.auto_identifier")

    class _DowhyVersionWarningFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            return "Generalized covariate adjustment identification is not supported" not in record.getMessage()

    logger.addFilter(_DowhyVersionWarningFilter())


def _aligned_proxy_target_columns(
    config: DictConfig,
    num_nodes: int,
) -> tuple[Optional[list[str]], Optional[str], Optional[str]]:
    alignment_cfg = config.extraction.get("alignment", {})
    target_mode = str(alignment_cfg.get("target_mode", ALIGNMENT_TARGET_SEM_TRUTH))

    if target_mode == ALIGNMENT_TARGET_SEM_TRUTH:
        prefix = str(alignment_cfg.get("output_prefix", "sem"))
        return [f"{prefix}_{idx}" for idx in range(int(num_nodes))], target_mode, None

    if target_mode == ALIGNMENT_TARGET_OBSERVABLE_STATE:
        input_size = int(config.data.input_size)
        output_size = int(config.data.output_size)
        if input_size + output_size != int(num_nodes):
            return (
                None,
                target_mode,
                "observable_state alignment requires input_size + output_size == num_nodes "
                f"(got {input_size + output_size} vs {num_nodes}).",
            )
        return observable_state_columns(input_size, output_size), target_mode, None

    return None, target_mode, f"Unsupported alignment target_mode: {target_mode}"


def _same_sign_rate(pred: pd.Series, true: pd.Series) -> Optional[float]:
    pred_num = pd.to_numeric(pred, errors="coerce")
    true_num = pd.to_numeric(true, errors="coerce")
    valid_mask = pred_num.notna() & true_num.notna()
    if not valid_mask.any():
        return None
    pred_sign = np.sign(pred_num.loc[valid_mask].values)
    true_sign = np.sign(true_num.loc[valid_mask].values)
    return float((pred_sign == true_sign).mean())


def _aligned_proxy_metrics(df: pd.DataFrame, effect_threshold: float) -> dict:
    if df.empty:
        return {
            "num_pairs_total": 0,
            "num_pairs_valid": 0,
            "mean_abs_delta": None,
            "median_abs_delta": None,
            "corr_aligned_vs_sem": None,
            "same_sign_rate": None,
            "same_sign_rate_strong_effect": None,
            "num_pairs_strong_effect": 0,
            "same_sign_rate_sem_effect_present": None,
            "num_pairs_sem_effect_present": 0,
        }

    ate_aligned = pd.to_numeric(df["ate_aligned_dowhy"], errors="coerce")
    ate_sem = pd.to_numeric(df["ate_sem_do"], errors="coerce")
    valid_mask = ate_aligned.notna() & ate_sem.notna()
    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        return {
            "num_pairs_total": int(len(df)),
            "num_pairs_valid": 0,
            "mean_abs_delta": None,
            "median_abs_delta": None,
            "corr_aligned_vs_sem": None,
            "same_sign_rate": None,
            "same_sign_rate_strong_effect": None,
            "num_pairs_strong_effect": 0,
            "same_sign_rate_sem_effect_present": None,
            "num_pairs_sem_effect_present": 0,
        }

    aligned_valid = ate_aligned.loc[valid_mask]
    sem_valid = ate_sem.loc[valid_mask]
    abs_delta = (aligned_valid - sem_valid).abs()
    corr = None
    if valid_count > 1:
        aligned_std = float(aligned_valid.std(ddof=0))
        sem_std = float(sem_valid.std(ddof=0))
        if aligned_std > 0.0 and sem_std > 0.0:
            corr_val = aligned_valid.corr(sem_valid)
            if not pd.isna(corr_val):
                corr = float(corr_val)

    strong_mask = sem_valid.abs() >= float(effect_threshold)
    strong_count = int(strong_mask.sum())
    same_sign_strong = None
    if strong_count > 0:
        same_sign_strong = _same_sign_rate(aligned_valid.loc[strong_mask], sem_valid.loc[strong_mask])

    sem_effect_present_rate = None
    sem_effect_count = 0
    if "sem_effect_present" in df.columns:
        sem_effect_col = df["sem_effect_present"].fillna(False).astype(bool)
        sem_effect_valid = sem_effect_col.loc[valid_mask]
        sem_effect_count = int(sem_effect_valid.sum())
        if sem_effect_count > 0:
            sem_effect_present_rate = _same_sign_rate(
                aligned_valid.loc[sem_effect_valid],
                sem_valid.loc[sem_effect_valid],
            )

    return {
        "num_pairs_total": int(len(df)),
        "num_pairs_valid": valid_count,
        "mean_abs_delta": float(abs_delta.mean()),
        "median_abs_delta": float(abs_delta.median()),
        "corr_aligned_vs_sem": corr,
        "same_sign_rate": _same_sign_rate(aligned_valid, sem_valid),
        "same_sign_rate_strong_effect": same_sign_strong,
        "num_pairs_strong_effect": strong_count,
        "same_sign_rate_sem_effect_present": sem_effect_present_rate,
        "num_pairs_sem_effect_present": sem_effect_count,
    }


def _run_aligned_proxy_dowhy_for_fold(
    config: DictConfig,
    fold: int,
    fold_dir: str,
    dataset: InterventionDataset,
    dag: DAG,
    num_nodes: int,
    disable_pbar: bool,
) -> tuple[list[dict], dict]:
    aligned_rows: list[dict] = []
    aligned_columns, target_mode, invalid_reason = _aligned_proxy_target_columns(config, num_nodes)
    fold_summary = {
        "fold": int(fold),
        "target_mode": target_mode,
        "status": "ok",
        "reason": None,
    }

    if invalid_reason is not None or aligned_columns is None:
        fold_summary["status"] = "skipped"
        fold_summary["reason"] = invalid_reason
        fold_summary.update(_aligned_proxy_metrics(pd.DataFrame(columns=_ALIGNED_PROXY_RESULT_COLUMNS), config.intervention.effect_threshold))
        return aligned_rows, fold_summary

    alignment_output_file = str(config.extraction.get("alignment", {}).get("output_file", "alignment_results.csv"))
    aligned_path = os.path.join(fold_dir, alignment_output_file)
    if not os.path.exists(aligned_path):
        fold_summary["status"] = "skipped"
        fold_summary["reason"] = f"Alignment output file not found: {aligned_path}"
        fold_summary.update(_aligned_proxy_metrics(pd.DataFrame(columns=_ALIGNED_PROXY_RESULT_COLUMNS), config.intervention.effect_threshold))
        return aligned_rows, fold_summary

    aligned_df = pd.read_csv(aligned_path, header=0)
    missing_cols = [col for col in aligned_columns if col not in aligned_df.columns]
    if missing_cols:
        fold_summary["status"] = "skipped"
        fold_summary["reason"] = f"Aligned data missing expected columns: {missing_cols}"
        fold_summary.update(_aligned_proxy_metrics(pd.DataFrame(columns=_ALIGNED_PROXY_RESULT_COLUMNS), config.intervention.effect_threshold))
        return aligned_rows, fold_summary
    aligned_df = aligned_df[aligned_columns].copy()
    aligned_std = {
        col: float(pd.to_numeric(aligned_df[col], errors="coerce").std(ddof=0))
        for col in aligned_columns
    }

    graph_dot = _dag_to_dot(dag, aligned_columns)
    pairs = [(i, j) for i in range(int(num_nodes)) for j in range(int(num_nodes)) if i != j]
    pair_iter = tqdm(
        pairs,
        desc="Aligned proxy pairs",
        unit="pair",
        file=sys.stdout,
        disable=disable_pbar,
        ascii=True,
    )

    intervention_cache = {}
    for i, j in pair_iter:
        treatment = aligned_columns[i]
        outcome = aligned_columns[j]

        if i not in intervention_cache:
            low_val, high_val = _get_intervention_values(aligned_df[treatment], config)
            low_seed, high_seed = _treatment_seeds(config, fold, i)
            data_low = dataset.simulate_intervention(
                num_samples=config.intervention.num_samples,
                cfg=config,
                intervention={i: low_val},
                seed=low_seed,
            )
            data_high = dataset.simulate_intervention(
                num_samples=config.intervention.num_samples,
                cfg=config,
                intervention={i: high_val},
                seed=high_seed,
            )
            intervention_cache[i] = (low_val, high_val, data_low, data_high)

        low_val, high_val, data_low, data_high = intervention_cache[i]
        ate_sem = float(data_high[:, j].mean() - data_low[:, j].mean())
        sem_ks_stat, sem_ks_p = stats.ks_2samp(data_high[:, j], data_low[:, j])
        sem_ks_stat = float(sem_ks_stat)
        sem_ks_p = float(sem_ks_p)
        sem_effect_by_ate = abs(ate_sem) >= float(config.intervention.effect_threshold)
        sem_effect_by_ks = sem_ks_p < float(config.intervention.alpha)
        sem_effect_present = sem_effect_by_ate and sem_effect_by_ks

        ate_aligned = None
        delta = None
        abs_delta = None
        dowhy_error = None
        treat_std = aligned_std.get(treatment, 0.0)
        outcome_std = aligned_std.get(outcome, 0.0)
        if (not np.isfinite(treat_std)) or (not np.isfinite(outcome_std)) or treat_std == 0.0 or outcome_std == 0.0:
            dowhy_error = (
                "Skipped DoWhy estimate due to degenerate aligned variance "
                f"(treatment_std={treat_std}, outcome_std={outcome_std})."
            )
        else:
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"statsmodels\..*")
                    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"scipy\.stats\..*")
                    dowhy_model = CausalModel(
                        data=aligned_df,
                        treatment=treatment,
                        outcome=outcome,
                        graph=graph_dot,
                    )
                    estimand = dowhy_model.identify_effect(proceed_when_unidentifiable=True)
                    estimate = dowhy_model.estimate_effect(
                        estimand,
                        method_name=config.intervention.estimate_method,
                        treatment_value=high_val,
                        control_value=low_val,
                        method_params={"init_params": {"need_conditional_estimates": False}},
                    )
                ate_aligned = float(estimate.value)
                delta = float(ate_aligned - ate_sem)
                abs_delta = float(abs(delta))
            except Exception as exc:
                dowhy_error = str(exc)

        aligned_rows.append({
            "fold": int(fold),
            "target_mode": target_mode,
            "treatment": treatment,
            "outcome": outcome,
            "low_value": float(low_val),
            "high_value": float(high_val),
            "ate_aligned_dowhy": ate_aligned,
            "ate_sem_do": ate_sem,
            "delta_aligned_minus_sem": delta,
            "abs_delta": abs_delta,
            "sem_ks_stat": sem_ks_stat,
            "sem_ks_p": sem_ks_p,
            "sem_effect_by_ate": sem_effect_by_ate,
            "sem_effect_by_ks": sem_effect_by_ks,
            "sem_effect_present": sem_effect_present,
            "dowhy_error": dowhy_error,
        })

    fold_summary.update(_aligned_proxy_metrics(pd.DataFrame(aligned_rows), config.intervention.effect_threshold))
    return aligned_rows, fold_summary


def run(_config: DictConfig) -> None:
    config = _config
    _configure_dowhy_logging()

    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()
    device = torch.device("mps" if use_mps else "cuda" if use_cuda else "cpu")
    torch.set_default_dtype(torch.float32)

    experiment_root = resolve_experiment_root(config)

    os.makedirs(experiment_root, exist_ok=True)

    all_rows = []
    summary = []
    aligned_proxy_rows = []
    aligned_proxy_summary = []
    compute_confusion = config.intervention.evaluate_pairs == "all"

    model_cfg = config.intervention.get("model_level", {})
    model_enabled = bool(model_cfg.get("enabled", False))
    prediction_source = model_cfg.get("prediction_source", "aligned_sem")
    _expected_alignment_target_mode(prediction_source)
    model_batch_size = int(model_cfg.get("batch_size", 256))
    model_effect_threshold = float(model_cfg.get("effect_threshold", config.intervention.effect_threshold))
    model_alpha = float(model_cfg.get("alpha", config.intervention.alpha))
    alignment_mapping_file = model_cfg.get("mapping_file", None)

    global_stats = None
    if model_enabled:
        global_stats = get_global_stats(config=config, use_folds=True, subset="train")

    folds = list(config.training.test_folds)
    total_folds = len(folds)
    disable_pbar = pbar_disabled(config)

    for fold_idx, test_fold in enumerate(folds, start=1):
        fold_dir = os.path.join(experiment_root, str(test_fold))
        with (fold_log_context(
            experiment_root,
            test_fold,
            "interventions.log",
            step_name="Interventions",
            fold_idx=fold_idx,
            num_folds=total_folds,
        )):
            print("Step: load dataset")

            dataset = hydra.utils.instantiate(config.data, subset="all", fold=test_fold)
            if not isinstance(dataset, InterventionDataset):
                raise ValueError("Dataset must implement InterventionDataset.simulate_intervention().")
            dag = dataset.dag
            dag_data = dataset.dag_data
            num_nodes = len(dag.nodes_set)
            node_names = [f"X{i}" for i in range(num_nodes)]
            df_obs = pd.DataFrame(dag_data, columns=node_names)
            outcome_std = _collect_outcome_std(dag_data)

            graph_dot = _dag_to_dot(dag, node_names)

            model = None
            model_prediction_source = prediction_source
            x_mean = x_std = y_mean = y_std = None
            alignment_mapping = None

            if model_enabled:
                print("Step: load model for model-level interventions")
                model = hydra.utils.instantiate(
                    config.model,
                    input_size=config.data.input_size,
                    output_size=config.data.output_size,
                )
                model_path = os.path.join(fold_dir, config.torch_model_name)
                model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
                model.to(device)
                model.eval()

                if global_stats is None:
                    raise ValueError("Global stats unavailable for model-level interventions.")
                x_mean, x_std, y_mean, y_std = global_stats[fold_idx - 1]
                x_mean = x_mean.to(device)
                x_std = x_std.to(device)
                y_mean = y_mean.to(device)
                y_std = y_std.to(device)

                if model_prediction_source in {"aligned_sem", "aligned_observable_state"}:
                    mapping_filename = alignment_mapping_file or config.extraction.alignment.mapping_file
                    mapping_path = os.path.join(fold_dir, mapping_filename)
                    alignment_mapping = _load_alignment_mapping(mapping_path)
                    if alignment_mapping is None:
                        print(f"Warning: alignment mapping not found at {mapping_path}. Falling back to outputs.")
                        model_prediction_source = "outputs"
                    else:
                        expected_target_mode = _expected_alignment_target_mode(model_prediction_source)
                        mapping_target_mode = alignment_mapping.get("target_mode", ALIGNMENT_TARGET_SEM_TRUTH)
                        if mapping_target_mode != expected_target_mode:
                            raise ValueError(
                                "Alignment mapping target_mode mismatch: "
                                f"prediction_source={model_prediction_source} expects {expected_target_mode}, "
                                f"but mapping provides {mapping_target_mode}."
                            )
                        if model_prediction_source == "aligned_observable_state":
                            target_columns = _validate_observable_state_mapping(
                                alignment_mapping,
                                input_size=int(config.data.input_size),
                                output_size=int(config.data.output_size),
                                num_nodes=int(num_nodes),
                            )
                            alignment_mapping["target_columns"] = target_columns

            pag = None
            if config.intervention.use_learned_pag:
                pag_adj_path = os.path.join(fold_dir, config.intervention.learned_pag_adj_file)
                if os.path.exists(pag_adj_path):
                    pag_adj = np.load(pag_adj_path)
                    pag = PAG(nodes_set=set(range(num_nodes)))
                    pag.init_from_adj_mat(pag_adj, nodes_order=list(range(num_nodes)))
                else:
                    print(f"Warning: learned PAG adjacency not found at {pag_adj_path}")

            pairs = []
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i == j:
                        continue
                    if config.intervention.evaluate_pairs == "parents_only":
                        if i not in dag.parents(j):
                            continue
                    pairs.append((i, j))

            print(f"Step: evaluate interventions ({len(pairs)} pairs)")
            pair_iter = tqdm(
                pairs,
                desc="Intervention pairs",
                unit="pair",
                file=sys.stdout,
                disable=disable_pbar,
                ascii=True,
            )

            intervention_cache = {}

            for i, j in pair_iter:
                treat = node_names[i]
                outcome = node_names[j]

                if i not in intervention_cache:
                    low_val, high_val = _get_intervention_values(df_obs[treat], config)
                    low_seed, high_seed = _treatment_seeds(config, test_fold, i)

                    data_low = dataset.simulate_intervention(
                        num_samples=config.intervention.num_samples,
                        cfg=config,
                        intervention={i: low_val},
                        seed=low_seed,
                    )
                    data_high = dataset.simulate_intervention(
                        num_samples=config.intervention.num_samples,
                        cfg=config,
                        intervention={i: high_val},
                        seed=high_seed,
                    )

                    model_preds = None
                    if model_enabled and model is not None:
                        x_low = _prepare_model_inputs(
                            data_low,
                            config.data.input_size,
                            device,
                            x_mean,
                            x_std,
                        )
                        x_high = _prepare_model_inputs(
                            data_high,
                            config.data.input_size,
                            device,
                            x_mean,
                            x_std,
                        )
                        y_low = _prepare_model_targets(
                            data_low,
                            config.data.input_size,
                            config.data.output_size,
                            device,
                            y_mean,
                            y_std,
                        )
                        y_high = _prepare_model_targets(
                            data_high,
                            config.data.input_size,
                            config.data.output_size,
                            device,
                            y_mean,
                            y_std,
                        )

                        if model_prediction_source in {"aligned_sem", "aligned_observable_state"}:
                            model_low = _predict_aligned_targets(
                                model,
                                x_low,
                                y_low,
                                model_batch_size,
                                config.extraction,
                                alignment_mapping,
                                progress=not disable_pbar,
                                progress_desc=f"align low fold {test_fold} treat {i}",
                            )
                            model_high = _predict_aligned_targets(
                                model,
                                x_high,
                                y_high,
                                model_batch_size,
                                config.extraction,
                                alignment_mapping,
                                progress=not disable_pbar,
                                progress_desc=f"align high fold {test_fold} treat {i}",
                            )
                            model_preds = (model_prediction_source, model_low, model_high)
                        elif model_prediction_source == "outputs":
                            model_low = _predict_model_outputs(
                                model,
                                x_low,
                                model_batch_size,
                                y_mean,
                                y_std,
                            )
                            model_high = _predict_model_outputs(
                                model,
                                x_high,
                                model_batch_size,
                                y_mean,
                                y_std,
                            )
                            model_preds = ("outputs", model_low, model_high)

                    intervention_cache[i] = (low_val, high_val, data_low, data_high, model_preds)

                low_val, high_val, data_low, data_high, model_preds = intervention_cache[i]

                ate_true = float(data_high[:, j].mean() - data_low[:, j].mean())
                ks_stat, ks_p = stats.ks_2samp(data_high[:, j], data_low[:, j])
                ks_stat = float(ks_stat)
                ks_p = float(ks_p)

                effect_by_ate = abs(ate_true) >= config.intervention.effect_threshold
                effect_by_ks = ks_p < config.intervention.alpha
                effect_present = effect_by_ate and effect_by_ks

                true_direct = i in dag.parents(j)
                true_total = dag.is_ancestor(descendant_node=j, tested_node=i) and i != j
                outcome_role = _outcome_role(j, int(config.data.input_size))

                dowhy_model = CausalModel(
                    data=df_obs,
                    treatment=treat,
                    outcome=outcome,
                    graph=graph_dot,
                )
                estimand = dowhy_model.identify_effect()
                estimate = dowhy_model.estimate_effect(
                    estimand,
                    method_name=config.intervention.estimate_method,
                    treatment_value=high_val,
                    control_value=low_val,
                    # We only consume ATE (estimate.value) in this pipeline.
                    # Disabling conditional estimates avoids a pandas GroupBy.apply
                    # deprecation warning emitted inside DoWhy.
                    method_params={"init_params": {"need_conditional_estimates": False}},
                )
                ate_est = float(estimate.value)
                ate_error = float(ate_est - ate_true)

                pag_predicts_direct = None
                pag_predicts_total = None
                agreement_direct = None
                agreement_total = None
                if pag is not None:
                    pag_predicts_direct = _pag_predicts_direct_effect(pag, i, j)
                    pag_predicts_total = _pag_predicts_total_effect(pag, i, j)
                    agreement_direct = (pag_predicts_direct == true_direct)
                    agreement_total = (pag_predicts_total == effect_present)

                model_ate = None
                model_ate_error = None
                model_ks_stat = None
                model_ks_p = None
                model_effect_by_ate = None
                model_effect_by_ks = None
                model_effect_present = None
                model_agreement = None
                model_source = None

                if model_preds is not None:
                    model_source, model_low, model_high = model_preds
                    model_index = None
                    if model_source == "aligned_sem":
                        if model_low.shape[1] > j:
                            model_index = j
                    elif model_source == "aligned_observable_state":
                        target_columns = alignment_mapping.get("target_columns") if alignment_mapping is not None else None
                        if target_columns is not None:
                            target_column = _observable_state_target_column(j, int(config.data.input_size))
                            if target_column in target_columns:
                                model_index = target_columns.index(target_column)
                    elif model_source == "outputs":
                        if j >= config.data.input_size:
                            model_index = j - config.data.input_size
                    if model_index is not None:
                        model_low_vals = model_low[:, model_index]
                        model_high_vals = model_high[:, model_index]
                        model_ate = float(model_high_vals.mean() - model_low_vals.mean())
                        model_ks_stat, model_ks_p = stats.ks_2samp(model_high_vals, model_low_vals)
                        model_ks_stat = float(model_ks_stat)
                        model_ks_p = float(model_ks_p)
                        model_effect_by_ate = abs(model_ate) >= model_effect_threshold
                        model_effect_by_ks = model_ks_p < model_alpha
                        model_effect_present = model_effect_by_ate and model_effect_by_ks
                        model_agreement = (model_effect_present == effect_present)
                        model_ate_error = float(model_ate - ate_true)

                std_scale = float(outcome_std[j]) if 0 <= j < outcome_std.size else 1.0
                ate_true_std = float(ate_true / std_scale)
                ate_est_std = float(ate_est / std_scale)
                model_ate_std = float(model_ate / std_scale) if model_ate is not None else None

                all_rows.append({
                    "fold": test_fold,
                    "treatment": treat,
                    "outcome": outcome,
                    "outcome_role": outcome_role,
                    "true_direct": true_direct,
                    "true_total": true_total,
                    "ate_true": ate_true,
                    "ate_true_std": ate_true_std,
                    "ate_est": ate_est,
                    "ate_est_std": ate_est_std,
                    "ate_error": ate_error,
                    "ks_stat": ks_stat,
                    "ks_p": ks_p,
                    "effect_by_ate": effect_by_ate,
                    "effect_by_ks": effect_by_ks,
                    "effect_present": effect_present,
                    "pag_predicts_direct": pag_predicts_direct,
                    "pag_predicts_total": pag_predicts_total,
                    "agreement_direct": agreement_direct,
                    "agreement_total": agreement_total,
                    "model_prediction_source": model_source,
                    "model_ate": model_ate,
                    "model_ate_std": model_ate_std,
                    "model_ate_error": model_ate_error,
                    "model_ks_stat": model_ks_stat,
                    "model_ks_p": model_ks_p,
                    "model_effect_by_ate": model_effect_by_ate,
                    "model_effect_by_ks": model_effect_by_ks,
                    "model_effect_present": model_effect_present,
                    "model_agreement": model_agreement,
                })

            fold_df = pd.DataFrame([r for r in all_rows if r["fold"] == test_fold])
            output_path = os.path.join(fold_dir, config.intervention.output_file)
            fold_df.to_csv(output_path, index=False)
            print(f"Saved intervention results to {output_path}")

            if not fold_df.empty:
                fold_confusion_total = _compute_confusion_metrics(
                    fold_df,
                    pred_col="pag_predicts_total",
                    true_col="effect_present",
                ) if compute_confusion else None
                fold_confusion_direct = _compute_confusion_metrics(
                    fold_df,
                    pred_col="pag_predicts_direct",
                    true_col="true_direct",
                ) if compute_confusion else None
                model_confusion = None
                if compute_confusion and model_enabled:
                    model_confusion = _compute_confusion_metrics(
                        fold_df,
                        pred_col="model_effect_present",
                        true_col="effect_present",
                    )
                summary_row = {
                    "fold": test_fold,
                    "agreement_rate_total": _mean_bool_column(fold_df, "agreement_total"),
                    "agreement_rate_direct": _mean_bool_column(fold_df, "agreement_direct"),
                    "mean_abs_ate_error": _mean_numeric_column(fold_df, "ate_error", absolute=True),
                    "mean_ks_p": _mean_numeric_column(fold_df, "ks_p"),
                    "confusion_total": fold_confusion_total,
                    "confusion_direct": fold_confusion_direct,
                    "model_agreement_rate": _mean_bool_column(fold_df, "model_agreement"),
                    "model_mean_abs_ate_error": _mean_numeric_column(fold_df, "model_ate_error", absolute=True),
                    "model_mean_ks_p": _mean_numeric_column(fold_df, "model_ks_p"),
                    "std_mae": (
                        float((fold_df["model_ate_std"] - fold_df["ate_true_std"]).abs().dropna().mean())
                        if {"model_ate_std", "ate_true_std"}.issubset(fold_df.columns)
                        and not (fold_df["model_ate_std"] - fold_df["ate_true_std"]).abs().dropna().empty
                        else None
                    ),
                    "eta_data": _mean_numeric_column(fold_df, "ate_true_std", absolute=True),
                    "eta_model": _mean_numeric_column(fold_df, "model_ate_std", absolute=True),
                    "model_confusion": model_confusion,
                }
                _add_role_summary_metrics(summary_row, fold_df, "input")
                _add_role_summary_metrics(summary_row, fold_df, "output")
                summary.append(summary_row)

            print("Step: aligned proxy DoWhy on aligned data (all ordered pairs)")
            aligned_rows_fold, aligned_summary_fold = _run_aligned_proxy_dowhy_for_fold(
                config=config,
                fold=int(test_fold),
                fold_dir=fold_dir,
                dataset=dataset,
                dag=dag,
                num_nodes=int(num_nodes),
                disable_pbar=disable_pbar,
            )
            aligned_proxy_rows.extend(aligned_rows_fold)
            aligned_proxy_summary.append(aligned_summary_fold)

    if compute_confusion and all_rows:
        all_df = pd.DataFrame(all_rows)
        global_confusion_total = _compute_confusion_metrics(
            all_df,
            pred_col="pag_predicts_total",
            true_col="effect_present",
        )
        global_confusion_direct = _compute_confusion_metrics(
            all_df,
            pred_col="pag_predicts_direct",
            true_col="true_direct",
        )
        model_global_confusion = None
        if model_enabled:
            model_global_confusion = _compute_confusion_metrics(
                all_df,
                pred_col="model_effect_present",
                true_col="effect_present",
            )
        summary.append({
            "fold": "global",
            "confusion_total": global_confusion_total,
            "confusion_direct": global_confusion_direct,
            "model_confusion": model_global_confusion,
        })

    summary_path = os.path.join(experiment_root, config.intervention.summary_file)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved intervention summary to {summary_path}")

    aligned_proxy_output_file = str(
        config.intervention.get("aligned_proxy_output_file", "aligned_proxy_dowhy_all_pairs.csv")
    )
    aligned_proxy_summary_file = str(
        config.intervention.get("aligned_proxy_summary_file", "aligned_proxy_dowhy_all_pairs_summary.json")
    )
    aligned_proxy_fold_summary_file = str(
        config.intervention.get("aligned_proxy_fold_summary_file", "aligned_proxy_dowhy_all_pairs_summary.csv")
    )

    aligned_proxy_df = pd.DataFrame(aligned_proxy_rows, columns=_ALIGNED_PROXY_RESULT_COLUMNS)
    aligned_proxy_output_path = os.path.join(experiment_root, aligned_proxy_output_file)
    aligned_proxy_df.to_csv(aligned_proxy_output_path, index=False)
    print(f"Saved aligned proxy DoWhy results to {aligned_proxy_output_path}")

    aligned_proxy_fold_df = pd.DataFrame(aligned_proxy_summary)
    aligned_proxy_fold_summary_path = os.path.join(experiment_root, aligned_proxy_fold_summary_file)
    aligned_proxy_fold_df.to_csv(aligned_proxy_fold_summary_path, index=False)
    print(f"Saved aligned proxy DoWhy fold summary to {aligned_proxy_fold_summary_path}")

    overall_aligned_proxy = _aligned_proxy_metrics(aligned_proxy_df, float(config.intervention.effect_threshold))
    overall_aligned_proxy.update({
        "num_folds_total": int(len(folds)),
        "num_folds_ok": int(sum(1 for row in aligned_proxy_summary if row.get("status") == "ok")),
        "num_folds_skipped": int(sum(1 for row in aligned_proxy_summary if row.get("status") == "skipped")),
    })
    aligned_proxy_summary_path = os.path.join(experiment_root, aligned_proxy_summary_file)
    with open(aligned_proxy_summary_path, "w") as f:
        json.dump(
            {
                "overall": overall_aligned_proxy,
                "per_fold": aligned_proxy_summary,
            },
            f,
            indent=2,
        )
    print(f"Saved aligned proxy DoWhy summary to {aligned_proxy_summary_path}")


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(_config: DictConfig) -> None:
    run(_config)


if __name__ == "__main__":
    main()
