"""Alignment utilities for mapping extracted representations to target states.

The alignment step fits a linear ridge regression that maps extracted features
to either the ground-truth SEM targets produced by the synthetic data generator
or an observable-state target composed of raw inputs and model predictions. This
keeps the alignment mapping reproducible and explicit for downstream causal
discovery and intervention analyses.
"""

import numpy as np
import pandas as pd
from data.utils import ConcatenatedDataset, StandardizedDataset

ALIGNMENT_TARGET_SEM_TRUTH = "sem_truth"
ALIGNMENT_TARGET_OBSERVABLE_STATE = "observable_state"
ALIGNMENT_TARGET_MODES = (
    ALIGNMENT_TARGET_SEM_TRUTH,
    ALIGNMENT_TARGET_OBSERVABLE_STATE,
)


def observable_state_columns(input_size: int, output_size: int) -> list[str]:
    """Return canonical observable-state column names [x, y_hat]."""
    return [f"input_{i}" for i in range(int(input_size))] + [f"output_{i}" for i in range(int(output_size))]


def alignment_target_columns(target_mode: str, config, alignment_cfg, n_targets: int) -> list[str]:
    """Return the aligned-output column names for the selected target mode."""
    if target_mode == ALIGNMENT_TARGET_OBSERVABLE_STATE:
        return observable_state_columns(config.data.input_size, config.data.output_size)

    prefix = alignment_cfg.get("output_prefix", "sem") if alignment_cfg is not None else "sem"
    return [f"{prefix}_{i}" for i in range(int(n_targets))]


def resolve_base_dataset(dataset):
    """Unwrap standardized datasets to access the underlying generator data."""
    if isinstance(dataset, StandardizedDataset):
        return dataset.base_ds
    return dataset


def collect_sem_targets(dataset) -> np.ndarray:
    """Collect SEM targets from datasets that expose the generator data.

    Requirements:
        - Dataset (or its base dataset) must define `dag_data` with raw SEM
          samples and `indices` that align with the current subset.
        - Supports concatenated datasets by recursively collecting targets.
    """
    base = resolve_base_dataset(dataset)
    if isinstance(base, ConcatenatedDataset):
        parts = [collect_sem_targets(part) for part in base.datasets]
        return np.concatenate(parts, axis=0)
    if not hasattr(base, "dag_data") or not hasattr(base, "indices"):
        raise ValueError("Alignment requires dataset with 'dag_data' and 'indices' attributes.")
    return np.asarray(base.dag_data[base.indices], dtype=float)


def select_alignment_columns(pd_data: pd.DataFrame, config, alignment_cfg) -> tuple[list[str], np.ndarray]:
    """Select feature columns used to fit the alignment mapping.

    The selection is deterministic by default, but can be randomized with a
    seed from `config.data.seed` when using the "random" selection strategy.

    Supported modes:
        - "inputs_outputs": Prefer explicit input/output columns if they exist.
        - "all" (default): Use the full column set before filtering.

    Filters:
        - include_prefixes: keep columns starting with any of these prefixes.
        - exclude_prefixes: drop columns starting with any of these prefixes.
        - max_features: cap the number of columns after filtering.

    Returns:
        (columns, values) where columns is the ordered list of selected names
        and values is the corresponding numpy array (n_samples, n_features).
    """
    columns_mode = alignment_cfg.get("columns_mode", "all")
    include_prefixes = alignment_cfg.get("include_prefixes", [])
    exclude_prefixes = alignment_cfg.get("exclude_prefixes", [])
    max_features = alignment_cfg.get("max_features")
    selection_strategy = alignment_cfg.get("selection_strategy", "first")

    if columns_mode == "inputs_outputs":
        input_cols = [f"input_{i}" for i in range(config.data.input_size)]
        output_cols = [f"output_{i}" for i in range(config.data.output_size)]
        expected = input_cols + output_cols
        missing = [col for col in expected if col not in pd_data.columns]
        if not missing:
            return expected, pd_data[expected].values

    columns = list(pd_data.columns)
    if include_prefixes:
        columns = [col for col in columns if any(col.startswith(p) for p in include_prefixes)]
    if exclude_prefixes:
        columns = [col for col in columns if not any(col.startswith(p) for p in exclude_prefixes)]

    if not columns:
        raise ValueError("Alignment column selection removed all features.")

    if max_features is not None and len(columns) > max_features:
        if selection_strategy == "random":
            seed = int(getattr(config.data, "seed", 0))
            rng = np.random.default_rng(seed)
            columns = sorted(rng.choice(columns, size=max_features, replace=False).tolist())
        else:
            columns = columns[:max_features]

    return columns, pd_data[columns].values


def standardize_array(values: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize features or targets column-wise and return (z, mean, std).

    Uses an epsilon floor to avoid division by zero for constant columns.
    """
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < eps, 1.0, std)
    return (values - mean) / std, mean, std


def fit_ridge_alignment(
    features: np.ndarray,
    targets: np.ndarray,
    alpha: float = 1e-3,
    fit_intercept: bool = True,
):
    """Fit a closed-form ridge regression mapping features -> targets.

    Args:
        features: Array of shape (n_samples, n_features).
        targets: Array of shape (n_samples, n_targets).
        alpha: Ridge regularization strength (applies to all weights except bias).
        fit_intercept: If True, append a bias column and do not regularize it.

    Returns:
        Weight matrix of shape (n_features + bias, n_targets) if `fit_intercept`
        else (n_features, n_targets).
    """
    if fit_intercept:
        ones = np.ones((features.shape[0], 1), dtype=features.dtype)
        features = np.concatenate([features, ones], axis=1)
    reg = alpha * np.eye(features.shape[1])
    if fit_intercept:
        reg[-1, -1] = 0.0
    weights = np.linalg.solve(features.T @ features + reg, features.T @ targets)
    return weights


def apply_ridge_alignment(features: np.ndarray, weights: np.ndarray, fit_intercept: bool = True) -> np.ndarray:
    """Apply the learned ridge mapping to new features."""
    if fit_intercept:
        ones = np.ones((features.shape[0], 1), dtype=features.dtype)
        features = np.concatenate([features, ones], axis=1)
    return features @ weights


def alignment_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute per-target R^2 scores; robust to constant targets."""
    denom = np.sum((y_true - y_true.mean(axis=0)) ** 2, axis=0)
    denom = np.where(denom == 0.0, 1.0, denom)
    return 1.0 - (np.sum((y_true - y_pred) ** 2, axis=0) / denom)
