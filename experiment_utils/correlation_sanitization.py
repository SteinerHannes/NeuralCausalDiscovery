import numpy as np


def sanitize_dataset_for_corr(dataset: np.ndarray, seed: int, eps: float = 1e-8) -> np.ndarray:
    """
    Add tiny deterministic noise to zero-variance columns so correlation-based CI tests stay finite.
    """
    data = np.asarray(dataset, dtype=float)
    if data.size == 0:
        return data

    std = data.std(axis=0)
    zero_var = std == 0
    if not np.any(zero_var):
        return data

    rng = np.random.default_rng(seed)
    stabilized = data.copy()
    stabilized[:, zero_var] += rng.normal(0.0, eps, size=(stabilized.shape[0], zero_var.sum()))
    print(f"Warning: {int(zero_var.sum())} constant columns detected; added tiny noise for correlation stability.")
    return stabilized
