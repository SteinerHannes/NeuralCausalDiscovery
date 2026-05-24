import os
from contextlib import contextmanager
from typing import Optional

from models.utils import Tee


def pbar_disabled(config) -> bool:
    try:
        return bool(config.training.debug.disable_bat_pbar)
    except Exception:
        return False


def resolve_experiment_root(config, cwd: Optional[str] = None) -> str:
    folds = [str(fold) for fold in config.training.test_folds]
    if not folds:
        raise ValueError("No configured test folds found in config.training.test_folds")

    def has_configured_fold_dirs(root: str) -> bool:
        return any(os.path.exists(os.path.join(root, fold)) for fold in folds)

    current_root = cwd if cwd is not None else os.getcwd()
    if has_configured_fold_dirs(current_root):
        return current_root

    fallback_root = config.experiment_root_fallback
    if has_configured_fold_dirs(fallback_root):
        return fallback_root

    raise ValueError("Experiment root not found")


@contextmanager
def fold_log_context(
    experiment_root,
    fold,
    log_name: str,
    step_name: Optional[str] = None,
    fold_idx: Optional[int] = None,
    num_folds: Optional[int] = None,
):
    log_path = os.path.join(experiment_root, str(fold), log_name)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with Tee(log_path, "w", 1, encoding="utf-8", newline="\n", proc_cr=True):
        if step_name:
            if fold_idx is not None and num_folds is not None:
                print(f"=== [{step_name}] Fold {fold} ({fold_idx}/{num_folds}) ===")
            else:
                print(f"=== [{step_name}] Fold {fold} ===")
        else:
            if fold_idx is not None and num_folds is not None:
                print(f"=== Fold {fold} ({fold_idx}/{num_folds}) ===")
            else:
                print(f"=== Fold {fold} ===")
        yield


@contextmanager
def step_log_context(experiment_root, log_name: str, step_name: str):
    log_path = os.path.join(experiment_root, log_name)
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    with Tee(log_path, "w", 1, encoding="utf-8", newline="\n", proc_cr=True):
        print(f"=== [{step_name}] ===")
        yield
