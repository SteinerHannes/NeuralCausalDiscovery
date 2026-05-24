import json
import os
from pathlib import Path
import hydra
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from typing import Optional
from data.utils import (
    STATS_FILE_NAME,
    ConcatenatedDataset,
    StandardizedDataset,
    compute_dataset_stats,
    load_dataset_stats,
)
from experiment_utils.device_utils import print_torch_runtime, resolve_torch_device
from experiment_utils.alignment import (
    ALIGNMENT_TARGET_OBSERVABLE_STATE,
    ALIGNMENT_TARGET_SEM_TRUTH,
    apply_ridge_alignment,
    alignment_target_columns,
    alignment_score,
    collect_sem_targets,
    fit_ridge_alignment,
    select_alignment_columns,
    standardize_array,
)
from experiment_utils.logging_utils import fold_log_context, pbar_disabled, resolve_experiment_root
from models.Extractable import Extractable


def _loader_runtime_kwargs(num_workers: int, persistent_workers: bool, prefetch_factor: Optional[int]) -> dict:
    kwargs = {
        "num_workers": int(num_workers),
        "persistent_workers": bool(persistent_workers) and int(num_workers) > 0,
    }
    if int(num_workers) > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def _resolve_fold_stats(config: DictConfig, experiment_root: str) -> dict:
    stats_by_fold = {}
    missing_folds = []
    for fold in config.training.test_folds:
        stats_path = os.path.join(experiment_root, str(fold), STATS_FILE_NAME)
        cached = load_dataset_stats(stats_path)
        if cached is None:
            missing_folds.append(fold)
        else:
            stats_by_fold[int(fold)] = cached

    if missing_folds:
        print(
            "Cached training statistics missing for folds "
            f"{missing_folds}. Recomputing from train split."
        )
    for fold in missing_folds:
        ds = hydra.utils.instantiate(config.data, subset="train", fold=fold)
        stats_by_fold[int(fold)] = compute_dataset_stats(ds)
    return stats_by_fold


def extract_data(
    model: Extractable,
    dataloader: torch.utils.data.DataLoader,
    extraction_config: DictConfig,
    progress: bool = False,
    progress_desc: Optional[str] = None,
) -> pd.DataFrame:
    assert isinstance(model, Extractable)
    method = extraction_config.get("method", "activations")

    prefix = extraction_config.get("prefix")

    if method == "inputs_outputs":
        return model.extract_inputs_outputs(
            dataloader,
            include_inputs=extraction_config.get("include_inputs", True),
            include_outputs=extraction_config.get("include_outputs", True),
            flatten=extraction_config.get("flatten", True),
            progress=progress,
            progress_desc=progress_desc,
        )
    if method == "pre_activations":
        return model.extract_pre_activations(
            dataloader,
            layer_types=extraction_config.get("layer_types"),
            include_inputs=extraction_config.get("include_inputs", True),
            include_outputs=extraction_config.get("include_outputs", True),
            flatten=extraction_config.get("flatten", True),
            prefix=prefix if prefix is not None else "pre",
            progress=progress,
            progress_desc=progress_desc,
        )
    if method == "input_gradients":
        return model.extract_input_gradients(
            dataloader,
            include_inputs=extraction_config.get("include_inputs", True),
            include_outputs=extraction_config.get("include_outputs", True),
            flatten=extraction_config.get("flatten", True),
            grad_target=extraction_config.get("grad_target", "sum"),
            progress=progress,
            progress_desc=progress_desc,
        )
    if method == "weights":
        return model.extract_weights(
            dataloader=dataloader,
            repeat_for_samples=extraction_config.get("weights_repeat", False),
            include_bias=extraction_config.get("weights_include_bias", True),
            prefix=prefix if prefix is not None else "param",
            progress=progress,
            progress_desc=progress_desc,
        )

    return model.extract_activations(
        dataloader,
        layer_types=extraction_config.get("layer_types"),
        include_inputs=extraction_config.get("include_inputs", True),
        include_outputs=extraction_config.get("include_outputs", True),
        flatten=extraction_config.get("flatten", True),
        prefix=prefix if prefix is not None else "act",
        progress=progress,
        progress_desc=progress_desc,
    )


def _predict_model_outputs_raw(
    model: Extractable,
    dataloader: torch.utils.data.DataLoader,
    y_mean: torch.Tensor,
    y_std: torch.Tensor,
) -> np.ndarray:
    """Predict model outputs on standardized inputs and return raw output units."""
    device = next(model.parameters()).device
    y_mean = torch.as_tensor(y_mean, dtype=torch.float32, device=device)
    y_std = torch.as_tensor(y_std, dtype=torch.float32, device=device)

    outputs = []
    with torch.inference_mode():
        for x, _ in dataloader:
            x = x.float().to(device)
            preds = model(x)
            preds = preds * y_std + y_mean
            outputs.append(preds.detach().cpu().numpy())

    if not outputs:
        return np.empty((0, int(y_mean.numel())), dtype=float)
    return np.concatenate(outputs, axis=0)


def run(_config: DictConfig) -> None:
    config = _config
    requested_device = config.training.get("device", "auto")
    device = resolve_torch_device(requested_device)
    # due to MPS limitations
    torch.set_default_dtype(torch.float32)
    pin_memory = device.type == "cuda"
    if bool(config.training.get("print_device_info", True)):
        print_torch_runtime(device)

    experiment_root = resolve_experiment_root(config)
    fold_stats = _resolve_fold_stats(config, experiment_root)

    extract_batch_size = int(config.extraction.get("batch_size", 256))
    extract_num_workers = int(config.extraction.get("num_workers", 0) or 0)
    extract_persistent_workers = bool(config.extraction.get("persistent_workers", False))
    extract_prefetch = config.extraction.get("prefetch_factor", None)
    extract_loader_kwargs = _loader_runtime_kwargs(
        num_workers=extract_num_workers,
        persistent_workers=extract_persistent_workers,
        prefetch_factor=extract_prefetch,
    )

    folds = list(config.training.test_folds)
    total_folds = len(folds)
    disable_pbar = pbar_disabled(config)

    for fold_idx, test_fold in enumerate(folds, start=1):
        with fold_log_context(
            experiment_root,
            test_fold,
            "extraction.log",
            step_name="Extraction",
            fold_idx=fold_idx,
            num_folds=total_folds,
        ):
            print("Step: load model")

            model = hydra.utils.instantiate(
                config.model,
                input_size=config.data.input_size,
                output_size=config.data.output_size
            )
            model_path = os.path.join(experiment_root, str(test_fold), config.torch_model_name)
            model.load_state_dict(torch.load(model_path, weights_only=True, map_location=device))
            model.to(device)
            model.eval()
            print(f"Loaded model from {model_path}")

            print("Step: prepare extraction dataset")
            extract_ds = hydra.utils.instantiate(
                config.data,
                subset="extract",
                fold=test_fold
            )

            x_mean, x_std, y_mean, y_std = fold_stats[int(test_fold)]
            extract_ds = StandardizedDataset(extract_ds, x_mean, x_std, y_mean, y_std)
            extract_loader = torch.utils.data.DataLoader(
                extract_ds,
                batch_size=extract_batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                drop_last=False,
                **extract_loader_kwargs,
            )

            print("Step: extract representations")
            extraction_data = extract_data(
                model,
                extract_loader,
                config.extraction,
                progress=not disable_pbar,
                progress_desc=f"extract fold {test_fold}",
            )

            output_file = os.path.join(experiment_root, str(test_fold), config.extraction.output_file)
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            extraction_data.to_csv(output_file, index=False)
            print(f"Saved extraction data to {output_file}")
            print(f"Shape: {extraction_data.shape}")

            alignment_cfg = config.extraction.get("alignment", {})
            if alignment_cfg.get("enabled", False):
                print("Step: alignment")
                # Fit a linear map from extracted features to the configured alignment
                # target, producing an aligned representation for causal discovery.
                # Fit on validation by default to avoid consuming the extraction split
                # (which is reserved for downstream causal discovery inputs).
                target_mode =  alignment_cfg.get("target_mode", ALIGNMENT_TARGET_SEM_TRUTH)
                fit_subset = alignment_cfg.get("fit_subset", "val")
                fit_subsets = [fit_subset] if isinstance(fit_subset, str) else list(fit_subset)
                if len(fit_subsets) == 1 and fit_subsets[0] in ("train_val", "train+val"):
                    fit_subsets = ["train", "val"]

                fit_datasets = [
                    hydra.utils.instantiate(config.data, subset=subset, fold=test_fold)
                    for subset in fit_subsets
                ]
                if len(fit_datasets) == 1:
                    fit_ds = fit_datasets[0]
                else:
                    fit_ds = ConcatenatedDataset(fit_datasets)

                fit_ds = StandardizedDataset(fit_ds, x_mean, x_std, y_mean, y_std)
                fit_loader = torch.utils.data.DataLoader(
                    fit_ds,
                    batch_size=extract_batch_size,
                    shuffle=False,
                    pin_memory=pin_memory,
                    drop_last=False,
                    **extract_loader_kwargs,
                )

                fit_features = extract_data(
                    model,
                    fit_loader,
                    config.extraction,
                    progress=not disable_pbar,
                    progress_desc=f"align fit fold {test_fold}",
                )
                # Column selection controls which extracted features are used for alignment.
                feature_columns, x_fit = select_alignment_columns(fit_features, config, alignment_cfg)
                if target_mode == ALIGNMENT_TARGET_OBSERVABLE_STATE:
                    sem_state_fit = collect_sem_targets(fit_ds)
                    x_fit_raw = sem_state_fit[:, :config.data.input_size]
                    y_hat_fit_raw = _predict_model_outputs_raw(model, fit_loader, y_mean=y_mean, y_std=y_std)
                    y_fit = np.concatenate([x_fit_raw, y_hat_fit_raw], axis=1)
                elif target_mode == ALIGNMENT_TARGET_SEM_TRUTH:
                    y_fit = collect_sem_targets(fit_ds)
                else:
                    raise ValueError(f"Unsupported target mode: {target_mode}")
                target_columns = alignment_target_columns(
                    target_mode,
                    config=config,
                    alignment_cfg=alignment_cfg,
                    n_targets=y_fit.shape[1],
                )
                if x_fit.shape[0] != y_fit.shape[0]:
                    raise ValueError(
                        "Alignment feature/target row mismatch: "
                        f"features={x_fit.shape[0]}, targets={y_fit.shape[0]}. "
                        "For extraction.method=weights, set extraction.weights_repeat=true "
                        "so features are repeated per sample."
                    )

                # Standardize for numerical stability and comparable scaling across folds.
                if alignment_cfg.get("standardize_features", True):
                    x_fit, x_mean_fit, x_std_fit = standardize_array(x_fit)
                else:
                    x_mean_fit = np.zeros(x_fit.shape[1], dtype=float)
                    x_std_fit = np.ones(x_fit.shape[1], dtype=float)

                # Targets correspond to SEM variables and can be standardized independently.
                if alignment_cfg.get("standardize_targets", True):
                    y_fit, y_mean_fit, y_std_fit = standardize_array(y_fit)
                else:
                    y_mean_fit = np.zeros(y_fit.shape[1], dtype=float)
                    y_std_fit = np.ones(y_fit.shape[1], dtype=float)

                alpha = float(alignment_cfg.get("ridge_alpha", 1e-3))
                fit_intercept = bool(alignment_cfg.get("fit_intercept", True))
                weights = fit_ridge_alignment(x_fit, y_fit, alpha=alpha, fit_intercept=fit_intercept)

                y_fit_pred = apply_ridge_alignment(x_fit, weights, fit_intercept=fit_intercept)
                r2 = alignment_score(y_fit, y_fit_pred)

                missing_cols = [col for col in feature_columns if col not in extraction_data.columns]
                if missing_cols:
                    raise ValueError(
                        f"Alignment columns missing from extraction data: {missing_cols}. "
                        "Ensure extraction settings match alignment feature selection."
                    )

                x_extract = extraction_data[feature_columns].values
                if alignment_cfg.get("standardize_features", True):
                    x_extract = (x_extract - x_mean_fit) / x_std_fit

                # Apply the fitted map to the extraction split and unstandardize targets.
                y_extract_pred = apply_ridge_alignment(x_extract, weights, fit_intercept=fit_intercept)
                if alignment_cfg.get("standardize_targets", True):
                    y_extract_pred = y_extract_pred * y_std_fit + y_mean_fit

                aligned_df = pd.DataFrame(
                    y_extract_pred,
                    columns=target_columns,
                )

                aligned_output_file = alignment_cfg.get("output_file", "alignment_results.csv")
                aligned_output_path = Path(os.path.join(experiment_root, str(test_fold), aligned_output_file))
                aligned_output_path.parent.mkdir(parents=True, exist_ok=True)
                aligned_df.to_csv(aligned_output_path, index=False)
                print(f"Saved aligned data to {aligned_output_path}")
                print(f"Shape: {aligned_df.shape}")

                metrics = {
                    "fold": int(test_fold),
                    "r2_mean": float(np.mean(r2)),
                    "r2_per_target": [float(val) for val in r2.tolist()],
                    "n_features": int(x_fit.shape[1]),
                    "n_targets": int(y_fit.shape[1]),
                    "target_mode": target_mode,
                    "target_columns": list(target_columns),
                }
                metrics_file = alignment_cfg.get("metrics_file", "alignment_metrics.json")
                metrics_path = Path(os.path.join(experiment_root, str(test_fold), metrics_file))
                with metrics_path.open("w") as f:
                    json.dump(metrics, f, indent=2)
                print(f"Saved alignment metrics to {metrics_path}")

                if alignment_cfg.get("save_mapping", True):
                    # Persist mapping for reproducibility and later inspection.
                    mapping_file = alignment_cfg.get("mapping_file", "alignment_mapping.npz")
                    mapping_path = Path(os.path.join(experiment_root, str(test_fold), mapping_file))
                    np.savez(
                        mapping_path,
                        weights=weights,
                        feature_columns=np.array(feature_columns, dtype=object),
                        x_mean=x_mean_fit,
                        x_std=x_std_fit,
                        y_mean=y_mean_fit,
                        y_std=y_std_fit,
                        fit_intercept=np.array([int(fit_intercept)], dtype=int),
                        target_mode=np.array([target_mode], dtype=object),
                        target_columns=np.array(target_columns, dtype=object),
                    )
                    print(f"Saved alignment mapping to {mapping_path}")


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(_config: DictConfig) -> None:
    run(_config)

if __name__ == "__main__":
    main()
