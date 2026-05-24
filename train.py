import os
import sys
import json
import math
import random
import time
import copy
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import pandas as pd
import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from data.utils import (
    STATS_FILE_NAME,
    StandardizedDataset,
    compute_dataset_stats,
    save_dataset_stats,
)
from experiment_utils.device_utils import print_torch_runtime, resolve_torch_device
from experiment_utils.logging_utils import fold_log_context

float_fmt = ".3f"
data_transfer_non_blocking = False

def get_run_path() -> str:
    #return hydra.utils.get_original_cwd()
    return Path(os.getcwd())


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _target_leaf(target: Optional[str]) -> Optional[str]:
    if not target:
        return None
    return str(target).split(".")[-1]


def _cfg_to_plain_dict(cfg: Optional[DictConfig]) -> Optional[dict]:
    if cfg is None:
        return None
    return OmegaConf.to_container(cfg, resolve=True)


def _compute_r2(res_sum_sq: float, y_sum: float, y_sq_sum: float, n_values: int) -> float:
    if n_values <= 0:
        return float("nan")
    ss_tot = y_sq_sum - ((y_sum * y_sum) / float(n_values))
    if np.isclose(ss_tot, 0.0):
        return 0.0
    return float(1.0 - (res_sum_sq / ss_tot))


def _loader_runtime_kwargs(config: DictConfig) -> dict:
    num_workers = int(config.training.get("num_workers", 0) or 0)
    persistent_workers = bool(config.training.get("persistent_workers", False)) and num_workers > 0
    prefetch_factor = config.training.get("prefetch_factor", None)
    kwargs = {
        "num_workers": num_workers,
        "persistent_workers": persistent_workers,
    }
    if num_workers > 0 and prefetch_factor is not None:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return kwargs


def _apply_known_overrides(base_cfg: Optional[DictConfig], overrides: Optional[dict]) -> Tuple[Optional[DictConfig], list]:
    if base_cfg is None or not overrides:
        return base_cfg, []

    skipped = []
    for key, value in overrides.items():
        if key in base_cfg:
            base_cfg[key] = value
        else:
            skipped.append(str(key))
    return base_cfg, skipped


def _resolve_training_runtime(config: DictConfig) -> dict:
    training_cfg = config.training
    model_name = _target_leaf(config.model.get("_target_")) or "UnknownModel"
    dataset_name = _target_leaf(config.data.get("_target_")) or "UnknownDataset"

    control_mode = str(getattr(training_cfg, "control_mode", "epochs"))
    configured_epochs = int(getattr(training_cfg, "epochs", 1) or 1)
    target_updates = getattr(training_cfg, "target_updates", None)

    profile_name = None
    profile_cfg = {}
    profile_selection = str(getattr(training_cfg, "profile_selection", "none") or "none")
    profiles_cfg = training_cfg.get("profiles")
    if profiles_cfg is not None and profile_selection != "none":
        if profile_selection == "auto":
            profile_by_model = training_cfg.get("profile_by_model", {})
            profile_name = profile_by_model.get(model_name)
            if not profile_name:
                profile_name = training_cfg.get("default_profile")
        else:
            profile_name = profile_selection

        if profile_name and profile_name in profiles_cfg:
            profile_cfg = OmegaConf.to_container(profiles_cfg[profile_name], resolve=True)
            control_mode = str(getattr(training_cfg, "profile_control_mode", control_mode))
            configured_epochs = int(profile_cfg.get("max_epochs", configured_epochs) or configured_epochs)
        else:
            profile_name = None
            profile_cfg = {}

    early_stopping_cfg = OmegaConf.to_container(training_cfg.get("early_stopping", {}), resolve=True)
    if profile_cfg.get("early_stopping"):
        early_stopping_cfg.update(profile_cfg.get("early_stopping", {}))

    optimizer_cfg = None
    scheduler_cfg = None
    optimizer_skipped = []
    scheduler_skipped = []
    if config.get("optimizer"):
        optimizer_cfg = OmegaConf.create(OmegaConf.to_container(config.optimizer, resolve=True))
    if config.get("scheduler"):
        scheduler_cfg = OmegaConf.create(OmegaConf.to_container(config.scheduler, resolve=True))

    if profile_cfg.get("optimizer"):
        optimizer_cfg, optimizer_skipped = _apply_known_overrides(optimizer_cfg, profile_cfg.get("optimizer"))
    if profile_cfg.get("scheduler"):
        scheduler_cfg, scheduler_skipped = _apply_known_overrides(scheduler_cfg, profile_cfg.get("scheduler"))

    return {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "profile_name": profile_name,
        "control_mode": control_mode,
        "configured_epochs": configured_epochs,
        "target_updates": None if target_updates is None else int(target_updates),
        "early_stopping": {
            "enabled": bool(early_stopping_cfg.get("enabled", False)),
            "min_epochs": int(early_stopping_cfg.get("min_epochs", 1) or 1),
            "patience": int(early_stopping_cfg.get("patience", 1) or 1),
            "min_delta": float(early_stopping_cfg.get("min_delta", 0.0) or 0.0),
            "restore_best_weights": bool(early_stopping_cfg.get("restore_best_weights", True)),
        },
        "optimizer_cfg": optimizer_cfg,
        "scheduler_cfg": scheduler_cfg,
        "optimizer_override_skipped_keys": optimizer_skipped,
        "scheduler_override_skipped_keys": scheduler_skipped,
    }


def train_epoch() -> Tuple[dict, list]:
    model.train()

    non_blocking = device.type == "cuda" and data_transfer_non_blocking
    loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    abs_error_sum = torch.zeros((), dtype=torch.float32, device=device)
    residual_sum_sq = torch.zeros((), dtype=torch.float32, device=device)
    y_sum = torch.zeros((), dtype=torch.float32, device=device)
    y_sq_sum = torch.zeros((), dtype=torch.float32, device=device)
    n_batches = 0
    n_values = 0
    for x, y in tqdm(
        train_loader,
        unit='bat',
        disable=config.training.debug.disable_bat_pbar,
        position=0,
        file=sys.stdout
    ):
        x = x.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
        y_true = y.to(device=device, dtype=torch.float32, non_blocking=non_blocking)

        y_pred = model(x)

        loss = criterion(y_pred, y_true)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if optimizer is not None:
            optimizer.step()

        with torch.no_grad():
            residual = (y_pred.detach() - y_true.detach()).to(dtype=torch.float32)
            y_detached = y_true.detach().to(dtype=torch.float32)
            loss_sum += loss.detach().to(dtype=torch.float32)
            abs_error_sum += torch.abs(residual).sum()
            residual_sum_sq += torch.square(residual).sum()
            y_sum += y_detached.sum()
            y_sq_sum += torch.square(y_detached).sum()
            n_batches += 1
            n_values += int(y_detached.numel())

    mean_loss = float((loss_sum / max(1, n_batches)).cpu().item()) if n_batches > 0 else float("nan")
    abs_error_sum_f = float(abs_error_sum.cpu().item())
    residual_sum_sq_f = float(residual_sum_sq.cpu().item())
    y_sum_f = float(y_sum.cpu().item())
    y_sq_sum_f = float(y_sq_sum.cpu().item())
    mae = float(abs_error_sum_f / n_values) if n_values > 0 else float("nan")
    r2 = _compute_r2(residual_sum_sq_f, y_sum_f, y_sq_sum_f, n_values)
    return {
        "loss": mean_loss,
        "mae": mae,
        "r2": r2,
        "n_values": int(n_values),
    }, []


def fit_regressor(training_runtime: dict, num_parameters: int) -> Tuple[list, dict]:
    batches_per_epoch = len(train_loader)
    control_mode = training_runtime["control_mode"]
    configured_epochs = int(training_runtime["configured_epochs"])
    target_updates = training_runtime["target_updates"]
    early_cfg = training_runtime["early_stopping"]

    if control_mode == "updates":
        if target_updates is not None and target_updates > 0:
            num_epochs = max(1, int(math.ceil(target_updates / max(1, batches_per_epoch))))
        else:
            num_epochs = max(1, configured_epochs)
    elif control_mode == "epochs":
        num_epochs = max(1, configured_epochs)
    else:
        raise ValueError(f"Unknown training.control_mode: {control_mode}")

    effective_epochs = int(num_epochs)
    effective_updates = int(effective_epochs * batches_per_epoch)

    training_summary = {
        "control_mode": control_mode,
        "configured_epochs": configured_epochs,
        "target_updates": None if target_updates is None else int(target_updates),
        "batch_size": int(config.training.batch_size),
        "batches_per_epoch": int(batches_per_epoch),
        "effective_epochs": effective_epochs,
        "effective_updates": effective_updates,
        "planned_epochs": effective_epochs,
        "planned_updates": effective_updates,
        "model_name": training_runtime["model_name"],
        "dataset_name": training_runtime["dataset_name"],
        "profile_name": training_runtime["profile_name"],
        "num_parameters": int(num_parameters),
        "early_stopping": dict(early_cfg),
        "optimizer_settings": _cfg_to_plain_dict(training_runtime["optimizer_cfg"]),
        "scheduler_settings": _cfg_to_plain_dict(training_runtime["scheduler_cfg"]),
        "optimizer_override_skipped_keys": list(training_runtime["optimizer_override_skipped_keys"]),
        "scheduler_override_skipped_keys": list(training_runtime["scheduler_override_skipped_keys"]),
    }

    pbar = tqdm(range(1, 1 + num_epochs), ncols=50, unit='ep', file=sys.stdout, ascii=True)
    epoch_rows = []
    cumulative_updates = 0
    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = None
    epochs_without_improvement = 0
    early_stopping_triggered = False
    stop_reason = "target_updates_budget" if control_mode == "updates" else "max_epochs"
    min_epochs = max(1, int(early_cfg.get("min_epochs", 1)))
    patience = max(1, int(early_cfg.get("patience", 1)))
    min_delta = float(early_cfg.get("min_delta", 0.0))
    early_enabled = bool(early_cfg.get("enabled", False))
    restore_best_weights = bool(early_cfg.get("restore_best_weights", True))

    for epoch in pbar:
        epoch_start_time = time.perf_counter()
        train_metrics, train_loss_list = train_epoch()

        val_metrics, val_loss_list = evaluate_regressor(
            config=config,
            model=model,
            dataloader=val_loader,
            device=device,
            criterion=criterion
        )

        pbar.write(
            f"TrnLoss={train_metrics['loss']:{float_fmt}} "
            f"ValLoss={val_metrics['loss']:{float_fmt}} "
            f"ValR2={val_metrics['r2']:{float_fmt}}"
        )

        if scheduler is not None:
            scheduler.step()

        lr = None
        if optimizer is not None and len(optimizer.param_groups) > 0:
            lr = optimizer.param_groups[0].get("lr")
        cumulative_updates += int(batches_per_epoch)
        epoch_seconds = float(time.perf_counter() - epoch_start_time)
        epoch_rows.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "val_mae": val_metrics["mae"],
            "train_r2": train_metrics["r2"],
            "val_r2": val_metrics["r2"],
            "lr": lr,
            "epoch_seconds": epoch_seconds,
            "cumulative_updates": cumulative_updates,
            "batches_per_epoch": int(batches_per_epoch),
            "effective_epochs": effective_epochs,
            "effective_updates": effective_updates
        })

        current_val_loss = float(val_metrics["loss"])
        if current_val_loss < (best_val_loss - min_delta):
            best_val_loss = current_val_loss
            best_epoch = int(epoch)
            epochs_without_improvement = 0
            if restore_best_weights:
                best_state_dict = copy.deepcopy(model.state_dict())
        elif early_enabled and epoch >= min_epochs:
            epochs_without_improvement += 1

        if early_enabled and epoch >= min_epochs and epochs_without_improvement >= patience:
            early_stopping_triggered = True
            stop_reason = "early_stopping"
            pbar.write(
                f"Early stopping at epoch {epoch}: "
                f"no validation improvement for {patience} epochs "
                f"(best epoch={best_epoch}, best val loss={best_val_loss:{float_fmt}})"
            )
            break

    if restore_best_weights and best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    actual_epochs = len(epoch_rows)
    actual_updates = int(actual_epochs * batches_per_epoch)
    training_summary["actual_epochs"] = int(actual_epochs)
    training_summary["actual_updates"] = int(actual_updates)
    training_summary["best_epoch"] = int(best_epoch) if best_epoch > 0 else None
    training_summary["best_val_loss"] = float(best_val_loss) if best_epoch > 0 else None
    training_summary["stop_reason"] = stop_reason
    training_summary["early_stopping_triggered"] = bool(early_stopping_triggered)

    torch.save(model.state_dict(), os.path.join(current_run_path, config.torch_model_name))
    return epoch_rows, training_summary


def evaluate_regressor(
        config: DictConfig,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device: torch.device) -> Tuple[dict, list]:

    model.eval()
    non_blocking = device.type == "cuda"
    loss_sum = torch.zeros((), dtype=torch.float32, device=device)
    abs_error_sum = torch.zeros((), dtype=torch.float32, device=device)
    residual_sum_sq = torch.zeros((), dtype=torch.float32, device=device)
    y_sum = torch.zeros((), dtype=torch.float32, device=device)
    y_sq_sum = torch.zeros((), dtype=torch.float32, device=device)
    n_batches = 0
    n_values = 0
    with torch.inference_mode():
        for x, y in tqdm(
            dataloader,
            unit="bat",
            disable=config.training.debug.disable_bat_pbar,
            position=0,
            file=sys.stdout
        ):
            x = x.to(device=device, dtype=torch.float32, non_blocking=non_blocking)
            y_true = y.to(device=device, dtype=torch.float32, non_blocking=non_blocking)

            y_pred = model(x)

            loss = criterion(y_pred, y_true)
            residual = (y_pred - y_true).to(dtype=torch.float32)
            loss_sum += loss.to(dtype=torch.float32)
            abs_error_sum += torch.abs(residual).sum()
            residual_sum_sq += torch.square(residual).sum()
            y_sum += y_true.sum()
            y_sq_sum += torch.square(y_true).sum()
            n_batches += 1
            n_values += int(y_true.numel())

    mean_loss = float((loss_sum / max(1, n_batches)).cpu().item()) if n_batches > 0 else float("nan")
    abs_error_sum_f = float(abs_error_sum.cpu().item())
    residual_sum_sq_f = float(residual_sum_sq.cpu().item())
    y_sum_f = float(y_sum.cpu().item())
    y_sq_sum_f = float(y_sq_sum.cpu().item())
    mae = float(abs_error_sum_f / n_values) if n_values > 0 else float("nan")
    r2 = _compute_r2(residual_sum_sq_f, y_sum_f, y_sq_sum_f, n_values)
    return {
        "loss": mean_loss,
        "mae": mae,
        "r2": r2,
        "n_values": int(n_values),
    }, []


def test_regressor(
        config: DictConfig,
        model: torch.nn.Module,
        dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        device: torch.device) -> Tuple[float, list]:

    metrics, losses = evaluate_regressor(
        config=config,
        model=model,
        dataloader=dataloader,
        criterion=criterion,
        device=device,
    )
    return float(metrics["loss"]), losses

def run(_config: DictConfig) -> None:
    global config
    config = _config
    pd.options.display.float_format = ('{:,' + float_fmt + '}').format

    global model, train_loader, val_loader, test_loader, optimizer, scheduler, criterion, current_run_path, device
    global data_transfer_non_blocking
    requested_device = config.training.get("device", "auto")
    device = resolve_torch_device(requested_device)
    # due to MPS limitations
    torch.set_default_dtype(torch.float32)
    pin_memory = device.type == "cuda"
    data_transfer_non_blocking = pin_memory
    if bool(config.training.get("print_device_info", True)):
        print_torch_runtime(device)
    loader_runtime_kwargs = _loader_runtime_kwargs(config)

    root_run_path = get_run_path()
    print(f"Output directory: {root_run_path}")

    scores = {}
    folds = list(config.training.test_folds)
    total_folds = len(folds)
    for fold_idx, test_fold in enumerate(folds, start=1):
        current_run_path = os.path.join(root_run_path, f"{test_fold}")
        os.makedirs(current_run_path, exist_ok=True)

        with fold_log_context(
            root_run_path,
            test_fold,
            "train.log",
            step_name="Train",
            fold_idx=fold_idx,
            num_folds=total_folds,
        ):
            fold_seed = config.training.seed + test_fold
            print(f"Step: set random seed ({fold_seed})")
            set_global_seed(fold_seed)
            train_generator = torch.Generator()
            train_generator.manual_seed(fold_seed)

            print("Step: build datasets")
            train_ds = hydra.utils.instantiate(
                config.data,
                subset="train",
                fold=test_fold
            )
            val_ds = hydra.utils.instantiate(
                config.data,
                subset="val",
                fold=test_fold
            )

            # compute stats on train split only
            x_mean, x_std, y_mean, y_std = compute_dataset_stats(train_ds)
            save_dataset_stats(
                os.path.join(current_run_path, STATS_FILE_NAME),
                x_mean,
                x_std,
                y_mean,
                y_std,
            )

            # wrap datasets so ALL data classes get standardized here
            train_ds = StandardizedDataset(train_ds, x_mean, x_std, y_mean, y_std)
            val_ds = StandardizedDataset(val_ds, x_mean, x_std, y_mean, y_std)

            train_loader = torch.utils.data.DataLoader(
                train_ds,
                batch_size=config.training.batch_size,
                shuffle=True,
                generator=train_generator,
                pin_memory=pin_memory,
                drop_last=False,
                **loader_runtime_kwargs,
            )

            val_loader = torch.utils.data.DataLoader(
                val_ds,
                batch_size=config.training.batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                drop_last=False,
                **loader_runtime_kwargs,
            )

            print("Step: initialize model")
            model = hydra.utils.instantiate(
                config.model,
                input_size=train_loader.dataset.input_size,
                output_size=train_loader.dataset.output_size
            )
            model = model.to(device)

            criterion = hydra.utils.instantiate(config.get('loss', {
                '_target_': 'torch.nn.MSELoss'
            }))
            criterion = criterion.to(device)

            training_runtime = _resolve_training_runtime(config)

            if training_runtime["optimizer_cfg"] is not None:
                optimizer = hydra.utils.instantiate(
                    training_runtime["optimizer_cfg"],
                    params=model.parameters()
                )
            else:
                optimizer = None

            if training_runtime["scheduler_cfg"] is not None and optimizer is not None:
                scheduler = hydra.utils.instantiate(
                    training_runtime["scheduler_cfg"],
                    optimizer=optimizer
                )
            else:
                scheduler = None

            print(
                "Step: training setup "
                f"(model={training_runtime['model_name']}, "
                f"profile={training_runtime['profile_name']}, "
                f"control_mode={training_runtime['control_mode']}, "
                f"epochs={training_runtime['configured_epochs']}, "
                f"target_updates={training_runtime['target_updates']})"
            )
            print("Step: training")
            num_parameters = int(sum(param.numel() for param in model.parameters()))
            epoch_rows, training_summary = fit_regressor(training_runtime=training_runtime, num_parameters=num_parameters)
            training_summary["seed"] = int(fold_seed)
            history_df = pd.DataFrame(epoch_rows)
            history_df.to_csv(
                os.path.join(current_run_path, "training_history.csv"),
                index=False
            )
            history_df.reindex(columns=[
                "epoch",
                "train_loss",
                "val_loss",
                "lr",
                "batches_per_epoch",
                "effective_epochs",
                "effective_updates",
            ]).to_csv(
                os.path.join(current_run_path, "train_losses.csv"),
                index=False
            )
            with open(os.path.join(current_run_path, "training_summary.json"), "w", encoding="utf-8") as f:
                json.dump(training_summary, f, indent=2)

            print("Step: evaluate test split")
            test_ds = hydra.utils.instantiate(
                config.data,
                subset="test",
                fold=test_fold
            )
            test_ds = StandardizedDataset(test_ds, x_mean, x_std, y_mean, y_std)

            test_loader = torch.utils.data.DataLoader(
                test_ds,
                batch_size=config.training.batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                drop_last=False,
                **loader_runtime_kwargs,
            )

            test_loss_mean, test_loss_list = test_regressor(
                config=config,
                model=model,
                dataloader=test_loader,
                criterion=criterion,
                device=device
            )

            scores[test_fold] = pd.Series(dict(TestLoss=test_loss_mean))

    scores = pd.concat(scores).unstack([-1])
    print(pd.concat((scores, scores.agg(['mean', 'std']))))
    scores.to_csv(os.path.join(root_run_path, "train_test_scores.csv"))


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(_config: DictConfig) -> None:
    global float_fmt
    float_fmt = ".3f"
    pd.options.display.float_format = ('{:,' + float_fmt + '}').format
    run(_config)

if __name__ == "__main__":
    main()
