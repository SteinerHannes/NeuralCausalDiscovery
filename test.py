import os
import hydra
import pandas as pd
import torch
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from data.utils import (
    STATS_FILE_NAME,
    StandardizedDataset,
    compute_dataset_stats,
    load_dataset_stats,
)
from experiment_utils.device_utils import print_torch_runtime, resolve_torch_device
from experiment_utils.logging_utils import fold_log_context, resolve_experiment_root
from train import test_regressor

float_fmt = ".3f"


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


def visualize_test_losses(test_losses: pd.Series, file_name: str, experiment_root: str, show: True) -> None:
    """
    Create and save a bar chart visualization of test losses per fold.

    Args:
        test_losses: Series containing test losses for each fold plus mean and std
        file_name: Name of the model file (without extension)
        experiment_root: Root directory where to save the visualization
        show: Show the visualization in a window
    """
    print(f'\nGenerating visualization for {file_name}...')

    # Extract numeric rows (exclude mean and std rows)
    fold_losses = test_losses.iloc[:-2]
    fold_ids = fold_losses.index.astype(int)
    losses = fold_losses.values

    # Get mean and std
    mean_loss = test_losses.loc['mean']
    std_loss = test_losses.loc['std']

    plt.figure(figsize=(10, 6))
    plt.bar(fold_ids, losses, color='steelblue', alpha=0.7, edgecolor='black')

    plt.axhline(mean_loss, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_loss:.3f}')
    plt.axhline(mean_loss + std_loss, color='orange', linestyle=':', linewidth=2, label='Mean ± Std')
    plt.axhline(mean_loss - std_loss, color='orange', linestyle=':', linewidth=2)

    plt.xlabel('Fold', fontsize=12)
    plt.ylabel('Test Loss', fontsize=12)
    plt.title(f'Test Loss per Fold', fontsize=14)
    plt.legend(loc='best')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    fig_path = os.path.join(experiment_root, f'test_loss_visualization_{file_name}.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f'Visualization saved to {fig_path}')
    if show:
        plt.show()
        plt.close()
    else:
        plt.close()


def run(_config: DictConfig) -> None:
    global config
    config = _config
    pd.options.display.float_format = ('{:,' + float_fmt + '}').format

    global model
    requested_device = config.training.get("device", "auto")
    device = resolve_torch_device(requested_device)
    # due to MPS limitations
    torch.set_default_dtype(torch.float32)
    pin_memory = device.type == "cuda"
    if bool(config.training.get("print_device_info", True)):
        print_torch_runtime(device)
    loader_runtime_kwargs = _loader_runtime_kwargs(config)

    experiment_root = resolve_experiment_root(config)
    fold_stats = _resolve_fold_stats(config, experiment_root)

    model = hydra.utils.instantiate(
        config.model,
        input_size=config.data.input_size,
        output_size=config.data.output_size,
    )
    model = model.to(device)

    criterion = hydra.utils.instantiate(config.get('loss', {
        '_target_': 'torch.nn.MSELoss'
    }))
    criterion = criterion.to(device)
    # Only one for now
    model_file_names = [config.torch_model_name]

    all_scores = []
    folds = list(config.training.test_folds)
    total_folds = len(folds)

    for fold_idx, test_fold in enumerate(folds, start=1):
        experiment = os.path.join(experiment_root, f'{test_fold}')
        with fold_log_context(
            experiment_root,
            test_fold,
            "test.log",
            step_name="Test",
            fold_idx=fold_idx,
            num_folds=total_folds,
        ):
            print("Step: prepare test split")
            x_mean, x_std, y_mean, y_std = fold_stats[int(test_fold)]
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

            for model_file_name in model_file_names:
                model_file = os.path.join(experiment, model_file_name)
                state_dict = torch.load(model_file, map_location=device)
                model.load_state_dict(state_dict)
                print('Testing', model_file)
                test_loss_mean, test_loss_list = test_regressor(
                    config=config,
                    model=model,
                    dataloader=test_loader,
                    device=device,
                    criterion=criterion
                )
                all_scores.append({
                    'fold': test_fold,
                    'model': model_file_name,
                    'metric': 'TestLoss',
                    'value': test_loss_mean
                })
                print(f"Test Loss: {test_loss_mean}")

    scores_df = pd.DataFrame(all_scores)
    scores = scores_df.pivot_table(
        index='fold',
        columns=['model', 'metric'],
        values='value'
    )

    stats = pd.concat([
        scores.mean().to_frame('mean').T,
        scores.std().to_frame('std').T
    ])
    scores = pd.concat([scores, stats])

    for model_file_name in model_file_names:
        file_name = os.path.splitext(model_file_name)[0]
        file_path = os.path.join(experiment_root, f'test_scores_{file_name}.csv')
        scores.to_csv(file_path)

        # Get test losses for current model
        test_losses = scores[(model_file_name, 'TestLoss')]
        visualize_test_losses(test_losses, file_name, experiment_root, show=False)

    print(scores)


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(_config: DictConfig) -> None:
    run(_config)


if __name__ == "__main__":
    float_fmt = ".3f"
    main()
