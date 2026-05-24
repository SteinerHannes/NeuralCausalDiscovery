import hydra
import numpy as np
import torch
from omegaconf import DictConfig

STATS_FILE_NAME = "dataset_stats.pt"


def build_split_indices(
    num_samples: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    extract_samples: int = 1000,
) -> dict[str, np.ndarray]:
    num_samples = int(num_samples)
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    train_ratio = float(train_ratio)
    val_ratio = float(val_ratio)
    test_ratio = float(test_ratio)
    extract_samples = int(extract_samples or 0)

    ratios = (train_ratio, val_ratio, test_ratio)
    if any(ratio < 0.0 or ratio > 1.0 for ratio in ratios):
        raise ValueError("Split ratios must be in [0, 1].")
    if not np.isclose(sum(ratios), 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must be 1.0")
    if extract_samples < 0:
        raise ValueError("extract_samples must be >= 0")

    main_pool = num_samples
    n_extract = extract_samples
    total_samples = main_pool + n_extract

    n_train = int(round(train_ratio * main_pool))
    n_val = int(round(val_ratio * main_pool))
    n_train = min(n_train, main_pool)
    n_val = min(n_val, main_pool - n_train)
    n_test = main_pool - n_train - n_val

    indices = np.arange(total_samples)
    train_end = n_train
    val_end = train_end + n_val
    test_end = val_end + n_test
    return {
        "train": indices[:train_end],
        "val": indices[train_end:val_end],
        "test": indices[val_end:test_end],
        "extract": indices[test_end:],
    }


class StandardizedDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, x_mean, x_std, y_mean, y_std):
        self.base_ds = base_ds
        self.input_size = base_ds.input_size
        self.output_size = base_ds.output_size
        self.x_mean = x_mean
        self.x_std = x_std
        self.y_mean = y_mean
        self.y_std = y_std

    def __len__(self):
        return len(self.base_ds)

    def __getitem__(self, idx):
        x, y = self.base_ds[idx]
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)

        x = (x - self.x_mean) / self.x_std
        y = (y - self.y_mean) / self.y_std
        return x, y

class ConcatenatedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        self.cumulative_sizes = np.cumsum([len(ds) for ds in datasets])

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        dataset_idx = np.searchsorted(self.cumulative_sizes, idx, side='right')
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

def compute_dataset_stats(ds: torch.utils.data.Dataset):
    # streaming mean/var for x and y
    n = 0
    x_mean = None
    x_M2 = None
    y_mean = None
    y_M2 = None

    for i in range(len(ds)):
        x, y = ds[i]
        x = torch.as_tensor(x, dtype=torch.float32)
        y = torch.as_tensor(y, dtype=torch.float32)

        n += 1
        if x_mean is None:
            x_mean = torch.zeros_like(x)
            x_M2 = torch.zeros_like(x)
            y_mean = torch.zeros_like(y)
            y_M2 = torch.zeros_like(y)

        # x
        dx = x - x_mean
        x_mean = x_mean + dx / n
        x_M2 = x_M2 + dx * (x - x_mean)

        # y
        dy = y - y_mean
        y_mean = y_mean + dy / n
        y_M2 = y_M2 + dy * (y - y_mean)

    if n < 2:
        x_var = torch.ones_like(x_mean)
        y_var = torch.ones_like(y_mean)
    else:
        x_var = x_M2 / (n - 1)
        y_var = y_M2 / (n - 1)

    x_std = torch.sqrt(torch.clamp(x_var, min=1e-8))
    y_std = torch.sqrt(torch.clamp(y_var, min=1e-8))

    return x_mean, x_std, y_mean, y_std

def get_global_stats(config: DictConfig, use_folds=True, subset="train") -> np.ndarray:
    if use_folds:
        res = []
        for test_fold in config.training.test_folds:
            ds = hydra.utils.instantiate(
                config.data,
                subset=subset,
                fold=test_fold
            )
            x_mean, x_std, y_mean, y_std = compute_dataset_stats(ds)
            res.append((x_mean, x_std, y_mean, y_std))

        return res
    else:
        full_ds = []
        for test_fold in config.training.test_folds:
            ds = hydra.utils.instantiate(
                config.data,
                subset=subset,
                fold=test_fold
            )
            full_ds.append(ds)

        full_ds = ConcatenatedDataset(full_ds)
        x_mean, x_std, y_mean, y_std = compute_dataset_stats(full_ds)
        return [(x_mean, x_std, y_mean, y_std)]


def save_dataset_stats(path: str, x_mean, x_std, y_mean, y_std) -> None:
    torch.save(
        {
            "x_mean": torch.as_tensor(x_mean).detach().cpu(),
            "x_std": torch.as_tensor(x_std).detach().cpu(),
            "y_mean": torch.as_tensor(y_mean).detach().cpu(),
            "y_std": torch.as_tensor(y_std).detach().cpu(),
        },
        path,
    )


def load_dataset_stats(path: str):
    try:
        payload = torch.load(path, map_location="cpu")
    except FileNotFoundError:
        return None

    required = ("x_mean", "x_std", "y_mean", "y_std")
    if not all(key in payload for key in required):
        raise ValueError(f"Invalid dataset stats file at {path}: missing required keys.")
    return (
        torch.as_tensor(payload["x_mean"], dtype=torch.float32),
        torch.as_tensor(payload["x_std"], dtype=torch.float32),
        torch.as_tensor(payload["y_mean"], dtype=torch.float32),
        torch.as_tensor(payload["y_std"], dtype=torch.float32),
    )
