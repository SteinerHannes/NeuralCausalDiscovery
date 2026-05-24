import random
import numpy as np
import torch
from experiment_utils.synthetic_graphs import (
    create_random_connected_dag,
    sample_data_from_dag,
    sample_edge_weights_from_dag,
)
from data.InterventionDataset import InterventionDataset
from data.utils import build_split_indices

_DATA_CACHE = {}

class LinearSEMDataset(InterventionDataset):
    def __init__(
        self,
        subset="train",
        seed=1,
        num_nodes=4,
        input_size=3,
        output_size=2,
        expected_neighborhood_size=2,
        num_dags_timeout=10,
        num_samples=10000,
        min_edge_weight=0.5,
        max_edge_weight=2.0,
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        extract_samples=1000,
        fold=0,
    ):
        super().__init__()
        if subset not in ["train", "val", "test", "extract", "all"]:
            raise ValueError(f"subset must be 'train', 'val', 'test', 'extract' or 'all', got {subset}")
        self.subset = subset

        self.input_size = input_size
        self.output_size = output_size
        self.min_edge_weight = min_edge_weight
        self.max_edge_weight = max_edge_weight
        self.num_samples = int(num_samples)
        self.extract_samples = int(extract_samples)

        cache_key = (
            seed,
            num_nodes,
            input_size,
            output_size,
            expected_neighborhood_size,
            num_dags_timeout,
            num_samples,
            min_edge_weight,
            max_edge_weight,
            float(train_ratio),
            float(val_ratio),
            float(test_ratio),
            int(extract_samples),
            fold,
        )

        if cache_key not in _DATA_CACHE:
            print(f"Key: {cache_key} - Generating new DAG and data...")
            print(f"Seed: {seed}, Fold: {fold}")
            random.seed(seed)
            dag = create_random_connected_dag(
                num_nodes=num_nodes,
                expected_neighborhood_size=expected_neighborhood_size,
                num_dags_timeout=num_dags_timeout,
            )
            if dag is None:
                raise ValueError(
                    "Failed to generate a connected DAG for "
                    f"seed={seed}, num_nodes={num_nodes}, "
                    f"expected_neighborhood_size={expected_neighborhood_size}, "
                    f"num_dags_timeout={num_dags_timeout}. "
                    "Try increasing num_dags_timeout or search for a valid seed "
                    "with data.linear_sem_benchmark.find_first_valid_linear_sem_seed(...)."
                )

            # Keep the linear SEM mechanism fixed for a benchmark seed and let the
            # fold index change only the observational realization.
            edge_weights = sample_edge_weights_from_dag(
                in_dag=dag,
                min_edge_weight=min_edge_weight,
                max_edge_weight=max_edge_weight,
                seed=int(seed),
            )
            data_seed = int(seed) + int(fold) * 1009
            total_num_samples = int(num_samples) + int(extract_samples)
            dag_data = sample_data_from_dag(
                in_dag=dag,
                num_samples=total_num_samples,
                min_edge_weight=min_edge_weight,
                max_edge_weight=max_edge_weight,
                edge_weights=edge_weights,
                seed=data_seed,
            )

            _DATA_CACHE[cache_key] = {
                "dag": dag,
                "dag_data": dag_data,
                "edge_weights": edge_weights,
                "splits": build_split_indices(
                    num_samples=num_samples,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    extract_samples=extract_samples,
                ),
            }

        entry = _DATA_CACHE[cache_key]
        self.dag = entry["dag"]
        self.dag_data = entry["dag_data"]
        self.edge_weights = entry.get("edge_weights")
        if subset == "all":
            self.indices = np.arange(len(self.dag_data))
        else:
            self.indices = entry["splits"][subset]

    def simulate_intervention(self, num_samples, cfg, intervention: dict, seed: int):
        if self.edge_weights is None:
            raise ValueError("SEM edge weights not found; LinearSEMDataset must store edge_weights.")
        noise_scale = cfg.intervention.noise_scale
        return sample_data_from_dag(
            in_dag=self.dag,
            num_samples=num_samples,
            min_edge_weight=self.min_edge_weight,
            max_edge_weight=self.max_edge_weight,
            edge_weights=self.edge_weights,
            noise_scale=noise_scale,
            intervention=intervention,
            seed=seed,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        data = self.dag_data[real_idx]
        x = data[0:-self.output_size]
        y = data[-self.output_size:]
        return x, y


if __name__ == "__main__":
    from plot_utils import draw_graph
    dag_data = LinearSEMDataset(subset="train", seed=1)
    draw_graph(dag_data.dag)
    print(dag_data.dag_data)
