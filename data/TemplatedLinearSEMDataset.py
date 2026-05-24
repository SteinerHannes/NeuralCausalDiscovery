from typing import Optional

import numpy as np

from data.InterventionDataset import InterventionDataset
from data.templated_sem_utils import (
    SUPPORTED_NOISE_DISTS,
    SUPPORTED_TOPOLOGIES,
    build_templated_dag,
    sample_signed_edge_weights,
    sample_unit_variance_noise,
    prepare_intervention_values,
    validate_observed_layout,
)
from data.utils import build_split_indices


_DATA_CACHE = {}


class TemplatedLinearSEMDataset(InterventionDataset):
    def __init__(
        self,
        subset="train",
        seed=1,
        topology="layered_sparse",
        num_nodes=6,
        input_size=3,
        output_size=3,
        num_samples=10000,
        min_edge_weight=0.5,
        max_edge_weight=1.5,
        noise_distribution="gaussian",
        noise_scale=1.0,
        train_ratio=0.80,
        val_ratio=0.10,
        test_ratio=0.10,
        extract_samples=1000,
        fold=0,
    ):
        super().__init__()
        if subset not in ["train", "val", "test", "extract", "all"]:
            raise ValueError(f"subset must be 'train', 'val', 'test', 'extract' or 'all', got {subset}")
        validate_observed_layout(num_nodes=num_nodes, input_size=input_size, output_size=output_size)
        if topology not in SUPPORTED_TOPOLOGIES:
            raise ValueError(f"Unsupported topology: {topology}")
        if noise_distribution not in SUPPORTED_NOISE_DISTS:
            raise ValueError(f"Unsupported noise distribution: {noise_distribution}")

        self.subset = subset
        self.seed = int(seed)
        self.topology = topology
        self.num_nodes = int(num_nodes)
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.num_samples = int(num_samples)
        self.extract_samples = int(extract_samples)
        self.min_edge_weight = float(min_edge_weight)
        self.max_edge_weight = float(max_edge_weight)
        self.noise_distribution = str(noise_distribution)
        self.noise_scale = float(noise_scale)

        cache_key = (
            self.seed,
            self.topology,
            self.num_nodes,
            self.input_size,
            self.output_size,
            self.num_samples,
            self.min_edge_weight,
            self.max_edge_weight,
            self.noise_distribution,
            self.noise_scale,
            float(train_ratio),
            float(val_ratio),
            float(test_ratio),
            int(extract_samples),
            int(fold),
        )

        if cache_key not in _DATA_CACHE:
            dag = build_templated_dag(num_nodes=self.num_nodes, topology=self.topology)
            weight_rng = np.random.default_rng(self.seed)
            edge_weights = sample_signed_edge_weights(
                dag=dag,
                rng=weight_rng,
                min_abs_weight=self.min_edge_weight,
                max_abs_weight=self.max_edge_weight,
            )
            total_num_samples = self.num_samples + self.extract_samples

            dag_data = _sample_linear_sem_data(
                dag=dag,
                edge_weights=edge_weights,
                num_samples=total_num_samples,
                noise_distribution=self.noise_distribution,
                noise_scale=self.noise_scale,
                seed=self.seed + int(fold) * 1009,
                intervention=None,
            )

            _DATA_CACHE[cache_key] = {
                "dag": dag,
                "dag_data": dag_data,
                "edge_weights": edge_weights,
                "splits": build_split_indices(
                    num_samples=self.num_samples,
                    train_ratio=train_ratio,
                    val_ratio=val_ratio,
                    test_ratio=test_ratio,
                    extract_samples=extract_samples,
                ),
            }

        entry = _DATA_CACHE[cache_key]
        self.dag = entry["dag"]
        self.dag_data = entry["dag_data"]
        self.edge_weights = entry["edge_weights"]
        if subset == "all":
            self.indices = np.arange(len(self.dag_data))
        else:
            self.indices = entry["splits"][subset]

    def simulate_intervention(self, num_samples, cfg, intervention: dict, seed: int):
        return _sample_linear_sem_data(
            dag=self.dag,
            edge_weights=self.edge_weights,
            num_samples=int(num_samples),
            noise_distribution=self.noise_distribution,
            noise_scale=float(cfg.intervention.noise_scale),
            seed=int(seed),
            intervention=intervention or {},
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        data = self.dag_data[real_idx]
        x = data[:self.input_size]
        y = data[self.input_size:self.input_size + self.output_size]
        return x, y


def _sample_linear_sem_data(
    dag,
    edge_weights,
    num_samples: int,
    noise_distribution: str,
    noise_scale: float,
    seed: int,
    intervention: Optional[dict],
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    data = sample_unit_variance_noise(
        rng=rng,
        distribution=noise_distribution,
        shape=(int(num_samples), len(dag.nodes_set)),
        noise_scale=float(noise_scale),
    )
    intervention = intervention or {}
    for node in dag.find_topological_order():
        if node in intervention:
            data[:, node] = prepare_intervention_values(int(num_samples), intervention[node])
            continue
        for parent in dag.parents(node):
            data[:, node] += edge_weights[(parent, node)] * data[:, parent]
    return np.asarray(data, dtype=float)
