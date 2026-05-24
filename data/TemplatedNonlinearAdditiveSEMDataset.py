import math
from typing import Optional

import numpy as np

from data.InterventionDataset import InterventionDataset
from data.templated_sem_utils import (
    SUPPORTED_TOPOLOGIES,
    build_templated_dag,
    prepare_intervention_values,
    sample_signed_edge_weights,
    sample_unit_variance_noise,
    validate_observed_layout,
)
from data.utils import build_split_indices


_DATA_CACHE = {}


class TemplatedNonlinearAdditiveSEMDataset(InterventionDataset):
    def __init__(
        self,
        subset="train",
        seed=1,
        topology="layered_sparse",
        num_nodes=6,
        input_size=3,
        output_size=3,
        num_samples=10000,
        mechanism_family="mixed",
        min_edge_weight=0.3,
        max_edge_weight=0.9,
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
        if mechanism_family != "mixed":
            raise ValueError("TemplatedNonlinearAdditiveSEMDataset currently supports only mechanism_family='mixed'.")

        self.subset = subset
        self.seed = int(seed)
        self.topology = topology
        self.num_nodes = int(num_nodes)
        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.num_samples = int(num_samples)
        self.extract_samples = int(extract_samples)
        self.mechanism_family = mechanism_family
        self.min_edge_weight = float(min_edge_weight)
        self.max_edge_weight = float(max_edge_weight)
        self.noise_scale = float(noise_scale)

        cache_key = (
            self.seed,
            self.topology,
            self.num_nodes,
            self.input_size,
            self.output_size,
            self.num_samples,
            self.mechanism_family,
            self.min_edge_weight,
            self.max_edge_weight,
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

            dag_data = _sample_nonlinear_additive_sem_data(
                dag=dag,
                edge_weights=edge_weights,
                num_samples=total_num_samples,
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
        return _sample_nonlinear_additive_sem_data(
            dag=self.dag,
            edge_weights=self.edge_weights,
            num_samples=int(num_samples),
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


def _sample_nonlinear_additive_sem_data(
    dag,
    edge_weights,
    num_samples: int,
    noise_scale: float,
    seed: int,
    intervention: Optional[dict],
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    noise = sample_unit_variance_noise(
        rng=rng,
        distribution="gaussian",
        shape=(int(num_samples), len(dag.nodes_set)),
        noise_scale=float(noise_scale),
    )
    data = np.zeros_like(noise, dtype=float)
    intervention = intervention or {}

    for node in dag.find_topological_order():
        if node in intervention:
            data[:, node] = prepare_intervention_values(int(num_samples), intervention[node])
            continue

        parents = sorted(dag.parents(node))
        if not parents:
            data[:, node] = noise[:, node]
            continue

        accum = np.zeros(int(num_samples), dtype=float)
        for parent in parents:
            transformed = _apply_mixed_transform(values=data[:, parent], parent=parent, child=node)
            accum += edge_weights[(parent, node)] * transformed
        data[:, node] = (accum / math.sqrt(len(parents))) + noise[:, node]

    if not np.isfinite(data).all():
        raise ValueError("Nonlinear additive SEM produced non-finite values.")
    return np.asarray(data, dtype=float)


def _apply_mixed_transform(values: np.ndarray, parent: int, child: int) -> np.ndarray:
    transform_id = (int(parent) + int(child)) % 3
    if transform_id == 0:
        return values
    if transform_id == 1:
        return values + 0.15 * np.square(values)
    return np.tanh(values)
