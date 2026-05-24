import math
from typing import Dict, Iterable, Tuple

import numpy as np
from graphical_models import DAG


SUPPORTED_TOPOLOGIES = ("chain", "fork_collider", "layered_sparse")
SUPPORTED_NOISE_DISTS = ("gaussian", "laplace", "student_t_df3")


def validate_observed_layout(num_nodes: int, input_size: int, output_size: int) -> None:
    if num_nodes < 6:
        raise ValueError("Templated SEM datasets require num_nodes >= 6.")
    if output_size != 3:
        raise ValueError("Templated SEM datasets require output_size == 3.")
    if input_size != num_nodes - output_size:
        raise ValueError("Templated SEM datasets require input_size == num_nodes - output_size.")


def build_templated_dag(num_nodes: int, topology: str) -> DAG:
    if topology not in SUPPORTED_TOPOLOGIES:
        raise ValueError(f"Unsupported topology: {topology}")

    dag = DAG(set(range(num_nodes)))
    edges = _topology_edges(num_nodes, topology)
    for parent, child in edges:
        dag.add_edges({int(parent)}, int(child))
    return dag


def _topology_edges(num_nodes: int, topology: str) -> Tuple[Tuple[int, int], ...]:
    if topology == "chain":
        return tuple((idx, idx + 1) for idx in range(num_nodes - 1))

    if topology == "fork_collider":
        edges = []
        for idx in range(0, num_nodes - 2, 2):
            edges.append((idx, idx + 2))
            edges.append((idx + 1, idx + 2))
            if idx + 3 < num_nodes:
                edges.append((idx + 2, idx + 3))
        return tuple(edges)

    early_nodes, middle_nodes, output_nodes = layered_sparse_partitions(num_nodes)
    edges = []
    for mid_idx, node in enumerate(middle_nodes):
        edges.append((early_nodes[mid_idx % len(early_nodes)], node))
        if len(early_nodes) > 1 and mid_idx % 2 == 0:
            edges.append((early_nodes[(mid_idx + 1) % len(early_nodes)], node))
        if mid_idx > 0:
            edges.append((middle_nodes[mid_idx - 1], node))

    output_parent_specs = (
        (early_nodes[0], middle_nodes[0] if middle_nodes else early_nodes[min(1, len(early_nodes) - 1)]),
        (
            early_nodes[len(early_nodes) // 2],
            middle_nodes[len(middle_nodes) // 2] if middle_nodes else early_nodes[(len(early_nodes) // 2 + 1) % len(early_nodes)],
        ),
        (
            early_nodes[-1],
            middle_nodes[-1] if middle_nodes else early_nodes[max(0, len(early_nodes) - 2)],
        ),
    )
    for output_node, (early_parent, middle_parent) in zip(output_nodes, output_parent_specs):
        edges.append((early_parent, output_node))
        edges.append((middle_parent, output_node))
    return tuple(edges)


def layered_sparse_partitions(num_nodes: int) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...]]:
    input_size = num_nodes - 3
    early_size = int(math.ceil(input_size / 2.0))
    early_nodes = tuple(range(early_size))
    middle_nodes = tuple(range(early_size, input_size))
    output_nodes = tuple(range(input_size, num_nodes))
    return early_nodes, middle_nodes, output_nodes


def sample_signed_edge_weights(
    dag: DAG,
    rng: np.random.Generator,
    min_abs_weight: float,
    max_abs_weight: float,
) -> Dict[Tuple[int, int], float]:
    edge_weights: Dict[Tuple[int, int], float] = {}
    for child in dag.find_topological_order():
        for parent in sorted(dag.parents(child)):
            sign = -1.0 if int(rng.integers(0, 2)) == 0 else 1.0
            magnitude = float(rng.uniform(min_abs_weight, max_abs_weight))
            edge_weights[(parent, child)] = sign * magnitude
    return edge_weights


def sample_unit_variance_noise(
    rng: np.random.Generator,
    distribution: str,
    shape: Tuple[int, int],
    noise_scale: float,
) -> np.ndarray:
    if distribution == "gaussian":
        base = rng.standard_normal(shape)
    elif distribution == "laplace":
        base = rng.laplace(0.0, 1.0 / math.sqrt(2.0), shape)
    elif distribution == "student_t_df3":
        base = rng.standard_t(3.0, size=shape) / math.sqrt(3.0)
    else:
        raise ValueError(f"Unsupported noise distribution: {distribution}")
    return np.asarray(base, dtype=float) * float(noise_scale)


def prepare_intervention_values(num_samples: int, intervention_value) -> np.ndarray:
    if np.isscalar(intervention_value):
        return np.full(num_samples, float(intervention_value), dtype=float)
    intervention_value = np.asarray(intervention_value, dtype=float)
    if intervention_value.shape[0] != num_samples:
        raise ValueError("Intervention value array must match num_samples")
    return intervention_value


def edge_list(dag: DAG) -> Tuple[Tuple[int, int], ...]:
    edges = []
    for child in sorted(dag.nodes_set):
        for parent in sorted(dag.parents(child)):
            edges.append((int(parent), int(child)))
    return tuple(sorted(edges))
