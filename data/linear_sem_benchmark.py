from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from experiment_utils.synthetic_graphs import create_random_connected_dag


@dataclass(frozen=True)
class OutputCoverage:
    output_node: int
    ancestor_inputs: Tuple[int, ...]
    non_ancestor_inputs: Tuple[int, ...]

    @property
    def is_valid(self) -> bool:
        return bool(self.ancestor_inputs) and bool(self.non_ancestor_inputs)

    def to_dict(self) -> dict:
        return {
            "output_node": int(self.output_node),
            "ancestor_inputs": list(self.ancestor_inputs),
            "non_ancestor_inputs": list(self.non_ancestor_inputs),
            "is_valid": bool(self.is_valid),
        }


@dataclass(frozen=True)
class InputCoverage:
    input_node: int
    child_nodes: Tuple[int, ...]
    descendant_outputs: Tuple[int, ...]

    @property
    def is_valid(self) -> bool:
        return bool(self.descendant_outputs)

    def to_dict(self) -> dict:
        return {
            "input_node": int(self.input_node),
            "child_nodes": list(self.child_nodes),
            "descendant_outputs": list(self.descendant_outputs),
            "is_valid": bool(self.is_valid),
        }


@dataclass(frozen=True)
class BenchmarkEvaluation:
    num_nodes: int
    input_size: int
    output_size: int
    input_nodes: Tuple[int, ...]
    output_nodes: Tuple[int, ...]
    inputs: Tuple[InputCoverage, ...]
    outputs: Tuple[OutputCoverage, ...]

    @property
    def is_valid(self) -> bool:
        return all(input_.is_valid for input_ in self.inputs) and all(output.is_valid for output in self.outputs)

    def to_dict(self) -> dict:
        return {
            "num_nodes": int(self.num_nodes),
            "input_size": int(self.input_size),
            "output_size": int(self.output_size),
            "input_nodes": list(self.input_nodes),
            "output_nodes": list(self.output_nodes),
            "inputs": [input_.to_dict() for input_ in self.inputs],
            "outputs": [output.to_dict() for output in self.outputs],
            "is_valid": bool(self.is_valid),
        }


@dataclass(frozen=True)
class SeedSearchResult:
    seed: int
    edge_list: Tuple[Tuple[int, int], ...]
    benchmark: BenchmarkEvaluation

    def to_dict(self) -> dict:
        return {
            "seed": int(self.seed),
            "edge_list": [[int(parent), int(child)] for parent, child in self.edge_list],
            "benchmark": self.benchmark.to_dict(),
        }


def _validate_linear_sem_shape(num_nodes: int, input_size: int, output_size: int) -> None:
    if int(num_nodes) <= 1:
        raise ValueError("num_nodes must be greater than 1.")
    if int(input_size) <= 0:
        raise ValueError("input_size must be positive.")
    if int(output_size) <= 0:
        raise ValueError("output_size must be positive.")
    if int(input_size) + int(output_size) != int(num_nodes):
        raise ValueError("Linear SEM benchmark expects input_size + output_size == num_nodes.")


def linear_sem_input_nodes(num_nodes: int, input_size: int, output_size: int) -> Tuple[int, ...]:
    _validate_linear_sem_shape(num_nodes=num_nodes, input_size=input_size, output_size=output_size)
    del num_nodes, output_size
    return tuple(range(int(input_size)))


def linear_sem_output_nodes(num_nodes: int, input_size: int, output_size: int) -> Tuple[int, ...]:
    _validate_linear_sem_shape(num_nodes=num_nodes, input_size=input_size, output_size=output_size)
    start = int(num_nodes) - int(output_size)
    return tuple(range(start, int(num_nodes)))


def dag_edge_list(dag) -> Tuple[Tuple[int, int], ...]:
    edges = []
    for child in sorted(dag.nodes_set):
        for parent in sorted(dag.parents(child)):
            edges.append((int(parent), int(child)))
    return tuple(edges)


def _dag_descendants(dag, node: int) -> Tuple[int, ...]:
    descendants = set()
    frontier = list(dag.find_children(node))
    while frontier:
        child = int(frontier.pop())
        if child in descendants:
            continue
        descendants.add(child)
        frontier.extend(dag.find_children(child))
    return tuple(sorted(descendants))


def evaluate_linear_sem_benchmark_dag(dag, num_nodes: int, input_size: int, output_size: int) -> BenchmarkEvaluation:
    input_nodes = linear_sem_input_nodes(num_nodes=num_nodes, input_size=input_size, output_size=output_size)
    output_nodes = linear_sem_output_nodes(num_nodes=num_nodes, input_size=input_size, output_size=output_size)

    inputs = []
    for input_node in input_nodes:
        child_nodes = tuple(sorted(int(node) for node in dag.find_children(input_node)))
        descendants = set(_dag_descendants(dag, input_node))
        descendant_outputs = tuple(node for node in output_nodes if node in descendants)
        inputs.append(
            InputCoverage(
                input_node=int(input_node),
                child_nodes=child_nodes,
                descendant_outputs=descendant_outputs,
            )
        )

    outputs = []
    for output_node in output_nodes:
        ancestors = set(dag.get_ancestors(output_node))
        ancestors.discard(output_node)
        ancestor_inputs = tuple(node for node in input_nodes if node in ancestors)
        non_ancestor_inputs = tuple(node for node in input_nodes if node not in ancestors)
        outputs.append(
            OutputCoverage(
                output_node=int(output_node),
                ancestor_inputs=ancestor_inputs,
                non_ancestor_inputs=non_ancestor_inputs,
            )
        )

    return BenchmarkEvaluation(
        num_nodes=int(num_nodes),
        input_size=int(input_size),
        output_size=int(output_size),
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        inputs=tuple(inputs),
        outputs=tuple(outputs),
    )


def is_valid_linear_sem_benchmark_dag(dag, num_nodes: int, input_size: int, output_size: int) -> bool:
    return evaluate_linear_sem_benchmark_dag(
        dag=dag,
        num_nodes=num_nodes,
        input_size=input_size,
        output_size=output_size,
    ).is_valid


def generate_linear_sem_dag(
    seed: int,
    num_nodes: int,
    expected_neighborhood_size: int,
    num_dags_timeout: int,
):
    random.seed(int(seed))
    return create_random_connected_dag(
        num_nodes=int(num_nodes),
        expected_neighborhood_size=int(expected_neighborhood_size),
        num_dags_timeout=int(num_dags_timeout),
    )


def find_first_valid_linear_sem_seed(
    num_nodes: int,
    input_size: int,
    output_size: int,
    expected_neighborhood_size: int,
    num_dags_timeout: int,
    seed_start: int,
    seed_end: int,
    dag_factory: Optional[Callable[[int], object]] = None,
) -> Optional[SeedSearchResult]:
    _validate_linear_sem_shape(num_nodes=num_nodes, input_size=input_size, output_size=output_size)
    if int(seed_end) < int(seed_start):
        raise ValueError("seed_end must be greater than or equal to seed_start.")

    for seed in range(int(seed_start), int(seed_end) + 1):
        dag = (
            dag_factory(seed)
            if dag_factory is not None
            else generate_linear_sem_dag(
                seed=seed,
                num_nodes=num_nodes,
                expected_neighborhood_size=expected_neighborhood_size,
                num_dags_timeout=num_dags_timeout,
            )
        )
        if dag is None:
            continue

        benchmark = evaluate_linear_sem_benchmark_dag(
            dag=dag,
            num_nodes=num_nodes,
            input_size=input_size,
            output_size=output_size,
        )
        if benchmark.is_valid:
            return SeedSearchResult(
                seed=int(seed),
                edge_list=dag_edge_list(dag),
                benchmark=benchmark,
            )

    return None
