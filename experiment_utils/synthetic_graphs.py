import random
import numpy as np
from graphical_models import DAG


def create_random_dag(num_nodes, expected_neighborhood_size):
    nodes_set = set(range(num_nodes))
    dag = DAG(nodes_set)

    # The probability of the presence of a directed edge is Bernoulli(expected_neighborhood_size/(num_nodes-1))
    for node_parent in range(num_nodes - 1):
        for node_child in range(node_parent + 1, num_nodes):
            is_edge = random.random() < (expected_neighborhood_size / (num_nodes - 1))
            if is_edge:
                dag.add_edges({node_parent}, node_child)

    return dag


def create_random_dag_max_fan(num_nodes, max_fan_in):
    """
    Create a random DAG with bounded fan-in (number of parents per node)
    :param num_nodes: Number of nodes in the graph
    :param max_fan_in: Maximal number of parents per child
    :return: A random DAG with bounded fan-in
    """
    nodes_set = set(range(num_nodes))
    dag = DAG(nodes_set)  # create an empty graph

    for current_node_id in range(1, num_nodes):
        num_parents = random.randint(0, min(current_node_id, max_fan_in))  # sample number of parents
        if num_parents > 0:
            parents = set(random.sample(range(current_node_id), num_parents))  # sample parents
            dag.add_edges(parents_set=parents, target_node=current_node_id)

    return dag


def create_random_connected_dag(num_nodes, expected_neighborhood_size, num_dags_timeout=10000):
    is_conn = False
    acc = 0
    while not is_conn:
        if num_dags_timeout < acc:
            return None  # timed out: did not find a connected DAG

        dag_rand = create_random_dag(num_nodes, expected_neighborhood_size)
        is_conn = dag_rand.is_graph_connected()
        acc += 1

    return dag_rand


def select_latent_variables(graph: DAG):
    """
    Find nodes in a DAG that can serve as latent confounders. They comply with:
        1. They don't have incoming edges (parentless)
        2. Each of the is a parent of at least two nodes
    :param graph: A DAG for which to find the possible latents
    :return: A set of varaibles that can serve as latent confounders
    """
    # find parentless nodes
    parentless_set = set()
    for node in graph.nodes_set:
        if len(graph.parents(node)) == 0:
            parentless_set.add(node)

    # find parentless that have at least 2 children
    parents_set = set()
    for parent in parentless_set:
        if len(graph.find_children(parent)) >= 2:
            parents_set.add(parent)

    return parents_set


def create_random_dag_with_latents(n_nodes, conn_coeff):
    # sample a connected DAG
    dag_samp = create_random_connected_dag(n_nodes, conn_coeff, num_dags_timeout=1000000)

    # find nodes that can serve as latents (parentless, and parents of at least two observed nodes)
    potential_latents = select_latent_variables(dag_samp)

    # sample 50% of the potential latents
    lat_set = set(
        random.sample(potential_latents, len(potential_latents) // 2)
    )
    obs_set = dag_samp.nodes_set - lat_set
    return dag_samp, obs_set, lat_set


def _prepare_intervention_values(num_samples, intervention_value):
    if np.isscalar(intervention_value):
        return np.full(num_samples, intervention_value, dtype=float)
    intervention_value = np.asarray(intervention_value, dtype=float)
    if intervention_value.shape[0] != num_samples:
        raise ValueError("Intervention value array must match num_samples")
    return intervention_value


def sample_edge_weights_from_dag(
    in_dag,
    min_edge_weight,
    max_edge_weight,
    seed: int,
):
    """
    Sample deterministic linear SEM edge weights for a DAG.
    The weight draw depends only on the supplied seed and DAG structure, not on
    the number of observational samples that may later be generated.
    """
    if seed is None:
        raise ValueError("sample_edge_weights_from_dag requires an explicit seed.")

    rng = np.random.default_rng(int(seed))
    edge_weights = {}
    topological_order = in_dag.find_topological_order()
    for node in topological_order:
        parents_set = in_dag.parents(node)
        for node_parent in parents_set:
            weight_sign = 2 * int(rng.integers(0, 2)) - 1
            weight = weight_sign * rng.uniform(min_edge_weight, max_edge_weight)
            edge_weights[(node_parent, node)] = weight
    return edge_weights


def sample_data_from_dag(
    in_dag,
    num_samples,
    min_edge_weight,
    max_edge_weight,
    seed:int,
    edge_weights=None,
    edge_weight_seed=None,
    return_weights=False,
    noise_scale=1.0,
    intervention=None,
):
    """
    Sample data from a linear SEM. A linear SEM is created from a DAG.
    A node is the sum of a normally distributed noise term and the weighted sum of the values of its parents.
    :param in_dag: The DAG structure of the linear SCM
    :param num_samples: number of samples (dataset records)
    :param min_edge_weight: lowest absolute value of edge weight (linear coefficient)
    :param max_edge_weight: highest absolute value of the weight (linear coefficient)
    :param edge_weights: optional mapping {(parent, child): weight} to reuse SEM weights
    :param edge_weight_seed: optional seed used to sample edge weights when edge_weights is None
    :param return_weights: if True, return (data, edge_weights)
    :param noise_scale: std dev for Gaussian noise
    :param intervention: optional dict {node: value} for do-intervention
    :param seed: Integer seed used for deterministic sampling (required)
    :return: Sampled dataset in the form of a 2D NumPy array (and optional weights)
    """
    if seed is None:
        raise ValueError("sample_data_from_dag requires an explicit seed.")
    rng = np.random.default_rng(int(seed))
    data = rng.standard_normal((num_samples, len(in_dag.nodes_set))) * noise_scale
    topological_order = in_dag.find_topological_order()

    if edge_weights is None:
        resolved_edge_weight_seed = int(seed) if edge_weight_seed is None else int(edge_weight_seed)
        edge_weights = sample_edge_weights_from_dag(
            in_dag=in_dag,
            min_edge_weight=min_edge_weight,
            max_edge_weight=max_edge_weight,
            seed=resolved_edge_weight_seed,
        )

    if intervention is None:
        intervention = {}

    for node in topological_order:
        if node in intervention:
            data[:, node] = _prepare_intervention_values(num_samples, intervention[node])
            continue

        parents_set = in_dag.parents(node)
        for node_parent in parents_set:
            weight = edge_weights[(node_parent, node)]
            data[:, node] += weight * data[:, node_parent]

    if return_weights:
        return data, edge_weights
    return data
