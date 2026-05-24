__all__ = [
    "create_random_connected_dag",
    "create_random_dag",
    "create_random_dag_max_fan",
    "create_random_dag_with_latents",
    "sample_data_from_dag",
    "select_latent_variables",
]

try:
    from .synthetic_graphs import (
        create_random_connected_dag,
        create_random_dag,
        create_random_dag_max_fan,
        create_random_dag_with_latents,
        sample_data_from_dag,
        select_latent_variables,
    )
except ModuleNotFoundError:
    # Allow lightweight utilities to be imported without the optional graph stack.
    pass
