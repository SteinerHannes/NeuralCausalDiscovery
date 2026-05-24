import os
import json
import sys
import numpy as np
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from tqdm import tqdm

from graphical_models import PAG
from metrics.performance_measures import dag_to_pag_comparison, pag_to_pag_comparison
from metrics.visual_comparison import compare_pags_against_dag
from experiment_utils.logging_utils import pbar_disabled, step_log_context, resolve_experiment_root


def _edge_counts_from_adj(adj: np.ndarray, i: int, j: int) -> dict:
    a = adj[i, j]
    b = adj[j, i]
    return {
        "present": int(a > 0 or b > 0),
        "directed_ij": int(a == 2 and b == 3),
        "directed_ji": int(a == 3 and b == 2),
        "partial_ij": int(a == 2 and b == 1),
        "partial_ji": int(a == 1 and b == 2),
    }


def _set_directed(adj: np.ndarray, parent: int, child: int) -> None:
    adj[parent, child] = 2
    adj[child, parent] = 3


def _set_partial(adj: np.ndarray, parent: int, child: int) -> None:
    adj[parent, child] = 2
    adj[child, parent] = 1


def _set_bidirected(adj: np.ndarray, i: int, j: int) -> None:
    adj[i, j] = 2
    adj[j, i] = 2


def _combine_model_pags(adj_mats: list[np.ndarray]) -> np.ndarray:
    if not adj_mats:
        raise ValueError("No adjacency matrices provided for PAG cleanup.")

    num_nodes = adj_mats[0].shape[0]
    for mat in adj_mats:
        if mat.shape != (num_nodes, num_nodes):
            raise ValueError("All PAG adjacency matrices must have the same shape.")

    new_adj = np.zeros((num_nodes, num_nodes), dtype=int)
    majority = (len(adj_mats) // 2) + 1

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            directed_ij = 0
            directed_ji = 0
            partial_ij = 0
            partial_ji = 0
            present = 0

            for adj in adj_mats:
                counts = _edge_counts_from_adj(adj, i, j)
                directed_ij += counts["directed_ij"]
                directed_ji += counts["directed_ji"]
                partial_ij += counts["partial_ij"]
                partial_ji += counts["partial_ji"]
                present += counts["present"]

            if directed_ij or directed_ji:
                if directed_ij > directed_ji:
                    _set_directed(new_adj, i, j)
                elif directed_ji > directed_ij:
                    _set_directed(new_adj, j, i)
                else:
                    parent, child = (i, j) if i < j else (j, i)
                    print(
                        f"Warning: conflicting directed edges between {i} and {j}; "
                        f"using {parent} -> {child} by tie-break."
                    )
                    _set_directed(new_adj, parent, child)
                continue

            if present >= majority:
                if partial_ij or partial_ji:
                    if partial_ij > partial_ji:
                        _set_partial(new_adj, i, j)
                    elif partial_ji > partial_ij:
                        _set_partial(new_adj, j, i)
                    else:
                        parent, child = (i, j) if i < j else (j, i)
                        print(
                            f"Warning: conflicting o-> edges between {i} and {j}; "
                            f"using {parent} o-> {child} by tie-break."
                        )
                        _set_partial(new_adj, parent, child)
                else:
                    _set_bidirected(new_adj, i, j)

    return new_adj


def run(_config: DictConfig) -> None:
    config = _config

    experiment_root = resolve_experiment_root(config)

    os.makedirs(experiment_root, exist_ok=True)

    disable_pbar = pbar_disabled(config)

    with step_log_context(experiment_root, "pag_cleanup.log", "PAG cleanup"):
        print("Step: load learned PAG adjacencies")
        learned_adj_mats = []
        fold_numbers = []
        fold_iter = tqdm(
            list(config.training.test_folds),
            desc="Loading folds",
            unit="fold",
            file=sys.stdout,
            disable=disable_pbar,
            ascii=True,
        )
        for test_fold in fold_iter:
            fold_dir = os.path.join(experiment_root, str(test_fold))
            adj_path = os.path.join(fold_dir, config.pag_cleanup.learned_pag_adj_file)
            if not os.path.exists(adj_path):
                print(f"Warning: missing learned PAG adjacency at {adj_path}; skipping fold {test_fold}.")
                continue
            learned_adj_mats.append(np.load(adj_path))
            fold_numbers.append(test_fold)

        if not learned_adj_mats:
            raise ValueError("No learned PAG adjacency matrices found for cleanup.")

        print("Step: combine model PAGs")
        combined_adj = _combine_model_pags(learned_adj_mats)
        combined_adj_path = os.path.join(experiment_root, config.pag_cleanup.combined_pag_adj_file)
        np.save(combined_adj_path, combined_adj)
        print(f"Saved combined model-level PAG adjacency to {combined_adj_path}")

        num_nodes = combined_adj.shape[0]
        combined_pag = PAG(nodes_set=set(range(num_nodes)))
        combined_pag.init_from_adj_mat(combined_adj, nodes_order=list(range(num_nodes)))

        reference_fold = fold_numbers[0]
        data_pag = None
        baseline_adj_path = os.path.join(
            experiment_root,
            str(reference_fold),
            config.pag_cleanup.baseline_pag_adj_file,
        )
        if os.path.exists(baseline_adj_path):
            baseline_adj = np.load(baseline_adj_path)
            data_pag = PAG(nodes_set=set(range(num_nodes)))
            data_pag.init_from_adj_mat(baseline_adj, nodes_order=list(range(num_nodes)))
        else:
            print(f"Warning: missing baseline PAG adjacency at {baseline_adj_path}; skipping data-level comparison.")

        print("Step: evaluate combined PAG")
        ground_truth_data = hydra.utils.instantiate(config.data, subset="all", fold=reference_fold)
        ground_truth_dag = ground_truth_data.dag

        dag_vs_combined = dag_to_pag_comparison(dag=ground_truth_dag, pag=combined_pag)
        dag_metrics_path = os.path.join(experiment_root, config.pag_cleanup.dag_vs_combined_metrics_file)
        with open(dag_metrics_path, "w") as f:
            json.dump(dag_vs_combined, f, indent=2)
        print(f"Saved DAG vs combined PAG metrics to {dag_metrics_path}")

        if data_pag is not None:
            data_vs_combined = pag_to_pag_comparison(reference_pag=data_pag, pag=combined_pag)
            data_metrics_path = os.path.join(experiment_root, config.pag_cleanup.data_vs_combined_metrics_file)
            with open(data_metrics_path, "w") as f:
                json.dump(data_vs_combined, f, indent=2)
            print(f"Saved data-level vs combined PAG metrics to {data_metrics_path}")

        if data_pag is not None:
            fig = compare_pags_against_dag(
                ground_truth_dag,
                data_pag,
                combined_pag,
                titles=("PAG (Data)", "PAG (Model - Combined)"),
                dag_title="Ground Truth DAG",
                show=False,
            )
            fig_path = os.path.join(experiment_root, config.pag_cleanup.combined_pag_plot_file)
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved combined PAG figure to {fig_path}")


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(_config: DictConfig) -> None:
    run(_config)


if __name__ == "__main__":
    main()
