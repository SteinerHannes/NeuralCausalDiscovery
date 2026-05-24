import os
import json
import sys
import numpy as np
import hydra
import pandas as pd
import matplotlib.pyplot as plt
from omegaconf import DictConfig, ListConfig
from typing import Optional
from tqdm import tqdm
from causal_discovery_algs import LearnStructOrderedICD
from graphical_models import PAG
from experiment_utils.logging_utils import fold_log_context, pbar_disabled, resolve_experiment_root
from experiment_utils.correlation_sanitization import sanitize_dataset_for_corr
from metrics.performance_measures import (
    dag_to_pag_comparison,
    compare_edge_details,
    pag_to_pag_comparison,
    compare_pag_edge_details,
)
from metrics.visual_comparison import (
    compare_pag_against_dag,
    compare_5pags_against_dag,
    compare_pags_against_dag,
    compare_pags_against_pag,
)

def _learn_pag_from_dataset(dataset, config):
    ci_test = hydra.utils.instantiate(config.causaldiscovery.ci_test, dataset=dataset)
    nodes = set(range(config.data.input_size + config.data.output_size))
    causal_order = list(range(config.data.input_size + config.data.output_size))
    causal_discovery = LearnStructOrderedICD(nodes_set=nodes, ci_test=ci_test, causal_order=causal_order)
    causal_discovery.learn_structure()
    return causal_discovery.graph

def _select_columns(pd_data: pd.DataFrame, config: DictConfig) -> np.ndarray:
    cd_cfg = config.get("causaldiscovery", {})
    columns_mode = cd_cfg.get("columns_mode", "all")
    include_prefixes = cd_cfg.get("include_prefixes", [])
    exclude_prefixes = cd_cfg.get("exclude_prefixes", [])
    max_features = cd_cfg.get("max_features")
    selection_strategy = cd_cfg.get("selection_strategy", "first")

    if columns_mode == "inputs_outputs":
        input_cols = [f"input_{i}" for i in range(config.data.input_size)]
        output_cols = [f"output_{i}" for i in range(config.data.output_size)]
        expected = input_cols + output_cols
        missing = [col for col in expected if col not in pd_data.columns]
        if missing:
            print(
                "Warning: Missing expected input/output columns for causal discovery. "
                "Falling back to full extraction data."
            )
            columns_mode = "all"
        else:
            return pd_data[expected].values

    columns = list(pd_data.columns)
    if include_prefixes:
        columns = [col for col in columns if any(col.startswith(p) for p in include_prefixes)]
    if exclude_prefixes:
        columns = [col for col in columns if not any(col.startswith(p) for p in exclude_prefixes)]

    if not columns:
        print("Warning: Column selection removed all features; using full extraction data.")
        columns = list(pd_data.columns)

    if max_features is not None and len(columns) > max_features:
        if selection_strategy == "random":
            seed = int(getattr(config.data, "seed", 0))
            rng = np.random.default_rng(seed)
            columns = sorted(rng.choice(columns, size=max_features, replace=False).tolist())
        else:
            columns = columns[:max_features]

    return pd_data[columns].values

def _subsample_dataset(dataset: np.ndarray, max_samples: Optional[int], strategy: str, seed: int) -> np.ndarray:
    if max_samples is None:
        return dataset
    if dataset.shape[0] <= max_samples:
        return dataset
    if strategy == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(dataset.shape[0], size=max_samples, replace=False)
        return dataset[idx]
    return dataset[:max_samples]

def _normalize_sample_sizes(value, fallback):
    if value is None:
        return [fallback]
    if isinstance(value, (list, tuple, ListConfig)):
        return list(value)
    return [value]

def _format_sample_size(size: Optional[int]) -> str:
    return "all" if size is None else str(size)

def _run_tag(sample_size: Optional[int], baseline_size: Optional[int]) -> str:
    return f"sample_{_format_sample_size(sample_size)}__baseline_{_format_sample_size(baseline_size)}"

def _target_info(cfg: Optional[DictConfig]) -> dict:
    if cfg is None:
        return {"name": None, "target": None}
    target = cfg.get("_target_")
    name = target.split(".")[-1] if target else None
    return {"name": name, "target": target}

def _build_metadata(
    config: DictConfig,
    sample_size: Optional[int],
    baseline_size: Optional[int],
    use_aligned: bool,
    alignment_cfg: DictConfig,
    fold: int,
) -> dict:
    ci_cfg = config.causaldiscovery.get("ci_test", {})
    ci_info = _target_info(ci_cfg)
    ci_info["threshold"] = ci_cfg.get("threshold")
    target_mode = None
    if alignment_cfg.get("enabled", False):
        target_mode = alignment_cfg.get("target_mode")
    return {
        "fold": fold,
        "dataset": _target_info(config.get("data")),
        "model": _target_info(config.get("model")),
        "ci_test": ci_info,
        "sample_size": sample_size,
        "baseline_sample_size": baseline_size,
        "alignment": {
            "enabled": alignment_cfg.get("enabled", False),
            "used_for_causaldiscovery": use_aligned,
            "status": "aligned" if use_aligned else "raw",
            "target_mode": target_mode,
        },
    }

def _init_run_state():
    return {
        "metrics": [],
        "baseline_metrics": [],
        "pag_vs_pag_metrics": [],
        "learned_pags": {},
        "baseline_pags": {},
        "ground_truth_dag": None,
    }

def run(_config: DictConfig) -> None:
    global config
    config = _config

    experiment_root = resolve_experiment_root(config)

    os.makedirs(experiment_root, exist_ok=True)

    cd_cfg = config.causaldiscovery
    sample_sizes = _normalize_sample_sizes(cd_cfg.get("sample_sizes"), cd_cfg.get("max_samples"))
    baseline_sample_sizes = _normalize_sample_sizes(cd_cfg.get("baseline_sample_sizes"), cd_cfg.get("baseline_max_samples"))
    multi_run = len(sample_sizes) > 1 or len(baseline_sample_sizes) > 1

    runs = {}

    folds = list(config.training.test_folds)
    total_folds = len(folds)
    disable_pbar = pbar_disabled(config)

    for fold_idx, test_fold in enumerate(folds, start=1):
        with fold_log_context(
            experiment_root,
            test_fold,
            "causaldiscovery.log",
            step_name="Causal Discovery",
            fold_idx=fold_idx,
            num_folds=total_folds,
        ):
            print("Step: load extraction data")

            fold_dir = os.path.join(experiment_root, str(test_fold))

            alignment_cfg = config.extraction.get("alignment", {})
            use_aligned = alignment_cfg.get("enabled", False) and alignment_cfg.get("use_aligned_for_causaldiscovery", True)
            extraction_filename = (
                alignment_cfg.get("output_file", config.extraction.output_file)
                if use_aligned
                else config.extraction.output_file
            )
            extraction_file = os.path.join(fold_dir, extraction_filename)
            print("Loading extraction data from:", extraction_file)
            pd_data = pd.read_csv(extraction_file, header=0)
            dataset = _select_columns(pd_data, config)
            base_seed = int(getattr(config.data, "seed", 0))
            dataset = sanitize_dataset_for_corr(dataset, seed=base_seed)

            print("Step: prepare ground-truth data")
            ground_truth_data = hydra.utils.instantiate(config.data, subset="all", fold=test_fold)
            ground_truth_graph = ground_truth_data.dag
            baseline_dataset = ground_truth_data.dag_data
            baseline_dataset = sanitize_dataset_for_corr(baseline_dataset, seed=base_seed + 1)

            run_pairs = [
                (sample_size, baseline_size)
                for sample_size in sample_sizes
                for baseline_size in baseline_sample_sizes
            ]
            run_iter = tqdm(
                run_pairs,
                desc="Causal discovery runs",
                unit="run",
                file=sys.stdout,
                disable=disable_pbar,
                ascii=True,
            )

            for sample_size, baseline_size in run_iter:
                run_tag = _run_tag(sample_size, baseline_size)
                run_state = runs.setdefault(run_tag, _init_run_state())
                run_dir = fold_dir if not multi_run else os.path.join(fold_dir, run_tag)
                os.makedirs(run_dir, exist_ok=True)
                print(f"Running {run_tag}")

                metadata = _build_metadata(
                    config=config,
                    sample_size=sample_size,
                    baseline_size=baseline_size,
                    use_aligned=use_aligned,
                    alignment_cfg=alignment_cfg,
                    fold=test_fold,
                )
                metadata_path = os.path.join(run_dir, "metadata.json")
                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)
                print(f"Saved metadata to {metadata_path}")

                sampled_dataset = _subsample_dataset(
                    dataset,
                    max_samples=sample_size,
                    strategy=config.causaldiscovery.get("sample_strategy", "first"),
                    seed=base_seed,
                )
                sampled_dataset = sanitize_dataset_for_corr(sampled_dataset, seed=base_seed + test_fold * 1000 + 11)

                learned_graph = _learn_pag_from_dataset(sampled_dataset, config)

                if config.causaldiscovery.mode == "model_level":
                    run_state["learned_pags"][test_fold] = learned_graph
                    adj_base = os.path.join(run_dir, "learned_graph")
                    if isinstance(learned_graph, PAG):
                        adj_mat = learned_graph.get_adj_mat()
                        np.save(adj_base + "_adj.npy", adj_mat)
                        print(f"Saved adjacency matrix to {adj_base}_adj.npy")

                sampled_baseline_dataset = _subsample_dataset(
                    baseline_dataset,
                    max_samples=baseline_size,
                    strategy=config.causaldiscovery.get("sample_strategy", "first"),
                    seed=base_seed,
                )
                sampled_baseline_dataset = sanitize_dataset_for_corr(
                    sampled_baseline_dataset,
                    seed=base_seed + test_fold * 1000 + 29,
                )
                baseline_graph = _learn_pag_from_dataset(sampled_baseline_dataset, config)

                # Store for combined visualization later
                run_state["learned_pags"][test_fold] = learned_graph
                run_state["baseline_pags"][test_fold] = baseline_graph
                if run_state["ground_truth_dag"] is None:
                    run_state["ground_truth_dag"] = ground_truth_graph

                metrics = dag_to_pag_comparison(dag=ground_truth_graph, pag=learned_graph)
                metrics['fold'] = test_fold
                run_state["metrics"].append(metrics)

                metrics_path = os.path.join(run_dir, "performance_metrics.json")
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2)
                print(f"Saved metrics to {metrics_path}")

                baseline_metrics = dag_to_pag_comparison(dag=ground_truth_graph, pag=baseline_graph)
                baseline_metrics['fold'] = test_fold
                run_state["baseline_metrics"].append(baseline_metrics)

                baseline_metrics_path = os.path.join(run_dir, "baseline_performance_metrics.json")
                with open(baseline_metrics_path, "w") as f:
                    json.dump(baseline_metrics, f, indent=2)
                print(f"Saved baseline metrics to {baseline_metrics_path}")

                pag_vs_pag_metrics = pag_to_pag_comparison(reference_pag=baseline_graph, pag=learned_graph)
                pag_vs_pag_metrics['fold'] = test_fold
                run_state["pag_vs_pag_metrics"].append(pag_vs_pag_metrics)

                pag_vs_pag_metrics_path = os.path.join(run_dir, "pag_vs_pag_metrics.json")
                with open(pag_vs_pag_metrics_path, "w") as f:
                    json.dump(pag_vs_pag_metrics, f, indent=2)
                print(f"Saved PAG-vs-PAG metrics to {pag_vs_pag_metrics_path}")

                differences, partials = compare_edge_details(dag=ground_truth_graph, pag=learned_graph, max_display=20)
                details_path = os.path.join(run_dir, "edge_comparison.txt")
                with open(details_path, "w") as f:
                    f.write("Differences:\n")
                    if not differences:
                        f.write("No differences found.\n")
                    for diff in differences:
                        f.write(diff + "\n")
                    f.write("\nPartial Orientations:\n")
                    if not partials:
                        f.write("No partial orientation differences found.\n")
                    for part in partials:
                        f.write(part + "\n")
                print(f"Saved edge comparison details to {details_path}")

                baseline_differences, baseline_partials = compare_edge_details(dag=ground_truth_graph, pag=baseline_graph, max_display=20)
                baseline_details_path = os.path.join(run_dir, "edge_comparison_baseline.txt")
                with open(baseline_details_path, "w") as f:
                    f.write("Differences:\n")
                    if not baseline_differences:
                        f.write("No differences found.\n")
                    for diff in baseline_differences:
                        f.write(diff + "\n")
                    f.write("\nPartial Orientations:\n")
                    if not baseline_partials:
                        f.write("No partial orientation differences found.\n")
                    for part in baseline_partials:
                        f.write(part + "\n")
                print(f"Saved baseline edge comparison details to {baseline_details_path}")

                pag_vs_pag_differences, pag_vs_pag_partials = compare_pag_edge_details(
                    reference_pag=baseline_graph,
                    pag=learned_graph,
                    max_display=20
                )
                pag_vs_pag_details_path = os.path.join(run_dir, "edge_comparison_pag_vs_pag.txt")
                with open(pag_vs_pag_details_path, "w") as f:
                    f.write("Differences:\n")
                    if not pag_vs_pag_differences:
                        f.write("No differences found.\n")
                    for diff in pag_vs_pag_differences:
                        f.write(diff + "\n")
                    f.write("\nPartial Orientations:\n")
                    if not pag_vs_pag_partials:
                        f.write("No partial orientation differences found.\n")
                    for part in pag_vs_pag_partials:
                        f.write(part + "\n")
                print(f"Saved PAG-vs-PAG edge comparison details to {pag_vs_pag_details_path}")

                fig = compare_pag_against_dag(ground_truth_graph, learned_graph, show=False)
                fig_path = os.path.join(run_dir, "pag_vs_dag.png")
                fig.savefig(fig_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved figure to {fig_path}")

                baseline_fig = compare_pag_against_dag(
                    ground_truth_graph,
                    baseline_graph,
                    titles=("Baseline PAG", "Ground Truth DAG"),
                    show=False
                )
                baseline_fig_path = os.path.join(run_dir, "baseline_pag_vs_dag.png")
                baseline_fig.savefig(baseline_fig_path, dpi=150, bbox_inches="tight")
                plt.close(baseline_fig)
                print(f"Saved baseline figure to {baseline_fig_path}")

                pags_vs_dag_fig = compare_pags_against_dag(
                    ground_truth_graph,
                    baseline_graph,
                    learned_graph,
                    titles=("PAG (Data) vs DAG", "PAG (Model) vs DAG"),
                    dag_title="Ground Truth DAG",
                    show=False
                )
                pags_vs_dag_path = os.path.join(run_dir, "pags_vs_dag.png")
                pags_vs_dag_fig.savefig(pags_vs_dag_path, dpi=150, bbox_inches="tight")
                plt.close(pags_vs_dag_fig)
                print(f"Saved combined DAG comparison to {pags_vs_dag_path}")

                pag_vs_pag_fig = compare_pags_against_pag(
                    baseline_graph,
                    learned_graph,
                    layout_graph=ground_truth_graph,
                    titles=("PAG (Data) vs PAG (Model)", "PAG (Model) vs PAG (Data)"),
                    show=False
                )
                pag_vs_pag_fig_path = os.path.join(run_dir, "pag_vs_pag.png")
                pag_vs_pag_fig.savefig(pag_vs_pag_fig_path, dpi=150, bbox_inches="tight")
                plt.close(pag_vs_pag_fig)
                print(f"Saved PAG-vs-PAG figure to {pag_vs_pag_fig_path}")

                adj_base = os.path.join(run_dir, "learned_graph")
                if isinstance(learned_graph, PAG):
                    adj_mat = learned_graph.get_adj_mat()
                    np.save(adj_base + "_adj.npy", adj_mat)
                    print(f"Saved adjacency matrix to {adj_base}_adj.npy")

                baseline_adj_base = os.path.join(run_dir, "baseline_graph")
                if isinstance(baseline_graph, PAG):
                    baseline_adj_mat = baseline_graph.get_adj_mat()
                    np.save(baseline_adj_base + "_adj.npy", baseline_adj_mat)
                    print(f"Saved baseline adjacency matrix to {baseline_adj_base}_adj.npy")

    for run_tag, run_state in runs.items():
        output_root = experiment_root if not multi_run else os.path.join(experiment_root, run_tag)
        os.makedirs(output_root, exist_ok=True)

        all_metrics = run_state["metrics"]
        all_baseline_metrics = run_state["baseline_metrics"]
        all_pag_vs_pag_metrics = run_state["pag_vs_pag_metrics"]
        all_learned_pags = run_state["learned_pags"]
        all_baseline_pags = run_state["baseline_pags"]
        ground_truth_dag = run_state["ground_truth_dag"]

        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            metrics_df = metrics_df.set_index('fold')

            summary_stats = pd.concat([
                metrics_df.mean().to_frame('mean').T,
                metrics_df.std().to_frame('std').T
            ])

            metrics_with_stats = pd.concat([metrics_df, summary_stats])

            summary_path = os.path.join(output_root, "performance_metrics_summary.csv")
            metrics_with_stats.to_csv(summary_path)
            print(f"Saved aggregated metrics to {summary_path}")
            print(metrics_with_stats)

        if all_baseline_metrics:
            baseline_df = pd.DataFrame(all_baseline_metrics)
            baseline_df = baseline_df.set_index('fold')

            baseline_stats = pd.concat([
                baseline_df.mean().to_frame('mean').T,
                baseline_df.std().to_frame('std').T
            ])

            baseline_with_stats = pd.concat([baseline_df, baseline_stats])

            baseline_summary_path = os.path.join(output_root, "baseline_metrics_summary.csv")
            baseline_with_stats.to_csv(baseline_summary_path)
            print(f"Saved baseline aggregated metrics to {baseline_summary_path}")
            print(baseline_with_stats)

        if all_pag_vs_pag_metrics:
            pag_vs_pag_df = pd.DataFrame(all_pag_vs_pag_metrics)
            pag_vs_pag_df = pag_vs_pag_df.set_index('fold')

            pag_vs_pag_stats = pd.concat([
                pag_vs_pag_df.mean().to_frame('mean').T,
                pag_vs_pag_df.std().to_frame('std').T
            ])

            pag_vs_pag_with_stats = pd.concat([pag_vs_pag_df, pag_vs_pag_stats])

            pag_vs_pag_summary_path = os.path.join(output_root, "pag_vs_pag_metrics_summary.csv")
            pag_vs_pag_with_stats.to_csv(pag_vs_pag_summary_path)
            print(f"Saved PAG-vs-PAG aggregated metrics to {pag_vs_pag_summary_path}")
            print(pag_vs_pag_with_stats)

        if len(all_learned_pags) == 5 and ground_truth_dag is not None:
            print("Creating combined visualization of all 5 folds...")

            fold_numbers = sorted(all_learned_pags.keys())
            pags = [all_learned_pags[fold] for fold in fold_numbers]

            titles = (
                f"PAG Fold {fold_numbers[0]}",
                f"PAG Fold {fold_numbers[1]}",
                f"PAG Fold {fold_numbers[2]}",
                f"PAG Fold {fold_numbers[3]}",
                f"PAG Fold {fold_numbers[4]}"
            )

            fig_combined = compare_5pags_against_dag(
                ground_truth_dag,
                pags[0], pags[1], pags[2], pags[3], pags[4],
                titles=titles,
                dag_title="Ground Truth DAG",
                show=False
            )

            combined_fig_path = os.path.join(output_root, "all_pags_vs_dag.png")
            fig_combined.savefig(combined_fig_path, dpi=150, bbox_inches="tight")
            plt.close(fig_combined)
            print(f"Saved combined figure to {combined_fig_path}")
        elif len(all_learned_pags) != 5:
            print(f"Warning: Expected 5 folds but got {len(all_learned_pags)}. Skipping combined visualization.")

        if len(all_baseline_pags) == 5 and ground_truth_dag is not None:
            print("Creating combined visualization of all 5 baseline folds...")

            baseline_fold_numbers = sorted(all_baseline_pags.keys())
            baseline_pags = [all_baseline_pags[fold] for fold in baseline_fold_numbers]

            baseline_titles = (
                f"Baseline PAG Fold {baseline_fold_numbers[0]}",
                f"Baseline PAG Fold {baseline_fold_numbers[1]}",
                f"Baseline PAG Fold {baseline_fold_numbers[2]}",
                f"Baseline PAG Fold {baseline_fold_numbers[3]}",
                f"Baseline PAG Fold {baseline_fold_numbers[4]}"
            )

            baseline_fig_combined = compare_5pags_against_dag(
                ground_truth_dag,
                baseline_pags[0], baseline_pags[1], baseline_pags[2], baseline_pags[3], baseline_pags[4],
                titles=baseline_titles,
                dag_title="Ground Truth DAG",
                show=False
            )

            baseline_combined_fig_path = os.path.join(output_root, "all_baseline_pags_vs_dag.png")
            baseline_fig_combined.savefig(baseline_combined_fig_path, dpi=150, bbox_inches="tight")
            plt.close(baseline_fig_combined)
            print(f"Saved combined baseline figure to {baseline_combined_fig_path}")
        elif len(all_baseline_pags) != 5:
            print(f"Warning: Expected 5 baseline folds but got {len(all_baseline_pags)}. Skipping combined baseline visualization.")


@hydra.main(version_base="1.3", config_path="config", config_name="config")
def main(_config: DictConfig) -> None:
    run(_config)

if __name__ == "__main__":
    main()
