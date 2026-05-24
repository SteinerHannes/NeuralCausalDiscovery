import os
from typing import List

import numpy as np
from graphical_models import PAG

from experiment_utils.reporting_utils import instantiate_dataset
from metrics.visual_comparison import compare_5pags_against_dag
from reporting.figures.common import (
    _dataset_label,
    _save_figure,
    set_reporting_theme,
)


FIGURE_DAG_FOLD_PAG_COMPARISON_TITLE = "Ground-Truth DAG vs Fold-Specific PAGs"


def _require_single_record(spec: dict, records: List[dict]) -> dict:
    if len(records) != 1:
        raise ValueError(
            f"{spec['id']}: expected exactly one experiment record, got {len(records)}."
        )
    return records[0]


def _load_pag_from_adj(path: str) -> PAG:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing PAG adjacency matrix: {path}")
    adj = np.load(path)
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError(f"Invalid PAG adjacency shape at {path}: expected square matrix, got {adj.shape}.")
    pag = PAG(nodes_set=set(range(adj.shape[0])))
    pag.init_from_adj_mat(adj, nodes_order=list(range(adj.shape[0])))
    return pag


def figure_dag_fold_pag_comparison(spec: dict, records: List[dict], output_dir: str) -> dict:
    set_reporting_theme(layout_profile="paper")

    record = _require_single_record(spec, records)
    cfg = record.get("cfg")
    if cfg is None:
        raise ValueError(f"{spec['id']}: reporting record is missing the resolved experiment config.")

    experiment_root = record.get("root")
    if not experiment_root:
        raise ValueError(f"{spec['id']}: reporting record is missing the experiment output root.")

    graph_adj_file = str(spec.get("graph_adj_file", "")).strip()
    if not graph_adj_file:
        raise ValueError(f"{spec['id']}: graph_adj_file must be configured.")

    pag_label_prefix = str(spec.get("pag_label_prefix", "PAG")).strip()
    if not pag_label_prefix:
        raise ValueError(f"{spec['id']}: pag_label_prefix must not be empty.")

    folds = sorted(int(fold) for fold in cfg.training.test_folds)
    if len(folds) != 5:
        raise ValueError(
            f"{spec['id']}: compare_5pags_against_dag requires exactly 5 folds, got {len(folds)}."
        )

    dataset_id = record.get("metadata", {}).get("dataset")
    dataset_label = _dataset_label(dataset_id)
    dag_title = str(spec.get("dag_title", f"Ground Truth DAG ({dataset_label})"))
    title = str(spec.get("title", FIGURE_DAG_FOLD_PAG_COMPARISON_TITLE))

    reference_dataset = instantiate_dataset(cfg, subset="all", fold=folds[0])
    ground_truth_dag = reference_dataset.dag

    pags = []
    source_files = []
    pag_titles = []
    for fold in folds:
        adj_path = os.path.join(experiment_root, str(fold), graph_adj_file)
        pags.append(_load_pag_from_adj(adj_path))
        source_files.append(adj_path)
        pag_titles.append(f"{pag_label_prefix} Fold {fold}")

    fig = compare_5pags_against_dag(
        ground_truth_dag,
        pags[0],
        pags[1],
        pags[2],
        pags[3],
        pags[4],
        titles=tuple(pag_titles),
        dag_title=dag_title,
        show=False,
    )

    source_kind = "data-level" if "data" in pag_label_prefix.lower() else "model-level"
    caption_lines = [
        (
            f"This figure compares the benchmark ground-truth DAG for {dataset_label} against the five fold-specific "
            f"{source_kind} PAG estimates from the single observable-state alignment experiment."
        ),
        (
            "The left panel in the first row shows the reference DAG, and the remaining panels show one PAG per test fold "
            "using the graph-coloring convention from the PAG-vs-DAG visual comparison utility."
        ),
        (
            "Fold labels are taken directly from the experiment's configured test folds so the visual appendix stays aligned "
            "with the exported per-fold structural metrics."
        ),
    ]

    all_source_files = source_files + list(record.get("source_files", []))
    return _save_figure(
        fig=fig,
        axes=fig.axes,
        figure_id=spec["id"],
        output_dir=output_dir,
        stats_rows=[],
        data_row_count=len(pags),
        source_files=all_source_files,
        title_generated=title,
        caption_lines=caption_lines,
        legend_mode="embedded_graph_legend",
        annotation_mode="caption_only",
        layout_profile="paper",
        apply_tight_layout=False,
        write_stats_csv=False,
        target_width=None,
        extra={
            "experiment_name": record.get("name"),
            "dataset": dataset_id,
            "graph_adj_file": graph_adj_file,
            "folds": folds,
        },
    )
