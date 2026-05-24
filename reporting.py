import json
import os
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from experiment_utils.reporting_utils import normalize_intervention_summary_payload, serialize_stats_value, \
    instantiate_dataset, collect_outcome_std, add_standardized_effects
from reporting.figures.figure_intervention_detection_quality import (
    figure_intervention_detection_quality,
)
from reporting.figures.figure_intervention_aligned_proxy_calibration import (
    figure_intervention_aligned_proxy_calibration,
)
from reporting.figures.figure_intervention_architecture_robustness import (
    figure_intervention_architecture_robustness,
)
from reporting.figures.figure_intervention_effect_calibration import figure_intervention_effect_calibration
from reporting.figures.figure_intervention_model_output_detection import (
    figure_intervention_model_output_detection,
)
from reporting.figures.figure_intervention_model_pag_consistency import (
    figure_intervention_model_pag_consistency,
)
from reporting.figures.figure_intervention_model_pag_consistency_benchmark_expansion import (
    figure_intervention_model_pag_consistency_benchmark_expansion,
)
from reporting.figures.figure_dag_fold_pag_comparison import (
    figure_dag_fold_pag_comparison,
)
from reporting.figures.figure_structure_intervention_coupling import (
    figure_structure_intervention_coupling,
)
from reporting.figures.figure_structural_alignment_target_truth import (
    figure_structural_alignment_target_truth,
)
from reporting.figures.figure_structural_architecture_robustness import (
    figure_structural_architecture_robustness,
)
from reporting.figures.figure_structural_ci_mismatch_sensitivity import (
    figure_structural_ci_mismatch_sensitivity,
)
from reporting.figures.figure_structural_ci_threshold_robustness import (
    figure_structural_ci_threshold_robustness,
)
from reporting.figures.figure_structural_data_model_truth_comparison import (
    figure_structural_data_model_truth_comparison,
)
from reporting.figures.figure_structural_data_truth_baseline import (
    figure_structural_data_truth_baseline,
)
from reporting.figures.figure_structural_extraction_modes_truth import (
    figure_structural_extraction_modes_truth,
)
from reporting.figures.figure_structural_extraction_internal_truth import (
    figure_structural_extraction_internal_truth,
)
from reporting.figures.figure_structural_node_count_robustness import (
    figure_structural_node_count_robustness,
)
from reporting.figures.figure_structural_sample_size_robustness import (
    figure_structural_sample_size_robustness,
)
from reporting.figures.figure_training_dynamics import figure_training_dynamics
from reporting.figures.table_model_parameter_counts import table_model_parameter_counts

REQUIRED_REPORTING_ARTIFACTS = [
    ".hydra/config.yaml",
    "performance_metrics_summary.csv",
    "baseline_metrics_summary.csv",
    "pag_vs_pag_metrics_summary.csv",
    "intervention_summary.json",
]


def _normalize_experiment_ref(experiment_ref: str) -> Tuple[str, str]:
    cleaned = experiment_ref.strip()
    if cleaned.endswith(".yaml"):
        cleaned = cleaned[:-5]
    if cleaned.startswith("experiment/"):
        cleaned = "experiments/" + cleaned[len("experiment/"):]
    elif not cleaned.startswith("experiments/"):
        cleaned = f"experiments/{cleaned}"
    experiment_name = cleaned.split("/")[-1]
    return cleaned, experiment_name


def _experiment_config_path(project_root: str, config_name: str) -> str:
    return os.path.join(project_root, "config", f"{config_name}.yaml")


def _expected_output_root(project_root: str, experiment_name: str) -> str:
    return os.path.join(project_root, "outputs", "experiments", experiment_name)


def _run_pipeline(project_root: str, config_name: str) -> None:
    pipeline_path = os.path.join(project_root, "pipeline.py")
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(f"pipeline.py not found at {pipeline_path}")

    cmd = [
        sys.executable,
        pipeline_path,
        "--config-name",
        config_name,
    ]
    print(f"Running pipeline for reporting preflight: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=project_root, check=True)


def _missing_artifacts(output_root: str, required_artifacts: Sequence[str]) -> List[str]:
    missing = []
    for artifact in required_artifacts:
        artifact_path = os.path.join(output_root, artifact)
        if not os.path.exists(artifact_path):
            missing.append(str(artifact))
    return missing


def _load_yaml_config(path: str, resolve: bool = True) -> Optional[DictConfig]:
    if not os.path.exists(path):
        return None
    cfg = OmegaConf.load(path)
    if not resolve:
        return cfg
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _load_experiment_definition(project_root: str, config_name: str) -> DictConfig:
    cfg_path = _experiment_config_path(project_root, config_name)
    cfg = _load_yaml_config(cfg_path, resolve=False)
    if cfg is None:
        raise FileNotFoundError(f"Experiment config not found: {cfg_path}")
    return cfg


def _training_config_name(experiment_cfg: DictConfig) -> str:
    defaults = OmegaConf.to_container(experiment_cfg.get("defaults", []), resolve=False)
    if not isinstance(defaults, list):
        return "Default"

    for entry in defaults:
        if not isinstance(entry, dict):
            continue
        for key, value in entry.items():
            normalized_key = str(key).replace("override ", "").lstrip("/")
            if normalized_key == "training":
                return str(value)
    return "Default"


def _experiment_test_folds(project_root: str, experiment_cfg: DictConfig) -> List[int]:
    training_cfg = experiment_cfg.get("training")
    if training_cfg and training_cfg.get("test_folds") is not None:
        return [int(fold) for fold in training_cfg.get("test_folds")]

    training_config_name = _training_config_name(experiment_cfg)
    training_cfg_path = os.path.join(project_root, "config", "training", f"{training_config_name}.yaml")
    fallback_training_cfg = _load_yaml_config(training_cfg_path, resolve=False)
    if fallback_training_cfg is None or fallback_training_cfg.get("test_folds") is None:
        experiment_name = str(experiment_cfg.get("experiment", {}).get("name", "<unknown>"))
        raise ValueError(
            f"Could not resolve training.test_folds for experiment {experiment_name} from {training_cfg_path}"
        )

    return [int(fold) for fold in fallback_training_cfg.get("test_folds")]


def _reuse_source_experiment_ref(experiment_cfg: DictConfig) -> Optional[str]:
    reuse_cfg = experiment_cfg.get("reuse")
    if not reuse_cfg or not bool(reuse_cfg.get("enabled", False)):
        return None

    source_experiment = reuse_cfg.get("source_experiment")
    if not source_experiment:
        return None

    config_name, _ = _normalize_experiment_ref(str(source_experiment))
    return config_name


def _reuse_source_ready(project_root: str, experiment_cfg: DictConfig) -> bool:
    reuse_cfg = experiment_cfg.get("reuse")
    if not reuse_cfg or not bool(reuse_cfg.get("enabled", False)):
        return True

    required_fold_files = list(reuse_cfg.get("required_fold_files", []))
    if not required_fold_files:
        raise ValueError("reuse.required_fold_files must not be empty when reuse.enabled=true.")

    source_path = reuse_cfg.get("source_path")
    if source_path:
        source_root = (
            str(source_path)
            if os.path.isabs(str(source_path))
            else os.path.join(project_root, str(source_path))
        )
    else:
        source_experiment = reuse_cfg.get("source_experiment")
        if not source_experiment:
            raise ValueError(
                "reuse.enabled=true requires either reuse.source_path or reuse.source_experiment."
            )
        _, source_name = _normalize_experiment_ref(str(source_experiment))
        source_root = _expected_output_root(project_root, source_name)

    if not os.path.isdir(source_root):
        return False

    folds = _experiment_test_folds(project_root, experiment_cfg)
    for fold in folds:
        fold_root = os.path.join(source_root, str(fold))
        if not os.path.isdir(fold_root):
            return False
        for rel_path in required_fold_files:
            if not os.path.exists(os.path.join(fold_root, rel_path)):
                return False
    return True


def _should_run_for_reporting(
    project_root: str,
    experiment_name: str,
    run_if_missing: bool,
    force_rerun: bool,
) -> bool:
    if force_rerun:
        return True
    if not run_if_missing:
        return False

    experiment_output_root = _expected_output_root(project_root, experiment_name)
    missing = _missing_artifacts(experiment_output_root, REQUIRED_REPORTING_ARTIFACTS)
    return len(missing) > 0


def _resolve_pipeline_run_order(
    project_root: str,
    experiment_refs: Sequence[str],
    run_if_missing: bool,
    force_rerun: bool,
) -> List[Tuple[str, str]]:
    experiment_cache: Dict[str, DictConfig] = {}
    ordered_runs: List[Tuple[str, str]] = []
    scheduled = set()
    visiting = set()

    def _load_cached(config_name: str) -> DictConfig:
        if config_name not in experiment_cache:
            experiment_cache[config_name] = _load_experiment_definition(project_root, config_name)
        return experiment_cache[config_name]

    def _schedule(ref: str) -> None:
        config_name, name = _normalize_experiment_ref(ref)
        if config_name in scheduled:
            return
        if config_name in visiting:
            raise ValueError(f"Cyclic experiment reuse dependency detected at {config_name}")

        visiting.add(config_name)
        experiment_cfg = _load_cached(config_name)
        dependency_ref = _reuse_source_experiment_ref(experiment_cfg)
        if dependency_ref and (force_rerun or not _reuse_source_ready(project_root, experiment_cfg)):
            _schedule(dependency_ref)

        visiting.remove(config_name)
        scheduled.add(config_name)
        ordered_runs.append((config_name, name))

    for ref in experiment_refs:
        config_name, name = _normalize_experiment_ref(ref)
        if _should_run_for_reporting(project_root, name, run_if_missing, force_rerun):
            _schedule(config_name)

    return ordered_runs


def _target_leaf_name(target: Optional[str]) -> Optional[str]:
    if not target:
        return None
    return target.split(".")[-1]


def _load_metrics_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, index_col=0)
    df.index = df.index.map(str)
    df = df[~df.index.isin(["mean", "std"])]
    fold_indices = []
    valid_rows = []
    for idx in df.index:
        try:
            fold_indices.append(int(float(idx)))
            valid_rows.append(True)
        except ValueError:
            valid_rows.append(False)
    df = df.loc[valid_rows]
    df.index = fold_indices
    return df.sort_index()


def _load_optional_csv(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def _load_intervention_summary(path: str, strict_schema: bool) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        payload = json.load(f)
    normalized_payload, _ = normalize_intervention_summary_payload(
        payload=payload,
        source_path=path,
        strict=strict_schema,
    )

    rows = []
    for item in normalized_payload:
        fold = item.get("fold")
        if fold in ("global", None):
            continue
        try:
            fold = int(fold)
        except (TypeError, ValueError):
            continue

        row = {"fold": fold}
        for key, value in item.items():
            if key == "fold":
                continue
            if isinstance(value, dict):
                for sub_key, sub_val in value.items():
                    row[f"{key}_{sub_key}"] = sub_val
            else:
                row[key] = value
        rows.append(row)

    if not rows:
        return None
    df = pd.DataFrame(rows)
    if "fold" not in df.columns:
        return None
    return df.set_index("fold").sort_index()


def _read_intervention_results(root: str, folds: Sequence[int]) -> Tuple[Optional[pd.DataFrame], List[str]]:
    frames = []
    source_files: List[str] = []
    for fold in folds:
        path = os.path.join(root, str(fold), "intervention_results.csv")
        if not os.path.exists(path):
            continue
        frame = pd.read_csv(path)
        frame["fold"] = int(fold)
        frames.append(frame)
        source_files.append(path)
    if not frames:
        return None, []
    return pd.concat(frames, ignore_index=True), source_files


def _read_aligned_proxy_artifacts(
    root: str,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[dict], List[str]]:
    pair_path = os.path.join(root, "aligned_proxy_dowhy_all_pairs.csv")
    summary_csv_path = os.path.join(root, "aligned_proxy_dowhy_all_pairs_summary.csv")
    summary_json_path = os.path.join(root, "aligned_proxy_dowhy_all_pairs_summary.json")

    pair_df = _load_optional_csv(pair_path)
    summary_df = _load_optional_csv(summary_csv_path)
    summary_json = _load_json(summary_json_path)
    source_files = [
        path
        for path, present in [
            (pair_path, pair_df is not None),
            (summary_csv_path, summary_df is not None),
            (summary_json_path, summary_json is not None),
        ]
        if present
    ]
    return pair_df, summary_df, summary_json, source_files


def _load_json(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def _read_training_history(root: str, folds: Sequence[int]) -> Tuple[Optional[pd.DataFrame], List[str]]:
    frames = []
    source_files: List[str] = []
    for fold in folds:
        preferred_path = os.path.join(root, str(fold), "training_history.csv")
        legacy_path = os.path.join(root, str(fold), "train_losses.csv")
        if os.path.exists(preferred_path):
            selected = preferred_path
        elif os.path.exists(legacy_path):
            selected = legacy_path
        else:
            continue

        frame = pd.read_csv(selected)
        if frame.empty:
            continue
        frame["fold"] = int(fold)
        if "epoch" in frame.columns:
            frame["epoch"] = pd.to_numeric(frame["epoch"], errors="coerce")
            frame = frame[frame["epoch"].notna()]
            frame["epoch"] = frame["epoch"].astype(int)
        frames.append(frame)
        source_files.append(selected)

    if not frames:
        return None, []
    return pd.concat(frames, ignore_index=True), source_files


def _read_training_summaries(root: str, folds: Sequence[int]) -> Tuple[Optional[pd.DataFrame], List[str]]:
    rows = []
    source_files: List[str] = []
    for fold in folds:
        path = os.path.join(root, str(fold), "training_summary.json")
        if not os.path.exists(path):
            continue
        with open(path, "r") as f:
            payload = json.load(f)
        if not isinstance(payload, dict):
            continue
        row = dict(payload)
        row["fold"] = int(fold)
        rows.append(row)
        source_files.append(path)

    if not rows:
        return None, []
    df = pd.DataFrame(rows)
    return df.set_index("fold").sort_index(), source_files


def _read_train_test_scores(root: str) -> Tuple[Optional[pd.DataFrame], List[str]]:
    path = os.path.join(root, "train_test_scores.csv")
    if not os.path.exists(path):
        return None, []

    frame = pd.read_csv(path)
    if frame.empty:
        return frame, [path]

    first_col = frame.columns[0]
    if "fold" not in frame.columns and (str(first_col).startswith("Unnamed") or str(first_col) == ""):
        frame = frame.rename(columns={first_col: "fold"})
    if "fold" in frame.columns:
        frame["fold"] = pd.to_numeric(frame["fold"], errors="coerce")
    if "TestLoss" in frame.columns:
        frame["TestLoss"] = pd.to_numeric(frame["TestLoss"], errors="coerce")
    if "test_loss" in frame.columns:
        frame["test_loss"] = pd.to_numeric(frame["test_loss"], errors="coerce")
    return frame, [path]


def _metadata_from_cfg(cfg: DictConfig) -> dict:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    data_cfg = cfg_dict.get("data", {}) if isinstance(cfg_dict, dict) else {}
    model_cfg = cfg_dict.get("model", {}) if isinstance(cfg_dict, dict) else {}
    extraction_cfg = cfg_dict.get("extraction", {}) if isinstance(cfg_dict, dict) else {}
    alignment_cfg = extraction_cfg.get("alignment", {}) if isinstance(extraction_cfg, dict) else {}
    cd_cfg = cfg_dict.get("causaldiscovery", {}) if isinstance(cfg_dict, dict) else {}
    ci_cfg = cd_cfg.get("ci_test", {}) if isinstance(cd_cfg, dict) else {}
    intervention_cfg = cfg_dict.get("intervention", {}) if isinstance(cfg_dict, dict) else {}
    model_level_cfg = intervention_cfg.get("model_level", {}) if isinstance(intervention_cfg, dict) else {}
    exp_cfg = cfg_dict.get("experiment", {}) if isinstance(cfg_dict, dict) else {}

    metadata = exp_cfg.get("metadata", {}) if isinstance(exp_cfg, dict) else {}
    dataset_target = _target_leaf_name(data_cfg.get("_target_")) if isinstance(data_cfg, dict) else None
    return {
        "dataset": _dataset_metadata_id(dataset_target, data_cfg if isinstance(data_cfg, dict) else {}),
        "model": _model_metadata_id(model_cfg if isinstance(model_cfg, dict) else {}),
        "alignment_enabled": bool(alignment_cfg.get("enabled", False)),
        "alignment_target_mode": (
            alignment_cfg.get("target_mode", "sem_truth") if bool(alignment_cfg.get("enabled", False)) else None
        ),
        "extraction_method": extraction_cfg.get("method", "activations") if isinstance(extraction_cfg, dict) else "activations",
        "sample_size": cd_cfg.get("max_samples") if isinstance(cd_cfg, dict) else None,
        "baseline_sample_size": cd_cfg.get("baseline_max_samples") if isinstance(cd_cfg, dict) else None,
        "ci_test": _target_leaf_name(ci_cfg.get("_target_")) if isinstance(ci_cfg, dict) else None,
        "ci_threshold": ci_cfg.get("threshold") if isinstance(ci_cfg, dict) else None,
        "intervention_alpha": intervention_cfg.get("alpha") if isinstance(intervention_cfg, dict) else None,
        "intervention_effect_threshold": (
            intervention_cfg.get("effect_threshold") if isinstance(intervention_cfg, dict) else None
        ),
        "model_alpha": model_level_cfg.get("alpha") if isinstance(model_level_cfg, dict) else None,
        "model_effect_threshold": (
            model_level_cfg.get("effect_threshold") if isinstance(model_level_cfg, dict) else None
        ),
        "evaluate_pairs": intervention_cfg.get("evaluate_pairs") if isinstance(intervention_cfg, dict) else None,
        "model_prediction_source": (
            model_level_cfg.get("prediction_source") if isinstance(model_level_cfg, dict) else None
        ),
        "exploratory": bool(metadata.get("exploratory", False)) if isinstance(metadata, dict) else False,
        "appendix_only": bool(metadata.get("appendix_only", False)) if isinstance(metadata, dict) else False,
        "num_nodes": data_cfg.get("num_nodes") if isinstance(data_cfg, dict) else None,
        "input_size": data_cfg.get("input_size") if isinstance(data_cfg, dict) else None,
        "output_size": data_cfg.get("output_size") if isinstance(data_cfg, dict) else None,
        "benchmark_family": _benchmark_family(dataset_target, data_cfg if isinstance(data_cfg, dict) else {}),
        "topology": data_cfg.get("topology") if isinstance(data_cfg, dict) else None,
        "noise_distribution": data_cfg.get("noise_distribution") if isinstance(data_cfg, dict) else None,
        "mechanism_family": data_cfg.get("mechanism_family") if isinstance(data_cfg, dict) else None,
    }


def _pascal_token(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return "".join(part.capitalize() for part in text.split("_") if part)


def _benchmark_family(dataset_target: Optional[str], data_cfg: dict) -> Optional[str]:
    if dataset_target == "TemplatedLinearSEMDataset":
        return "templated_linear"
    if dataset_target == "TemplatedNonlinearAdditiveSEMDataset":
        return "templated_nonlinear_additive"
    return dataset_target


def _dataset_metadata_id(dataset_target: Optional[str], data_cfg: dict) -> Optional[str]:
    if dataset_target == "TemplatedLinearSEMDataset":
        topology = _pascal_token(data_cfg.get("topology"))
        noise = str(data_cfg.get("noise_distribution", "gaussian"))
        dataset_id = f"{topology}LinearSEM"
        if noise == "laplace":
            return f"{dataset_id}_Laplace"
        if noise == "student_t_df3":
            return f"{dataset_id}_StudentT3"
        return dataset_id
    if dataset_target == "TemplatedNonlinearAdditiveSEMDataset":
        topology = _pascal_token(data_cfg.get("topology"))
        return f"{topology}NonlinearSEM"
    return dataset_target


def _model_metadata_id(model_cfg: dict) -> Optional[str]:
    explicit_model_id = str(model_cfg.get("model_id", "")).strip() if isinstance(model_cfg, dict) else ""
    if explicit_model_id:
        return explicit_model_id

    model_target = _target_leaf_name(model_cfg.get("_target_")) if isinstance(model_cfg, dict) else None
    if model_target == "MLPModel":
        hidden_sizes = list(model_cfg.get("hidden_sizes") or [])
        activation = str(model_cfg.get("activation", "")).lower()
        dropout = float(model_cfg.get("dropout", 0.0))
        use_batch_norm = bool(model_cfg.get("use_batch_norm", False))
        if hidden_sizes == [256, 128, 64] and activation == "silu" and use_batch_norm and np.isclose(dropout, 0.1):
            return "TabularMLP"
    return model_target


def _collect_unique_refs(figure_specs: List[DictConfig]) -> List[str]:
    refs = []
    seen = set()

    def _maybe_add(value):
        if isinstance(value, str) and (value.startswith("experiment/") or value.startswith("experiments/") or value.endswith(".yaml")):
            _, name = _normalize_experiment_ref(value)
            if name not in seen:
                seen.add(name)
                refs.append(value)

    def _walk(value):
        if isinstance(value, list):
            for item in value:
                _walk(item)
        elif isinstance(value, dict):
            for item in value.values():
                _walk(item)
        elif isinstance(value, str):
            _maybe_add(value)

    for spec in figure_specs:
        spec_dict = OmegaConf.to_container(spec, resolve=True)
        _walk(spec_dict)
    return refs


def _load_record(project_root: str, experiment_ref: str, strict_schema: bool) -> Optional[dict]:
    _, experiment_name = _normalize_experiment_ref(experiment_ref)
    experiment_root = _expected_output_root(project_root, experiment_name)
    exp_cfg_path = os.path.join(experiment_root, ".hydra", "config.yaml")
    exp_cfg = _load_yaml_config(exp_cfg_path)
    if exp_cfg is None:
        return None

    metadata = _metadata_from_cfg(exp_cfg)
    folds = list(exp_cfg.training.test_folds)

    performance_df = _load_metrics_csv(os.path.join(experiment_root, "performance_metrics_summary.csv"))
    baseline_df = _load_metrics_csv(os.path.join(experiment_root, "baseline_metrics_summary.csv"))
    pag_vs_pag_df = _load_metrics_csv(os.path.join(experiment_root, "pag_vs_pag_metrics_summary.csv"))
    intervention_summary_df = _load_intervention_summary(
        os.path.join(experiment_root, "intervention_summary.json"),
        strict_schema=strict_schema,
    )
    pair_df, pair_source_files = _read_intervention_results(experiment_root, folds)
    aligned_proxy_pair_df, aligned_proxy_summary_df, aligned_proxy_summary_json, aligned_proxy_source_files = (
        _read_aligned_proxy_artifacts(experiment_root)
    )
    training_history_df, training_history_files = _read_training_history(experiment_root, folds)
    training_summary_df, training_summary_files = _read_training_summaries(experiment_root, folds)
    train_test_scores_df, train_test_score_files = _read_train_test_scores(experiment_root)

    if pair_df is not None:
        outcome_std_by_fold = {}
        for fold in folds:
            dataset = instantiate_dataset(exp_cfg, subset="all", fold=fold)
            outcome_std_by_fold[fold] = collect_outcome_std(dataset.dag_data)
        add_standardized_effects(pair_df, outcome_std_by_fold)

    record = {
        "name": experiment_name,
        "root": experiment_root,
        "cfg": exp_cfg,
        "metadata": metadata,
        "performance_df": performance_df,
        "baseline_df": baseline_df,
        "pag_vs_pag_df": pag_vs_pag_df,
        "intervention_summary_df": intervention_summary_df,
        "pair_df": pair_df,
        "aligned_proxy_pair_df": aligned_proxy_pair_df,
        "aligned_proxy_summary_df": aligned_proxy_summary_df,
        "aligned_proxy_summary_json": aligned_proxy_summary_json,
        "training_history_df": training_history_df,
        "training_summary_df": training_summary_df,
        "train_test_scores_df": train_test_scores_df,
        "dag_vs_combined": _load_json(os.path.join(experiment_root, "dag_vs_combined_pag_metrics.json")),
        "data_vs_combined": _load_json(os.path.join(experiment_root, "data_vs_combined_pag_metrics.json")),
        "source_files": [
            os.path.join(experiment_root, "performance_metrics_summary.csv"),
            os.path.join(experiment_root, "baseline_metrics_summary.csv"),
            os.path.join(experiment_root, "pag_vs_pag_metrics_summary.csv"),
            os.path.join(experiment_root, "intervention_summary.json"),
        ] + pair_source_files,
        "aligned_proxy_source_files": aligned_proxy_source_files,
        "training_source_files": training_history_files + training_summary_files + train_test_score_files,
    }
    return record


def _records_by_refs(records: Dict[str, dict], refs: Sequence[str]) -> List[dict]:
    out = []
    for ref in refs:
        _, name = _normalize_experiment_ref(ref)
        if name in records:
            out.append(records[name])
    return out


def run(_config: DictConfig) -> None:
    config = _config
    project_root = hydra.utils.get_original_cwd()
    output_root = os.getcwd()

    strict_schema = bool(config.reporting.get("strict_intervention_schema", True))
    run_if_missing = bool(config.reporting.get("run_if_missing", False))
    force_rerun = bool(config.reporting.get("force_rerun", False))
    figure_specs = list(config.reporting.figure_specs)
    enabled_figure_specs = [spec for spec in figure_specs if bool(spec.get("enabled", True))]

    all_refs = _collect_unique_refs(enabled_figure_specs)
    pipeline_runs = []
    pipeline_run_order = _resolve_pipeline_run_order(
        project_root=project_root,
        experiment_refs=all_refs,
        run_if_missing=run_if_missing,
        force_rerun=force_rerun,
    )
    if pipeline_run_order:
        planned_names = ", ".join(name for _, name in pipeline_run_order)
        print(f"Resolved reporting preflight run order: {planned_names}")
    for config_name, name in pipeline_run_order:
        _run_pipeline(project_root, config_name)
        pipeline_runs.append(name)

    records_by_name: Dict[str, dict] = {}
    missing_experiments = []

    for ref in all_refs:
        _, name = _normalize_experiment_ref(ref)
        record = _load_record(project_root, ref, strict_schema=strict_schema)
        if record is None:
            missing_experiments.append(name)
            continue
        records_by_name[name] = record

    figures_output_dir = os.path.join(output_root, "figures")
    tables_output_dir = os.path.join(output_root, "tables")
    os.makedirs(figures_output_dir, exist_ok=True)

    manifest = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "report_name": str(config.reporting.name),
        "manifest_schema_version": "2.0",
        "p_value_method": "benjamini_hochberg",
        "strict_intervention_schema": strict_schema,
        "run_if_missing": run_if_missing,
        "force_rerun": force_rerun,
        "pipeline_runs": pipeline_runs,
        "missing_experiments": missing_experiments,
        "figures": [],
    }

    for spec_cfg in figure_specs:
        spec = OmegaConf.to_container(spec_cfg, resolve=True)
        if not spec.get("enabled", True):
            continue
        kind = spec["kind"]

        if kind == "structural_data_truth_baseline":
            records = _records_by_refs(records_by_name, spec.get("experiments", []))
            entry = figure_structural_data_truth_baseline(spec, records, figures_output_dir)
        elif kind == "structural_alignment_target_truth":
            records = _records_by_refs(records_by_name, spec.get("experiments", []))
            entry = figure_structural_alignment_target_truth(spec, records, figures_output_dir)
        elif kind == "structural_data_model_truth_comparison":
            records = _records_by_refs(records_by_name, spec.get("experiments", []))
            entry = figure_structural_data_model_truth_comparison(spec, records, figures_output_dir)
        elif kind == "dag_fold_pag_comparison":
            records = _records_by_refs(records_by_name, spec.get("experiments", []))
            entry = figure_dag_fold_pag_comparison(spec, records, figures_output_dir)
        elif kind == "structural_extraction_internal_truth":
            records = _records_by_refs(records_by_name, spec.get("experiments", []))
            entry = figure_structural_extraction_internal_truth(spec, records, figures_output_dir)
        elif kind == "structural_extraction_modes_truth":
            records = _records_by_refs(records_by_name, spec.get("experiments", []))
            entry = figure_structural_extraction_modes_truth(spec, records, figures_output_dir)
        elif kind == "structural_sample_size_robustness":
            records = _records_by_refs(records_by_name, spec.get("experiments", []))
            entry = figure_structural_sample_size_robustness(spec, records, figures_output_dir)
        elif kind == "structural_ci_threshold_robustness":
            records = _records_by_refs(records_by_name, spec.get("experiments", []))
            entry = figure_structural_ci_threshold_robustness(spec, records, figures_output_dir)
        elif kind == "intervention_detection_quality":
            records = _records_by_refs(records_by_name, spec.get("experiments", []))
            entry = figure_intervention_detection_quality(spec, records, figures_output_dir)
        elif kind == "intervention_model_output_detection":
            records = _records_by_refs(records_by_name, spec.get("experiments", []))
            entry = figure_intervention_model_output_detection(spec, records, figures_output_dir)
        elif kind == "intervention_model_pag_consistency":
            records = _records_by_refs(records_by_name, spec.get("experiments", []))
            entry = figure_intervention_model_pag_consistency(spec, records, figures_output_dir)
        elif kind == "intervention_model_pag_consistency_benchmark_expansion":
            records = _records_by_refs(records_by_name, spec.get("experiments", []))
            entry = figure_intervention_model_pag_consistency_benchmark_expansion(
                spec,
                records,
                figures_output_dir,
            )
        elif kind == "intervention_aligned_proxy_calibration":
            records = _records_by_refs(records_by_name, spec.get("experiments", []))
            entry = figure_intervention_aligned_proxy_calibration(spec, records, figures_output_dir)
        elif kind == "effect_calibration":
            records = _records_by_refs(records_by_name, spec.get("experiments", []))
            entry = figure_intervention_effect_calibration(spec, records, figures_output_dir)
        elif kind == "structure_intervention_coupling":
            records = _records_by_refs(records_by_name, spec.get("experiments", []))
            entry = figure_structure_intervention_coupling(spec, records, figures_output_dir)
        elif kind == "structural_architecture_robustness":
            records = _records_by_refs(records_by_name, spec.get("experiments", []))
            entry = figure_structural_architecture_robustness(spec, records, figures_output_dir)
        elif kind == "intervention_architecture_robustness":
            records = _records_by_refs(records_by_name, spec.get("experiments", []))
            entry = figure_intervention_architecture_robustness(spec, records, figures_output_dir)
        elif kind == "structural_ci_mismatch_sensitivity":
            proper = _records_by_refs(records_by_name, spec.get("proper_experiments", []))
            mismatch = _records_by_refs(records_by_name, spec.get("mismatch_experiments", []))
            entry = figure_structural_ci_mismatch_sensitivity(spec, proper, mismatch, figures_output_dir)
        elif kind == "structural_node_count_robustness":
            records = _records_by_refs(records_by_name, spec.get("experiments", []))
            entry = figure_structural_node_count_robustness(spec, records, figures_output_dir)
        elif kind == "model_parameter_counts":
            entry = table_model_parameter_counts(spec, project_root, tables_output_dir)
        elif kind == "training_dynamics":
            records = _records_by_refs(records_by_name, spec.get("experiments", []))
            entry = figure_training_dynamics(spec, records, figures_output_dir)
        else:
            raise ValueError(f"Unknown figure kind: {kind}")

        entry["figure_kind"] = kind
        manifest["figures"].append(entry)

    manifest_path = os.path.join(output_root, "figure_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=serialize_stats_value)

    summary_path = os.path.join(output_root, "report_summary.json")
    with open(summary_path, "w") as f:
        json.dump(
            {
                "created_at": datetime.utcnow().isoformat() + "Z",
                "name": str(config.reporting.name),
                "run_if_missing": run_if_missing,
                "force_rerun": force_rerun,
                "pipeline_runs": pipeline_runs,
                "missing_experiments": missing_experiments,
                "figure_manifest": manifest_path,
            },
            f,
            indent=2,
        )

    print(f"Saved thesis figure manifest to {manifest_path}")
    print(f"Saved thesis report summary to {summary_path}")


@hydra.main(version_base="1.3", config_path="config", config_name="reporting_config")
def main(config: DictConfig) -> None:
    run(config)


if __name__ == "__main__":
    main()
