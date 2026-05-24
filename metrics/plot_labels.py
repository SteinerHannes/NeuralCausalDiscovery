import re
from typing import Any, Optional


_METRIC_LABELS = {
    "edge_precision": "Adjacency precision",
    "edge_recall": "Adjacency recall",
    "edge_f1": "Edge $F_1$ Score",
    "endpoint_precision": "Endpoint-mark precision",
    "endpoint_recall": "Endpoint-mark recall",
    "endpoint_f1": "Endpoint-mark $F_1$",
    "orientation_accuracy": "Strict directed orientation recall",
    "agreement_rate": "Agreement rate",
    "interventional_agreement": "Interventional agreement",
    "missing_edges": "Missing edges",
    "extra_edges": "Extra edges",
    "correct_edges": "Correct edges",
    "orientation_errors": "Orientation errors",
    "partial_orientations": "Partially oriented edges",
    "correct_orientations": "Correctly oriented edges",
    "shd_strict": "SHD (strict)",
    "shd_partial": "SHD (partial)",
    "shd_edges_only": "Structural Hamming Distance (edges only)",
    "agreement_rate_total": "Agreement rate (confirmed intervention-effect criterion)",
    "agreement_rate_direct": "Agreement rate (directed edges)",
    "mean_abs_ate_error": "Mean absolute ATE error",
    "mean_ks_p": "Mean KS p-value",
    "model_agreement_rate": "Model agreement rate (confirmed intervention-effect criterion)",
    "model_mean_abs_ate_error": "Model mean absolute ATE error",
    "model_mean_ks_p": "Model mean KS p-value",
    "ate_true": "True ATE",
    "ate_est": "DoWhy ATE estimate",
    "model_ate": "Model ATE estimate",
    "ate_error": "DoWhy ATE error",
    "model_ate_error": "Model ATE error",
    "ks_p": "KS p-value",
    "model_ks_p": "Model KS p-value",
    "ate_true_std": "True ATE (standardized)",
    "ate_est_std": "DoWhy ATE estimate (standardized)",
    "model_ate_std": "Model ATE estimate (standardized)",
    "std_mae_output": "Standardized output-effect MAE",
    "effect_present": "Confirmed intervention effect present",
    "model_effect_present": "Model confirmed intervention effect present",
    "model_agreement": "Model agreement (confirmed intervention-effect criterion)",
    "effect_positive_rate_output": "Output confirmed-effect prevalence",
    "pag_precision_output": "PAG precision (confirmed intervention effect, output)",
    "pag_recall_output": "PAG recall (confirmed intervention effect, output)",
    "pag_specificity_output": "PAG specificity (confirmed intervention effect, output)",
    "pag_f1_output": "PAG $F_1$ (confirmed intervention effect, output)",
    "precision": "Precision",
    "recall": "Recall",
    "specificity": "Specificity",
    "f1": "$F_1$ score",
    "pag_miss_rate": "PAG miss rate",
    "pag_overclaim_rate": "PAG overclaim rate",
    "model_precision_output": "Model precision (confirmed intervention effect, output)",
    "model_recall_output": "Model recall (confirmed intervention effect, output)",
    "model_specificity_output": "Model specificity (confirmed intervention effect, output)",
    "model_f1_output": "Model $F_1$ (confirmed intervention effect, output)",
    "train_loss": "Training loss",
    "val_loss": "Validation loss",
    "train_mae": "Training MAE",
    "val_mae": "Validation MAE",
    "train_r2": "Training R2",
    "val_r2": "Validation R2",
    "best_val_loss": "Best validation loss",
    "best_epoch_mean": "Best epoch (mean)",
    "early_stop_rate": "Early-stop rate",
}

_GROUP_KEY_LABELS = {
    "sample_size": "Sample size (n)",
    "baseline_sample_size": "Baseline sample size (n)",
    "alignment_enabled": "Alignment",
    "alignment_target_mode": "Alignment target",
    "extraction_method": "Extraction method",
    "mode": "Mode",
    "model": "Model architecture",
    "dataset": "Dataset",
    "ci_test": "Conditional independence test",
    "ci_threshold": "CI significance threshold",
    "experiment": "Experiment",
}

_GROUP_NAME_LABELS = {
    "io_robustness_linear": "I/O Robustness (Linear SEM)",
    "linear_sem_node_count_robustness": "Linear SEM Node-Count Robustness",
    "templated_sem_dataset_expansion": "Templated SEM Dataset Expansion",
    "templated_sem_model_expansion": "Templated SEM Model Expansion",
    "layered_sparse_node_count_robustness": "Layered Sparse Node-Count Robustness",
}

_SOURCE_LABELS = {
    "performance": "PAG vs DAG reference metrics",
    "pag_vs_pag": "PAG vs PAG metrics",
    "intervention": "Interventional fidelity metrics",
    "dowhy": "DoWhy",
    "model": "Model",
}

_ACRONYM_TOKENS = {
    "ate": "ATE",
    "ci": "CI",
    "dag": "DAG",
    "dowhy": "DoWhy",
    "f1": "$F_1$",
    "ks": "KS",
    "mlp": "MLP",
    "pag": "PAG",
    "sem": "SEM",
    "shd": "SHD",
    "std": "standardized",
}

_GROUP_VALUE_LABELS = {
    "align_on": "Alignment enabled",
    "align_off": "Alignment disabled",
    "on": "Alignment enabled",
    "off": "No alignment",
    "proper": "Proper CI test",
    "mismatch": "Mismatched CI test",
    "activations": "Activations",
    "pre_activations": "Pre-activations",
    "input_gradients": "Input gradients",
    "inputs_outputs": "Inputs/outputs",
    "weights": "Weights (exploratory)",
    "sem_truth": "SEM-target alignment",
    "observable_state": "Observable-state alignment",
    "true": "Yes",
    "false": "No",
}

_EXPERIMENT_LABELS = {
    "linear_sem_io_3_2_shallow_mlp_align": "Linear SEM (I/O 3:2)",
    "linear_sem_io_4_2_shallow_mlp_align": "Linear SEM (I/O 4:2)",
    "linear_sem_io_5_2_shallow_mlp_align": "Linear SEM (I/O 5:2)",
    "linear_sem_nodes_7_shallow_mlp_align": "7 nodes",
    "linear_sem_nodes_7_shallow_mlp_align_observable_state": "7 nodes",
    "linear_sem_nodes_8_shallow_mlp_align": "8 nodes",
    "linear_sem_nodes_8_shallow_mlp_align_observable_state": "8 nodes",
    "linear_sem_nodes_9_shallow_mlp_align": "9 nodes",
    "linear_sem_nodes_9_shallow_mlp_align_observable_state": "9 nodes",
    "linear_sem_nodes_10_shallow_mlp_align": "10 nodes",
    "linear_sem_nodes_10_shallow_mlp_align_observable_state": "10 nodes",
    "chain_linear_sem_shallow_mlp_align_observable_state": "Chain linear SEM",
    "fork_collider_linear_sem_shallow_mlp_align_observable_state": "Fork-collider linear SEM",
    "layered_sparse_linear_sem_shallow_mlp_align_observable_state": "Layered sparse linear SEM",
    "layered_sparse_linear_sem_laplace_shallow_mlp_align_observable_state": "Layered sparse linear SEM (Laplace)",
    "layered_sparse_linear_sem_student_t3_shallow_mlp_align_observable_state": "Layered sparse linear SEM (Student-t)",
    "chain_nonlinear_sem_shallow_mlp_align_observable_state": "Chain nonlinear SEM",
    "fork_collider_nonlinear_sem_shallow_mlp_align_observable_state": "Fork-collider nonlinear SEM",
    "layered_sparse_nonlinear_sem_shallow_mlp_align_observable_state": "Layered sparse nonlinear SEM",
    "layered_sparse_linear_sem_nodes_8_shallow_mlp_align_observable_state": "8 nodes",
    "layered_sparse_linear_sem_nodes_10_shallow_mlp_align_observable_state": "10 nodes",
    "layered_sparse_linear_sem_nodes_12_shallow_mlp_align_observable_state": "12 nodes",
    "layered_sparse_linear_sem_nodes_14_shallow_mlp_align_observable_state": "14 nodes",
    "layered_sparse_nonlinear_sem_nodes_8_shallow_mlp_align_observable_state": "8 nodes",
    "layered_sparse_nonlinear_sem_nodes_10_shallow_mlp_align_observable_state": "10 nodes",
    "layered_sparse_nonlinear_sem_nodes_12_shallow_mlp_align_observable_state": "12 nodes",
    "layered_sparse_nonlinear_sem_nodes_14_shallow_mlp_align_observable_state": "14 nodes",
}

_BOOLEAN_COLUMN_CONTEXT = {
    "effect_present": ("Confirmed intervention effect", "No confirmed intervention effect"),
    "model_effect_present": ("Confirmed intervention effect", "No confirmed intervention effect"),
    "agreement_total": ("Agreement", "Disagreement"),
    "model_agreement": ("Agreement", "Disagreement"),
}


def _humanize_identifier(value: str) -> str:
    text = str(value).strip()
    if not text:
        return text

    if "_" in text:
        parts = [part for part in text.split("_") if part]
    else:
        parts = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text).split()

    formatted = []
    for part in parts:
        token = part.lower()
        if token in _ACRONYM_TOKENS:
            formatted.append(_ACRONYM_TOKENS[token])
        else:
            formatted.append(part.capitalize())
    return " ".join(formatted)


def metric_label(metric: str) -> str:
    return _METRIC_LABELS.get(metric, _humanize_identifier(metric))


def group_key_label(group_key: str) -> str:
    return _GROUP_KEY_LABELS.get(group_key, _humanize_identifier(group_key))


def group_name_label(group_name: str) -> str:
    return _GROUP_NAME_LABELS.get(group_name, _humanize_identifier(group_name))


def source_label(source: str) -> str:
    return _SOURCE_LABELS.get(source, _humanize_identifier(source))


def metric_axis_label() -> str:
    return "Metric value"


def group_value_label(value: Any, group_key: Optional[str] = None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, bool):
        if group_key == "alignment_enabled":
            return "Alignment enabled" if value else "Alignment disabled"
        return "Yes" if value else "No"

    text = str(value).strip()
    lowered = text.lower()
    if lowered in _GROUP_VALUE_LABELS and group_key in {"alignment_enabled", "alignment_target_mode", "mode", "extraction_method"}:
        return _GROUP_VALUE_LABELS[lowered]
    if lowered in ("none", "nan", ""):
        return "N/A"

    if group_key == "experiment":
        if lowered in _EXPERIMENT_LABELS:
            return _EXPERIMENT_LABELS[lowered]
        return _humanize_identifier(text)

    if group_key in {"model", "dataset", "ci_test"}:
        return _humanize_identifier(text)
    return text


def combined_source_metric_title(source: str, metric: str) -> str:
    return f"{source_label(source)}: {metric_label(metric)}"


def boolean_value_label(column: str, value: bool) -> str:
    true_label, false_label = _BOOLEAN_COLUMN_CONTEXT.get(column, ("True", "False"))
    return true_label if bool(value) else false_label
