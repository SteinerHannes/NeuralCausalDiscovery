import os
from typing import Callable, Dict, List, Sequence, Tuple

import pandas as pd

from models.BottleneckMLP import BottleneckMLP
from models.DeepMLP import DeepMLP
from models.GatedMLP import GatedMLP
from models.MLPModel import MLPModel
from models.ResidualMLP import ResidualMLP
from models.ShallowMLP import ShallowMLP


DEFAULT_MODEL_PARAMETER_TABLE_ORDER = [
    "ShallowMLP",
    "DeepMLP",
    "BottleneckMLP",
    "GatedMLP",
    "ResidualMLP",
    "TabularMLP",
]


def _make_shallow(input_size: int, output_size: int):
    return ShallowMLP(
        input_size=input_size,
        hidden1_size=32,
        hidden2_size=32,
        output_size=output_size,
        activation="relu",
    )


def _make_deep(input_size: int, output_size: int):
    return DeepMLP(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=(64, 64, 64, 64),
        activation="gelu",
        dropout=0.1,
        use_batch_norm=False,
        bias=True,
    )


def _make_bottleneck(input_size: int, output_size: int):
    return BottleneckMLP(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=(128, 16, 128),
        activation="silu",
        dropout=0.0,
        use_batch_norm=False,
        bias=True,
    )


def _make_gated(input_size: int, output_size: int):
    return GatedMLP(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=(64, 64, 64, 64),
        activation="silu",
        dropout=0.1,
        use_batch_norm=False,
        bias=True,
    )


def _make_residual(input_size: int, output_size: int):
    return ResidualMLP(
        input_size=input_size,
        output_size=output_size,
        hidden_size=128,
        num_blocks=3,
        activation="gelu",
        dropout=0.0,
        bias=True,
    )


def _make_tabular(input_size: int, output_size: int):
    return MLPModel(
        input_size=input_size,
        output_size=output_size,
        hidden_sizes=(256, 128, 64),
        activation="silu",
        dropout=0.1,
        use_batch_norm=True,
        bias=True,
        model_id="TabularMLP",
    )


MODEL_PARAMETER_SPECS: Dict[str, dict] = {
    "ShallowMLP": {
        "model_family": "Shallow MLP",
        "hidden_structure": "32--32",
        "components": "ReLU",
        "factory": _make_shallow,
    },
    "DeepMLP": {
        "model_family": "Deep MLP",
        "hidden_structure": "64--64--64--64",
        "components": "GELU, dropout 0.1",
        "factory": _make_deep,
    },
    "BottleneckMLP": {
        "model_family": "Bottleneck MLP",
        "hidden_structure": "128--16--128",
        "components": "SiLU",
        "factory": _make_bottleneck,
    },
    "GatedMLP": {
        "model_family": "Gated MLP",
        "hidden_structure": "64--64--64--64",
        "components": "Gated blocks, SiLU, dropout 0.1",
        "factory": _make_gated,
    },
    "ResidualMLP": {
        "model_family": "Residual MLP",
        "hidden_structure": "128 units, 3 residual blocks",
        "components": "Residual blocks, GELU",
        "factory": _make_residual,
    },
    "TabularMLP": {
        "model_family": "Tabular MLP",
        "hidden_structure": "256--128--64",
        "components": "SiLU, batch normalization, dropout 0.1",
        "factory": _make_tabular,
    },
}


def _count_model_parameters(model) -> Tuple[int, int, int]:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    return int(trainable), int(total), int(total - trainable)


def _model_parameter_rows(
    project_root: str,
    model_order: Sequence[str],
    input_size: int,
    output_size: int,
) -> Tuple[List[dict], List[str]]:
    rows = []
    source_files = [os.path.join(project_root, "reporting", "figures", "table_model_parameter_counts.py")]
    for model_id in model_order:
        if model_id not in MODEL_PARAMETER_SPECS:
            raise ValueError(f"Unknown model_id for parameter table: {model_id}")
        model_spec = MODEL_PARAMETER_SPECS[model_id]
        factory: Callable[[int, int], object] = model_spec["factory"]
        model = factory(int(input_size), int(output_size))
        trainable, total, non_trainable = _count_model_parameters(model)
        rows.append(
            {
                "model_id": model_id,
                "model_family": model_spec["model_family"],
                "input_size": int(input_size),
                "output_size": int(output_size),
                "hidden_structure": model_spec["hidden_structure"],
                "components": model_spec["components"],
                "trainable_parameters": trainable,
                "total_parameters": total,
                "non_trainable_parameters": non_trainable,
            }
        )
    return rows, source_files


def table_model_parameter_counts(spec: dict, project_root: str, output_dir: str) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    table_id = str(spec["id"])
    model_order = list(spec.get("model_order", DEFAULT_MODEL_PARAMETER_TABLE_ORDER))
    input_size = int(spec.get("input_size", 3))
    output_size = int(spec.get("output_size", 3))

    rows, source_files = _model_parameter_rows(
        project_root=project_root,
        model_order=model_order,
        input_size=input_size,
        output_size=output_size,
    )
    if not rows:
        raise ValueError(f"{table_id}: no model parameter rows were generated.")

    csv_path = os.path.join(output_dir, f"{table_id}.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    return {
        "figure_id": table_id,
        "figure_path": None,
        "png_path": None,
        "pdf_path": None,
        "stats_path": None,
        "table_path": csv_path,
        "notes_path": None,
        "row_count": len(rows),
        "matched_n": [],
        "test_used": [],
        "raw_p_values": [],
        "adjusted_q_values": [],
        "axis_limits": [],
        "source_files": sorted(source_files),
        "title_generated": str(spec.get("title", "Model Parameter Counts")),
        "legend_mode": "none",
        "annotation_mode": "none",
        "layout_profile": "table_only",
    }
