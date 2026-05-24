#!/usr/bin/env python3
"""Re-run only interventions for an existing experiment output directory.

Usage examples:
  python scripts/rerun_interventions.py linear_sem_shallow_mlp_align
  python scripts/rerun_interventions.py linear_sem_shallow_mlp_align --folds 1,3,5
  python scripts/rerun_interventions.py /abs/path/to/outputs/experiments/my_exp
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from omegaconf import OmegaConf


def _parse_folds(value: Optional[str]) -> Optional[list[int]]:
    if value is None:
        return None
    chunks = [part.strip() for part in value.split(",")]
    chunks = [part for part in chunks if part]
    if not chunks:
        return None
    folds = [int(part) for part in chunks]
    if len(set(folds)) != len(folds):
        raise ValueError(f"Duplicate folds are not allowed: {folds}")
    return folds


def _resolve_experiment_dir(project_root: Path, experiment_arg: str) -> Path:
    direct = Path(experiment_arg).expanduser().resolve()
    if direct.exists():
        cfg = direct / ".hydra" / "config.yaml"
        if cfg.exists():
            return direct
        raise FileNotFoundError(f"Experiment path exists but .hydra/config.yaml is missing: {direct}")

    by_name = (project_root / "outputs" / "experiments" / experiment_arg).resolve()
    cfg = by_name / ".hydra" / "config.yaml"
    if by_name.exists() and cfg.exists():
        return by_name

    raise FileNotFoundError(
        "Could not resolve experiment directory. "
        f"Tried: {direct} and {by_name}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run only interventions for an already completed experiment."
    )
    parser.add_argument(
        "experiment",
        help="Experiment name under outputs/experiments or absolute/relative path to experiment dir.",
    )
    parser.add_argument(
        "--folds",
        default=None,
        help="Optional comma-separated folds (e.g. 1,3,5). Default: all folds from experiment config.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.append(str(project_root))

    # Optional local causality-lab checkout next to this repo.
    causality_lab_root = project_root.parent / "causality-lab"
    if causality_lab_root.exists() and str(causality_lab_root) not in sys.path:
        sys.path.append(str(causality_lab_root))

    experiment_dir = _resolve_experiment_dir(project_root, args.experiment)
    cfg_path = experiment_dir / ".hydra" / "config.yaml"
    cfg = OmegaConf.load(cfg_path)

    folds = _parse_folds(args.folds)
    if folds is not None:
        cfg.training.test_folds = folds

    # interventions.run resolves experiment root from cwd/fallback; run from the experiment dir.
    old_cwd = Path.cwd()
    os.chdir(experiment_dir)
    try:
        from interventions import run as interventions_run

        print(f"Re-running interventions in: {experiment_dir}")
        print(f"Using config: {cfg_path}")
        print(f"Folds: {list(cfg.training.test_folds)}")
        interventions_run(cfg)
        print("Done.")
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
