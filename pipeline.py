import os
import sys
import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import hydra
from omegaconf import DictConfig

from train import run as train_run
from test import run as test_run
from extraction import run as extraction_run
from causaldiscovery import run as causaldiscovery_run
from pag_cleanup import run as pag_cleanup_run
from interventions import run as interventions_run
from experiment_utils.device_utils import describe_torch_runtime, resolve_torch_device
from experiment_utils.reuse_artifacts import bootstrap_reuse_artifacts


def _run_step(step_idx: int, total_steps: int, label: str, enabled: bool, fn, config: DictConfig) -> None:
    print(f"\n=== Step {step_idx}/{total_steps}: {label} ===")
    if not enabled:
        print(f"Skipped {label} (disabled in config).")
        return
    print(f"Starting {label}...")
    fn(config)
    print(f"Finished {label}.")


def _resolve_device_summary(config: DictConfig) -> dict:
    requested = str(config.training.get("device", "auto"))
    try:
        resolved = resolve_torch_device(requested).type
        error = None
    except Exception as exc:
        resolved = None
        error = str(exc)
    return {
        "requested": requested,
        "resolved": resolved,
        "error": error,
    }


def _collect_runtime_report(config: DictConfig, output_dir: str) -> dict:
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": output_dir,
        "python": {
            "version": sys.version.replace("\n", " "),
            "executable": sys.executable,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "torch_runtime": describe_torch_runtime(),
        "device": _resolve_device_summary(config)
    }
    return report


def _log_runtime_report(config: DictConfig, output_dir: str) -> None:
    report = _collect_runtime_report(config, output_dir)
    print("=== Runtime Environment ===")
    print(
        f"Python={report['python']['version']} "
        f"platform={report['python']['platform']} "
        f"machine={report['python']['machine']}"
    )
    print(
        f"Torch={report['torch_runtime']['torch_version']} "
        f"mps_built={report['torch_runtime']['mps_built']} "
        f"mps_available={report['torch_runtime']['mps_available']} "
        f"cuda_available={report['torch_runtime']['cuda_available']}"
    )
    print(
        f"Device requested={report['device']['requested']} "
        f"resolved={report['device']['resolved']} "
        f"error={report['device']['error']}"
    )

    runtime_file = os.path.join(output_dir, "runtime_environment.json")
    with open(runtime_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved runtime environment to {runtime_file}")


@hydra.main(version_base="1.3", config_path="config", config_name="experiments/linear_sem_shallow_mlp_align.yaml")
def main(config: DictConfig) -> None:
    output_dir = os.getcwd()
    print(f"Pipeline output directory: {output_dir}")
    _log_runtime_report(config, output_dir)
    bootstrap_reuse_artifacts(config, output_dir)

    pipeline = config.get("pipeline", {})
    steps = [
        ("train", pipeline.get("run_train", True), train_run),
        ("test", pipeline.get("run_test", True), test_run),
        ("extraction", pipeline.get("run_extraction", True), extraction_run),
        ("causal discovery", pipeline.get("run_causaldiscovery", True), causaldiscovery_run),
        ("PAG cleanup", pipeline.get("run_pag_cleanup", True), pag_cleanup_run),
        ("interventions", pipeline.get("run_interventions", True), interventions_run),
    ]
    total_steps = len(steps)
    for step_idx, (label, enabled, fn) in enumerate(steps, start=1):
        _run_step(step_idx, total_steps, label, enabled, fn, config)


if __name__ == "__main__":
    main()
