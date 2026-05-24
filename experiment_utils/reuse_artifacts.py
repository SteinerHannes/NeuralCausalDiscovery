import json
import os
import shutil
from datetime import datetime
from typing import Dict, List, Tuple

import hydra
from omegaconf import DictConfig


def _is_same_target(dst: str, src: str) -> bool:
    if not os.path.exists(dst):
        return False
    try:
        return os.path.samefile(dst, src)
    except OSError:
        return False


def _resolve_source_root(reuse_cfg: DictConfig) -> str:
    source_path = reuse_cfg.get("source_path")
    if source_path:
        if os.path.isabs(source_path):
            return source_path
        return os.path.join(hydra.utils.get_original_cwd(), source_path)

    source_experiment = reuse_cfg.get("source_experiment")
    if source_experiment:
        return os.path.join(
            hydra.utils.get_original_cwd(),
            "outputs",
            "experiments",
            source_experiment,
        )

    raise ValueError("reuse.enabled=true requires either reuse.source_path or reuse.source_experiment.")


def _link_or_copy(src: str, dst: str, mode: str) -> None:
    if mode == "symlink":
        os.symlink(src, dst)
        return
    if mode == "copy":
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)
        return
    if mode == "hardlink":
        if os.path.isdir(src):
            raise ValueError(f"Hardlink mode does not support directories: {src}")
        os.link(src, dst)
        return
    raise ValueError(f"Unsupported reuse mode: {mode}")


def bootstrap_reuse_artifacts(config: DictConfig, experiment_root: str) -> Dict:
    reuse_cfg = config.get("reuse")
    if not reuse_cfg or not bool(reuse_cfg.get("enabled", False)):
        return {"enabled": False}

    mode = str(reuse_cfg.get("mode", "symlink"))
    required_fold_files = list(reuse_cfg.get("required_fold_files", []))
    if not required_fold_files:
        raise ValueError("reuse.required_fold_files must not be empty when reuse.enabled=true.")

    source_root = _resolve_source_root(reuse_cfg)
    if not os.path.isdir(source_root):
        raise ValueError(f"Reuse source root does not exist: {source_root}")

    folds = list(config.training.test_folds)
    missing: List[Tuple[int, str, str]] = []
    for fold in folds:
        source_fold_dir = os.path.join(source_root, str(fold))
        if not os.path.isdir(source_fold_dir):
            missing.append((int(fold), "<fold_dir>", source_fold_dir))
            continue
        for rel_path in required_fold_files:
            src = os.path.join(source_fold_dir, rel_path)
            if not os.path.exists(src):
                missing.append((int(fold), rel_path, src))
    if missing:
        sample = ", ".join([f"fold {fold}: {rel_path}" for fold, rel_path, _ in missing[:5]])
        raise ValueError(f"Missing required reuse artifacts ({len(missing)}): {sample}")

    manifest = {
        "enabled": True,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "mode": mode,
        "source_root": source_root,
        "required_fold_files": required_fold_files,
        "folds": [],
    }

    for fold in folds:
        source_fold_dir = os.path.join(source_root, str(fold))
        target_fold_dir = os.path.join(experiment_root, str(fold))
        os.makedirs(target_fold_dir, exist_ok=True)

        fold_manifest = {
            "fold": int(fold),
            "files": [],
        }
        for rel_path in required_fold_files:
            src = os.path.join(source_fold_dir, rel_path)
            dst = os.path.join(target_fold_dir, rel_path)
            dst_parent = os.path.dirname(dst)
            os.makedirs(dst_parent, exist_ok=True)

            if os.path.exists(dst):
                if mode == "copy":
                    if os.path.isdir(src) == os.path.isdir(dst):
                        status = "existing"
                    else:
                        raise ValueError(
                            f"Target already exists and differs from source type for fold {fold}: {dst}"
                        )
                elif _is_same_target(dst, src):
                    status = "existing"
                else:
                    raise ValueError(
                        f"Target already exists and differs from source for fold {fold}: {dst}"
                    )
            else:
                _link_or_copy(src, dst, mode)
                status = "created"

            fold_manifest["files"].append(
                {
                    "relative_path": rel_path,
                    "source": src,
                    "target": dst,
                    "status": status,
                }
            )

        manifest["folds"].append(fold_manifest)

    manifest_file = str(reuse_cfg.get("manifest_file", "reuse_manifest.json"))
    manifest_path = os.path.join(experiment_root, manifest_file)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Reuse bootstrap complete. Manifest written to {manifest_path}")
    return manifest
