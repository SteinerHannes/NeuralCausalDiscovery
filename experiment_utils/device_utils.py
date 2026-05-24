from typing import Optional

import torch


def resolve_torch_device(requested: Optional[str] = "auto") -> torch.device:
    req = str(requested or "auto").lower()
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()

    if req == "auto":
        if has_mps:
            return torch.device("mps")
        if has_cuda:
            return torch.device("cuda")
        return torch.device("cpu")

    if req == "mps":
        if not has_mps:
            raise RuntimeError(
                "training.device is set to 'mps', but MPS is unavailable in this environment."
            )
        return torch.device("mps")

    if req == "cuda":
        if not has_cuda:
            raise RuntimeError(
                "training.device is set to 'cuda', but CUDA is unavailable in this environment."
            )
        return torch.device("cuda")

    if req == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unknown training.device value: {requested}")


def describe_torch_runtime() -> dict:
    return {
        "torch_version": torch.__version__,
        "mps_built": bool(torch.backends.mps.is_built()),
        "mps_available": bool(torch.backends.mps.is_available()),
        "cuda_available": bool(torch.cuda.is_available()),
    }


def print_torch_runtime(device: torch.device) -> None:
    info = describe_torch_runtime()
    print(
        "Torch runtime: "
        f"version={info['torch_version']} "
        f"device={device.type} "
        f"mps_built={info['mps_built']} "
        f"mps_available={info['mps_available']} "
        f"cuda_available={info['cuda_available']}"
    )
