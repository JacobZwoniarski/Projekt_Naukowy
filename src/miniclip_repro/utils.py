from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def select_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def make_run_dir(output_root: str | Path, prefix: str = "run") -> Path:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(output_root) / f"{prefix}-{timestamp}"
    counter = 1
    while run_dir.exists():
        run_dir = Path(output_root) / f"{prefix}-{timestamp}-{counter}"
        counter += 1
    run_dir.mkdir(parents=True)
    return run_dir


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("Cannot write an empty CSV.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def package_versions() -> dict[str, str]:
    versions = {"python": ".".join(map(str, os.sys.version_info[:3])), "torch": torch.__version__}
    try:
        import torchvision

        versions["torchvision"] = torchvision.__version__
    except Exception:
        versions["torchvision"] = "unavailable"
    try:
        import datasets

        versions["datasets"] = datasets.__version__
    except Exception:
        versions["datasets"] = "unavailable"
    return versions


def to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}

