from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "output_dir": "outputs",
    "data": {
        "dataset_id": "jxie/flickr8k",
        "cache_dir": "data/hf_cache",
        "image_size": 128,
        "num_workers": 0,
    },
    "model": {
        "embed_dim": 256,
        "text_width": 256,
        "text_layers": 2,
        "text_heads": 4,
        "text_ff_dim": 512,
        "text_dropout": 0.1,
        "max_length": 32,
        "min_token_freq": 1,
        "max_vocab_size": 20000,
    },
    "training": {
        "epochs": 6,
        "batch_size": 64,
        "lr": 3e-4,
        "weight_decay": 0.01,
        "grad_clip_norm": 1.0,
        "log_interval": 20,
    },
    "eval": {
        "batch_size": 128,
        "retrieval_batch_size": 256,
        "cifar10_limit": 1000,
        "cifar10_root": "data/cifar10",
        "fast_dev": {
            "train_examples": 64,
            "validation_examples": 16,
            "test_examples": 16,
            "cifar10_limit": 64,
        },
        "prompt_templates": [
            "a photo of a {label}",
            "a blurry photo of a {label}",
            "a close-up photo of a {label}",
            "a small photo of a {label}",
        ],
    },
}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        user_config = yaml.safe_load(handle) or {}
    return deep_update(DEFAULT_CONFIG, user_config)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def apply_fast_dev_overrides(config: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(config)
    cfg["training"]["epochs"] = 1
    cfg["training"]["batch_size"] = min(int(cfg["training"]["batch_size"]), 16)
    cfg["eval"]["batch_size"] = min(int(cfg["eval"]["batch_size"]), 32)
    cfg["eval"]["cifar10_limit"] = int(cfg["eval"]["fast_dev"]["cifar10_limit"])
    return cfg

