from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from tqdm.auto import tqdm

from .config import load_config
from .data import build_image_transform
from .model import MiniCLIP, load_checkpoint
from .tokenizer import CaptionVocabulary
from .utils import ensure_dir, select_device, write_csv, write_json


CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def _article(label: str) -> str:
    return "an" if label[0].lower() in {"a", "e", "i", "o", "u"} else "a"


def _format_prompt(template: str, label: str) -> str:
    return template.format(label=label, article=_article(label))


@torch.no_grad()
def encode_prompt_set(
    model: MiniCLIP,
    vocab: CaptionVocabulary,
    prompts_by_class: list[list[str]],
    config: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    class_features: list[torch.Tensor] = []
    max_length = int(config["model"]["max_length"])
    model.eval()
    for prompts in prompts_by_class:
        tokens = torch.tensor([vocab.encode(prompt, max_length) for prompt in prompts], dtype=torch.long, device=device)
        prompt_features = model.encode_text(tokens)
        prompt_features = torch.nn.functional.normalize(prompt_features.mean(dim=0, keepdim=True), dim=-1)
        class_features.append(prompt_features.squeeze(0).cpu())
    return torch.stack(class_features, dim=0)


@torch.no_grad()
def evaluate_prompt_accuracy(
    model: MiniCLIP,
    class_features: torch.Tensor,
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    device: torch.device,
) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    class_features = class_features.to(device)
    correct = 0
    total = 0
    model.eval()
    for images, targets in tqdm(loader, desc="CIFAR-10 zero-shot", leave=False):
        images = images.to(device)
        targets = targets.to(device)
        image_features = model.encode_image(images)
        logits = model.logit_scale.exp().clamp(max=100.0) * image_features @ class_features.t()
        predictions = logits.argmax(dim=1)
        correct += predictions.eq(targets).sum().item()
        total += targets.numel()
    return 100.0 * correct / max(total, 1)


def build_prompt_variants(config: dict[str, Any]) -> dict[str, list[list[str]]]:
    return {
        "class_name_only": [[label] for label in CIFAR10_CLASSES],
        "photo_prompt": [[f"a photo of {_article(label)} {label}"] for label in CIFAR10_CLASSES],
        "prompt_ensemble": [
            [_format_prompt(template, label) for template in config["eval"]["prompt_templates"]]
            for label in CIFAR10_CLASSES
        ],
    }


def write_prompt_plot(rows: list[dict[str, str]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    names = [row["prompt_variant"] for row in rows]
    accuracies = [float(row["accuracy"]) for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(names, accuracies, color=["#4c78a8", "#f58518", "#54a24b"])
    ax.set_ylabel("Zero-shot accuracy (%)")
    ax.set_title("CIFAR-10 prompt ablation")
    ax.set_ylim(0, max(100.0, max(accuracies) + 5.0))
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(output_dir / "prompt_ablation.png", dpi=160)
    plt.close(fig)


def run_zeroshot_eval(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    fast_dev_run: bool = False,
    device_name: str | None = None,
) -> list[dict[str, str]]:
    model, vocab, checkpoint_config, _checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = load_config(config_path)
    config["model"] = checkpoint_config["model"]
    device = select_device(device_name)
    model.to(device)

    transform = build_image_transform(int(config["data"]["image_size"]))
    cifar = CIFAR10(root=str(config["eval"]["cifar10_root"]), train=False, transform=transform, download=True)
    limit = int(config["eval"]["fast_dev"]["cifar10_limit"] if fast_dev_run else config["eval"]["cifar10_limit"])
    if limit > 0 and limit < len(cifar):
        cifar = Subset(cifar, range(limit))

    prompt_variants = build_prompt_variants(config)
    rows: list[dict[str, str]] = []
    for name, prompts_by_class in prompt_variants.items():
        class_features = encode_prompt_set(model, vocab, prompts_by_class, config, device)
        accuracy = evaluate_prompt_accuracy(
            model,
            class_features,
            cifar,
            batch_size=int(config["eval"]["batch_size"]),
            device=device,
        )
        rows.append({"prompt_variant": name, "accuracy": f"{accuracy:.4f}", "num_examples": str(len(cifar))})

    out_dir = ensure_dir(output_dir if output_dir else Path(checkpoint_path).parent)
    write_csv(rows, out_dir / "prompt_ablation.csv")
    write_json({"rows": rows}, out_dir / "prompt_ablation.json")
    write_prompt_plot(rows, out_dir)
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate CIFAR-10 zero-shot prompt variants.")
    parser.add_argument("--config", default="configs/flickr8k.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    rows = run_zeroshot_eval(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        fast_dev_run=args.fast_dev_run,
        device_name=args.device,
    )
    for row in rows:
        print(f"{row['prompt_variant']}: {row['accuracy']}%")


if __name__ == "__main__":
    main()

