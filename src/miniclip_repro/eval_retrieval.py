from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import load_config
from .data import (
    CaptionEvaluationDataset,
    ImageEvaluationDataset,
    collate_images,
    collate_texts,
    load_raw_splits,
)
from .metrics import format_retrieval_markdown, retrieval_recall_at_k
from .model import MiniCLIP, load_checkpoint
from .tokenizer import CaptionVocabulary
from .utils import ensure_dir, select_device, write_csv, write_json


@torch.no_grad()
def encode_images(
    model: MiniCLIP,
    rows: Any,
    config: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    dataset = ImageEvaluationDataset(rows, image_size=int(config["data"]["image_size"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["eval"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        collate_fn=collate_images,
    )
    chunks: list[torch.Tensor] = []
    model.eval()
    for batch in tqdm(loader, desc="Encode images", leave=False):
        images = batch["images"].to(device)
        chunks.append(model.encode_image(images).cpu())
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def encode_captions(
    model: MiniCLIP,
    rows: Any,
    vocab: CaptionVocabulary,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = CaptionEvaluationDataset(rows, vocab=vocab, max_length=int(config["model"]["max_length"]))
    loader = DataLoader(
        dataset,
        batch_size=int(config["eval"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["data"]["num_workers"]),
        collate_fn=collate_texts,
    )
    text_chunks: list[torch.Tensor] = []
    image_index_chunks: list[torch.Tensor] = []
    model.eval()
    for batch in tqdm(loader, desc="Encode captions", leave=False):
        texts = batch["texts"].to(device)
        text_chunks.append(model.encode_text(texts).cpu())
        image_index_chunks.append(batch["image_indices"].cpu())
    return torch.cat(text_chunks, dim=0), torch.cat(image_index_chunks, dim=0)


def evaluate_retrieval(
    model: MiniCLIP,
    rows: Any,
    vocab: CaptionVocabulary,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    image_features = encode_images(model, rows, config, device)
    text_features, text_to_image = encode_captions(model, rows, vocab, config, device)
    similarity = image_features @ text_features.t()
    return retrieval_recall_at_k(similarity, text_to_image, ks=(1, 5, 10))


def write_retrieval_artifacts(metrics: dict[str, float], output_dir: str | Path, split_name: str) -> None:
    output_dir = ensure_dir(output_dir)
    row = {
        "split": split_name,
        "text_retrieval_r@1": f"{metrics['text_retrieval_r@1']:.4f}",
        "text_retrieval_r@5": f"{metrics['text_retrieval_r@5']:.4f}",
        "text_retrieval_r@10": f"{metrics['text_retrieval_r@10']:.4f}",
        "image_retrieval_r@1": f"{metrics['image_retrieval_r@1']:.4f}",
        "image_retrieval_r@5": f"{metrics['image_retrieval_r@5']:.4f}",
        "image_retrieval_r@10": f"{metrics['image_retrieval_r@10']:.4f}",
        "mean_r@1": f"{metrics['mean_r@1']:.4f}",
    }
    write_csv([row], output_dir / "retrieval_table.csv")
    write_json(metrics, output_dir / "retrieval_metrics.json")
    markdown = format_retrieval_markdown(metrics, title=f"Flickr8k {split_name}")
    (output_dir / "retrieval_table.md").write_text(markdown, encoding="utf-8")


def run_retrieval_eval(
    config_path: str | Path,
    checkpoint_path: str | Path,
    output_dir: str | Path | None = None,
    split: str = "test",
    fast_dev_run: bool = False,
    synthetic_data: bool = False,
    device_name: str | None = None,
) -> dict[str, float]:
    model, vocab, checkpoint_config, _checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    config = load_config(config_path)
    config["model"] = checkpoint_config["model"]
    if synthetic_data:
        config["data"]["dataset_id"] = "synthetic"
    device = select_device(device_name)
    model.to(device)
    splits = load_raw_splits(config, fast_dev_run=fast_dev_run, synthetic_data=synthetic_data)
    rows = splits[split]
    metrics = evaluate_retrieval(model, rows, vocab, config, device)
    out_dir = Path(output_dir) if output_dir else Path(checkpoint_path).parent
    write_retrieval_artifacts(metrics, out_dir, split_name=split)
    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Mini-CLIP retrieval on Flickr8k.")
    parser.add_argument("--config", default="configs/flickr8k.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", default="test", choices=["validation", "test"])
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    metrics = run_retrieval_eval(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        split=args.split,
        fast_dev_run=args.fast_dev_run,
        synthetic_data=args.synthetic_data,
        device_name=args.device,
    )
    print(format_retrieval_markdown(metrics))


if __name__ == "__main__":
    main()

