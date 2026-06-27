from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import apply_fast_dev_overrides, load_config, save_config
from .data import Flickr8kTrainingDataset, all_captions, collate_pairs, load_raw_splits
from .eval_retrieval import evaluate_retrieval
from .losses import symmetric_clip_loss
from .metrics import format_retrieval_markdown
from .model import MiniCLIP, save_checkpoint
from .tokenizer import CaptionVocabulary, VocabularySpec
from .utils import (
    ensure_dir,
    make_run_dir,
    package_versions,
    seed_worker,
    select_device,
    set_seed,
    to_device,
    write_json,
)


def build_train_loader(
    rows: Any,
    vocab: CaptionVocabulary,
    config: dict[str, Any],
    seed: int,
) -> tuple[Flickr8kTrainingDataset, DataLoader]:
    dataset = Flickr8kTrainingDataset(
        rows,
        vocab=vocab,
        image_size=int(config["data"]["image_size"]),
        max_length=int(config["model"]["max_length"]),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=True,
        num_workers=int(config["data"]["num_workers"]),
        collate_fn=collate_pairs,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        drop_last=True,
    )
    return dataset, loader


def train_model(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    fast_dev_run: bool = False,
    synthetic_data: bool = False,
    device_name: str | None = None,
) -> tuple[Path, Path]:
    config = load_config(config_path)
    if fast_dev_run:
        config = apply_fast_dev_overrides(config)
    if synthetic_data:
        config["data"]["dataset_id"] = "synthetic"
        config["data"]["num_workers"] = 0

    run_dir = ensure_dir(output_dir) if output_dir else make_run_dir(config["output_dir"], prefix="miniclip")
    save_config(config, run_dir / "config.yaml")
    write_json({"fast_dev_run": fast_dev_run, "synthetic_data": synthetic_data}, run_dir / "run_flags.json")

    seed = int(config["seed"])
    set_seed(seed)
    device = select_device(device_name)
    start_time = time.time()

    print(f"Loading dataset: {config['data']['dataset_id']}")
    splits = load_raw_splits(config, fast_dev_run=fast_dev_run, synthetic_data=synthetic_data)
    vocab = CaptionVocabulary.build(
        all_captions(splits["train"]),
        VocabularySpec(
            min_freq=int(config["model"]["min_token_freq"]),
            max_size=int(config["model"]["max_vocab_size"]),
        ),
    )
    vocab.save(run_dir / "vocab.json")

    model = MiniCLIP.from_config(config, vocab).to(device)
    train_dataset, train_loader = build_train_loader(splits["train"], vocab, config, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    total_steps = max(1, int(config["training"]["epochs"]) * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    history: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_checkpoint = run_dir / "checkpoint_best.pt"
    last_checkpoint = run_dir / "checkpoint_last.pt"

    for epoch in range(int(config["training"]["epochs"])):
        train_dataset.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        step_count = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=True)
        for step, batch in enumerate(progress, start=1):
            batch = to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            logits_per_image, logits_per_text = model(batch["images"], batch["texts"])
            loss = symmetric_clip_loss(logits_per_image, logits_per_text)
            loss.backward()
            grad_clip = float(config["training"].get("grad_clip_norm", 0.0))
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            step_count += 1
            if step % int(config["training"]["log_interval"]) == 0 or step == 1:
                progress.set_postfix(loss=f"{running_loss / step_count:.4f}")

        avg_loss = running_loss / max(step_count, 1)
        val_metrics = evaluate_retrieval(model, splits["validation"], vocab, config, device)
        epoch_metrics = {"epoch": epoch + 1, "train_loss": avg_loss, **val_metrics}
        history.append(epoch_metrics)
        print(format_retrieval_markdown(val_metrics, title=f"Flickr8k validation epoch {epoch + 1}"))

        save_checkpoint(last_checkpoint, model, config, vocab, epoch + 1, epoch_metrics)
        if val_metrics["mean_r@1"] > best_score:
            best_score = val_metrics["mean_r@1"]
            save_checkpoint(best_checkpoint, model, config, vocab, epoch + 1, epoch_metrics)

    elapsed = time.time() - start_time
    summary = {
        "config_path": str(config_path),
        "run_dir": str(run_dir),
        "best_checkpoint": str(best_checkpoint),
        "last_checkpoint": str(last_checkpoint),
        "best_validation_mean_r@1": best_score,
        "history": history,
        "elapsed_seconds": elapsed,
        "device": str(device),
        "versions": package_versions(),
    }
    write_json(summary, run_dir / "metrics.json")
    return run_dir, best_checkpoint


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Mini-CLIP on Flickr8k.")
    parser.add_argument("--config", default="configs/flickr8k.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--fast-dev-run", action="store_true")
    parser.add_argument("--synthetic-data", action="store_true")
    parser.add_argument("--device", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_dir, checkpoint = train_model(
        config_path=args.config,
        output_dir=args.output_dir,
        fast_dev_run=args.fast_dev_run,
        synthetic_data=args.synthetic_data,
        device_name=args.device,
    )
    print(f"Run directory: {run_dir}")
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()

