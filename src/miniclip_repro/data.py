from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .tokenizer import CaptionVocabulary


CAPTION_COLUMNS = ("caption_0", "caption_1", "caption_2", "caption_3", "caption_4")


def row_captions(row: dict[str, Any]) -> list[str]:
    return [str(row[column]) for column in CAPTION_COLUMNS if column in row and row[column]]


def all_captions(rows: Sequence[dict[str, Any]]) -> list[str]:
    captions: list[str] = []
    for row in rows:
        captions.extend(row_captions(row))
    return captions


def build_image_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(image_size + 16),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
        ]
    )


def _to_rgb(image: Image.Image) -> Image.Image:
    return image.convert("RGB") if image.mode != "RGB" else image


def _limit_split(rows: Any, limit: int | None) -> Any:
    if limit is None:
        return rows
    limit = min(limit, len(rows))
    if hasattr(rows, "select"):
        return rows.select(range(limit))
    return list(rows)[:limit]


def load_raw_splits(config: dict[str, Any], fast_dev_run: bool = False, synthetic_data: bool = False) -> dict[str, Any]:
    if synthetic_data or config["data"].get("dataset_id") == "synthetic":
        splits = make_synthetic_splits()
    else:
        from datasets import load_dataset

        cache_dir = Path(config["data"]["cache_dir"])
        dataset = load_dataset(config["data"]["dataset_id"], cache_dir=str(cache_dir))
        validation_key = "validation" if "validation" in dataset else "dev"
        splits = {
            "train": dataset["train"],
            "validation": dataset[validation_key],
            "test": dataset["test"],
        }

    if fast_dev_run:
        limits = config["eval"]["fast_dev"]
        splits = {
            "train": _limit_split(splits["train"], int(limits["train_examples"])),
            "validation": _limit_split(splits["validation"], int(limits["validation_examples"])),
            "test": _limit_split(splits["test"], int(limits["test_examples"])),
        }
    return splits


def make_synthetic_splits() -> dict[str, list[dict[str, Any]]]:
    palette = [
        ("red", (220, 48, 48)),
        ("green", (40, 170, 80)),
        ("blue", (45, 95, 210)),
        ("yellow", (230, 200, 40)),
        ("purple", (135, 75, 180)),
        ("orange", (230, 130, 40)),
    ]

    def make_rows(start: int, count: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for offset in range(count):
            idx = start + offset
            color_name, rgb = palette[idx % len(palette)]
            image = Image.new("RGB", (96, 96), rgb)
            captions = [
                f"a {color_name} square",
                f"plain {color_name} image",
                f"synthetic sample {idx}",
                f"{color_name} color patch",
                f"small {color_name} picture",
            ]
            rows.append({"image": image, **{column: caption for column, caption in zip(CAPTION_COLUMNS, captions)}})
        return rows

    return {"train": make_rows(0, 48), "validation": make_rows(48, 16), "test": make_rows(64, 16)}


class Flickr8kTrainingDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        vocab: CaptionVocabulary,
        image_size: int,
        max_length: int,
    ):
        self.rows = rows
        self.vocab = vocab
        self.transform = build_image_transform(image_size)
        self.max_length = max_length
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[index]
        captions = row_captions(row)
        caption = captions[(self.epoch + index) % len(captions)]
        image = self.transform(_to_rgb(row["image"]))
        token_ids = torch.tensor(self.vocab.encode(caption, self.max_length), dtype=torch.long)
        return {"images": image, "texts": token_ids}


class ImageEvaluationDataset(Dataset):
    def __init__(self, rows: Sequence[dict[str, Any]], image_size: int):
        self.rows = rows
        self.transform = build_image_transform(image_size)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = self.transform(_to_rgb(self.rows[index]["image"]))
        return {"images": image, "indices": torch.tensor(index, dtype=torch.long)}


class CaptionEvaluationDataset(Dataset):
    def __init__(self, rows: Sequence[dict[str, Any]], vocab: CaptionVocabulary, max_length: int):
        self.examples: list[tuple[int, str]] = []
        self.vocab = vocab
        self.max_length = max_length
        for image_index, row in enumerate(rows):
            for caption in row_captions(row):
                self.examples.append((image_index, caption))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_index, caption = self.examples[index]
        token_ids = torch.tensor(self.vocab.encode(caption, self.max_length), dtype=torch.long)
        return {"texts": token_ids, "image_indices": torch.tensor(image_index, dtype=torch.long)}


def collate_pairs(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "images": torch.stack([item["images"] for item in batch]),
        "texts": torch.stack([item["texts"] for item in batch]),
    }


def collate_images(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "images": torch.stack([item["images"] for item in batch]),
        "indices": torch.stack([item["indices"] for item in batch]),
    }


def collate_texts(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "texts": torch.stack([item["texts"] for item in batch]),
        "image_indices": torch.stack([item["image_indices"] for item in batch]),
    }

