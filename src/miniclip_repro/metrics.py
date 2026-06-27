from __future__ import annotations

from typing import Iterable

import torch


def retrieval_recall_at_k(
    similarity: torch.Tensor,
    text_to_image: torch.Tensor,
    ks: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    """Compute CLIP-style image-to-text and text-to-image recall.

    similarity has shape [num_images, num_texts]. text_to_image[j] is the
    correct image index for text j. An image can have multiple valid captions.
    """
    if similarity.ndim != 2:
        raise ValueError("similarity must be a 2D tensor.")
    num_images, num_texts = similarity.shape
    if text_to_image.shape[0] != num_texts:
        raise ValueError("text_to_image length must match the number of text embeddings.")

    ks = tuple(sorted(int(k) for k in ks))
    max_k = min(max(ks), max(num_images, num_texts))
    text_to_image = text_to_image.cpu()

    image_top = similarity.cpu().topk(k=min(max_k, num_texts), dim=1).indices
    text_top = similarity.cpu().t().topk(k=min(max_k, num_images), dim=1).indices

    results: dict[str, float] = {}
    for k in ks:
        image_hits = 0
        for image_idx in range(num_images):
            positive_texts = text_to_image.eq(image_idx)
            if positive_texts[image_top[image_idx, : min(k, num_texts)]].any():
                image_hits += 1
        text_hits = text_top[:, : min(k, num_images)].eq(text_to_image[:, None]).any(dim=1).sum().item()
        results[f"text_retrieval_r@{k}"] = 100.0 * image_hits / num_images
        results[f"image_retrieval_r@{k}"] = 100.0 * text_hits / num_texts

    results["mean_r@1"] = 0.5 * (results.get("text_retrieval_r@1", 0.0) + results.get("image_retrieval_r@1", 0.0))
    return results


def format_retrieval_markdown(metrics: dict[str, float], title: str = "Flickr8k retrieval") -> str:
    headers = [
        "Dataset",
        "Text R@1",
        "Text R@5",
        "Text R@10",
        "Image R@1",
        "Image R@5",
        "Image R@10",
    ]
    values = [
        title,
        f"{metrics['text_retrieval_r@1']:.2f}",
        f"{metrics['text_retrieval_r@5']:.2f}",
        f"{metrics['text_retrieval_r@10']:.2f}",
        f"{metrics['image_retrieval_r@1']:.2f}",
        f"{metrics['image_retrieval_r@5']:.2f}",
        f"{metrics['image_retrieval_r@10']:.2f}",
    ]
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            "| " + " | ".join(values) + " |",
            "",
        ]
    )

