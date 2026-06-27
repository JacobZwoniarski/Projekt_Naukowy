from __future__ import annotations

import torch
from torch.nn import functional as F


def symmetric_clip_loss(logits_per_image: torch.Tensor, logits_per_text: torch.Tensor) -> torch.Tensor:
    if logits_per_image.shape[0] != logits_per_image.shape[1]:
        raise ValueError("CLIP batch loss expects one matching text for each image in the batch.")
    targets = torch.arange(logits_per_image.shape[0], device=logits_per_image.device)
    image_loss = F.cross_entropy(logits_per_image, targets)
    text_loss = F.cross_entropy(logits_per_text, targets)
    return 0.5 * (image_loss + text_loss)

