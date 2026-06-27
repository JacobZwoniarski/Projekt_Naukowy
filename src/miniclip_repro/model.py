from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

from .tokenizer import CaptionVocabulary


class ImageEncoder(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        try:
            backbone = models.resnet18(weights=None)
        except TypeError:
            backbone = models.resnet18(pretrained=False)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.projection = nn.Linear(feature_dim, embed_dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.projection(self.backbone(images))


class TextEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        width: int,
        embed_dim: int,
        max_length: int,
        layers: int,
        heads: int,
        ff_dim: int,
        dropout: float,
        pad_id: int,
        eos_id: int,
    ):
        super().__init__()
        self.max_length = max_length
        self.pad_id = pad_id
        self.eos_id = eos_id
        self.token_embedding = nn.Embedding(vocab_size, width, padding_idx=pad_id)
        self.position_embedding = nn.Parameter(torch.empty(max_length, width))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.final_norm = nn.LayerNorm(width)
        self.projection = nn.Linear(width, embed_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.position_embedding, std=0.01)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.shape[1] > self.max_length:
            raise ValueError(f"Expected sequence length <= {self.max_length}, got {token_ids.shape[1]}.")
        padding_mask = token_ids.eq(self.pad_id)
        positions = self.position_embedding[: token_ids.shape[1]].unsqueeze(0)
        hidden = self.token_embedding(token_ids) + positions
        hidden = self.transformer(hidden, src_key_padding_mask=padding_mask)
        hidden = self.final_norm(hidden)

        eos_mask = token_ids.eq(self.eos_id)
        eos_positions = eos_mask.float().argmax(dim=1)
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), eos_positions]
        return self.projection(pooled)


class MiniCLIP(nn.Module):
    def __init__(self, image_encoder: ImageEncoder, text_encoder: TextEncoder, logit_scale_init: float = 1 / 0.07):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init).log())

    @classmethod
    def from_config(cls, config: dict[str, Any], vocab: CaptionVocabulary) -> "MiniCLIP":
        model_cfg = config["model"]
        image_encoder = ImageEncoder(embed_dim=int(model_cfg["embed_dim"]))
        text_encoder = TextEncoder(
            vocab_size=len(vocab),
            width=int(model_cfg["text_width"]),
            embed_dim=int(model_cfg["embed_dim"]),
            max_length=int(model_cfg["max_length"]),
            layers=int(model_cfg["text_layers"]),
            heads=int(model_cfg["text_heads"]),
            ff_dim=int(model_cfg["text_ff_dim"]),
            dropout=float(model_cfg["text_dropout"]),
            pad_id=vocab.pad_id,
            eos_id=vocab.eos_id,
        )
        return cls(image_encoder=image_encoder, text_encoder=text_encoder)

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.image_encoder(images), dim=-1)

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_encoder(token_ids), dim=-1)

    def forward(self, images: torch.Tensor, token_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(images)
        text_features = self.encode_text(token_ids)
        logit_scale = self.logit_scale.exp().clamp(max=100.0)
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        return logits_per_image, logits_per_text


def save_checkpoint(
    path: str | Path,
    model: MiniCLIP,
    config: dict[str, Any],
    vocab: CaptionVocabulary,
    epoch: int,
    metrics: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config,
            "vocab": vocab.to_dict(),
            "epoch": epoch,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: str | Path, map_location: torch.device | str = "cpu") -> tuple[MiniCLIP, CaptionVocabulary, dict[str, Any], dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=map_location)
    vocab = CaptionVocabulary.from_dict(checkpoint["vocab"])
    config = checkpoint["config"]
    model = MiniCLIP.from_config(config, vocab)
    model.load_state_dict(checkpoint["model_state"])
    return model, vocab, config, checkpoint

