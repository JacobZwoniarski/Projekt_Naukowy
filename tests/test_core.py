from __future__ import annotations

import torch

from miniclip_repro.losses import symmetric_clip_loss
from miniclip_repro.metrics import retrieval_recall_at_k
from miniclip_repro.tokenizer import CaptionVocabulary, VocabularySpec


def test_vocabulary_is_deterministic() -> None:
    captions = ["A dog runs", "A cat runs", "A dog jumps"]
    vocab_a = CaptionVocabulary.build(captions, VocabularySpec(min_freq=1, max_size=20))
    vocab_b = CaptionVocabulary.build(captions, VocabularySpec(min_freq=1, max_size=20))

    assert vocab_a.token_to_id == vocab_b.token_to_id
    assert vocab_a.encode("a dog", max_length=6) == vocab_b.encode("a dog", max_length=6)
    assert vocab_a.encode("unknown token", max_length=5)[1] == vocab_a.unk_id


def test_symmetric_clip_loss_is_finite() -> None:
    logits = torch.tensor([[4.0, 1.0], [0.5, 3.0]])
    loss = symmetric_clip_loss(logits, logits.t())

    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_retrieval_recall_perfect_embeddings() -> None:
    image_features = torch.eye(3)
    text_features = torch.vstack([torch.eye(3), torch.eye(3)])
    text_to_image = torch.tensor([0, 1, 2, 0, 1, 2])
    similarity = image_features @ text_features.t()

    metrics = retrieval_recall_at_k(similarity, text_to_image, ks=(1, 2, 3))

    assert metrics["text_retrieval_r@1"] == 100.0
    assert metrics["image_retrieval_r@1"] == 100.0
    assert metrics["mean_r@1"] == 100.0

