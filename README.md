# Mini-CLIP Reproduction

Laptop-scale reproduction of the main idea from Radford et al. 2021, *Learning Transferable Visual Models From Natural Language Supervision*: train a dual image/text encoder with a symmetric contrastive loss, then evaluate retrieval and prompt-based zero-shot transfer.

The full CLIP setup is far outside the course constraints, so this project uses Flickr8k and a small model trained from scratch.

## One-command run

```bash
uv run python -m miniclip_repro.run_all --config configs/flickr8k.yaml
```

Recommended full experiment:

```bash
uv run python -m miniclip_repro.train --config configs/flickr8k_strong.yaml --output-dir outputs/flickr8k-strong-160-b128
uv run python -m miniclip_repro.eval_retrieval --config configs/flickr8k_strong.yaml --checkpoint outputs/flickr8k-strong-160-b128/checkpoint_last.pt --output-dir outputs/flickr8k-strong-160-b128-last
uv run python -m miniclip_repro.eval_zeroshot --config configs/flickr8k_strong.yaml --checkpoint outputs/flickr8k-strong-160-b128/checkpoint_best.pt --output-dir outputs/flickr8k-strong-160-b128
```

Quick smoke run:

```bash
uv run python -m miniclip_repro.run_all --config configs/flickr8k.yaml --fast-dev-run
```

The run writes checkpoints, metrics and tables under `outputs/<run_id>/`.

## Current headline result

The strongest laptop-scale configuration is `configs/flickr8k_strong.yaml`: 160 px images, batch size 128, light image augmentation, 50 epochs, warmup and cosine learning-rate decay. It trains from scratch, without ImageNet or CLIP-pretrained weights, and finished in 2142.9 seconds on Apple MPS.

Flickr8k test retrieval with `checkpoint_last.pt`:

| Split | Text R@1 | Text R@5 | Text R@10 | Image R@1 | Image R@5 | Image R@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flickr8k test | 4.10 | 14.90 | 20.50 | 3.84 | 14.02 | 21.52 |

CIFAR-10 prompt ablation with `checkpoint_best.pt`:

| Prompt variant | Accuracy |
| --- | ---: |
| Class name only | 13.70 |
| `a photo of a {label}` | 16.00 |
| Prompt ensemble | 14.20 |

The result is strongest as an image-text retrieval reproduction. Zero-shot transfer is above random chance but remains a limitation of training a small CLIP-like model from scratch on Flickr8k.

## Main artifacts

- `retrieval_table.csv` and `retrieval_table.md`: Flickr8k image-text retrieval, adapting CLIP Table 13 style to this smaller dataset.
- `prompt_ablation.csv` and `prompt_ablation.png`: CIFAR-10 zero-shot prompt ablation, adapting the prompt-engineering comparison from CLIP Figure 4.
- `metrics.json`: training loss, validation retrieval and final artifact locations.

## Useful commands

```bash
uv run python -m miniclip_repro.train --config configs/flickr8k.yaml
uv run python -m miniclip_repro.eval_retrieval --config configs/flickr8k.yaml --checkpoint outputs/<run_id>/checkpoint_best.pt
uv run python -m miniclip_repro.eval_zeroshot --config configs/flickr8k.yaml --checkpoint outputs/<run_id>/checkpoint_best.pt
uv run pytest
```

For local code smoke tests without downloading Flickr8k:

```bash
uv run python -m miniclip_repro.run_all --config configs/flickr8k.yaml --fast-dev-run --synthetic-data --skip-zeroshot
```
