# Mini-CLIP Reproduction

Laptop-scale reproduction of the main idea from Radford et al. 2021, *Learning Transferable Visual Models From Natural Language Supervision*: train a dual image/text encoder with a symmetric contrastive loss, then evaluate retrieval and prompt-based zero-shot transfer.

The full CLIP setup is far outside the course constraints, so this project uses Flickr8k and a small model trained from scratch.

## One-command run

```bash
uv run python -m miniclip_repro.run_all --config configs/flickr8k.yaml
```

Quick smoke run:

```bash
uv run python -m miniclip_repro.run_all --config configs/flickr8k.yaml --fast-dev-run
```

The run writes checkpoints, metrics and tables under `outputs/<run_id>/`.

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

