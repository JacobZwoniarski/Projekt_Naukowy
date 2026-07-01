# Documentation and Methodology Notes

## 1. Project Goal

This project is a laptop-scale reproduction of the main idea from Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision*. The original CLIP system trains image and text encoders with a contrastive objective on a very large collection of image-text pairs. This implementation uses a smaller Mini-CLIP model trained from scratch on Flickr8k.

The goal is not to match the full CLIP results. The goal is to verify whether the same learning procedure can produce measurable image-text alignment under course-scale compute constraints.

The project contains three main parts:

- image-text data preparation,
- training a dual image/text encoder,
- evaluating image-text retrieval and simple zero-shot transfer.

## 2. Data

The main dataset is `jxie/flickr8k`, loaded through Hugging Face `datasets`. Each example contains one image and up to five captions (`caption_0` ... `caption_4`). The implementation uses `train`, `validation`, and `test` splits. If the dataset exposes `dev` instead of `validation`, the code maps `dev` to validation.

During training, each image is paired with one caption. The caption is selected deterministically from the available captions based on the epoch and example index. This lets the model see different captions for the same image across epochs without adding extra random sampling logic.

Image preprocessing:

- images are converted to RGB,
- the base evaluation transform uses `Resize(image_size + 16)` followed by `CenterCrop(image_size)`,
- `configs/flickr8k_strong.yaml` uses light training augmentation with `RandomResizedCrop` and `RandomHorizontalFlip`,
- normalization uses the image mean and standard deviation commonly used by CLIP.

Text preprocessing:

- tokenization uses a simple regular expression, `[a-z0-9]+`,
- captions are lowercased,
- the vocabulary is built only from training captions,
- sequences contain `<bos>`, `<eos>`, `<pad>`, and `<unk>` special tokens,
- the configured maximum sequence length is 32 tokens.

For technical smoke tests without downloading Flickr8k, the code also supports synthetic data: simple colored-square images with matching captions.

## 3. Model

The model uses a dual-encoder architecture similar to CLIP:

- an image encoder maps images to feature vectors,
- a text encoder maps captions to feature vectors,
- both vectors are projected into a shared embedding space,
- both representations are L2-normalized,
- image-text similarity is computed by scaled dot product.

The image encoder is a `torchvision` ResNet-18 trained from scratch (`weights=None`). Its classification layer is removed and replaced with a linear projection into the shared embedding space.

The text encoder contains:

- token embeddings,
- learned positional embeddings,
- a small `TransformerEncoder`,
- final layer normalization,
- a linear projection into the shared embedding space.

The text representation is taken from the hidden state at the `<eos>` token. In the current configurations, the text Transformer has 2 layers, width 256, 4 attention heads, and feed-forward dimension 512.

## 4. Objective Function

Training uses the symmetric CLIP contrastive loss. For a batch of `N` matched image-caption pairs, the model computes an `N x N` similarity matrix. Diagonal entries are positive pairs; off-diagonal entries are in-batch negatives.

The loss averages two classification directions:

- image-to-text: predict the matching text for each image,
- text-to-image: predict the matching image for each text.

In code, this is:

```text
targets = [0, 1, ..., N-1]
loss = 0.5 * (
    cross_entropy(logits_per_image, targets)
    + cross_entropy(logits_per_text, targets)
)
```

## 5. Training Procedure

Training is configured through YAML files in `configs/`. The base config `configs/flickr8k.yaml` is a quick starting point. The recommended full experiment is `configs/flickr8k_strong.yaml`.

Important `flickr8k_strong` settings:

- image size: 160,
- batch size: 128,
- epochs: 50,
- optimizer: `AdamW`,
- learning rate: `0.0004`,
- weight decay: `0.01`,
- gradient clipping: `1.0`,
- warmup: 250 steps,
- scheduler: warmup plus cosine decay,
- validation monitor: `mean_r@1`.

During training, the run directory stores:

- `config.yaml` with the resolved run configuration,
- `vocab.json` with the caption vocabulary,
- `checkpoint_last.pt`,
- `checkpoint_best.pt`,
- `metrics.json` with training history, elapsed time, device, and package versions.

Randomness is controlled by `seed = 42`. The code sets seeds for `random`, `numpy`, and `torch`, and also sets CUDA seeds when CUDA is available. It disables cuDNN benchmarking and requests deterministic cuDNN behavior where applicable.

## 6. Evaluation

### 6.1 Image-Text Retrieval

The main evaluation checks whether the model retrieves the correct text for an image and the correct image for a text. For the validation or test split, the code encodes all images and all captions, then computes the full similarity matrix.

Reported metrics:

- `Text R@1`, `Text R@5`, `Text R@10`: percentage of images for which at least one correct caption appears in the top-k retrieved captions,
- `Image R@1`, `Image R@5`, `Image R@10`: percentage of captions for which the correct image appears in the top-k retrieved images,
- `mean_r@1`: average of `Text R@1` and `Image R@1`.

Retrieval artifacts:

- `retrieval_table.csv`,
- `retrieval_table.md`,
- `retrieval_metrics.json`.

### 6.2 CIFAR-10 Zero-Shot Prompt Ablation

The second evaluation is a simplified zero-shot transfer test. The model is not fine-tuned on CIFAR-10. Instead, CIFAR-10 class names are converted into text prompts, encoded by the text tower, and compared with image embeddings.

The code compares three prompt variants:

- class name only,
- a single `a photo of a/an {label}` prompt,
- an average over multiple prompt templates from the config.

Prompt-ablation artifacts:

- `prompt_ablation.csv`,
- `prompt_ablation.json`,
- `prompt_ablation.png`.

This should be interpreted as an additional generalization check, not as a direct comparison with full CLIP. The model is trained from scratch on a small image-caption dataset, so zero-shot transfer is expected to be weak.

## 7. Reference Results

According to the current README, the strongest configuration is `configs/flickr8k_strong.yaml`. On Flickr8k test with `checkpoint_last.pt`, the recorded result is:

| Split | Text R@1 | Text R@5 | Text R@10 | Image R@1 | Image R@5 | Image R@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Flickr8k test | 4.10 | 14.90 | 20.50 | 3.84 | 14.02 | 21.52 |

For CIFAR-10 prompt ablation with `checkpoint_best.pt`, the recorded result is:

| Prompt variant | Accuracy |
| --- | ---: |
| Class name only | 13.70 |
| `a photo of a {label}` | 16.00 |
| Prompt ensemble | 14.20 |

Interpretation: the model learns useful image-caption alignment, but zero-shot transfer remains limited. This matches the expected limitations of training a small CLIP-like model from scratch on Flickr8k.

## 8. Reproducibility

Recommended one-command reproduction:

```bash
uv run python -m miniclip_repro.reproduce --config configs/flickr8k_strong.yaml --output-dir outputs/flickr8k-strong-160-b128
```

Equivalent expanded commands:

```bash
uv run python -m miniclip_repro.train --config configs/flickr8k_strong.yaml --output-dir outputs/flickr8k-strong-160-b128
uv run python -m miniclip_repro.eval_retrieval --config configs/flickr8k_strong.yaml --checkpoint outputs/flickr8k-strong-160-b128/checkpoint_last.pt --output-dir outputs/flickr8k-strong-160-b128/retrieval_last
uv run python -m miniclip_repro.eval_zeroshot --config configs/flickr8k_strong.yaml --checkpoint outputs/flickr8k-strong-160-b128/checkpoint_best.pt --output-dir outputs/flickr8k-strong-160-b128/zeroshot_best
```

Quick smoke test without full data:

```bash
uv run python -m miniclip_repro.run_all --config configs/flickr8k.yaml --fast-dev-run --synthetic-data --skip-zeroshot
```

Unit tests:

```bash
uv run pytest
```

## 9. Limitations

Main methodological limitations:

- Flickr8k is very small compared with the data scale used by original CLIP,
- the model is trained from scratch, without ImageNet or CLIP-pretrained weights,
- the tokenizer is simple and does not use BPE or the original CLIP tokenizer,
- ResNet-18 and the small text Transformer have much lower capacity than CLIP models,
- the CIFAR-10 zero-shot evaluation uses the configured subset size,
- results may vary with hardware backend, package versions, and numeric precision.

The results should therefore be interpreted as a reproduction of CLIP's mechanism and experimental procedure, not as a direct quality comparison with the original CLIP model.
