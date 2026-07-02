# Mini-CLIP reproduction slides

## Slide 1 - Title

Mini-CLIP: reproducing CLIP-style image-text contrastive learning at laptop scale

Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision*

## Slide 2 - What CLIP claims

- Replace fixed class-label supervision with natural-language supervision.
- Train image and text encoders on matched image-text pairs.
- Use text prompts at test time to define visual tasks.
- Main claim: broad image-text pre-training enables transferable zero-shot visual recognition.

## Slide 3 - Why this matters

- Standard classifiers are tied to a fixed label set.
- CLIP turns recognition into image-text matching.
- A new class can be introduced by writing text, not retraining the classifier.
- This is useful for retrieval, open-vocabulary recognition and flexible downstream tasks.

## Slide 4 - What we reproduced

- Core dual-encoder architecture.
- Symmetric contrastive loss.
- Retrieval evaluation in the style of CLIP Table 13.
- Prompt ablation inspired by CLIP prompt engineering.
- Small-scale substitution: Flickr8k instead of web-scale WIT.

## Slide 5 - Method

For a batch of matched pairs:

```text
image_i -> image encoder -> normalized vector z_i
text_i  -> text encoder  -> normalized vector u_i
similarity_ij = exp(scale) * z_i dot u_j
```

The correct matches are on the diagonal of the similarity matrix.

## Slide 6 - Loss

- Compute image-to-text cross entropy.
- Compute text-to-image cross entropy.
- Average both directions.

```text
L = 0.5 * (CE(S, targets) + CE(S^T, targets))
targets = [0, 1, ..., N-1]
```

## Slide 7 - Implementation

- Image encoder: ResNet-18 from scratch.
- Text encoder: 2-layer Transformer.
- Embedding dimension: 256.
- Tokenizer: lowercase regex vocabulary.
- Training: AdamW, 50 epochs, batch size 128.
- Seed: 42.

## Slide 8 - Dataset and constraints

- Original CLIP: about 400M web image-text pairs.
- Our run: Flickr8k.
- Original eval table: Flickr30k and MSCOCO retrieval.
- Our eval: Flickr8k retrieval.
- This is a method reproduction, not a scale reproduction.

## Slide 9 - Main result

Caption: Reproduction-style adaptation of Table 13 in Radford et al. (2021), evaluated on Flickr8k instead of Flickr30k/MSCOCO.

| Model | Text R@1 | Text R@5 | Text R@10 | Image R@1 | Image R@5 | Image R@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mini-CLIP | 4.10 | 14.90 | 20.50 | 3.84 | 14.02 | 21.52 |

## Slide 10 - Gap to the paper

| System | Data | Text R@1 | Image R@1 |
| --- | --- | ---: | ---: |
| CLIP Table 13, Flickr30k | Web-scale WIT | 88.0 | 68.7 |
| Mini-CLIP, Flickr8k | Flickr8k only | 4.10 | 3.84 |
| Cluster replication | Flickr8k only | 4.10 | 4.00 |

The same objective works, but scale changes the outcome.

## Slide 11 - Ablation

Caption: Prompt ablation inspired by the prompt-engineering discussion in Radford et al. (2021), adapted to Mini-CLIP.

| Prompt variant | CIFAR-10 accuracy |
| --- | ---: |
| Class name only | 13.70 |
| `a photo of a {label}` | 16.00 |
| Prompt ensemble | 14.20 |

Cluster replication keeps the same ranking but is lower: 11.40, 14.00, and 12.70.

## Slide 12 - Course connections

- Chapters 21-25: CNN image encoder.
- Chapter 26: cross-entropy in the contrastive loss.
- Chapter 27: AdamW and weight decay.
- Chapter 30: PyTorch training and evaluation loops.
- Chapters 37-40: attention and Transformer text encoder.
- Chapter 42: tokenization.

## Slide 13 - Takeaways

- The CLIP training loop is simple and reproducible.
- Mini-CLIP learns measurable image-text alignment.
- Retrieval is stable across the original run and the cluster replication.
- Zero-shot transfer is weaker and more sensitive.
- Prompt engineering helps only slightly here.
- The project highlights CLIP's dependence on data scale, model capacity, and broad pre-training.

## Slide 14 - Reproduction command

```bash
python -m miniclip_repro.reproduce \
  --config configs/flickr8k_strong.yaml \
  --output-dir outputs/flickr8k-strong-160-b128
```

Outputs include checkpoints, metrics, retrieval tables and prompt-ablation artifacts.

## Slide 15 - Q&A prompts

- What changes most if batch size is reduced?
- Would pretrained ResNet weights close the gap?
- Why does prompt ensembling not help here?
- Which limitation is due to data and which is due to architecture?
