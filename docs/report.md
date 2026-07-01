# Mini-CLIP: a laptop-scale reproduction of CLIP-style contrastive image-text learning

## 1. Paper summary

The reproduced paper is *Learning Transferable Visual Models From Natural Language Supervision* by Radford et al. (2021). The central claim is that visual representations can be learned from natural language supervision at web scale. Instead of training a classifier on a fixed label set such as ImageNet classes, CLIP trains an image encoder and a text encoder to identify which text caption belongs to which image. After pre-training, classification can be expressed as a retrieval problem: class names or prompt templates are encoded as text, images are encoded by the vision tower, and the class with the highest image-text similarity is selected.

This matters because it changes the interface between visual recognition and supervision. A conventional supervised classifier is tied to a predefined taxonomy. CLIP shows that a model trained on broad image-text pairs can transfer to many downstream datasets without dataset-specific supervised training. The paper reports non-trivial zero-shot transfer across many vision benchmarks and shows that natural-language prompts can act as flexible task descriptions.

The novelty is not a new convolution or transformer block. The key contribution is the combination of a simple contrastive objective, a dual-encoder architecture, large-scale noisy image-text data, and prompt-based zero-shot evaluation. In this project we reproduce the mechanism at a smaller scale. The original paper trains on roughly 400 million image-text pairs, while this project trains from scratch on Flickr8k. Therefore the goal is not to match CLIP's absolute numbers. The goal is to reproduce the experimental pattern: learn aligned image and text embeddings, evaluate retrieval, and test prompt sensitivity.

Primary reference:

> Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision. ICML 2021.

## 2. Reproduced result

The closest result in the original paper is Table 13, where Radford et al. report zero-shot text and image retrieval on Flickr30k and MSCOCO. Our project mirrors the structure of that table, but the dataset is changed to Flickr8k and the model is trained from scratch. This makes the comparison methodologically useful but not numerically equivalent.

Caption for the reproduced table:

**Reproduction-style adaptation of Table 13 in Radford et al. (2021), evaluated on Flickr8k instead of Flickr30k/MSCOCO.**

| System | Training data | Eval data | Text R@1 | Text R@5 | Text R@10 | Image R@1 | Image R@5 | Image R@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CLIP zero-shot, paper Table 13 | Web-scale WIT, about 400M pairs | Flickr30k | 88.0 | 98.7 | 99.4 | 68.7 | 90.6 | 95.2 |
| Mini-CLIP, this project | Flickr8k train split | Flickr8k test | 4.10 | 14.90 | 20.50 | 3.84 | 14.02 | 21.52 |

The gap is expected. CLIP has web-scale data, larger encoders, better tokenization, more compute, and evaluates zero-shot transfer from broad pre-training. Mini-CLIP is trained only on Flickr8k, uses a ResNet-18 image encoder and a small Transformer text encoder, and has no ImageNet or CLIP-pretrained weights.

## 3. Method

### 3.1 Dual encoder formulation

The model receives a minibatch of matched image-text pairs:

```text
(image_1, text_1), ..., (image_N, text_N)
```

The image encoder maps each image to a vector:

```text
v_i = f_image(image_i)
```

The text encoder maps each caption to a vector:

```text
t_i = f_text(text_i)
```

Both vectors are projected into the same embedding dimension and normalized:

```text
z_i = v_i / ||v_i||
u_i = t_i / ||t_i||
```

The similarity between image `i` and text `j` is the scaled dot product:

```text
s_ij = exp(logit_scale) * z_i dot u_j
```

The learned temperature parameter controls how sharp the similarity distribution is. The implementation clamps the exponentiated scale to a maximum of 100 for numerical stability.

### 3.2 Symmetric contrastive loss

For a batch of `N` correct pairs, the diagonal entries of the similarity matrix are positives. All off-diagonal entries are negatives inside the same batch. The loss is computed in both directions.

Image-to-text loss:

```text
L_i2t = CE(S, [0, 1, ..., N-1])
```

Text-to-image loss:

```text
L_t2i = CE(S^T, [0, 1, ..., N-1])
```

Final loss:

```text
L = 0.5 * (L_i2t + L_t2i)
```

This is the same conceptual objective used by CLIP: the model is trained to assign the highest similarity to the matching pair in the batch.

### 3.3 Model architecture

The image tower is a ResNet-18 from `torchvision` with random initialization. The classification head is replaced by a linear projection into the shared embedding space.

The text tower is a small Transformer encoder. Captions are lowercased and tokenized with a simple regex tokenizer. The sequence includes `<bos>` and `<eos>` markers, and the representation at `<eos>` is projected into the shared embedding space.

The final model differs from the original CLIP in several ways:

| Component | Original CLIP | This project |
| --- | --- | --- |
| Data scale | about 400M image-text pairs | Flickr8k |
| Image tower | ResNet or ViT family | ResNet-18 |
| Text tower | CLIP Transformer with BPE tokenizer | 2-layer Transformer with regex vocabulary |
| Initialization | trained at web scale | trained from scratch |
| Main eval | many zero-shot datasets | Flickr8k retrieval and CIFAR-10 prompt ablation |

### 3.4 Course-book connections

The implementation uses standard building blocks normally introduced across the neural-networks course material:

- vector representations and dot-product similarity,
- cross-entropy classification loss,
- stochastic gradient descent variants and AdamW,
- convolutional image encoders,
- attention and Transformer encoders,
- regularization through weight decay, dropout and data augmentation.

The final PDF should replace this bullet list with exact chapter numbers from the course book once the book title/edition is known.

## 4. Experimental setup

### 4.1 Dataset and splits

The project uses `jxie/flickr8k` through Hugging Face `datasets`. Each image has up to five captions. The code uses the train, validation and test splits exposed by the dataset. If the dataset exposes `dev` instead of `validation`, the code maps `dev` to validation.

During training, each image contributes one caption per epoch. The selected caption is rotated deterministically as a function of epoch and example index. This uses more of the caption set over training while keeping the data pipeline simple.

### 4.2 Preprocessing

Image preprocessing:

- convert to RGB,
- resize and center crop for evaluation,
- random resized crop and horizontal flip for the strong training configuration,
- normalize with CLIP image mean and standard deviation.

Text preprocessing:

- lowercase captions,
- tokenize alphanumeric spans with `[a-z0-9]+`,
- build vocabulary only from train captions,
- add `<bos>`, `<eos>`, `<pad>` and `<unk>`,
- pad or truncate to 32 tokens.

### 4.3 Hyperparameters

The headline run uses `configs/flickr8k_strong.yaml`.

| Hyperparameter | Value |
| --- | ---: |
| Seed | 42 |
| Image size | 160 |
| Embedding dimension | 256 |
| Text width | 256 |
| Text layers | 2 |
| Text attention heads | 4 |
| Text FF dimension | 512 |
| Dropout | 0.1 |
| Batch size | 128 |
| Epochs | 50 |
| Optimizer | AdamW |
| Learning rate | 0.0004 |
| Weight decay | 0.01 |
| Gradient clipping | 1.0 |
| Warmup steps | 250 |
| LR schedule | warmup + cosine decay |

### 4.4 Training budget and hardware

The README result was produced on Apple MPS in 2142.9 seconds. A cluster replication should record:

- GPU model,
- CPU allocation,
- memory allocation,
- wall time,
- CUDA/PyTorch versions,
- final `metrics.json`.

The code writes package versions into `metrics.json`, which should be included in the final artifact archive.

### 4.5 What was copied and what changed

Copied from the paper:

- dual image/text encoder design,
- symmetric contrastive objective,
- normalized embeddings and learned logit scale,
- retrieval-style evaluation,
- prompt-based zero-shot evaluation idea.

Changed for course constraints:

- Flickr8k replaces web-scale WIT pre-training,
- Flickr8k retrieval replaces Flickr30k/MSCOCO retrieval,
- ResNet-18 and a small Transformer replace full CLIP encoders,
- regex tokenizer replaces CLIP BPE,
- training is from scratch with no CLIP or ImageNet weights,
- CIFAR-10 prompt ablation is limited to a small evaluation subset by default.

## 5. Results

### 5.1 Retrieval

The main result is image-text retrieval on Flickr8k.

| Model | Eval split | Text R@1 | Text R@5 | Text R@10 | Image R@1 | Image R@5 | Image R@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Mini-CLIP strong | Flickr8k test | 4.10 | 14.90 | 20.50 | 3.84 | 14.02 | 21.52 |

The result is low compared with CLIP, but it is above a random retrieval baseline and shows that the model has learned a measurable shared embedding. The stronger interpretation is qualitative rather than competitive: the CLIP loss and dual-encoder training loop work even at small scale, but scale is essential for the high zero-shot performance reported by Radford et al.

### 5.2 Side-by-side with paper

| Experiment | Paper result | Our result | Interpretation |
| --- | ---: | ---: | --- |
| Flickr-style text retrieval R@1 | 88.0 on Flickr30k | 4.10 on Flickr8k | Same retrieval format, not same training scale or eval dataset |
| Flickr-style image retrieval R@1 | 68.7 on Flickr30k | 3.84 on Flickr8k | Large gap caused by data/model/compute scale |
| Prompted zero-shot classification | Broad zero-shot transfer across many datasets | 16.00% on CIFAR-10 subset | Above random chance, but weak transfer |

The paper comparison should not be read as a failed attempt to match CLIP. It is evidence for the scaling argument: the same objective is easy to implement and does learn alignment, but the transfer behavior of CLIP depends heavily on massive and diverse pre-training.

## 6. Ablation

The project includes a prompt ablation inspired by CLIP's prompt-engineering discussion. The trained model is evaluated on CIFAR-10 without supervised fine-tuning. Each class is represented by text prompts, and images are classified by highest image-text similarity.

Caption for the ablation figure/table:

**Prompt ablation inspired by the prompt-engineering discussion in Radford et al. (2021), adapted to a Mini-CLIP model trained on Flickr8k.**

| Prompt variant | Accuracy |
| --- | ---: |
| Class name only | 13.70 |
| `a photo of a {label}` | 16.00 |
| Prompt ensemble | 14.20 |

The best prompt in this run is the simple photo prompt. The ensemble does not improve the result, unlike what is often expected for larger CLIP models. This is a useful negative result: prompt ensembling only helps if the text encoder has learned robust enough semantics for the additional templates to average meaningful class representations.

## 7. Limitations and reproducibility notes

Important limitations:

- Flickr8k is tiny relative to CLIP's web-scale pre-training.
- The model is trained from scratch.
- The tokenizer is simple and likely loses useful linguistic structure.
- The image encoder is small.
- The text encoder is small.
- The evaluation datasets do not match the paper exactly.
- Cluster results may differ from Apple MPS results due to backend differences.

Implementation details invented or simplified:

- exact small-model architecture,
- caption rotation rule across epochs,
- augmentation profile,
- tokenizer design,
- CIFAR-10 prompt set,
- output artifact layout.

Sensitivity concerns:

- contrastive learning depends strongly on batch size because in-batch negatives define the classification problem,
- prompt ablation is sensitive to the tokenizer and training captions,
- small-data training can vary across seeds,
- retrieval R@1 is noisy when the dataset and model are small.

The code fixes the default seed at 42 and writes run metadata, configuration, vocabulary, checkpoints, metrics and generated tables. For a final submission, the exact cluster run directory should be archived or included in the repository release.

## 8. Reproduction command

The final result is reproduced with:

```bash
uv run python -m miniclip_repro.reproduce --config configs/flickr8k_strong.yaml --output-dir outputs/flickr8k-strong-160-b128
```

Without `uv`:

```bash
python -m miniclip_repro.reproduce --config configs/flickr8k_strong.yaml --output-dir outputs/flickr8k-strong-160-b128
```

Expected artifacts:

- `outputs/flickr8k-strong-160-b128/metrics.json`,
- `outputs/flickr8k-strong-160-b128/checkpoint_best.pt`,
- `outputs/flickr8k-strong-160-b128/checkpoint_last.pt`,
- `outputs/flickr8k-strong-160-b128/retrieval_last/retrieval_table.md`,
- `outputs/flickr8k-strong-160-b128/retrieval_last/retrieval_metrics.json`,
- `outputs/flickr8k-strong-160-b128/zeroshot_best/prompt_ablation.csv`,
- `outputs/flickr8k-strong-160-b128/zeroshot_best/prompt_ablation.png`.

## 9. Conclusion

This project reproduces CLIP's core learning mechanism under course-scale constraints. The model learns an aligned embedding space that supports image-text retrieval, and the prompt ablation shows limited but measurable zero-shot behavior. The large gap to the original paper is itself the main scientific lesson: CLIP's method is simple, but its strongest claims emerge from scale, data diversity, and careful evaluation design.
