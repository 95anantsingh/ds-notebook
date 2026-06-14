(transformer-variants-qa)=
# Transformer Variants: Interview Q&A

> Reference implementations: {ref}`GPT Architecture <gpt-architecture>` · {ref}`BERT Architecture <bert-architecture>` · {ref}`ViT Architecture <vit-architecture>`

---

## Architecture Overview

::::{grid} 2

:::{grid-item-card} BERT {bdg-info}`Encoder-only`
**Bidirectional** — every token attends to every other token

Trained with **MLM** (masked language modeling)

Best for: classification, NER, extractive QA
:::

:::{grid-item-card} GPT {bdg-warning}`Decoder-only`
**Causal** — position i only attends to positions ≤ i

Trained with **next-token prediction**

Best for: generation, chat, code completion
:::

:::{grid-item-card} T5 {bdg-secondary}`Encoder-Decoder`
**Bidirectional encoder** + **causal decoder** with cross-attention

Trained with **span corruption**

Best for: translation, summarization, generative QA
:::

:::{grid-item-card} ViT {bdg-success}`Encoder-only (vision)`
**Bidirectional** over image patches (no causal mask)

Trained with **supervised / DINO / MAE**

Best for: image classification, visual representations
:::

::::

---

## Architecture Comparison Table

| Model | Attention | Positional Enc. | Pre-training | Use Case |
|-------|-----------|----------------|--------------|----------|
| **BERT** | Bidirectional (no mask) | Learned absolute | MLM + NSP | Classification, NER, QA extraction |
| **GPT-2/3** | Causal (masked) | Learned absolute | Next-token prediction | Text generation, chat |
| **T5** | Bidirectional (enc) + Causal (dec) | Relative (T5 bias) | Span corruption | Translation, summarization |
| **ViT** | Bidirectional (no mask) | Learned absolute (1D) | Supervised / DINO / MAE | Image classification |
| **LLaMA** | Causal + GQA | RoPE | Next-token prediction | Generation, instruction-following |

---

## BERT (Encoder-Only) {bdg-info}`encoder-only`

**Architecture**: N encoder blocks, bidirectional attention (no causal mask), 12/24 layers.

**Inputs**:
```
[CLS] token₁ token₂ ... [SEP]   (single sentence)
[CLS] sentA ... [SEP] sentB ... [SEP]   (sentence pair)
Input = token_embed + segment_embed + position_embed
```

**Pre-training objectives**:
- **MLM** (Masked Language Modeling): 15% of tokens masked; predict original token from context
- **NSP** (Next Sentence Prediction): predict if sentence B follows sentence A (now considered less useful)

**Fine-tuning patterns**:
- Classification: `last_hidden[:, 0, :]` → linear head (use `[CLS]` token)
- Token classification (NER): `last_hidden` → linear over all positions
- QA (SQuAD): predict start/end span positions with two linear heads

**Q: Why is BERT bidirectional and GPT is not?**

```{toggle}
BERT uses MLM — it predicts masked tokens using both left and right context simultaneously, so bidirectional attention is fine. GPT uses next-token prediction — at inference, future tokens don't exist, so causal masking must be applied during training to match inference conditions.
```

**Q: What does the `[CLS]` token do?**

```{toggle}
Prepended to every sequence. Its final hidden state is trained to aggregate sentence-level information and is used as the sequence representation for classification tasks. It has no semantic meaning of its own — the training objective forces it to capture global context.
```

---

## GPT (Decoder-Only) {bdg-warning}`decoder-only`

**Architecture**: N decoder blocks with causal self-attention only (no cross-attention). Causal mask prevents position i from attending to positions j > i.

**Key implementation details**:
- Causal mask built once as `register_buffer`, sliced per forward pass
- Weight tying: `lm_head.weight = tok_embed.weight`
- GPT-2+ uses pre-LayerNorm; original GPT uses post-LayerNorm

**At inference**:
- Generate one token at a time (autoregressive)
- KV cache stores past K/V tensors so attention doesn't recompute from scratch each step

**Q: How do you implement causal masking?**

````{toggle}
```python
mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
# mask[i,j] = True when j > i — blocks future tokens
scores = scores.masked_fill(mask, float('-inf'))
```
Register as a buffer in `__init__` so it moves with the model; slice to `[:T, :T]` in `forward`.
````

**Q: What's KV caching and why does it matter?**

```{toggle}
At step `t`, token `t` only adds one new Q/K/V row. Without cache, attention recomputes K/V for all `t` past tokens on every step → O(T²) total work. With cache, past K/V are stored and reused → O(T) total work for the first T tokens. Memory cost: `2 × num_layers × num_heads × d_head × seq_len × dtype_bytes` per sequence.
```

**Q: How does GPT differ from a transformer decoder in the original paper?**

```{toggle}
Original decoder has three sub-layers: causal self-attention + **cross-attention** (attends to encoder output) + FFN. GPT drops cross-attention entirely — there's no encoder. GPT conditions on context solely through its causal self-attention over the concatenated prompt + generated tokens.
```

---

## T5 (Encoder-Decoder) {bdg-secondary}`seq2seq`

**Architecture**: Separate encoder (bidirectional) and decoder (causal + cross-attention) stacks, similar to the original transformer. All tasks framed as text-to-text.

**Positional encoding**: Relative position biases added to attention logits (not absolute position embeddings in the input).

**Pre-training**: Span corruption — random spans of input text replaced with sentinel tokens; model must regenerate the spans.

---

## ViT (Vision Transformer) {bdg-success}`vision`

**Architecture**: Standard encoder stack applied to image patches.

**Patch embedding**:
```
Image (B, C, H, W)
  → patches: (B, N, patch_size² × C)   where N = (H/P) × (W/P)
  → linear projection: (B, N, d_model)
  → prepend [CLS]: (B, N+1, d_model)
  → add learned position embed
  → N encoder blocks
  → LayerNorm
  → MLP head on [CLS] position
```

**Q: How does ViT differ from a CNN?**

```{toggle}
CNNs have inductive biases for locality (convolution) and translation invariance (weight sharing). ViT has neither — it treats the image as a flat sequence of patches. ViT typically requires more data or stronger pre-training (ImageNet-21k, JFT, self-supervised like MAE or DINO) to match CNN performance.
```

**Q: What's the efficient way to implement patch embedding?**

````{toggle}
```python
# nn.Conv2d with kernel_size=patch_size, stride=patch_size extracts non-overlapping patches
# and projects in one operation — equivalent to unfold + linear
self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
# (B, C, H, W) → (B, d_model, H/P, W/P) → flatten → (B, N, d_model)
```
````

**Q: What happens if image size isn't divisible by patch size?**

```{toggle}
It breaks — `patch_size` must evenly divide both H and W. ViT implementations enforce this via assertion or padding. At inference on a different resolution, the position embeddings must be interpolated since they were trained for a fixed number of patches.
```

---

## LLaMA / Modern LLM Differences from GPT-2

| | GPT-2 | LLaMA |
|--|-------|-------|
| **Attention** | MHA | GQA (grouped query) |
| **Positional enc.** | Learned absolute | RoPE |
| **Normalization** | Post-attn LayerNorm | Pre-attn RMSNorm |
| **Activation** | GELU | SwiGLU |
| **Bias** | Linear bias enabled | No bias in linear layers |
