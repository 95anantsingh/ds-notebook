# Component Reference

A living reference for every component available when writing notes on this site.
Each block below shows the rendered output followed by the source that produced it.

---

## Admonitions

Use admonitions to call attention to important content.

```{note}
Use `{note}` for supplementary information that is helpful but not critical.
```

```{tip}
Use `{tip}` for shortcuts, best practices, or suggestions that improve understanding.
```

```{important}
Use `{important}` for information that must not be overlooked.
```

```{warning}
Use `{warning}` when a reader could make a costly mistake if they skip this.
```

```{caution}
Use `{caution}` for potential issues that are less severe than a warning.
```

```{danger}
Use `{danger}` for actions that can cause data loss, crashes, or irreversible damage.
```

```{attention}
Use `{attention}` as a general eye-catcher with no specific severity implied.
```

```{hint}
Use `{hint}` for small clues that help without giving away the full answer.
```

```{error}
Use `{error}` to document a known error condition or a common mistake.
```

Custom title and body:

```{warning} Mixed Precision Gotcha
When using `torch.autocast`, gradients for `float16` operations can underflow to zero.
Wrap the loss scaling step inside `torch.cuda.amp.GradScaler`.
```

---

## Math

### Inline Math

Inline equations render mid-sentence: the softmax function is $\sigma(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$, and the dot-product attention score is $\text{score}(q, k) = \frac{q \cdot k}{\sqrt{d_k}}$.

### Block Equations

$$
\mathcal{L} = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)
$$

### Multi-line with Alignment

$$
\begin{align}
\mu &= \frac{1}{n} \sum_{i=1}^{n} x_i \\
\sigma^2 &= \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2 \\
\hat{x}_i &= \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
\end{align}
$$

### Transformer Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

### Matrix Notation

$$
W = \begin{pmatrix} w_{11} & w_{12} & \cdots & w_{1n} \\ w_{21} & w_{22} & \cdots & w_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ w_{m1} & w_{m2} & \cdots & w_{mn} \end{pmatrix}
$$

---

## Code Blocks

All code blocks automatically get a copy button from `sphinx_copybutton`.

### Python

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.d_k)
        q, k, v = qkv.unbind(dim=2)
        scale = self.d_k ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        return self.out((attn @ v).reshape(B, T, C))
```

### With Line Numbers and Highlights

```{code-block} python
:linenos:
:emphasize-lines: 3,8,9

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # scale

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    weights = F.softmax(scores, dim=-1)  # normalize
    return torch.matmul(weights, v)      # weighted sum
```

---

## Mermaid Diagrams

All diagrams are written in Markdown and rendered in the browser — no image files needed.

### Flowchart — Training Loop

```{mermaid}
flowchart TD
    A([Start Training]) --> B[Load Batch]
    B --> C[Forward Pass]
    C --> D[Compute Loss]
    D --> E{Loss < threshold?}
    E -->|No| F[Backward Pass]
    F --> G[Clip Gradients]
    G --> H[Optimizer Step]
    H --> I[Zero Gradients]
    I --> B
    E -->|Yes| J([Save Checkpoint])
```
---

## Toggle Buttons

Use toggles to hide worked examples, derivations, or answers so the page stays scannable.

### Basic Toggle — Hidden by Default

```{toggle}
**Answer:** The gradient of the cross-entropy loss with respect to the softmax input $z_i$ is:

$$\frac{\partial \mathcal{L}}{\partial z_i} = \hat{y}_i - y_i$$

This is why softmax + cross-entropy is so convenient — the gradient is just the prediction error.
```

### Toggle — Always Starts Collapsed

`{toggle}` always starts collapsed — `:show:` is unreliable because the content is hidden by CSS before JavaScript runs. Use it only when you want content hidden by default.

```{toggle}
**Answer:** The gradient of the cross-entropy loss with respect to the softmax input $z_i$ is:

$$\frac{\partial \mathcal{L}}{\partial z_i} = \hat{y}_i - y_i$$

This is why softmax + cross-entropy is so convenient — the gradient is just the prediction error.
```

### Shown by Default — Use Dropdown Instead

For content that starts open and can be collapsed, use `{dropdown}` with `:open:` from sphinx-design. It renders open in the HTML itself without JavaScript.

::::{dropdown} Click to hide the derivation
:open:

**Derivation of Backprop through LayerNorm:**

Given $\hat{x} = \frac{x - \mu}{\sigma}$, the gradient with respect to $x$ requires the chain rule through both $\mu$ and $\sigma$:

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{1}{\sigma} \left( \frac{\partial \mathcal{L}}{\partial \hat{x}_i} - \frac{1}{n}\sum_j \frac{\partial \mathcal{L}}{\partial \hat{x}_j} - \frac{\hat{x}_i}{n} \sum_j \frac{\partial \mathcal{L}}{\partial \hat{x}_j} \hat{x}_j \right)$$
::::

### Toggle Inside an Admonition

:::{note}
Flash Attention rewrites the attention computation to be IO-aware.

```{toggle} Show memory complexity comparison
| Method          | Memory       | Passes over HBM |
|-----------------|-------------|-----------------|
| Standard Attn   | $O(N^2)$    | $O(N^2 / M)$    |
| Flash Attention | $O(N)$      | $O(N^2 / M)$    |
| Flash Attn 2    | $O(N)$      | $O(N^2 / M)$    |

Where $N$ = sequence length, $M$ = SRAM size.
```
:::

---

## Design Components

### Tabs

Use tabs to show the same concept across frameworks or to separate long alternatives.

::::{tab-set}

:::{tab-item} PyTorch
```python
import torch.nn.functional as F

output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True,
)
```
:::

:::{tab-item} JAX
```python
import jax.numpy as jnp

def dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = jnp.matmul(q, k.swapaxes(-2, -1)) / jnp.sqrt(d_k)
    if mask is not None:
        scores = jnp.where(mask, scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(weights, v)
```
:::

:::{tab-item} NumPy
```python
import numpy as np

def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    return weights @ V
```
:::

::::

### Cards

Use cards to organize related concepts side-by-side.

::::{grid} 2

:::{grid-item-card} Encoder-only Models
**Examples:** BERT, RoBERTa, DeBERTa

Best for tasks that require understanding the full context simultaneously — classification, NER, extractive QA.

Trained with Masked Language Modelling (MLM).
:::

:::{grid-item-card} Decoder-only Models
**Examples:** GPT-4, LLaMA, Mistral

Best for generation tasks. Process tokens left-to-right with causal masking.

Trained with next-token prediction (CLM).
:::

:::{grid-item-card} Encoder-Decoder Models
**Examples:** T5, BART, mT5

Best for seq2seq tasks — translation, summarisation, generative QA.

Encoder builds full context; decoder attends to it via cross-attention.
:::

:::{grid-item-card} Mixture of Experts
**Examples:** Mixtral, Switch Transformer

Scales parameter count without scaling compute. A router sends each token to $k$ of $N$ expert FFN layers.

Active params ≪ total params.
:::

::::

### Cards with Links

```{note}
For `:link-type: doc`, paths with spaces in directory names are not supported by sphinx-design.
Use `:link-type: url` with a relative `.html` path, or only link to docs whose full path has no spaces.
```

::::{grid} 2

:::{grid-item-card} Inference Notes
:link: ../Interview/Inference/1-inference
:link-type: doc

Click this card to navigate to a page. Add `:link:` (doc path) and `:link-type: doc` to any card to make the whole card clickable.
:::

:::{grid-item-card} NLP Index
:link: ../Interview/NLP/index
:link-type: doc

Cards can also link to section index pages, not just leaf documents.
:::

::::

### Dropdowns

::::{dropdown} When to use BF16 vs FP16
BF16 and FP16 both use 16 bits, but allocate them differently:

| Format | Exponent bits | Mantissa bits | Max value |
|--------|--------------|---------------|-----------|
| FP16   | 5            | 10            | 65504     |
| BF16   | 8            | 7             | ~3.4×10³⁸ |

**Use BF16** on Ampere+ GPUs (A100, H100, RTX 30xx+) — same dynamic range as FP32, avoids overflow during training.

**Use FP16** only when targeting older hardware (V100, T4) that lacks BF16 support. Requires loss scaling to prevent underflow.
::::

::::{dropdown} KV Cache — How it works
:open:

During autoregressive generation, every new token attends to all previous tokens. Without a cache, keys and values are recomputed from scratch at each step — $O(T^2)$ total work.

With a KV cache:
1. On the first forward pass (prefill), compute and **store** $K, V$ for the entire prompt.
2. On each decode step, compute $K, V$ for only the **new token** and append to the cache.
3. Attention uses the full cached $K, V$ — $O(T)$ work per step.

Memory cost: `2 × num_layers × num_heads × d_head × seq_len × dtype_bytes` per sequence.
::::

### Badges

Use badges inline to tag content with status or category labels.

{bdg-primary}`primary` {bdg-secondary}`secondary` {bdg-success}`stable` {bdg-info}`info` {bdg-warning}`experimental` {bdg-danger}`deprecated` {bdg-dark}`dark` {bdg-light}`light`

Example in context: Flash Attention 2 {bdg-success}`stable` is the current default in HuggingFace Transformers {bdg-info}`≥4.36`.

Outline variants: {bdg-primary-line}`outline primary` {bdg-success-line}`outline success` {bdg-warning-line}`outline warning`

---

## Tables

### Simple Table

| Optimiser  | Momentum | Adaptive LR | Weight Decay | Notes                        |
|-----------|----------|-------------|--------------|------------------------------|
| SGD       | ✓        | ✗           | L2           | Needs careful LR tuning      |
| Adam      | ✓        | ✓           | Decoupled*   | *use AdamW for correct WD    |
| AdamW     | ✓        | ✓           | ✓            | Default for transformer LM   |
| Adafactor | ✗        | ✓           | ✓            | Low memory; used in T5       |
| LION      | ✓        | ✗           | ✓            | Sign-based; memory efficient |

### Grid Table (complex spans)

```{list-table} Quantisation Format Comparison
:header-rows: 1
:widths: 15 10 10 15 30 20

* - Format
  - Bits
  - Type
  - Hardware
  - Typical Use
  - Accuracy Impact
* - FP32
  - 32
  - Float
  - All
  - Training baseline
  - None (reference)
* - BF16
  - 16
  - Float
  - A100, H100
  - Mixed-precision training
  - Negligible
* - FP8
  - 8
  - Float
  - H100
  - Training + inference
  - Small
* - INT8
  - 8
  - Int
  - All
  - Post-training inference
  - Small–medium
* - INT4
  - 4
  - Int
  - All
  - Compressed inference
  - Medium
* - GPTQ
  - 4
  - Int
  - All
  - LLM weight-only quant
  - Medium (depends on group size)
```

### Spanning Cells

Neither pipe tables nor `{list-table}` support `colspan`/`rowspan`. Use a raw HTML `<table>` block directly in the Markdown file.

<table>
  <thead>
    <tr>
      <th rowspan="2">Format</th>
      <th rowspan="2">Bits</th>
      <th colspan="2">Supported On</th>
      <th rowspan="2">Notes</th>
    </tr>
    <tr>
      <th>Training</th>
      <th>Inference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><strong>Float</strong></td>
      <td>32 (FP32)</td>
      <td>✓</td><td>✓</td>
      <td>Full precision baseline</td>
    </tr>
    <tr>
      <td>16 (BF16)</td>
      <td>✓</td><td>✓</td>
      <td>Same range as FP32; preferred on Ampere+</td>
    </tr>
    <tr>
      <td>8 (FP8)</td>
      <td>✓</td><td>✓</td>
      <td>H100+ only; two variants: E4M3 and E5M2</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>Int</strong></td>
      <td>8 (INT8)</td>
      <td>—</td><td>✓</td>
      <td>Post-training quantisation</td>
    </tr>
    <tr>
      <td>4 (INT4)</td>
      <td>—</td><td>✓</td>
      <td>Weight-only; requires dequant at runtime</td>
    </tr>
    <tr>
      <td>2 (INT2)</td>
      <td>—</td><td>✓</td>
      <td>Experimental; high accuracy loss</td>
    </tr>
  </tbody>
</table>

---

## Cross-References

Because `autosectionlabel` is enabled with `autosectionlabel_prefix_document = True`, every heading gets an anchor prefixed by its document path.

- Link to a heading on this page: {ref}`pages/Demo/1-demo:Mermaid Diagrams`
- Link to a heading on this page with custom text: {ref}`jump to diagrams <pages/Demo/1-demo:Mermaid Diagrams>`
- Link to a heading in another file: `` {ref}`pages/Natural Language Processing/Attentions/index:Attentions` ``
- Link to another page by path: {doc}`../Natural_Language_Processing/Attentions/index`

---

## Figures

```{figure} ../../assets/favicon.png
:width: 64px
:align: center
:alt: DS Notebook logo

*Figure 1.* Use `{figure}` to add a caption and alt text to any image.
```

---

## Block Quotes and Definitions

> "Attention is all you need."
> — Vaswani et al., 2017

Definition list (term followed by indented definition):

Autoregressive
: Generates one token at a time, conditioning each output on all previous outputs.

KV Cache
: A memory buffer storing the key and value tensors from past decode steps to avoid recomputation.

**Perplexity**
: $\text{PPL} = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T} \log p(w_t \mid w_{<t})\right)$ — lower is better.
