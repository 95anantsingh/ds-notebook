(scaled-dot-product-attention)=
# Scaled Dot-Product Attention

> **Paper**: "Attention Is All You Need" (Vaswani et al., 2017)
> **Interview cheat sheet**: {ref}`Attention Implementation Guide <attention-implementation-guide>`
> **Flash Attention deep dive**: [Flash Attention](../Attentions/3-flash-attention.md)

The fundamental building block of the transformer. Computes a weighted sum of values, where weights are determined by the compatibility of queries with keys.

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

```{mermaid}
---
title: Scaled Dot-Product Attention
---
flowchart TD
    Q["Q  (B, H, T_q, d_k)"]
    K["K  (B, H, T_k, d_k)"]
    V["V  (B, H, T_k, d_v)"]

    subgraph compute["Attention Computation"]
        MM1["Q @ Kᵀ  (B, H, T_q, T_k)"]
        SCALE["÷ √d_k"]
        MASK{"mask?"}
        FILL["masked_fill −∞"]
        SOFT["softmax(dim=−1)"]
        DROP["Dropout  (training only)"]
        MM2["@ V  (B, H, T_q, d_v)"]
    end

    OUT["Output  (B, H, T_q, d_v)"]

    Q --> MM1
    K --> MM1
    MM1 --> SCALE --> MASK
    MASK -->|yes| FILL --> SOFT
    MASK -->|no| SOFT
    SOFT --> DROP --> MM2
    V --> MM2
    MM2 --> OUT
```

---

## Standalone Implementation

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(
    q: torch.Tensor,     # (B, H, T_q, d_k)
    k: torch.Tensor,     # (B, H, T_k, d_k)
    v: torch.Tensor,     # (B, H, T_k, d_v)
    mask: torch.Tensor | None = None,   # (B, 1, T_q, T_k) or (T_q, T_k), bool
    dropout_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (output, attention_weights) with shapes (B,H,T_q,d_v) and (B,H,T_q,T_k)."""
    d_k = q.size(-1)

    # Step 1: dot product — (B, H, T_q, T_k)
    scores = q @ k.transpose(-2, -1)

    # Step 2: scale
    scores = scores / d_k**0.5

    # Step 3: mask (before softmax — additive masking)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    # Step 4: softmax over key dimension
    attn_weights = F.softmax(scores, dim=-1)   # (B, H, T_q, T_k)

    # Step 5: dropout (training only)
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p)

    # Step 6: weighted sum of values
    output = attn_weights @ v                  # (B, H, T_q, d_v)

    return output, attn_weights
```

---

## Mask Variants

```{important}
Always apply the mask **before** softmax using additive masking. Fill blocked positions with `float('-inf')` so they become exactly 0 after softmax. Applying a mask after softmax silently produces incorrect attention distributions and incorrect gradients.
```

### Additive mask (preferred)

Fill masked positions with `-inf` **before** softmax. After softmax those positions become exactly 0.

```python
scores = scores.masked_fill(bool_mask, float('-inf'))
attn = F.softmax(scores, dim=-1)
```

### Multiplicative mask (avoid)

Multiply scores by 0/1 after softmax. Does **not** zero out the gradient path cleanly and produces non-zero attention for masked positions.

### Causal mask

```python
T = seq_len
causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
# causal_mask[i, j] = True when j > i  →  block future tokens
```

### Padding mask

```python
# key_padding_mask: (B, T_k) — True for PAD positions
padding_mask = key_padding_mask[:, None, None, :]   # broadcast to (B, 1, 1, T_k)
scores = scores.masked_fill(padding_mask, float('-inf'))
```

---

## Cross-Attention

In encoder-decoder architectures, Q comes from the decoder state and K/V come from the encoder output:

```python
def cross_attention(decoder_hidden, encoder_output, encoder_padding_mask=None):
    # decoder_hidden:  (B, H, T_dec, d_k)
    # encoder_output:  (B, H, T_enc, d_k)
    q = W_q(decoder_hidden)
    k = W_k(encoder_output)
    v = W_v(encoder_output)
    mask = encoder_padding_mask  # mask over encoder (key) positions
    return scaled_dot_product_attention(q, k, v, mask)
```

---

## Manual vs PyTorch 2.0+

::::{tab-set}

:::{tab-item} Manual (interview ready)
```python
def scaled_dot_product_attention(q, k, v, mask=None, dropout_p=0.0):
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / d_k**0.5
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    attn = F.softmax(scores, dim=-1)
    if dropout_p > 0.0:
        attn = F.dropout(attn, p=dropout_p)
    return attn @ v
```
:::

:::{tab-item} PyTorch 2.0+ {bdg-info}`recommended`
```python
# Dispatches to FlashAttention on CUDA automatically — no boilerplate needed
out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)

# Explicit float additive mask (NOT boolean):
float_mask = torch.zeros(T, T).masked_fill(causal_mask, float('-inf'))
out = F.scaled_dot_product_attention(q, k, v, attn_mask=float_mask)
```
:::

::::

```{warning}
`F.scaled_dot_product_attention` expects `attn_mask` as a **float** additive mask, not a boolean mask. Passing a boolean mask silently treats `True` as `1.0` and produces wrong attention scores. Convert first: `mask.float().masked_fill(mask, float('-inf'))`.
```

---

## Why Scaling Matters

Without scaling, the variance of $QK^T$ grows linearly with $d_k$ (assuming unit-normal Q, K). Large logits push softmax into saturation:

```
d_k = 64  →  std(QK^T) ≈ 8  →  without scaling, most gradient flows through one key
d_k = 64  →  std(QK^T / sqrt(64)) ≈ 1  →  softmax stays in a healthy gradient regime
```
