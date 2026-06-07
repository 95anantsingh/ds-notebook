(attention-implementation-guide)=
# Attention Implementation Guide

> Reference implementations: {ref}`Scaled Dot-Product Attention <scaled-dot-product-attention>` · {ref}`Multi-Head Attention <multi-head-attention>`

---

## Scaled Dot-Product Attention

The core formula: $\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right) V$

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(q, k, v, mask=None):
    # q, k, v: (B, H, T, d_k)
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / d_k**0.5  # (B, H, T, T)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))  # additive: fill before softmax
    attn = F.softmax(scores, dim=-1)                      # (B, H, T, T)  — dim=-1 is critical
    return attn @ v                                        # (B, H, T, d_k)
```

### Why `sqrt(d_k)` scaling?

```{important}
Without `sqrt(d_k)` scaling, dot products grow large → softmax saturates → attention weights become near-one-hot → gradients vanish → the model learns slowly or not at all. Always scale by `d_k**0.5`.
```

---

## Causal Mask Construction

```python
T = 10
# True where attention should be BLOCKED (upper triangle excl. diagonal)
causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
# Shape: (T, T) — broadcast works with (B, H, T, T)
```

Register as a buffer so it moves with the model device:

```python
class MyDecoder(nn.Module):
    def __init__(self, max_len):
        super().__init__()
        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask', mask)

    def forward(self, x):
        T = x.size(1)
        mask = self.causal_mask[:T, :T]   # slice to actual sequence length
        ...
```

---

## Common Gotchas

| Mistake | Symptom | Fix |
|---------|---------|-----|
| Mask applied **after** softmax | Model leaks future tokens silently | Apply before softmax with `masked_fill` |
| Wrong `dim` in softmax | Scores sum to 1 across wrong axis | Always `dim=-1` (across key positions) |
| Missing `sqrt(d_k)` | Loss spikes or NaN early in training | Divide by `d_k**0.5` |
| Forgetting batch dims | Shape error on batched input | Shape is `(B, H, T, d_k)` in MHA, not `(T, d_k)` |
| `view` after `transpose` | RuntimeError: non-contiguous | Call `.contiguous()` before `.view()` |
| `float('-inf')` mask on all tokens | NaN after softmax | Ensure at least one unmasked position per row |

```{warning}
If every position in a row is masked with `float('-inf')`, softmax produces `NaN` (0/0). This can happen with padding-only sequences or incorrect mask construction. Always verify at least one position per query is unmasked.
```

---

## Q&A

**Q: What is the time and memory complexity of attention?**

```{toggle}
Time: $O(T^2 \cdot d_k)$. Memory: $O(T^2)$ for the attention matrix. FlashAttention reduces memory to $O(T)$ via tiled recomputation while keeping the same time complexity.
```

**Q: Why multi-head attention instead of one large head?**

```{toggle}
Different heads learn to attend in different representation subspaces (syntax, coreference, positional relationships, etc.). A single large head collapses all that into one pattern. MHA also effectively runs H attention functions in parallel at the same cost as one full-dimension function (because each head works in d_k = d_model/H dimensions).
```

**Q: What happens if you forget `sqrt(d_k)`?**

```{toggle}
The dot products grow large → softmax saturates → attention weights become near-one-hot → gradients vanish (only one key gets signal) → the model learns slowly or not at all.
```

**Q: Difference between self-attention and cross-attention?**

```{toggle}
Self-attention: Q, K, V all come from the same sequence. Cross-attention: Q comes from one sequence (e.g. decoder state), K and V come from another (e.g. encoder output). The mask shape differs: self-attention uses a square (T, T) mask; cross-attention uses a rectangular (T_dec, T_enc) mask.
```

**Q: Why is the causal mask applied in decoders but not encoders?**

```{toggle}
Encoders process the full input bidirectionally — every token can see every other token. Decoders generate autoregressively; position $i$ must not attend to positions $j > i$ (future tokens don't exist yet at inference time). Applying the causal mask during training matches the inference condition.
```

---

## PyTorch 2.0+ Shortcut

```{tip}
Since PyTorch 2.0, `F.scaled_dot_product_attention` is built-in and dispatches to FlashAttention on CUDA automatically. In an interview: know it exists and use it if allowed; if asked to implement from scratch, write the manual version above.
```

::::{tab-set}

:::{tab-item} Use in production
```python
# is_causal=True builds the causal mask internally
out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
```
:::

:::{tab-item} Write from scratch (interview)
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

::::
