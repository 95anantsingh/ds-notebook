(positional-encoding)=
# Positional Encoding

> **Why it's needed**: Transformers process tokens in parallel via attention — without positional info, "cat sat on mat" and "mat on sat cat" would produce identical representations.

---

## Implementations

:::::{tab-set}

::::{tab-item} Sinusoidal {bdg-secondary}`original paper`

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \quad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

```python
import torch
import torch.nn as nn
import math


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                   # (max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()     # (max_len, 1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )                                                     # (d_model/2,)

        pe[:, 0::2] = torch.sin(pos * div)                   # even indices
        pe[:, 1::2] = torch.cos(pos * div)                   # odd indices
        pe = pe.unsqueeze(0)                                  # (1, max_len, d_model)

        self.register_buffer('pe', pe)                        # not a parameter, moves with model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
```

:::{dropdown} Wondering how `div` equals the inverse of the denominator?
The code computes the **inverse frequency** $1 / 10000^{2i/d_{\text{model}}}$ in log-space for stability and speed.

Using the identity $a^b = \exp(b \cdot \ln a)$:

$$\frac{1}{10000^{2i/d_{\text{model}}}} = \exp\!\left(-\frac{2i}{d_{\text{model}}} \cdot \ln 10000\right) = \exp\!\left(2i \cdot \frac{-\ln 10000}{d_{\text{model}}}\right)$$

Mapping to the code:

- `torch.arange(0, d_model, 2)` → $2i$ for $i = 0, 1, \ldots, d_{\text{model}}/2 - 1$
- `-math.log(10000.0) / d_model` → $-\ln(10000) / d_{\text{model}}$
- `torch.exp(...)` wraps the product

So `pos * div` is exactly $pos / 10000^{2i/d_{\text{model}}}$. Doing it via `exp` avoids `pow`, stays numerically stable for large `d_model`, and vectorizes cleanly.
:::

**Why it generalizes**: The relative positions are encoded in the phase difference between sin/cos curves, so the model can extrapolate beyond training sequence lengths.
::::

::::{tab-item} Learned Absolute {bdg-info}`BERT, GPT-2`

```python
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.pe = nn.Embedding(max_len, d_model)   # learned, updated via backprop
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d_model)
        T = x.size(1)
        positions = torch.arange(T, device=x.device)     # (T,)
        x = x + self.pe(positions)                        # (B, T, d_model)
        return self.dropout(x)
```

Simpler and typically works as well as sinusoidal within the training length. Does **not** generalize to sequences longer than `max_len`.
::::

:::::


## RoPE (Rotary Positional Embedding)

Used in: LLaMA, Mistral, Falcon, GPT-NeoX.

**Key idea**: Instead of adding position to the input, rotate Q and K vectors before computing attention. The rotation matrix encodes absolute position; relative position appears naturally in the dot product.

- Applied per attention head, directly to Q and K (not to V or the input embedding)
- Better length extrapolation than learned absolute PE
- See [RoPE deep dive](../Embeddings/1-rope.md) for the full implementation

```python
# Conceptual: rotate q and k by position-dependent angle θ_pos
q_rot = rotate(q, pos)
k_rot = rotate(k, pos)
scores = q_rot @ k_rot.transpose(-2, -1)   # relative position encoded in the dot product
```

---

## ALiBi (Attention with Linear Biases)

Used in: BLOOM, MPT.

**Key idea**: Add a negative bias proportional to the distance between tokens, directly to attention logits. No positional embedding in the input.

```python
# bias[i, j] = -|i - j| * slope_h   (slope depends on head h)
alibi_bias = -torch.abs(positions_q.unsqueeze(-1) - positions_k.unsqueeze(-2)) * slope
scores = scores + alibi_bias
```

Strong length extrapolation (can generalize far beyond training length).

---

## Comparison

| Method | Used In | Strengths | Weaknesses |
|--------|---------|-----------|------------|
| Sinusoidal | Original Transformer, T5 | No parameters, extrapolates | Underperforms learned on short tasks |
| Learned absolute | BERT, GPT-2, ViT | Simple, effective at training length | Fails beyond `max_len` |
| RoPE | LLaMA, Mistral, GPT-NeoX | Extrapolates well, efficient | Slightly more complex to implement |
| ALiBi | BLOOM, MPT | Strong extrapolation, no PE parameters | Not compatible with prefix caching strategies |

```{note}
**Interview Q: Why does GPT use learned PE while the original transformer uses sinusoidal?**

GPT is trained and used within a fixed context window; it never needs to generalize beyond training length, so learned PE is simpler and equally effective. The original transformer was designed with length generalization in mind.
```
