# Multi-Query Attention (MQA)

A memory-bandwidth-optimized variant of Multi-Head Attention introduced in 2019. MQA uses a single shared key head and a single shared value head across all query heads, dramatically reducing the KV cache size and enabling faster autoregressive decoding.

---

## Intuition

In standard MHA, every head maintains its own K and V matrices — but during incremental decoding, the GPU spends most of its time waiting for KV cache reads, not on computation. MQA eliminates $H - 1$ copies of K and V: all query heads share one K and one V head. This reduces KV cache size by $H\times$ and makes decoding memory-bandwidth-bound on a much smaller footprint. Quality is slightly lower than MHA but the inference speedup is significant.

```{mermaid}
flowchart TD
    A([Input X\nB × T × d_model]) --> B["W_Q → Q\n(B, T, H × d_k)"]
    A --> C["W_K → K\n(B, T, 1 × d_k)"]
    A --> D["W_V → V\n(B, T, 1 × d_v)"]
    B --> E["Split Q into H heads\n(B, H, T, d_k)"]
    C --> F["Broadcast K to H heads\n(B, H, T, d_k)"]
    D --> G["Broadcast V to H heads\n(B, H, T, d_v)"]
    E --> H[Scaled dot-product per head]
    F --> H
    G --> H
    H --> I["Concat + W_O\n(B, T, d_model)"]
    I --> J([Output])
```

---

## Theory

> **Paper:** [Fast Transformer Decoding: One Write-Head is All You Need — Shazeer (2019)](https://arxiv.org/abs/1911.02150)

$$
Q = X W_Q,\quad Q \in \mathbb{R}^{B \times T \times H \cdot d_k}
\quad\text{(H separate Q projections)}
$$

$$
K = X W_K,\quad K \in \mathbb{R}^{B \times T \times d_k}
\quad\text{(1 shared K projection)}
$$

$$
V = X W_V,\quad V \in \mathbb{R}^{B \times T \times d_v}
\quad\text{(1 shared V projection)}
$$

Each query head $h$ attends using the same K and V:

$$
\text{head}_h = \text{Attention}(Q_h,\, K,\, V),
\quad Q_h \in \mathbb{R}^{B \times T \times d_k}
$$

$$
\text{MQA}(X) = \text{Concat}(\text{head}_1,\ldots,\text{head}_H)\, W_O
\in \mathbb{R}^{B \times T \times d_{\text{model}}}
$$

### Complexity

| Property | Value |
|---|---|
| Time | $O(T^2 \cdot d_k \cdot H)$ — same as MHA |
| KV cache | $O(T \cdot d_k \cdot L)$ — $H\times$ smaller than MHA's $O(T \cdot d_k \cdot H \cdot L)$ |
| Parameters | $W_Q \in \mathbb{R}^{d \times Hd_k}$, $W_K, W_V \in \mathbb{R}^{d \times d_k}$, $W_O \in \mathbb{R}^{Hd_k \times d}$ |

---

## PyTorch Module

```{code-block} python
:linenos:
:emphasize-lines: 20,21,22

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.H   = num_heads
        self.d_k = d_model // num_heads  # head dim

        # Q has H heads; K and V have only 1 head
        self.W_q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, self.d_k,             bias=False)  # single head
        self.W_v = nn.Linear(d_model, self.d_k,             bias=False)  # single head
        self.W_o = nn.Linear(num_heads * self.d_k, d_model, bias=False)  # d_model == H*d_k
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.W_q(x).view(B, T, self.H, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        k = self.W_k(x).view(B, T, 1,      self.d_k).transpose(1, 2)  # (B, 1, T, d_k)
        v = self.W_v(x).view(B, T, 1,      self.d_k).transpose(1, 2)  # (B, 1, T, d_k)

        # k and v broadcast across H heads automatically inside sdpa
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, H, T, d_k)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)
        return self.W_o(out)                                     # (B, T, d_model)
```

The key insight is that `k` and `v` have shape `(B, 1, T, d_k)` — PyTorch's `scaled_dot_product_attention` broadcasts the single K/V head across all H query heads automatically.

---

## Modern Usage

MQA is used during inference primarily. Many frameworks support it via explicit repeat of K/V:

```python
# Expand single K/V head to match H query heads before calling sdpa
k = k.expand(B, self.H, T, self.d_k)  # (B, H, T, d_k) — no data copy (stride trick)
v = v.expand(B, self.H, T, self.d_k)  # (B, H, T, d_k)

out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

Using `.expand()` instead of `.repeat()` avoids allocating a new tensor — the expanded tensor shares the same storage.

**Flash Attention 2+** natively supports MQA/GQA with the `num_heads_k` parameter:

```python
from flash_attn import flash_attn_func

# q: (B, T, H, d_k),  k/v: (B, T, 1, d_k)
out = flash_attn_func(q, k, v, causal=True)  # (B, T, H, d_k)
```

---

```{warning} Common Pitfalls
- **Training quality degradation**: MQA can hurt perplexity, especially on tasks requiring diverse attention patterns. Consider GQA (G=2 or 4) as a middle ground.
- **`.expand()` vs `.repeat()`**: Use `.expand()` to broadcast without copying; `.repeat()` allocates a full H-copies tensor and wastes memory — the opposite of what MQA is trying to achieve.
- **Mixed precision KV cache**: Store KV cache in `float16`/`bfloat16`, not `float32`; the memory saving of MQA is halved otherwise.
- **Checkpoint compatibility**: MHA checkpoints cannot be loaded into an MQA model without discarding K/V weights — plan the architecture before training.
```

```{tip} Tips
- MQA reduces KV cache by exactly $H\times$ compared to MHA — for H=32 (LLaMA-scale), this is a 32× reduction in KV memory.
- PaLM (2022) and Falcon use MQA. LLaMA 2 70B uses GQA (the successor) rather than MQA.
- The compute savings are primarily at *inference*, not training — training throughput barely changes.
- MQA is a special case of GQA with G=1 (one group, one K/V head per group).
```
