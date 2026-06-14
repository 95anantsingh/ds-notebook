# Multi-Head Attention

The core operation of the Transformer (2017). Instead of running a single attention function, Multi-Head Attention projects queries, keys, and values into $H$ lower-dimensional subspaces, runs attention in each subspace independently, and concatenates the results. Each head can learn to attend to different aspects of the input simultaneously.

---

## Intuition

A single attention head produces one weighted average of values — a single "view" of the input. Multiple heads let the model attend to positional information in one head, syntactic structure in another, and semantic similarity in a third, all at the same time. The final projection mixes these perspectives back into the model dimension.

```{mermaid}
flowchart TD
    A([Input X\nB × T × d_model]) --> B[W_Q\nd_model → d_model]
    A --> C[W_K\nd_model → d_model]
    A --> D[W_V\nd_model → d_model]
    B --> E[Split into H heads\nB × H × T × d_k]
    C --> F[Split into H heads\nB × H × T × d_k]
    D --> G[Split into H heads\nB × H × T × d_v]
    E --> H[Scaled dot-product\nattention per head]
    F --> H
    G --> H
    H --> I[Concat heads\nB × T × d_model]
    I --> J[W_O\nd_model → d_model]
    J --> K([Output\nB × T × d_model])
```

---

## Theory

> **Paper:** [Attention Is All You Need — Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762)

$$
Q = X W_Q,\quad K = X W_K,\quad V = X W_V
$$

$$
W_Q, W_K \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}, \quad
W_V \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}
$$

Each head $i$ operates on a $d_k$-dimensional slice ($d_k = d_{\text{model}} / H$):

$$
\text{head}_i = \text{Attention}\!\left(Q_i,\, K_i,\, V_i\right), \quad
Q_i, K_i, V_i \in \mathbb{R}^{B \times T \times d_k}
$$

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H)\, W_O
\in \mathbb{R}^{B \times T \times d_{\text{model}}}
$$

$$
W_O \in \mathbb{R}^{H d_v \times d_{\text{model}}}
$$

In the standard Transformer $d_v = d_k = d_{\text{model}} / H$, so $W_O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}$ and each head's value dimension equals its key/query dimension.

### Complexity

| Property | Value |
|---|---|
| Time | $O(T^2 \cdot d_{\text{model}})$ — same as single-head |
| Memory | $O(T^2 \cdot H)$ for attention matrices across all heads |
| Parameters | $4\, d_{\text{model}}^2$ (three in-projections + one out-projection) |

---

## PyTorch Module

```{code-block} python
:linenos:
:emphasize-lines: 21,22,23,24

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.H   = num_heads
        self.d_k = d_model // num_heads  # head dimension
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,                   # (B, T, d_model)
        mask: torch.Tensor | None = None,  # (B, T, T) bool, True = keep
    ) -> torch.Tensor:
        B, T, _ = x.shape

        # Project and split into heads
        q = self.W_q(x).view(B, T, self.H, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        k = self.W_k(x).view(B, T, self.H, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        v = self.W_v(x).view(B, T, self.H, self.d_k).transpose(1, 2)  # (B, H, T, d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)  # (B, 1, T, T) — broadcast over heads

        # Scaled dot-product attention across all heads in one batched call
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, H, T, d_k)

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)
        return self.W_o(out)                                    # (B, T, d_model)
```

---

## Modern Usage

PyTorch provides `nn.MultiheadAttention` which handles projections, head splitting, and the output projection:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(
    embed_dim=512,
    num_heads=8,
    dropout=0.1,
    batch_first=True,   # expect (B, T, d_model); default is (T, B, d_model)
)

# Self-attention
out, weights = mha(query=x, key=x, value=x)  # out: (B, T, 512)

# With key padding mask (True = ignore this position)
out, weights = mha(x, x, x, key_padding_mask=padding_mask)  # padding_mask: (B, T)
```

For new code, the module above using `F.scaled_dot_product_attention` is often preferable — it avoids the legacy `batch_first=False` default and is more transparent.

---

```{warning} Common Pitfalls
- **d_model % num_heads ≠ 0**: Head dimension `d_k = d_model / H` must be an integer. Common safe combinations: (512, 8), (768, 12), (1024, 16).
- **Forgetting `contiguous()`**: After `.transpose(1, 2)` the tensor is non-contiguous; `.view()` will fail without `.contiguous()` (or use `.reshape()` instead).
- **Mask shape mismatch**: The mask for `F.scaled_dot_product_attention` must be `(B, H, T_q, T_k)` or broadcastable to that shape.
- **bias=True in projections**: The original Transformer uses no bias in Q/K/V projections; some implementations add it. Be consistent with the architecture you're replicating.
```

```{tip} Tips
- $H$ heads in parallel has the *same total compute* as one head of size $d_{\text{model}}$ — multi-head is free in FLOPs, but adds expressiveness by covering different subspaces.
- Typical configurations: GPT-2 small uses (768, 12), BERT base uses (768, 12), GPT-3 uses (12288, 96).
- The four $d_{\text{model}}^2$ weight matrices ($W_Q$, $W_K$, $W_V$, $W_O$) are usually the largest chunk of parameters per transformer block.
- To reduce the KV cache size at inference, MHA is often replaced by MQA or GQA — see those pages.
```
