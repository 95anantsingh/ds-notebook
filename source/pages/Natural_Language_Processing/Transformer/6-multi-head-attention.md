(multi-head-attention)=
# Multi-Head Attention

> **Depends on**: {ref}`Scaled Dot-Product Attention <scaled-dot-product-attention>`

Instead of a single attention function over full `d_model`-dimensional Q/K/V, MHA projects into `H` smaller subspaces of dimension `d_k = d_model // H`, computes attention in each, then concatenates and projects back.

$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H)\, W^O$$
$$\text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)$$

```{mermaid}
---
title: Multi-Head Attention
---
flowchart TD
    X["Input  (B, T, d_model)"]

    subgraph project["Projection  —  d_k = d_model ÷ H"]
        QKV["qkv_proj: Linear → (B, T, 3·d_model)"]
        CHUNK["chunk(3, dim=-1)"]
        Q["Q  (B, T, d_model)"]
        K["K  (B, T, d_model)"]
        V["V  (B, T, d_model)"]
    end

    subgraph heads["Split Heads  →  (B, H, T, d_k)"]
        SH["view + transpose(1,2)"]
        ATTN["Scaled Dot-Product Attention\n× H heads in parallel"]
    end

    subgraph merge["Merge Heads  →  (B, T, d_model)"]
        MH["transpose(1,2).contiguous().view"]
        OUT["out_proj: Linear → (B, T, d_model)"]
    end

    RESULT["Output  (B, T, d_model)"]

    X --> QKV
    QKV --> CHUNK
    CHUNK --> Q & K & V
    Q & K & V --> SH
    SH --> ATTN
    ATTN --> MH
    MH --> OUT
    OUT --> RESULT
```

---

## Implementation

```{tip}
Use a single fused `qkv_proj` (one `3·d_model` linear) rather than three separate linears. Fewer kernel launches and better GPU utilization.
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Fused QKV projection (more efficient than three separate linears)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                      # (B, T, d_model) — self-attention
        context: torch.Tensor | None = None,   # (B, S, d_model) — cross-attention encoder output
        mask: torch.Tensor | None = None,      # (B, 1, T, T) or (T, T), bool
    ) -> torch.Tensor:
        B, T, _ = x.shape
        is_cross = context is not None
        kv_src = context if is_cross else x

        if is_cross:
            # Separate projections for cross-attention
            q = self._split_heads(nn.Linear(self.d_model, self.d_model)(x), B)
            k = self._split_heads(nn.Linear(self.d_model, self.d_model)(kv_src), B)
            v = self._split_heads(nn.Linear(self.d_model, self.d_model)(kv_src), B)
        else:
            # Fused QKV for self-attention
            qkv = self.qkv_proj(x)                     # (B, T, 3*d_model)
            q, k, v = qkv.chunk(3, dim=-1)             # each: (B, T, d_model)
            q = self._split_heads(q, B)                # (B, H, T, d_k)
            k = self._split_heads(k, B)
            v = self._split_heads(v, B)

        # Scaled dot-product attention
        d_k = q.size(-1)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)   # (B, H, T, T)
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))          # (B, H, T, T)
        out = attn @ v                                           # (B, H, T, d_k)

        # Merge heads and project
        out = self._merge_heads(out, B, T)                      # (B, T, d_model)
        return self.out_proj(out)

    def _split_heads(self, x: torch.Tensor, B: int) -> torch.Tensor:
        # (B, T, d_model) → (B, H, T, d_k)
        T = x.size(1)
        return x.view(B, T, self.num_heads, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor, B: int, T: int) -> torch.Tensor:
        # (B, H, T, d_k) → (B, T, d_model)
        return x.transpose(1, 2).contiguous().view(B, T, self.d_model)
```

---

## Shape Walkthrough

| Step | Operation | Shape |
|------|-----------|-------|
| Input | `x` | `(B, T, d_model)` |
| QKV projection | `qkv_proj(x)` | `(B, T, 3·d_model)` |
| After chunk | `q`, `k`, `v` | `(B, T, d_model)` each |
| Split heads | `.view(...).transpose(1,2)` | `(B, H, T, d_k)` |
| Attention scores | `q @ k.T` / sqrt(d_k) | `(B, H, T, T)` |
| Attention weights | softmax | `(B, H, T, T)` |
| Context vectors | `attn @ v` | `(B, H, T, d_k)` |
| Merge heads | `.transpose(1,2).view(...)` | `(B, T, d_model)` |
| Output projection | `out_proj(...)` | `(B, T, d_model)` |

**Critical**: `d_k = d_model // H`, so `H × d_k = d_model`. The merge step recovers the original dimension.

---

## The Split/Merge Trick

```python
# Split: (B, T, d_model) → (B, H, T, d_k)
x.view(B, T, H, d_k).transpose(1, 2)

# Merge: (B, H, T, d_k) → (B, T, d_model)
x.transpose(1, 2).contiguous().view(B, T, d_model)
#          ↑ must call contiguous() after transpose before view
```

---

## Grouped Query Attention (GQA)

Used in LLaMA 2/3, Mistral. `num_kv_heads < num_heads` — queries are grouped and share K/V heads:

```python
# num_kv_heads = num_heads // groups
k = k.repeat_interleave(groups, dim=1)   # expand K from (B, kv_H, T, d_k) to (B, H, T, d_k)
v = v.repeat_interleave(groups, dim=1)
```

Multi-Query Attention (MQA) is the extreme case: `num_kv_heads = 1`.

---

```{note}
- **Bias**: Most modern LLMs omit bias in QKV projections (`bias=False`) — it doesn't add expressivity when LayerNorm is applied before the sublayer.
- **Efficiency**: The fused `qkv_proj` (one `3·d_model` linear) is faster than three separate linears due to fewer kernel launches.
- **`F.scaled_dot_product_attention`**: Replace the manual attention block with this for FlashAttention on CUDA (PyTorch ≥ 2.0).
```
