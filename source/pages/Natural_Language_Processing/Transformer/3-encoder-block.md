(encoder-block)=
# Transformer Encoder Block

> **Depends on**: {ref}`Multi-Head Attention <multi-head-attention>` · {ref}`Positional Encoding <positional-encoding>`

Each encoder block applies two sub-layers with residual connections and normalization:
1. Multi-head self-attention
2. Position-wise feed-forward network (FFN)

```{mermaid}
---
title: Transformer Encoder Block (Pre-Norm)
---
flowchart TD
    X["Input  (B, T, d_model)"]

    subgraph attn_block["Sub-layer 1 — Self-Attention"]
        LN1["LayerNorm 1"]
        MHA["Multi-Head Self-Attention"]
        DROP1["Dropout"]
        ADD1(["+ residual"])
    end

    subgraph ff_block["Sub-layer 2 — Feed-Forward"]
        LN2["LayerNorm 2"]
        FFN["Feed-Forward Network\nLinear(d→4d) → GELU → Linear(4d→d)"]
        DROP2["Dropout"]
        ADD2(["+ residual"])
    end

    OUT["Output  (B, T, d_model)"]

    X --> LN1
    LN1 --> MHA
    MHA --> DROP1
    DROP1 --> ADD1
    X -. skip .-> ADD1

    ADD1 --> LN2
    LN2 --> FFN
    FFN --> DROP2
    DROP2 --> ADD2
    ADD1 -. skip .-> ADD2

    ADD2 --> OUT
```

```{tip}
**Pre-norm is the modern default.** GPT-2, LLaMA, and most contemporary LLMs use pre-LayerNorm. It stabilizes training at depth by keeping the residual stream unnormalized (a "highway" for gradients). Only the original "Attention Is All You Need" paper used post-norm.
```

---

## Pre-Norm vs Post-Norm

::::{tab-set}

:::{tab-item} Pre-Norm {bdg-success}`modern default`
```python
import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model     # convention: hidden dim = 4× model dim
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.attn   = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ff     = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm1  = nn.LayerNorm(d_model)
        self.norm2  = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm: normalize BEFORE sublayer, add residual AFTER
        x = x + self.dropout(self.attn(self.norm1(x), mask=mask))   # (B, T, d_model)
        x = x + self.dropout(self.ff(self.norm2(x)))                 # (B, T, d_model)
        return x
```
:::

:::{tab-item} Post-Norm {bdg-warning}`original paper`
```python
def forward(self, x, mask=None):
    # Post-norm: normalize AFTER residual addition
    x = self.norm1(x + self.dropout(self.attn(x, mask=mask)))
    x = self.norm2(x + self.dropout(self.ff(x)))
    return x
```

Used in the original "Attention Is All You Need". Less common now due to instability at depth — with post-norm, gradients must pass through LayerNorm at every layer which can vanish in very deep networks.
:::

::::

---

## Feed-Forward Network Details

The FFN is applied **independently to each token position** — there is no interaction between positions in this sublayer (that happens in attention).

```
x: (B, T, d_model)
  → Linear(d_model → 4·d_model)    # expand
  → activation
  → Dropout
  → Linear(4·d_model → d_model)    # contract
```

::::{tab-set}

:::{tab-item} GELU {bdg-info}`GPT-2+`
```python
self.act = nn.GELU()
out = self.fc2(self.dropout(self.act(self.fc1(x))))
```
Smooth approximation to ReLU; empirically better than ReLU for transformers.
:::

:::{tab-item} ReLU {bdg-secondary}`original paper`
```python
self.act = nn.ReLU()
out = self.fc2(self.dropout(self.act(self.fc1(x))))
```
Original transformer uses ReLU. Simpler but produces dead neurons at scale.
:::

:::{tab-item} SwiGLU {bdg-info}`LLaMA`
```python
# Gated linear unit — two projections, one acts as a gate
gate = F.silu(self.fc_gate(x))    # (B, T, d_ff)
out  = gate * self.fc_up(x)       # element-wise gate
out  = self.fc_down(out)
```
LLaMA-style. Uses `d_ff ≈ 8/3 × d_model` (not 4×) to keep parameter count comparable.
:::

::::

---

## Full Encoder Stack

```python
class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_len, dropout=0.1):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model)
        self.pos_enc = LearnedPositionalEncoding(d_model, max_len, dropout)
        self.layers  = nn.ModuleList([
            EncoderBlock(d_model, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)   # final norm (pre-norm convention)

    def forward(self, x, mask=None):
        # x: (B, T) token ids
        x = self.pos_enc(self.embed(x))     # (B, T, d_model)
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)                 # (B, T, d_model)
```

---

## Pre-norm vs Post-norm Summary

| | Pre-norm | Post-norm |
|--|---------|-----------|
| **LayerNorm placement** | Before sublayer | After residual add |
| **Used by** | GPT-2+, LLaMA, most modern LLMs | Original transformer |
| **Training stability** | More stable at depth | Requires careful warmup |
| **Residual highway** | Unobstructed | Passes through norm |
