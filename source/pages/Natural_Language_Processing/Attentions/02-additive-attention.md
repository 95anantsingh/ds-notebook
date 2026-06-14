# Additive (Bahdanau) Attention

The first attention mechanism, introduced in 2015 for neural machine translation. It allowed seq2seq models to learn *where* in the source sentence to look when generating each target word, removing the information bottleneck of a fixed-length encoder vector.

---

## Intuition

In a classic encoder-decoder model, the decoder receives one fixed summary vector from the encoder — a severe bottleneck for long sentences. Additive attention replaces this with a dynamic context vector: at each decoding step, a small alignment network scores every encoder hidden state against the current decoder state, and the context is a weighted sum of encoder states based on those scores.

```{mermaid}
flowchart TD
    A([Encoder hidden states\nh₁ … hₙ]) --> C["Alignment score\ne_ij = vᵀ tanh(W₁hⱼ + W₂sᵢ)"]
    B([Decoder state sᵢ]) --> C
    C --> D["softmax → α_ij\n(attention weights)"]
    D --> E["cᵢ = Σⱼ α_ij hⱼ\n(context vector)"]
    E --> F([Decoder output])
```

---

## Theory

> **Paper:** [Neural Machine Translation by Jointly Learning to Align and Translate — Bahdanau et al. (2015)](https://arxiv.org/abs/1409.0473)

Let $\mathbf{h}_j \in \mathbb{R}^{d_h}$ be the $j$-th encoder hidden state and $\mathbf{s}_i \in \mathbb{R}^{d_s}$ be the decoder state at step $i$.

$$
e_{ij} = \mathbf{v}^\top \tanh\!\bigl(\mathbf{W}_1\, \mathbf{h}_j + \mathbf{W}_2\, \mathbf{s}_i\bigr)
\quad \mathbf{v} \in \mathbb{R}^{d_a},\; \mathbf{W}_1 \in \mathbb{R}^{d_a \times d_h},\; \mathbf{W}_2 \in \mathbb{R}^{d_a \times d_s}
$$

$$
\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
\quad \boldsymbol{\alpha}_i \in \mathbb{R}^{T_x}
$$

$$
\mathbf{c}_i = \sum_{j=1}^{T_x} \alpha_{ij}\, \mathbf{h}_j
\quad \mathbf{c}_i \in \mathbb{R}^{d_h}
$$

The parameters $\mathbf{W}_1$, $\mathbf{W}_2$, and $\mathbf{v}$ are the *alignment model* — a small MLP trained end-to-end with the encoder and decoder.

### Complexity

| Property | Value |
|---|---|
| Time | $O(T_x \cdot T_y \cdot d_a)$ for one sequence pair |
| Memory | $O(T_x \cdot T_y)$ for the alignment matrix |
| Extra params | $\mathbf{W}_1 \in \mathbb{R}^{d_a \times d_h}$, $\mathbf{W}_2 \in \mathbb{R}^{d_a \times d_s}$, $\mathbf{v} \in \mathbb{R}^{d_a}$ |

---

## PyTorch Module

```{code-block} python
:linenos:
:emphasize-lines: 17,18,19

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    """Bahdanau-style additive (content-based) attention."""

    def __init__(self, query_dim: int, key_dim: int, hidden_dim: int):
        super().__init__()
        self.W_query = nn.Linear(query_dim, hidden_dim, bias=False)
        self.W_key   = nn.Linear(key_dim,   hidden_dim, bias=False)
        self.v       = nn.Linear(hidden_dim, 1,          bias=False)

    def forward(
        self,
        query: torch.Tensor,  # (B, d_s)        — decoder state
        keys:  torch.Tensor,  # (B, T_x, d_h)   — encoder hidden states
        values: torch.Tensor, # (B, T_x, d_v)   — usually same as keys
        mask: torch.Tensor | None = None,  # (B, T_x) bool, True = keep
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Expand query to match key sequence length
        q = self.W_query(query).unsqueeze(1)          # (B, 1, d_a)
        k = self.W_key(keys)                          # (B, T_x, d_a)
        energy = self.v(torch.tanh(q + k)).squeeze(-1) # (B, T_x)

        if mask is not None:
            energy = energy.masked_fill(~mask, float("-inf"))  # (B, T_x)

        weights = F.softmax(energy, dim=-1)            # (B, T_x)
        context = (weights.unsqueeze(1) @ values).squeeze(1)  # (B, d_v)
        return context, weights
```

---

## Modern Usage

Additive attention was **superseded by scaled dot-product attention** for most tasks — dot-product is faster (uses matrix multiplication rather than per-pair additions) and parallelizes trivially across sequence positions.

**When you'd still reach for additive attention:**

- **Interpretable alignment weights** are required (e.g., aligning source and target tokens for analysis or post-hoc inspection).
- **Lightweight CPU inference** on small encoder-decoder models where the MLP overhead is negligible compared to the RNN body.
- **Key and query have very different dimensions** ($d_q \neq d_k$) and you don't want to add projection layers — the separate $\mathbf{W}_1$, $\mathbf{W}_2$ matrices handle this naturally.
- Replicating the exact architecture of **classic NMT papers** (Bahdanau, Luong) for research comparison.

```python
# Modern equivalent for most tasks:
import torch.nn.functional as F

# Project to the same dimension first, then use dot-product
q = nn.Linear(query_dim, d_k)(query)   # (B, d_k)
k = nn.Linear(key_dim,   d_k)(keys)    # (B, T_x, d_k)
v = keys                                # (B, T_x, d_v)

context = F.scaled_dot_product_attention(
    q.unsqueeze(1), k, v
).squeeze(1)  # (B, d_v)
```

---

```{warning} Common Pitfalls
- **Broadcasting bug**: `q + k` only works if shapes broadcast correctly. `query` must be expanded to `(B, 1, d_a)` before adding to `keys` of shape `(B, T_x, d_a)`.
- **Gradient flow**: The tanh non-linearity can saturate for large pre-activations; keep `hidden_dim` moderate (32–128) and consider initializing $\mathbf{v}$ with a small norm.
- **Memory for long sequences**: The alignment matrix is $O(T_x \cdot T_y)$; for long documents this grows quickly even though there's no explicit $T^2$ quadratic bottleneck.
```

```{tip} Tips
- The alignment model introduces extra parameters ($\mathbf{W}_1$, $\mathbf{W}_2$, $\mathbf{v}$) unlike scaled dot-product, which is parameter-free.
- Bahdanau used a *bidirectional* LSTM encoder; the "hidden states" $h_j$ are the concatenated forward and backward states.
- The attention weights $\alpha_{ij}$ form an **alignment matrix** — visualizing it as a heatmap shows which source tokens the model attends to at each decoding step.
- Luong (2015) simplified this by replacing the MLP with a plain dot-product or bilinear score: $e_{ij} = \mathbf{s}_i^\top \mathbf{W}\, \mathbf{h}_j$.
```
