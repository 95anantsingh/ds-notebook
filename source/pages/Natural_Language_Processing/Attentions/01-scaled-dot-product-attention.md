# Scaled Dot-Product Attention

The foundational attention operation introduced in [*Attention Is All You Need*](arxiv:1706.03762) {bdg-info}`NeurIPS 2017`. Every modern transformer variant -- including Multi-Head, Causal, Cross, MQA, and GQA -- is built on this single function.

---

## Intuition

:::::{grid} 1 1 1 2
:gutter: 3

::::{grid-item}
:columns: 12 12 12 7
:child-align: start

Think of attention as a soft dictionary lookup -- you have a **query** (what you're looking for), **keys** (what each token advertises for matching), and **values** (the actual content to retrieve). The dot product between Q and K is a similarity score: high when two tokens are relevant to each other, low otherwise. Softmax converts all scores for a query into a probability distribution (weights that sum to 1), and the output is a weighted blend of values -- so tokens with high Q-K similarity contribute more of their content to the result.

**Key insight: Q and K determine *how much* each token contributes; V determines *what* it contributes.**

**What each matrix represents:**

- **Query (Q)** -- the "question" a token is asking. Token $i$ broadcasts: *"I need information of this kind."*
- **Key (K)** -- each token's "advertisement." Token $j$ broadcasts: *"This is the kind of information I can offer."*
- **Value (V)** -- the actual content to share. Once token $j$ is deemed relevant, $V_j$ is what gets mixed into the output.

:::{dropdown} Why scale?
:color: primary
:icon: info
:animate: fade-in-slide-down
As $d_k$ grows, dot products grow in magnitude and softmax saturates -- collapsing to near one-hot weights where gradients vanish. Dividing by $\sqrt{d_k}$ keeps scores in a range where softmax stays well-behaved.
:::

:::{dropdown} Why $\sqrt{d_k}$ specifically?
:color: primary
:icon: info
:animate: fade-in-slide-down
If $q$ and $k$ are i.i.d. zero-mean unit-variance vectors, each term $q_i k_i$ has variance 1. Their dot product sums $d_k$ such terms, so $\text{Var}(q \cdot k) = d_k$ and $\text{std}(q \cdot k) = \sqrt{d_k}$. Dividing by $\sqrt{d_k}$ brings the standard deviation back to 1 -- the natural normalizer that cancels exactly the growth introduced by the dimensionality.
:::


::::

::::{grid-item}
:child-align: center
:columns: 12 12 12 5

```{figure} assets/scaled-dot-product.png
:width: 60%
:align: center
:alt: Scaled dot-product attention -- soft lookup over Q, K, V
:name: fig-sdpa

*Scaled dot-product attention as a soft lookup.*
```
:::
:::::

---

## Deep Dive

:::::{grid} 1 1 1 2
:gutter: 3

::::{grid-item}
:columns: 12 12 12 7

$$
Q \in \mathbb{R}^{B \times T \times d_k}, \quad
K \in \mathbb{R}^{B \times T \times d_k}, \quad
V \in \mathbb{R}^{B \times T \times d_v}
$$

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
\quad \in \mathbb{R}^{B \times T \times d_v}
$$


Where,

:$Q$: Query matrix -- what the current position is looking for
:$K$: Key matrix -- what each position advertises for matching
:$V$: Value matrix -- the content to retrieve
:$B$: Batch size
:$T$: Sequence length ($T_q = T_k = T$ in self-attention; separate lengths in cross-attention)
:$d_k$: Key/query dimension per head; also the scale factor
:$d_v$: Value dimension per head


::::

::::{grid-item}
:columns: 12 12 12 5
:child-align: center

:::{container} mermaid-w-xs
```{mermaid}
flowchart TD
    Q([Q]) --> E["$$QK^T / \sqrt{d_k}$$<br>(scores)"]
    K([K]) --> E
    E --> F["softmax<br>(weights)"]
    F --> G["weights · V<br>(output)"]
    V([V]) --> G
    G --> H([Output])
```
:::

::::
:::::


### Example

Given that -- $B = 3$, $T = 3$, $d_k = 4$, $d_v = 2$. The following four steps run independently on every batch element (the two depth slices in each figure); the figure above shows $Q$, $K^\top$, and the resulting scores.

::::{grid} 1 1 1 2
:gutter: 3

:::{grid-item}
:columns: 12 12 12 4
:child-align: center
**Step 1 -- Scores**: dot every query row with every key row.
$$\text{scores} = QK^\top \quad \in \mathbb{R}^{B \times T \times T}$$
:::

:::{grid-item}
:columns: 12 12 12 8
:child-align: start
```{glue} scores_plot
:doc: assets/scaled-dot-product.ipynb
```
:::
::::

::::{grid} 1 1 1 2
:gutter: 3

:::{grid-item}
:columns: 12 12 12 4
:child-align: center
**Step 2 -- Scale**: divide by $\sqrt{d_k} = 2$ to keep the variance near 1 before softmax.
$$\text{scaled} = \frac{QK^\top}{\sqrt{d_k}}$$
:::

:::{grid-item}
:columns: 12 12 12 8
:child-align: start
```{glue} scaled_plot
:doc: assets/scaled-dot-product.ipynb
```
:::
::::

::::{grid} 1 1 1 2
:gutter: 3

:::{grid-item}
:columns: 12 12 12 4
:child-align: center
**Step 3 -- Softmax** (row-wise): each row becomes a probability distribution summing to 1.
$$W = \text{softmax}(\text{scaled})$$
:::

:::{grid-item}
:columns: 12 12 12 8
:child-align: start
```{glue} weights_plot
:doc: assets/scaled-dot-product.ipynb
```
:::
::::

::::{grid} 1 1 1 2
:gutter: 3

:::{grid-item}
:columns: 12 12 12 4
:child-align: center
**Step 4 -- Output**: blend the value rows by the attention weights.
$$\text{Output} = WV \quad \in \mathbb{R}^{B \times T \times d_v}$$
:::

:::{grid-item}
:columns: 12 12 12 8
:child-align: start
```{glue} output_plot
:doc: assets/scaled-dot-product.ipynb
```
:::
::::

### Complexity

::::{grid} 1 1 1 2
:gutter: 3

:::{grid-item}
:columns: 12 12 12 6
:child-align: center

:Time: $O(T^2 \cdot d_k)$
:Memory: $O(T^2)$ -- attention matrix always materialized
:Extra params: None -- weight matrices live in the calling module

:::

:::{grid-item}
:columns: 12 12 12 6
:child-align: start

:Parallelizable: Yes -- all positions computed simultaneously
:Bottleneck: Quadratic in $T$ -- dominates for long sequences
:Attention matrix: Always stored in full; eliminated by Flash Attention

:::
::::
---

## Implementation

::::{tab-set}

:::{tab-item} PyTorch

```{code-block} python
:linenos:
:emphasize-lines: 20,23,25,26

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,  # (B, T_q, d_k)
        k: torch.Tensor,  # (B, T_k, d_k)
        v: torch.Tensor,  # (B, T_k, d_v)
        mask: torch.Tensor | None = None,  # (B, T_q, T_k) bool, True = keep
    ) -> tuple[torch.Tensor, torch.Tensor]:
        d_k = q.size(-1)
        scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)  # (B, T_q, T_k)

        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))  # (B, T_q, T_k)

        weights = self.dropout(F.softmax(scores, dim=-1))  # (B, T_q, T_k)
        output = weights @ v                                # (B, T_q, d_v)
        return output, weights
```

Use this when you need to inspect attention weights for visualization or interpretability.

:::

:::{tab-item} PyTorch Optimized
```python
```
:::

:::{tab-item} PyTorch Built-in

{bdg-success}`PyTorch 2.0+` &nbsp; `F.scaled_dot_product_attention` dispatches automatically to the most efficient kernel -- Flash Attention, Memory-Efficient Attention, or math fallback -- depending on hardware and inputs.

```python
import torch.nn.functional as F

# (B, T_q, d_k) -- or with heads: (B, H, T_q, d_k)
output = F.scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,  # float mask (−∞ for masked positions) or bool (False = mask out)
    dropout_p=0.0,
    is_causal=False,  # set True to apply causal mask automatically
)  # (B, T_q, d_v)
```

```{note}
`F.scaled_dot_product_attention` returns **only the output tensor** -- it does not expose attention weights. Use the manual implementation when you need weights for visualization or interpretability.
```

Prefer this over hand-rolled implementations whenever possible -- numerically identical but significantly faster and more memory-efficient.

:::

:::{tab-item} JAX

```python
import jax
import jax.numpy as jnp


def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = jnp.matmul(q, k.swapaxes(-2, -1)) / jnp.sqrt(d_k)  # (B, T_q, T_k)
    if mask is not None:
        scores = jnp.where(mask, scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)  # (B, T_q, T_k)
    return jnp.matmul(weights, v)  # (B, T_q, d_v)
```

:::

::::

:::{admonition} Pitfalls
:class: warning
- **Overflow before scaling**: Always divide by $\sqrt{d_k}$ *before* softmax, never after.
- **Mask convention**: `masked_fill` expects `True` to mean *keep* or *mask out*, depending on your convention -- be consistent. PyTorch's built-in uses `False = mask out` for boolean masks.
- **-inf vs large negative**: Use `float("-inf")` not `-1e9`; `-1e9` can cause NaN in mixed-precision (`bfloat16` dynamic range is ~3.4 × 10³⁸ so it's fine, but `float16` overflows at ~65504).
- **Attention to padding tokens**: Don't forget to pass a padding mask if your batch has variable-length sequences; unmasked padding positions will leak signal.
:::

```{tip}
- The output shape is always `(B, T_q, d_v)` -- same batch and query-length dimensions, but value's depth dimension.
- This is the *building block* -- {doc}`Multi-Head Attention <03-multi-head-attention>` runs $H$ copies of this in parallel, each on a $d_k = d_{\text{model}} / H$ slice.
- Time complexity is $O(T^2)$ in sequence length, the primary bottleneck for long contexts (see {doc}`Flash Attention <09-flash-attention>`).
- `F.scaled_dot_product_attention` is the production API; the manual implementation above is for understanding.
```
