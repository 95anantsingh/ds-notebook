# Scaled Dot-Product Attention

The foundational attention operation introduced in *Attention Is All You Need* (2017) {bdg-info}`NeurIPS 2017`. Every modern transformer variant — including Multi-Head, Causal, Cross, MQA, and GQA — is built on this single function.

---

## Intuition

::::{grid} 1 1 1 2
:gutter: 3

:::{grid-item}
:columns: 12 12 12 7
:child-align: start

Think of attention as a soft dictionary lookup. You have a **query** (what you're looking for), a set of **keys** (what each entry can be matched against), and **values** (the content to retrieve). The dot product between the query and each key produces a relevance score; softmax turns these into a probability distribution; the output is a weighted sum of values.

**What each matrix represents:**

- **Query (Q)** — the "question" a token is asking. Token $i$ broadcasts: *"I need information of this kind."*
- **Key (K)** — each token's "advertisement." Token $j$ broadcasts: *"This is the kind of information I can offer."*
- **Value (V)** — the actual content to share. Once token $j$ is deemed relevant, $V_j$ is what gets mixed into the output.

**Concrete example:** In *"The cat sat on the mat because **it** was tired"*, when computing attention for *it*:

- The query of *it* asks: *"who or what am I referring to?"*
- *cat* has a key that scores high against that query
- So the output for *it* absorbs a large portion of *cat*'s value — its semantic content

The key insight: **Q and K determine *how much* each token contributes; V determines *what* it contributes.** You can change what information flows without changing who attends to whom.
:::

:::{grid-item}
:child-align: center
:columns: 12 12 12 5

```{figure} assets/scaled-dot-product.png
:width: 60%
:align: center
:alt: Scaled dot-product attention — soft lookup over Q, K, V
:name: fig-sdpa

*Scaled dot-product attention as a soft lookup.*
```
:::
::::

---

## Theory

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
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
\quad \in \mathbb{R}^{B \times T \times d_v}
$$

Where,

:$Q$: Query matrix — what the current position is looking for
:$K$: Key matrix — what each position advertises for matching
:$V$: Value matrix — the content to retrieve
:$B$: Batch size
:$T$: Sequence length ($T_q = T_k = T$ in self-attention; separate lengths in cross-attention)
:$d_k$: Key/query dimension per head; also the scale factor
:$d_v$: Value dimension per head

The $\frac{1}{\sqrt{d_k}}$ scaling prevents the dot products from growing large in magnitude as $d_k$ increases, which would push softmax into regions of vanishing gradients.

**Paper:** [Attention Is All You Need — Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) {bdg-info}`NeurIPS 2017`

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

**Why does the scale matter?**

Without scaling, as $d_k$ grows the dot products grow in variance proportional to $d_k$. Concretely, if $q$ and $k$ are i.i.d. zero-mean unit-variance vectors, then $q \cdot k \sim \mathcal{N}(0, d_k)$. After softmax, the distribution becomes peaked on one entry and gradients nearly vanish. Dividing by $\sqrt{d_k}$ brings the variance back to 1 regardless of head dimension.

### Example

$T = 2$ tokens, $d_k = 2$. Let $Q = K = I_2$ and $V = \begin{bmatrix}1 & 2\\ 3 & 4\end{bmatrix}$.

**Step 1 -- Scores**: $$QK^\top = I_2$$

**Step 2 -- Scale**:
$$\displaystyle\frac{QK^\top}{\sqrt{2}} = \begin{bmatrix}0.71 & 0\\ 0 & 0.71\end{bmatrix}$$

**Step 3 -- Softmax** row-wise:
$$W = \begin{bmatrix}0.67 & 0.33\\ 0.33 & 0.67\end{bmatrix}$$
Token 0 attends 67% to itself and 33% to token 1; token 1 attends symmetrically.

**Step 4 -- Output** $WV$:
$$\text{Output} = \begin{bmatrix}0.67 & 0.33\\ 0.33 & 0.67\end{bmatrix}\begin{bmatrix}1 & 2\\ 3 & 4\end{bmatrix} = \begin{bmatrix}1.66 & 2.66\\ 2.34 & 3.34\end{bmatrix}$$

### Complexity

| Property | Value |
|---|---|
| Time | $O(T^2 \cdot d_k)$ |
| Memory | $O(T^2)$ — attention matrix always materialized |
| Extra params | None — weight matrices live in the calling module |
| Parallelizable | Yes — all positions computed simultaneously |
| Bottleneck | Quadratic in $T$ — dominates for long sequences |
| Attention matrix | Always stored in full; eliminated by Flash Attention |

---

## Implementation

::::{tab-set}

:::{tab-item} PyTorch (Manual)

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

:::{tab-item} PyTorch Built-in

{bdg-success}`PyTorch 2.0+` &nbsp; `F.scaled_dot_product_attention` dispatches automatically to the most efficient kernel — Flash Attention, Memory-Efficient Attention, or math fallback — depending on hardware and inputs.

```python
import torch.nn.functional as F

# (B, T_q, d_k) — or with heads: (B, H, T_q, d_k)
output = F.scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,   # float mask (−∞ for masked positions) or bool (False = mask out)
    dropout_p=0.0,
    is_causal=False,  # set True to apply causal mask automatically
)  # (B, T_q, d_v)
```

```{note}
`F.scaled_dot_product_attention` returns **only the output tensor** — it does not expose attention weights. Use the manual implementation when you need weights for visualization or interpretability.
```

Prefer this over hand-rolled implementations whenever possible — numerically identical but significantly faster and more memory-efficient.

:::

:::{tab-item} JAX

```python
import jax
import jax.numpy as jnp

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = jnp.matmul(q, k.swapaxes(-2, -1)) / jnp.sqrt(d_k)    # (B, T_q, T_k)
    if mask is not None:
        scores = jnp.where(mask, scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)                     # (B, T_q, T_k)
    return jnp.matmul(weights, v)                                 # (B, T_q, d_v)
```

:::

::::

:::{admonition} Pitfalls
:class: warning
- **Overflow before scaling**: Always divide by $\sqrt{d_k}$ *before* softmax, never after.
- **Mask convention**: `masked_fill` expects `True` to mean *keep* or *mask out*, depending on your convention — be consistent. PyTorch's built-in uses `False = mask out` for boolean masks.
- **-inf vs large negative**: Use `float("-inf")` not `-1e9`; `-1e9` can cause NaN in mixed-precision (`bfloat16` dynamic range is ~3.4 × 10³⁸ so it's fine, but `float16` overflows at ~65504).
- **Attention to padding tokens**: Don't forget to pass a padding mask if your batch has variable-length sequences; unmasked padding positions will leak signal.
:::

```{tip}
- The output shape is always `(B, T_q, d_v)` — same batch and query-length dimensions, but value's depth dimension.
- This is the *building block* — {doc}`Multi-Head Attention <03-multi-head-attention>` runs $H$ copies of this in parallel, each on a $d_k = d_{\text{model}} / H$ slice.
- Time complexity is $O(T^2)$ in sequence length, the primary bottleneck for long contexts (see {doc}`Flash Attention <09-flash-attention>`).
- `F.scaled_dot_product_attention` is the production API; the manual implementation above is for understanding.
```
