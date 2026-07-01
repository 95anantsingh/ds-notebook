# Multi-Query Attention (MQA)

A memory-bandwidth-optimized variant of Multi-Head Attention introduced in [*Fast Transformer Decoding: One Write-Head is All You Need*](arxiv:1911.02150) {bdg-info}`arXiv 2019`. MQA keeps $H$ independent query heads but collapses keys and values to a single shared head, reducing KV cache size by $H\times$ and making autoregressive decoding dramatically faster.

---

## Intuition

:::::{grid} 1 1 1 2
:gutter: 3

::::{grid-item}
:columns: 12 12 12 7
:child-align: start

In standard MHA, every head maintains its own K and V matrices -- but during incremental decoding, the GPU stalls waiting for KV cache reads, not on compute. MQA eliminates $H - 1$ copies of K and V: all query heads share a single K and a single V. This reduces the KV cache by $H\times$ and turns a memory-bandwidth bottleneck into a much lighter one, with only a modest quality cost.

**Key insight: during decoding, the bottleneck is reading K and V from memory, not arithmetic. Sharing one K/V head cuts those reads by $H\times$ -- the primary speedup.**

**What changes relative to MHA:**

- **Query ($Q$)** -- unchanged: $H$ independent projections; each head still learns to look for different patterns.
- **Key ($K$)** -- collapsed to 1 shared head; all $H$ query heads query the same key space.
- **Value ($V$)** -- collapsed to 1 shared head; all $H$ query heads retrieve from the same value space.
- **KV cache** -- stores 2 tensors instead of $2H$; memory footprint shrinks by $H\times$.

:::{dropdown} Why share K and V, not Q?
:color: primary
:icon: info
:animate: fade-in-slide-down
Query heads embody different "questions" -- what the model is looking for. Each head $h$ uses its own $W_Q^h$ to project $X$ into a different semantic subspace, so heads specialize in distinct patterns. Keys and values describe "what each token offers" -- sharing them means all heads search the same token representations, with diversity coming entirely from the query projections. Sharing Q instead would collapse all heads to identical lookups, destroying multi-head expressiveness entirely.
:::

:::{dropdown} How does sharing K/V speed up inference?
:color: primary
:icon: info
:animate: fade-in-slide-down
During autoregressive decoding the model generates one token at a time. For each new token, it reads all past K and V vectors from the KV cache in GPU memory. With $H$ heads and a sequence of length $T$, MHA reads $2 \cdot H \cdot T \cdot d_k$ floats per step -- a memory-bandwidth bottleneck, not a compute bottleneck. MQA reduces this to $2 \cdot T \cdot d_k$ floats (one K/V head), cutting bandwidth pressure by $H\times$ and making decoding significantly faster.
:::

::::

::::{grid-item}
:child-align: center
:columns: 12 12 12 5

```{mermaid}
flowchart TD
    X(["Input X"]) --> Q["W_Q → H query heads<br>(B, T, H·d_k)"]
    X --> K["W_K → 1 key head<br>(B, T, d_k)  ← shared"]
    X --> V["W_V → 1 value head<br>(B, T, d_k)  ← shared"]
    Q --> S["Attention per head<br>(K, V shared across all H)"]
    K --> S
    V --> S
    S --> O["Concat + W_O<br>(B, T, d_model)"]
    O --> OUT([Output])
```

::::
:::::

---

## Deep Dive

:::::{grid} 1 1 1 2
:gutter: 3

::::{grid-item}
:columns: 12 12 12 7

$$
Q = X W_Q \in \mathbb{R}^{B \times T \times H \cdot d_k}
\quad \text{($H$ query heads)}
$$

$$
K = X W_K \in \mathbb{R}^{B \times T \times d_k}
\quad \text{(1 shared key head)}
$$

$$
V = X W_V \in \mathbb{R}^{B \times T \times d_v}
\quad \text{(1 shared value head)}
$$

Each query head $h$ attends using the same $K$ and $V$:

$$
\text{head}_h = \text{Attention}(Q_h,\, K,\, V),
\quad Q_h \in \mathbb{R}^{B \times T \times d_k}
$$

$$
\text{MQA}(X) = \text{Concat}(\text{head}_1,\ldots,\text{head}_H)\, W_O
\in \mathbb{R}^{B \times T \times d_{\text{model}}}
$$

Where,

:$X$: Input sequence
:$Q$: Query matrix -- $H$ independent heads
:$K$, $V$: Key and value matrices -- 1 shared head each
:$Q_h$: Query for head $h$; $K$ and $V$ are the same for all $h$
:$H$: Number of query heads
:$d_k$: Head dimension; $d_{\text{model}} = H \cdot d_k$
:$B$: Batch size; $T$: sequence length
:$W_Q \in \mathbb{R}^{d \times H d_k}$: Full-rank query projection
:$W_K, W_V \in \mathbb{R}^{d \times d_k}$: Rank-reduced key/value projections
:$W_O \in \mathbb{R}^{H d_k \times d}$: Output projection

::::

::::{grid-item}
:columns: 12 12 12 5
:child-align: center

:::{container} w-xs
```{mermaid}
flowchart TD
    Q(["Q (B,H,T,d_k)"]) --> A["$$\text{softmax}\!\left(\frac{Q_h K^T}{\sqrt{d_k}}\right)V$$<br>(broadcast K,V across H)"]
    K(["K (B,1,T,d_k)"]) --> A
    V(["V (B,1,T,d_k)"]) --> A
    A --> C["Concat heads<br>(B,T,H·d_k)"]
    C --> O(["W_O → output<br>(B,T,d_model)"])
```
:::

::::
:::::

### Complexity

::::{grid} 1 1 1 2
:gutter: 3

:::{grid-item}
:columns: 12 12 12 6
:child-align: center

:Time: $O(T^2 \cdot d_k \cdot H)$ -- same as MHA
:KV cache: $O(T \cdot d_k \cdot L)$ -- $H\times$ smaller than MHA
:Parameters: Fewer -- $W_K, W_V \in \mathbb{R}^{d \times d_k}$ vs $\mathbb{R}^{d \times H d_k}$

:::

:::{grid-item}
:columns: 12 12 12 6
:child-align: start

:Training compute: Unchanged -- same FLOPs as MHA
:Decoding bandwidth: Reduced $H\times$ -- dominant cost at inference
:Quality: Slightly below MHA; GQA (G=2–8) is a better middle ground

:::
::::

---

## Implementation

::::{tab-set}

:::{tab-item} PyTorch

```{code-block} python
:linenos:
:emphasize-lines: 22,23,24

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.H   = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, num_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, self.d_k,             bias=False)
        self.W_v = nn.Linear(d_model, self.d_k,             bias=False)
        self.W_o = nn.Linear(num_heads * self.d_k, d_model, bias=False)
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
        return self.W_o(out)
```

`k` and `v` have shape `(B, 1, T, d_k)` -- PyTorch's `scaled_dot_product_attention` broadcasts the single K/V head across all H query heads automatically, with no extra allocation.

:::

:::{tab-item} Flash Attention 2

{bdg-success}`flash-attn ≥2.0` &nbsp; Flash Attention 2 natively supports MQA/GQA -- pass K/V with a single head and it handles the broadcast internally with fused CUDA kernels.

```python
from flash_attn import flash_attn_func

# Shapes use (B, T, H, d_k) — seq_len before heads (flash-attn convention)
# q: (B, T, H, d_k)    — all H query heads
# k, v: (B, T, 1, d_k) — single shared head
out = flash_attn_func(q, k, v, causal=True)  # (B, T, H, d_k)
```

Prefer this over the PyTorch module for production inference -- the fused kernel eliminates HBM traffic for the attention score matrix on top of the KV cache savings.

:::

::::

---

## When to Use

MQA targets the **inference memory-bandwidth bottleneck** -- it does not help training throughput.

**Reach for MQA when:**

- **Maximizing decoding throughput** -- KV cache reads dominate inference time; $H\times$ reduction in cache size translates directly to higher tokens/sec.
- **GPU memory is constrained** -- serving large models with long contexts; MQA lets you fit more sequences per device.
- **Latency is critical** -- MQA is strictly faster than MHA at inference with minimal accuracy cost for many tasks.

**Consider GQA instead when:**

- Quality matters more than raw speed -- GQA (G=2–8 groups) recovers most of MQA's throughput benefit while closing much of the quality gap to MHA.
- You are training from scratch -- GQA is now the default in most modern LLMs (LLaMA 2 70B, Mistral, Gemma).

**Models using MQA:** PaLM, Falcon, StarCoder.

---

:::{admonition} Pitfalls
:class: warning
- **Training quality degradation**: MQA can hurt perplexity, especially on tasks requiring diverse attention patterns. GQA with G=2–4 is usually a better default for new training runs.
- **`.expand()` vs `.repeat()`**: When manually broadcasting K/V, use `.expand()` (zero-copy stride trick) not `.repeat()` (allocates a full H-copies tensor) -- the opposite of what MQA is trying to achieve.
- **Mixed-precision KV cache**: Store the cache in `float16`/`bfloat16`, not `float32`; the memory saving of MQA is halved otherwise.
- **Checkpoint compatibility**: MHA checkpoints cannot be loaded into an MQA model without discarding K/V weights -- fix the architecture before training begins.
:::

```{tip}
- MQA reduces the KV cache by exactly $H\times$ vs MHA -- for H=32 (LLaMA-scale), that is a 32× reduction in KV memory per layer.
- MQA is a special case of GQA with G=1 (one group, one K/V head shared by all query heads).
- The compute savings are inference-only -- training FLOPs and throughput are unchanged.
- PyTorch's `F.scaled_dot_product_attention` broadcasts `(B, 1, T, d_k)` K/V to `(B, H, T, d_k)` Q automatically -- no explicit `.expand()` needed.
```
