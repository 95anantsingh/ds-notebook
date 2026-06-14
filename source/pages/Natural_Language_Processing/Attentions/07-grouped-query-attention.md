# Grouped-Query Attention (GQA)

A generalization between Multi-Head Attention (H K/V heads) and Multi-Query Attention (1 K/V head), introduced in 2023. GQA divides the $H$ query heads into $G$ groups; each group shares one K head and one V head. This interpolates quality (close to MHA) and inference speed (close to MQA). Used in LLaMA 2 70B, LLaMA 3, Mistral, Gemma, and most modern LLMs.

---

## Intuition

MQA's single K/V head is an extreme — all query heads share one key and one value, which can degrade quality. GQA finds a practical middle ground: group the H query heads into G groups (e.g., G=8 for LLaMA 2 70B with H=64). Each group of $H/G$ query heads attends to one shared K and V head. KV cache shrinks by $H/G\times$ while attention diversity is maintained across groups.

```{mermaid}
flowchart TD
    A([Input X]) --> B["W_Q → Q\n(B, T, H × d_k)\nH query heads"]
    A --> C["W_K → K\n(B, T, G × d_k)\nG key heads"]
    A --> D["W_V → V\n(B, T, G × d_v)\nG value heads"]
    B --> E["Split Q into G groups\neach group: (B, H/G, T, d_k)"]
    C --> F["One K head per group\n(B, 1, T, d_k)"]
    D --> G["One V head per group\n(B, 1, T, d_v)"]
    E --> H[Attention within each group]
    F --> H
    G --> H
    H --> I["Concat all heads + W_O\n(B, T, d_model)"]
    I --> J([Output])
```

---

## Theory

> **Paper:** [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints — Ainslie et al. (2023)](https://arxiv.org/abs/2305.13245)

Let $H$ be the number of query heads and $G$ the number of groups ($G$ divides $H$):

$$
Q \in \mathbb{R}^{B \times T \times H \cdot d_k}, \quad
K \in \mathbb{R}^{B \times T \times G \cdot d_k}, \quad
V \in \mathbb{R}^{B \times T \times G \cdot d_v}
$$

Each group $g$ contains $H_g = H / G$ query heads that share one K/V head:

$$
\text{head}_{g,h} = \text{Attention}(Q_{g,h},\, K_g,\, V_g),
\quad Q_{g,h} \in \mathbb{R}^{B \times T \times d_k}
$$

$$
\text{GQA}(X) = \text{Concat}(\text{head}_{1,1},\ldots,\text{head}_{G, H_g})\, W_O
\in \mathbb{R}^{B \times T \times d_{\text{model}}}
$$

Special cases:
- $G = H$: reduces to standard MHA
- $G = 1$: reduces to MQA

### Complexity

| Property | Value |
|---|---|
| Time | $O(T^2 \cdot d_k \cdot H)$ — same as MHA |
| KV cache | $O(T \cdot d_k \cdot G \cdot L \cdot 2)$ — reduced by $H/G$ vs MHA |
| Parameters | $W_Q \in \mathbb{R}^{d \times H d_k}$ (same as MHA); $W_K, W_V \in \mathbb{R}^{d \times G d_k}$ (smaller by $H/G$) |

---

## PyTorch Module

```{code-block} python
:linenos:
:emphasize-lines: 28,29,30

import torch
import torch.nn as nn
import torch.nn.functional as F


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_kv_heads: int, dropout: float = 0.0):
        super().__init__()
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        self.H    = num_heads
        self.G    = num_kv_heads                 # number of K/V heads (groups)
        self.reps = num_heads // num_kv_heads    # query heads per group
        self.d_k  = d_model // num_heads         # per-head dim

        self.W_q = nn.Linear(d_model, num_heads    * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)  # G heads
        self.W_v = nn.Linear(d_model, num_kv_heads * self.d_k, bias=False)  # G heads
        self.W_o = nn.Linear(d_model, d_model,                 bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.W_q(x).view(B, T, self.H, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        k = self.W_k(x).view(B, T, self.G, self.d_k).transpose(1, 2)  # (B, G, T, d_k)
        v = self.W_v(x).view(B, T, self.G, self.d_k).transpose(1, 2)  # (B, G, T, d_k)

        # Repeat K/V to match H query heads: (B, G, T, d_k) → (B, H, T, d_k)
        k = k.repeat_interleave(self.reps, dim=1)  # (B, H, T, d_k)
        v = v.repeat_interleave(self.reps, dim=1)  # (B, H, T, d_k)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, H, T, d_k)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)
        return self.W_o(out)                                     # (B, T, d_model)
```

---

## Modern Usage

**Upcycling from an MHA checkpoint** — the paper shows that GQA models can be initialized from trained MHA checkpoints by mean-pooling the $H/G$ K/V heads within each group, then fine-tuning briefly:

```python
# Convert MHA checkpoint to GQA by mean-pooling K/V heads per group
# old_k_weight: (H * d_k, d_model)  →  new_k_weight: (G * d_k, d_model)
old_k = checkpoint["W_k"].view(H, d_k, d_model)      # (H, d_k, d_model)
new_k = old_k.view(G, H // G, d_k, d_model).mean(1)  # (G, d_k, d_model)
new_k = new_k.view(G * d_k, d_model)
```

**Flash Attention 2** natively handles GQA without the `repeat_interleave` call:

```python
from flash_attn import flash_attn_func

# q: (B, T, H, d_k),  k/v: (B, T, G, d_k)
out = flash_attn_func(q, k, v, causal=True)  # handles GQA automatically
```

---

```{warning} Common Pitfalls
- **`repeat_interleave` duplicates memory**: If you expand K/V using `repeat_interleave` before passing to `scaled_dot_product_attention`, you've negated the memory savings. Flash Attention handles GQA without expansion — prefer that for production.
- **Group size must divide evenly**: `num_heads % num_kv_heads == 0` must hold. Mistral 7B uses H=32, G=8 (ratio 4); LLaMA 2 70B uses H=64, G=8 (ratio 8).
- **KV cache stores G heads, not H**: A common implementation bug is allocating the KV cache for H heads — only G are needed.
```

```{tip} Tips
- GQA is now the default in virtually all new large LLMs: LLaMA 2 (70B), LLaMA 3 (all sizes), Mistral 7B, Gemma, Command R.
- LLaMA 2 7B/13B use full MHA (G=H); only 70B uses GQA. LLaMA 3 uses GQA at all sizes.
- At G=8, H=32: KV cache is 4× smaller than MHA with minimal quality loss (< 0.3 perplexity on LM benchmarks per the original paper).
- The paper recommends *uptrained* GQA (start from MHA checkpoint, not scratch) for best quality.
```
