# Sliding Window Attention

A local attention mechanism that restricts each token to attend only to a fixed-size window of $w$ neighboring tokens. This reduces the attention complexity from $O(T^2)$ to $O(T \cdot w)$, enabling efficient processing of very long sequences. Used in Mistral 7B and Longformer.

---

## Intuition

For many tasks, a token's most relevant context is nearby — the next few words, the surrounding sentences. Global $O(T^2)$ attention computes scores between every pair of tokens, but most of those pairs are far apart and contribute little. Sliding window attention constrains each token to attend only within a local window of size $w$.

Two variants exist:
- **Symmetric** (Longformer, bidirectional): token $i$ attends to $[i - w/2,\; i + w/2]$.
- **Causal** (Mistral, GPT-style): token $i$ attends only to past tokens $[i - w,\; i]$.

```{mermaid}
flowchart TD
    A([Tokens t₁ … tₙ]) --> B[Q, K, V projections]
    B --> C["Build band mask\nsymmetric: [i-w/2, i+w/2]\ncausal: [i-w, i]"]
    C --> D["Masked score matrix\n(B, H, T, T) — band structure"]
    D --> E[softmax per row\nonly w non-masked positions]
    E --> F[Weighted sum of V]
    F --> G([Output — O\(T·w\) memory])
```

---

## Theory

> **Paper:** [Longformer: The Long-Document Transformer — Beltagy et al. (2020)](https://arxiv.org/abs/2004.05150); also core to [Mistral 7B — Jiang et al. (2023)](https://arxiv.org/abs/2310.06825)

Define a window of size $w$. Two variants:

**Symmetric** (Longformer, bidirectional encoders) — $w$ must be even:

$$
M_{ij} = \begin{cases} 0 & \text{if } |i - j| \leq w/2 \\ -\infty & \text{otherwise} \end{cases}
$$

**Causal** (Mistral, GPT-style autoregressive decoders) — token $i$ attends only to the preceding $w$ tokens:

$$
M_{ij} = \begin{cases} 0 & \text{if } 0 \leq i - j < w \\ -\infty & \text{otherwise} \end{cases}
$$

The attention output for both variants uses the standard formula with the band mask:

$$
\text{SWA}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V
\in \mathbb{R}^{B \times T \times d_v}
$$

In practice, the band structure is exploited so only the $w$ non-masked scores per row are computed.

### Complexity

| Property | Value |
|---|---|
| Time | $O(T \cdot w \cdot d_k)$ — linear in sequence length |
| Memory | $O(T \cdot w)$ — proportional to window size, not $T^2$ |
| Effective receptive field | $O(w \cdot L)$ across $L$ transformer layers (stacked local windows) |

---

## PyTorch Module

```{code-block} python
:linenos:
:emphasize-lines: 6,7,8,9,10,11,12,13,14,25,36,37

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_sliding_window_mask(
    seq_len: int, window: int, causal: bool, device: torch.device
) -> torch.Tensor:
    """Additive mask (0 = keep, -inf = block) for sliding window attention."""
    i = torch.arange(seq_len, device=device).unsqueeze(1)   # (T, 1)
    j = torch.arange(seq_len, device=device).unsqueeze(0)   # (1, T)
    if causal:
        # token i attends to j in [i-window+1, i]  (left-only, no future)
        in_window = (i - j >= 0) & (i - j < window)         # (T, T)
    else:
        # token i attends to j in [i-window//2, i+window//2]  (symmetric)
        in_window = (i - j).abs() <= window // 2             # (T, T)
    return torch.where(in_window, 0.0, float("-inf"))        # (T, T) additive mask


class SlidingWindowAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        window_size: int,
        causal: bool = False,   # True for autoregressive (Mistral-style)
        dropout: float = 0.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.H           = num_heads
        self.d_k         = d_model // num_heads  # head dim
        self.window_size = window_size           # w
        self.causal      = causal

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.W_q(x).view(B, T, self.H, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        k = self.W_k(x).view(B, T, self.H, self.d_k).transpose(1, 2)  # (B, H, T, d_k)
        v = self.W_v(x).view(B, T, self.H, self.d_k).transpose(1, 2)  # (B, H, T, d_k)

        # Build sliding window mask (additive, shape broadcastable to B×H×T×T)
        sw_mask = make_sliding_window_mask(
            T, self.window_size, self.causal, x.device
        )  # (T, T)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=sw_mask,   # (T, T) broadcasts to (B, H, T, T)
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, H, T, d_k)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, d_model)
        return self.W_o(out)                                     # (B, T, d_model)
```

```{note}
The mask-based implementation above is $O(T^2)$ in memory because it materializes the full score matrix before masking. For true $O(T \cdot w)$ memory efficiency, use Flash Attention with the `window_size` parameter.
```

---

## Modern Usage

Flash Attention 2+ provides native sliding window attention that avoids materializing the $T \times T$ score matrix:

```python
from flash_attn import flash_attn_func

# q, k, v: (B, T, H, d_k)
out = flash_attn_func(
    q, k, v,
    causal=True,
    window_size=(-1, 0),
    # window_size convention: (left, right)
    #   left  = -1 → attend all past tokens (up to window limit)
    #   right =  0 → no future tokens (causal)
    #   For symmetric bidirectional: window_size=(w//2, w//2)
    #   For full causal attention (no window): window_size=(-1, 0) without window_size arg
)  # (B, T, H, d_k)
```

Mistral 7B uses window_size=4096 with `causal=True` — each token attends to the preceding 4096 tokens. The effective context extends beyond 4096 via stacked layers; the theoretical receptive field after $L$ layers is $w \times L$.

**Combining with global tokens** (Longformer style) — a few "global" tokens (e.g., `[CLS]`) attend to all positions while local tokens use sliding windows:

```python
# Longformer-style: create a mask that is 0 everywhere for global token rows,
# and the band mask for all other rows. Pass as attn_mask.
```

---

```{warning} Common Pitfalls
- **Window size must be even**: If $w$ is odd, the half-window $w/2$ is non-integer. Use even values (512, 1024, 2048, 4096).
- **Information bottleneck**: Tokens more than $w/2$ positions apart cannot communicate directly within one layer. Deep networks compensate with stacked windows, but tasks requiring very long-range dependencies (e.g., book-level reasoning) may still be hurt.
- **Naive mask is $O(T^2)$ in memory**: The mask-based PyTorch implementation materializes the full attention matrix. For sequences > 8k tokens, use Flash Attention's native sliding window.
- **Causal vs symmetric windows**: Mistral uses a causal left-only window (`window_size=(-1, 0)` in FlashAttention convention). Longformer for encoding tasks uses a symmetric centered window.
```

```{tip} Tips
- Window size $w = 4096$ is Mistral 7B's setting; combined with GQA and RoPE, it achieves strong long-context performance.
- The effective receptive field after $L$ layers is $w \times L$ — for 32 layers and $w = 4096$, all tokens in a 131k context can influence each other.
- Sliding window attention is the local component; combine with global tokens (Longformer) or retrieval (RAG) for tasks needing unbounded context.
- Memory scales as $O(T \cdot w)$ — at $T = 100K$ and $w = 4096$, the attention matrix is ~4×10⁸ entries vs 10¹⁰ for full attention (25× reduction).
```
