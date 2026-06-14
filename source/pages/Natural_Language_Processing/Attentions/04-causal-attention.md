# Causal (Masked) Self-Attention

The attention variant used in autoregressive (decoder-only) models like GPT. A causal mask ensures that each position can only attend to itself and earlier positions, preserving the left-to-right generation constraint during training and making it equivalent to sequential generation at inference.

---

## Intuition

During language model training, the full sequence is available, but the model must not "cheat" by looking at future tokens when predicting the next one. A causal mask enforces this by setting attention scores for future positions to $-\infty$ before the softmax, driving their weights to zero. The model processes all positions in parallel (efficient training) while behaving as if it generates one token at a time.

```{mermaid}
flowchart TD
    A([Input tokens\nt₁ t₂ t₃ … tₙ]) --> B[Q, K, V projections]
    B --> C["Score matrix\nQ · Kᵀ / √d_k\n(B, H, T, T)"]
    C --> D["Apply causal mask:\nset upper-triangle to -∞"]
    D --> E["softmax per row\n(each token only sees past)"]
    E --> F["weights · V"]
    F --> G([Output — each position\nattends to t₁…tᵢ only])
```

---

## Theory

> **Paper:** [Attention Is All You Need — Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) (decoder self-attention); widely used in GPT-series models.

Let $M \in \{0, -\infty\}^{T \times T}$ be the causal mask:

$$
M_{ij} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}
$$

The masked attention score matrix is:

$$
\tilde{S} = \frac{Q K^\top}{\sqrt{d_k}} + M
\quad \in \mathbb{R}^{B \times H \times T \times T}
$$

$$
\text{CausalAttention}(Q, K, V) = \text{softmax}\!\left(\tilde{S}\right) V
\quad \in \mathbb{R}^{B \times H \times T \times d_v}
$$

After adding $M$, any position $i$ has $-\infty$ for all $j > i$, which becomes 0 after softmax.

### Complexity

| Property | Value |
|---|---|
| Time | $O(T^2 \cdot d_k)$ — same as standard attention |
| Memory | $O(T^2)$ for the attention matrix |
| KV cache at inference | $O(T \cdot d_k \cdot L)$ — stores past K/V to avoid recomputation |

---

## PyTorch Module

```{code-block} python
:linenos:
:emphasize-lines: 12,13,30

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.H   = num_heads
        self.d_k = d_model // num_heads  # head dim

        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o   = nn.Linear(d_model, d_model,     bias=False)
        self.dropout = dropout

        # Pre-built causal mask — upper triangle is True (masked out)
        # register_buffer: moves to device with the module but is not a parameter
        mask = torch.ones(max_seq_len, max_seq_len, dtype=torch.bool).triu(diagonal=1)
        self.register_buffer("causal_mask", mask)  # (max_T, max_T)
        # When applying manually: slice to actual T → causal_mask[:T, :T]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape  # (B, T, d_model)

        qkv = self.W_qkv(x)                                     # (B, T, 3*d_model)
        q, k, v = qkv.split(x.size(-1), dim=-1)                 # 3 × (B, T, d_model)
        q = q.view(B, T, self.H, self.d_k).transpose(1, 2)      # (B, H, T, d_k)
        k = k.view(B, T, self.H, self.d_k).transpose(1, 2)      # (B, H, T, d_k)
        v = v.view(B, T, self.H, self.d_k).transpose(1, 2)      # (B, H, T, d_k)

        # is_causal=True builds the causal mask internally (most efficient path)
        out = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )  # (B, H, T, d_k)

        return self.W_o(out.transpose(1, 2).contiguous().view(B, T, -1))  # (B, T, d_model)
```

---

## Modern Usage

Pass `is_causal=True` to `F.scaled_dot_product_attention` — this is the cleanest and most efficient path, as PyTorch (and FlashAttention) can exploit the triangular structure to skip computing masked entries entirely.

```python
import torch.nn.functional as F

# q, k, v: (B, H, T, d_k)
out = F.scaled_dot_product_attention(
    q, k, v,
    is_causal=True,   # no explicit mask needed
    dropout_p=0.0,
)  # (B, H, T, d_k)
```

**KV cache for inference** — at generation time, only the new token's Q is computed; K and V for all past tokens are retrieved from cache:

```python
# Pseudo-code for single-token generation step
new_k = W_k(new_token)      # (B, 1, d_model)
new_v = W_v(new_token)      # (B, 1, d_model)
k_cache = torch.cat([k_cache, new_k], dim=1)  # (B, t+1, d_model)
v_cache = torch.cat([v_cache, new_v], dim=1)  # (B, t+1, d_model)

out = F.scaled_dot_product_attention(
    W_q(new_token), k_cache, v_cache, is_causal=False  # attending to all past, not causal
)
```

---

```{warning} Common Pitfalls
- **`is_causal=True` assumes square Q/K**: When using a KV cache, `T_q = 1` and `T_k > 1`; passing `is_causal=True` would mask incorrectly. Use `is_causal=False` and pass no mask (the KV cache already limits the keys to past tokens).
- **Register buffer device sync**: The pre-built `causal_mask` must be on the same device as inputs. Using `register_buffer` handles this automatically via `.to(device)`.
- **Attention sink**: In practice, the first token often receives disproportionate attention weight ("attention sink"). This is a known phenomenon in GPT-style models — not a bug, but worth knowing when interpreting attention weights.
```

```{tip} Tips
- The only difference from standard MHA is `is_causal=True` (or an equivalent upper-triangular mask) — everything else is identical.
- During training, the full sequence is processed in one forward pass; the causal mask makes each position see only past context, which is equivalent to running left-to-right.
- GPT-2, GPT-3, LLaMA, Mistral, Falcon, and essentially every modern decoder-only LLM uses this exact mechanism.
- KV cache reduces per-token generation cost from $O(T^2)$ to $O(T)$ in sequence length.
```
