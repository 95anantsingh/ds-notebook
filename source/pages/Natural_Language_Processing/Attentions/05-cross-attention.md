# Cross-Attention

The attention variant used in encoder-decoder architectures. Queries come from the decoder, while keys and values come from the encoder — allowing the decoder to selectively retrieve information from the full source sequence at every generation step. Used in T5, BART, Whisper, and diffusion model U-Nets.

---

## Intuition

In a translation model, the decoder must know *which source words are relevant* when generating the next target word. Cross-attention provides this link: the decoder's current state generates queries that "search" through the encoder's outputs (keys and values), retrieving the most relevant source context. The decoder's own sequence length and the encoder's sequence length are independent.

```{mermaid}
flowchart TD
    A([Encoder outputs\nB × T_src × d_model]) --> C[W_K → Keys]
    A --> D[W_V → Values]
    B([Decoder hidden states\nB × T_tgt × d_model]) --> E[W_Q → Queries]
    E --> F["Q · Kᵀ / √d_k\n(B, H, T_tgt, T_src)"]
    C --> F
    F --> G[softmax per row]
    G --> H["weights · V\n(B, H, T_tgt, d_v)"]
    D --> H
    H --> I[W_O projection]
    I --> J([Output\nB × T_tgt × d_model])
```

---

## Theory

> **Paper:** [Attention Is All You Need — Vaswani et al. (2017)](https://arxiv.org/abs/1706.03762) (Transformer decoder layer)

$$
Q = X_{\text{dec}}\, W_Q, \quad Q \in \mathbb{R}^{B \times T_{\text{tgt}} \times d_k}
$$

$$
K = X_{\text{enc}}\, W_K, \quad K \in \mathbb{R}^{B \times T_{\text{src}} \times d_k}
$$

$$
V = X_{\text{enc}}\, W_V, \quad V \in \mathbb{R}^{B \times T_{\text{src}} \times d_v}
$$

$$
\text{CrossAttention}(X_{\text{dec}}, X_{\text{enc}}) =
\text{softmax}\!\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V
\in \mathbb{R}^{B \times T_{\text{tgt}} \times d_v}
$$

The attention score matrix has shape $\mathbb{R}^{B \times H \times T_{\text{tgt}} \times T_{\text{src}}}$ — it is *not* square when source and target lengths differ.

### Complexity

| Property | Value |
|---|---|
| Time | $O(T_{\text{tgt}} \cdot T_{\text{src}} \cdot d_k)$ |
| Memory | $O(T_{\text{tgt}} \cdot T_{\text{src}})$ for attention matrix |
| KV cache at inference | Encoder K/V can be computed once and reused across all decoding steps |

---

## PyTorch Module

```{code-block} python
:linenos:
:emphasize-lines: 24,25,26,27

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.H   = num_heads
        self.d_k = d_model // num_heads  # head dim

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(
        self,
        x_dec: torch.Tensor,                     # (B, T_tgt, d_model) — decoder states
        x_enc: torch.Tensor,                     # (B, T_src, d_model) — encoder outputs
        enc_mask: torch.Tensor | None = None,    # (B, T_tgt, T_src) bool, True = keep
                                                 # For source padding mask (B, T_src):
                                                 #   enc_mask = ~pad_mask.view(B,1,1,T_src)
    ) -> torch.Tensor:
        B, T_tgt, _ = x_dec.shape
        T_src = x_enc.size(1)

        q = self.W_q(x_dec).view(B, T_tgt, self.H, self.d_k).transpose(1, 2)  # (B, H, T_tgt, d_k)
        k = self.W_k(x_enc).view(B, T_src, self.H, self.d_k).transpose(1, 2)  # (B, H, T_src, d_k)
        v = self.W_v(x_enc).view(B, T_src, self.H, self.d_k).transpose(1, 2)  # (B, H, T_src, d_k)

        if enc_mask is not None:
            enc_mask = enc_mask.unsqueeze(1)  # (B, 1, T_tgt, T_src)

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=enc_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,  # cross-attention is never causal
        )  # (B, H, T_tgt, d_k)

        out = out.transpose(1, 2).contiguous().view(B, T_tgt, -1)  # (B, T_tgt, d_model)
        return self.W_o(out)                                         # (B, T_tgt, d_model)
```

---

## Modern Usage

In encoder-decoder models, the encoder K/V projections are constant across all decoding steps — compute them once and reuse:

```python
# Compute encoder KV once
enc_k = model.W_k(encoder_out)  # (B, T_src, d_model)
enc_v = model.W_v(encoder_out)  # (B, T_src, d_model)

# Reuse at every decoding step
for step in range(max_length):
    dec_q = model.W_q(decoder_state)  # (B, 1, d_model)
    out = F.scaled_dot_product_attention(dec_q, enc_k, enc_v, is_causal=False)
    # ...
```

**In diffusion U-Nets** (e.g., Stable Diffusion), cross-attention connects spatial features (queries from the image feature map) with text conditioning (keys and values from a CLIP/text encoder):

```python
# x: (B, T_spatial, d_model) — flattened spatial features
# context: (B, T_text, d_text) — text encoder outputs
out = cross_attn(x_dec=x, x_enc=context)  # (B, T_spatial, d_model)
```

---

```{warning} Common Pitfalls
- **Mask shape**: The attention mask is `(B, T_tgt, T_src)`, not square. A common mistake is reusing a square self-attention mask.
- **`is_causal=True` is wrong here**: Cross-attention never needs a causal mask — the decoder can attend to any encoder position.
- **Encoder padding**: If encoder sequences are padded to the same length, pass an encoder key padding mask so the decoder doesn't attend to padding tokens.
- **Forgetting to cache encoder KV**: Recomputing encoder K/V at every decoding step wastes compute proportional to the number of decoder layers × encoder length.
```

```{tip} Tips
- Cross-attention appears as the *middle* sublayer in each Transformer decoder block: after causal self-attention, before the feed-forward layer.
- In T5, encoder and decoder share the same dimension, so no projection mismatch. In encoder-decoder models with different widths, add a projection.
- Whisper uses cross-attention to align audio encoder frames with text decoder positions — the alignment pattern is nearly monotonic for well-formed speech.
- Diffusion U-Nets use cross-attention to inject text conditioning at every spatial resolution level.
```
