(decoder-block)=
# Transformer Decoder Block

> **Depends on**: {ref}`Multi-Head Attention <multi-head-attention>` · {ref}`Encoder Block <encoder-block>`

Two decoder variants exist for different use cases:

| Variant | Architecture | Used By |
|---------|-------------|---------|
| **GPT-style (decoder-only)** | Causal self-attention + FFN | GPT, LLaMA, Mistral |
| **Encoder-decoder style** | Causal self-attn + cross-attn + FFN | Original transformer, T5, BART |

```{tip}
**GPT-style is the most common interview target.** Unless the question specifically involves translation or summarization, default to the decoder-only (GPT) variant. Encoder-decoder blocks add cross-attention and require a separate encoder output — only needed for seq2seq tasks.
```

---

## GPT-Style Decoder Block (Most Common Interview Target)

No cross-attention. The causal mask enforces autoregressive generation.

```python
import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .encoder_block import FeedForward


class GPTDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.attn    = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ff      = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor | None = None) -> torch.Tensor:
        # Pre-norm, causal self-attention only
        x = x + self.dropout(self.attn(self.norm1(x), mask=causal_mask))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x
```

---

## Encoder-Decoder Decoder Block (Original Transformer)

Three sub-layers: causal self-attention → cross-attention → FFN.

```python
class EncoderDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        self.ff         = FeedForward(d_model, d_ff, dropout=dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.norm3      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,               # decoder input:  (B, T_dec, d_model)
        enc_out: torch.Tensor,          # encoder output: (B, T_enc, d_model)
        causal_mask: torch.Tensor | None = None,     # (T_dec, T_dec) or (B,1,T_dec,T_dec)
        enc_padding_mask: torch.Tensor | None = None # (B, 1, 1, T_enc)
    ) -> torch.Tensor:
        # causal_mask shape: (T_dec, T_dec) — applied to self-attention only
        # enc_padding_mask shape: (B, 1, 1, T_enc) — applied to cross-attention only
        # 1. Causal self-attention on decoder sequence
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=causal_mask))

        # 2. Cross-attention: Q from decoder, K/V from encoder
        x = x + self.dropout(self.cross_attn(
            self.norm2(x),
            context=enc_out,
            mask=enc_padding_mask,
        ))

        # 3. Feed-forward
        x = x + self.dropout(self.ff(self.norm3(x)))
        return x
```

```{warning}
**Do not swap the two masks.** The causal mask `(T_dec, T_dec)` goes to self-attention to prevent attending to future tokens. The encoder padding mask `(B, 1, 1, T_enc)` goes to cross-attention to ignore `[PAD]` tokens in the encoder output. Swapping them causes silent, hard-to-debug attention errors.
```

---

## Causal Mask Utility

```python
def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Returns (seq_len, seq_len) bool mask. True = blocked (upper triangle)."""
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
```

Registered in the parent model as a buffer so it stays on the right device:

```python
class GPT(nn.Module):
    def __init__(self, ..., max_len: int):
        ...
        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask', mask)

    def forward(self, x):
        T = x.size(1)
        mask = self.causal_mask[:T, :T]   # slice to actual length
        ...
```

---

## Key Differences: GPT vs Original Decoder

| | GPT-style | Encoder-decoder style |
|--|----------|-----------------------|
| **Self-attention** | Causal (masked) | Causal (masked) |
| **Cross-attention** | None | Yes — Q from decoder, K/V from encoder |
| **Input** | Token + position embedding | Same |
| **Use case** | Unconditional generation, chat | Translation, summarization, Q&A |
| **Examples** | GPT-2/3/4, LLaMA | Original transformer, T5, BART |

**Why no cross-attention in GPT?** GPT is decoder-only — there's no separate encoder. The model learns to condition on the prompt context entirely through its causal self-attention over the concatenated prompt + generated tokens.
