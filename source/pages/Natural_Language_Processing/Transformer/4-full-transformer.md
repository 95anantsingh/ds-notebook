(full-transformer)=
# Full Transformer Architecture

> **Depends on**: {ref}`Encoder Block <encoder-block>` · {ref}`Decoder Block <decoder-block>` · {ref}`Positional Encoding <positional-encoding>` · {ref}`Multi-Head Attention <multi-head-attention>`

This page assembles the building blocks into two complete models: a bidirectional `TransformerEncoder` (BERT-style) and an autoregressive `GPT` (decoder-only).

---

## TransformerEncoder (BERT-style)

```python
import torch
import torch.nn as nn
from .encoder_block import EncoderBlock
from .positional_encoding import LearnedPositionalEncoding


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        max_len: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed   = nn.Embedding(vocab_size, d_model)
        self.pos_enc = LearnedPositionalEncoding(d_model, max_len, dropout)
        self.layers  = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor, padding_mask: torch.Tensor | None = None):
        # input_ids: (B, T) — integer token ids
        # padding_mask: (B, T) bool — True for PAD positions
        x = self.pos_enc(self.embed(input_ids))   # (B, T, d_model)

        attn_mask = None
        if padding_mask is not None:
            # Expand to (B, 1, 1, T) for broadcasting over (B, H, T, T)
            attn_mask = padding_mask[:, None, None, :]

        for layer in self.layers:
            x = layer(x, mask=attn_mask)

        return self.norm(x)                        # (B, T, d_model)
```

---

## GPT (Decoder-Only)

```python
from .decoder_block import GPTDecoderBlock


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        max_len: int,
        d_ff: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.dropout   = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            GPTDecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: share embedding and unembedding matrices
        self.lm_head.weight = self.tok_embed.weight

        # Causal mask — registered as buffer so it moves with .to(device)
        causal_mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask', causal_mask)

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = 0.02
                # Scale residual projections by 1/sqrt(2*n_layers) — GPT-2 convention
                if 'out_proj' in name or 'fc2' in name:
                    std /= (2 * len(self.layers)) ** 0.5
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # input_ids: (B, T)
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device)   # (T,)

        x = self.dropout(self.tok_embed(input_ids) + self.pos_embed(positions))  # (B, T, d_model)

        causal_mask = self.causal_mask[:T, :T]                  # (T, T) — slice to actual length

        for layer in self.layers:
            x = layer(x, causal_mask=causal_mask)

        x = self.norm(x)                                        # (B, T, d_model)
        return self.lm_head(x)                                  # (B, T, vocab_size) — logits
```

---

## Weight Tying

```{important}
Sharing the token embedding and the output (language model) head matrix is standard in all modern autoregressive LLMs. Always include it when implementing GPT from scratch — it's a signal of implementation quality in an interview.
```

Sharing the token embedding and the output (language model) head matrix:

```python
self.lm_head.weight = self.tok_embed.weight
```

- Halves the parameter count for the embedding tables (largest in small models)
- Enforces that tokens with similar embeddings also get similar output logits
- Standard in GPT-2, LLaMA, and most autoregressive LLMs

---

## Weight Initialization (GPT-2 Convention)

- All weights: `N(0, 0.02)`
- Residual projection weights (attention `out_proj`, FFN `fc2`): scaled by `1 / sqrt(2 * n_layers)`

**Why scale residual projections?** At initialization, each residual block adds `std=0.02` to the residual stream. With `N` layers, the stream's variance grows as `N × 0.02²`. Scaling by `1/sqrt(2N)` keeps the stream's variance constant at depth.

---

## Autoregressive Generation

```python
@torch.no_grad()
def generate(model: GPT, prompt_ids: torch.Tensor, max_new_tokens: int, temperature: float = 1.0) -> torch.Tensor:
    # prompt_ids: (1, T_prompt)
    model.eval()
    x = prompt_ids

    for _ in range(max_new_tokens):
        logits = model(x)               # (1, T, vocab_size)
        next_logits = logits[:, -1, :]  # last position: (1, vocab_size)

        if temperature != 1.0:
            next_logits = next_logits / temperature

        probs = torch.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)   # (1, 1)
        x = torch.cat([x, next_token], dim=1)

    return x
```

For production, add a KV cache to avoid recomputing past keys/values on every step (the cache stores `(B, H, T_past, d_k)` tensors for each layer).
