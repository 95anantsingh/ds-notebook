(gpt-architecture)=
# GPT Architecture

> **Depends on**: {ref}`Full Transformer <full-transformer>` · {ref}`Decoder Block <decoder-block>`
> **Interview Q&A**: {ref}`Transformer Variants Q&A <transformer-variants-qa>`

GPT is a decoder-only autoregressive language model. There is no encoder; the model conditions on context through causal self-attention over the concatenated prompt and generated tokens.

---

## Complete Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class GPTConfig:
    vocab_size:  int   = 50257
    max_len:     int   = 1024
    d_model:     int   = 768
    num_heads:   int   = 12
    num_layers:  int   = 12
    d_ff:        int   = 3072     # 4 × d_model
    dropout:     float = 0.1


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.num_heads == 0
        self.num_heads = cfg.num_heads
        self.d_k = cfg.d_model // cfg.num_heads

        self.qkv  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)

        # Causal mask stored as buffer — not a parameter, but moves with .to(device)
        mask = torch.triu(torch.ones(cfg.max_len, cfg.max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask', mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, dk = self.num_heads, self.d_k

        q, k, v = self.qkv(x).chunk(3, dim=-1)                   # each: (B, T, D)
        q = q.view(B, T, H, dk).transpose(1, 2)                  # (B, H, T, dk)
        k = k.view(B, T, H, dk).transpose(1, 2)
        v = v.view(B, T, H, dk).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dk)        # (B, H, T, T)
        scores = scores.masked_fill(self.causal_mask[:T, :T], float('-inf'))
        attn   = self.attn_drop(F.softmax(scores, dim=-1))

        out = attn @ v                                             # (B, H, T, dk)
        out = out.transpose(1, 2).contiguous().view(B, T, D)      # (B, T, D)
        return self.resid_drop(self.proj(out))


class GPTBlock(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn  = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff    = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff, bias=False),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model, bias=False),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))   # pre-norm
        x = x + self.ff(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop      = nn.Dropout(cfg.dropout)
        self.blocks    = nn.ModuleList([GPTBlock(cfg) for _ in range(cfg.num_layers)])
        self.norm      = nn.LayerNorm(cfg.d_model)
        self.lm_head   = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, '_is_residual'):
                std /= math.sqrt(2 * len(self.blocks))
            nn.init.normal_(module.weight, 0.0, std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, 0.0, 0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos  = torch.arange(T, device=input_ids.device)
        x = self.drop(self.tok_embed(input_ids) + self.pos_embed(pos))   # (B, T, D)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))   # (B, T, vocab_size) logits
```

---

## KV Cache

Without cache, generating T tokens requires O(T²) attention ops. With cache, each new token only computes Q for the new position; K/V for all previous positions are reused.

```{tip}
In an interview, mention KV caching unprompted when explaining autoregressive generation. It shows you understand the inference-time bottleneck. Memory cost: `2 × num_layers × num_heads × d_head × seq_len × dtype_bytes` per sequence.
```

```python
# Cache: list of (k_cache, v_cache) per layer, each (B, H, T_past, d_k)
kv_cache = [(None, None)] * num_layers

def forward_with_cache(x, kv_cache, layer_idx):
    # Compute Q/K/V for new token(s) only
    q, k, v = ...
    if kv_cache[layer_idx][0] is not None:
        k = torch.cat([kv_cache[layer_idx][0], k], dim=2)  # append new K
        v = torch.cat([kv_cache[layer_idx][1], v], dim=2)
    kv_cache[layer_idx] = (k, v)
    # Attend: Q is (B,H,1,dk), K/V are (B,H,T_past+1,dk)
    out = scaled_dot_product_attention(q, k, v)
    return out, kv_cache
```

---

## Text Generation

```python
@torch.no_grad()
def generate(model, prompt_ids, max_new_tokens, temperature=1.0, top_k=50):
    model.eval()
    x = prompt_ids   # (1, T_prompt)

    for _ in range(max_new_tokens):
        logits = model(x)[:, -1, :]         # (1, vocab_size)
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, -1:]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)   # (1, 1)
        x = torch.cat([x, next_token], dim=1)

    return x
```

---

## GPT-2 vs LLaMA Key Differences

```{note}
Interviewers often ask you to compare GPT-2 and LLaMA. The four key differences are: RMSNorm instead of LayerNorm, RoPE instead of learned PE, GQA instead of MHA, and SwiGLU instead of GELU. All four reduce memory or improve scaling without changing the overall architecture.
```

| | GPT-2 | LLaMA 2/3 |
|--|-------|----------|
| Normalization | Pre-LayerNorm | Pre-RMSNorm |
| Positional enc. | Learned absolute | RoPE |
| Attention | MHA | GQA (num_kv_heads < num_heads) |
| Activation | GELU | SwiGLU |
| Context length | 1024 | 4096–128k |
