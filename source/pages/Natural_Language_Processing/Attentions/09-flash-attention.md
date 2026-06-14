# Flash Attention

An IO-aware exact attention algorithm introduced in 2022. Flash Attention computes the same result as standard scaled dot-product attention but restructures the computation to minimize data movement between GPU HBM (high-bandwidth memory) and SRAM (on-chip cache), cutting both memory usage and wall-clock time — especially for long sequences.

---

## Intuition

Standard attention is **memory-bandwidth bound**, not compute bound. The bottleneck is reading and writing the $T \times T$ attention matrix to HBM: for a 4096-token sequence, that's 16M floats just for the score matrix — far more data movement than needed.

Flash Attention avoids materializing the full score matrix by splitting Q, K, V into tiles that fit in SRAM, computing attention block-by-block, and accumulating the output. It uses the **online softmax trick** to fuse the softmax and the value aggregation into a single pass, so the $T \times T$ matrix never exists in full.

```{mermaid}
flowchart TD
    A([Q, K, V in HBM]) --> B["Tile Q into blocks\nQ₁, Q₂, … Qₙ"]
    A --> C["Tile K, V into blocks\nK₁…Kₘ, V₁…Vₘ"]
    B --> D["For each Q block:\nloop over K/V blocks in SRAM"]
    C --> D
    D --> E["Online softmax:\nupdate running max + sum"]
    E --> F["Accumulate weighted V\nin SRAM"]
    F --> G["Write output block\nto HBM once"]
    G --> H([Output O in HBM])
```

---

## Theory

> **Paper:** [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness — Dao et al. (2022)](https://arxiv.org/abs/2205.14135)

The math is identical to standard attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

The key insight is the **online softmax** formulation. For a row of scores $\mathbf{x}$, softmax can be computed incrementally across tiles without storing the full row:

$$
m^{(j)} = \max\!\left(m^{(j-1)},\; \max_i x_i^{(j)}\right)
\qquad \text{(running max)}
$$

$$
\ell^{(j)} = e^{m^{(j-1)} - m^{(j)}} \ell^{(j-1)} + \sum_i e^{x_i^{(j)} - m^{(j)}}
\qquad \text{(running normalizer)}
$$

$$
O^{(j)} = \frac{e^{m^{(j-1)} - m^{(j)}} O^{(j-1)} + e^{X^{(j)} - m^{(j)}} V^{(j)}}{\ell^{(j)}}
\qquad \text{(running output)}
$$

This produces the exact softmax result without ever storing the full $T \times T$ score matrix.

### Complexity

| Property | Standard Attention | Flash Attention |
|---|---|---|
| Memory | $O(T^2)$ — full score matrix | $O(T)$ — only tile buffers in SRAM |
| HBM reads/writes | $\Theta(T^2)$ | $\Theta(T^2 d / M)$ — $M/d$ times fewer |
| Compute FLOPs | $O(T^2 d)$ | $O(T^2 d)$ — identical, just reorganized |

$M$ = SRAM capacity; $d$ = head dimension. When $d \ll M$ (typical), IO cost drops dramatically.

---

## PyTorch / Flash Attention API

Flash Attention ships as a separate CUDA extension. PyTorch 2.0+ also dispatches to it automatically via `F.scaled_dot_product_attention` when inputs are fp16/bf16 on Ampere+.

```{code-block} python
:linenos:
:emphasize-lines: 5,6,7,8

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func

# q, k, v: (B, T, H, d_k)  — must be fp16 or bf16
out = flash_attn_func(
    q, k, v,
    causal=False,   # True for autoregressive / decoder-only models
    dropout_p=0.0,
)  # (B, T, H, d_k)

# PyTorch 2.0+ — dispatches to Flash Attention automatically on supported hardware
# q, k, v: (B, H, T, d_k)
out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

**Variable-length sequences** (pack sequences without padding — saves compute):

```python
from flash_attn import flash_attn_varlen_func

# cu_seqlens: (B+1,) cumulative sequence lengths
out = flash_attn_varlen_func(
    q.view(-1, H, d_k),   # (total_tokens, H, d_k)
    k.view(-1, H, d_k),
    v.view(-1, H, d_k),
    cu_seqlens_q=cu_seqlens,    # (B+1,) int32
    cu_seqlens_k=cu_seqlens,
    max_seqlen_q=max_seqlen,
    max_seqlen_k=max_seqlen,
    causal=True,
)
```

---

## Modern Usage

Install:
```bash
pip install flash-attn --no-build-isolation
```

PyTorch 2.0+ dispatches to Flash Attention automatically — no code changes needed:

```python
# Verify Flash Attention is being used
with torch.backends.cuda.sdp_kernel(
    enable_flash=True, enable_math=False, enable_mem_efficient=False
):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

---

```{warning} Common Pitfalls
- **dtype must be fp16 or bf16**: Flash Attention does not support float32. Cast with `.half()` or `.to(torch.bfloat16)`.
- **Head dimension constraint**: FA1 requires `d_k` to be a power of 2 and ≤ 256. FA2/3 relax this; prefer those for new code.
- **No attention weight output**: `flash_attn_func` does not return attention weights. Use standard attention if you need weights for visualization.
- **Backward recomputation**: FA stores only the softmax statistics (not the full score matrix) and recomputes during backward. This saves memory but adds backward FLOPs — training is still faster because larger batches fit.
- **Sequence length padding**: For best performance, align `T` to a multiple of 128.
```

```{tip} Tips
- Flash Attention achieves **2–4× wall-clock speedup** and **5–20× memory reduction** vs standard attention at sequence lengths ≥ 1K.
- The algorithm is exact — output is numerically identical to standard attention (up to floating-point reordering).
- `is_causal=True` triggers a causal path that skips the upper-triangle entirely, giving ~2× the throughput of non-causal at the same sequence length.
- Flash Attention 1 introduced the core algorithm; Flash Attention 2 (2023) and 3 (2024) optimize it further — see the next two pages.
```
