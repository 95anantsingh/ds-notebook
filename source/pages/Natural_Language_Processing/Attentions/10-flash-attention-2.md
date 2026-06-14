# Flash Attention 2

A major revision of Flash Attention (2023) that keeps the same IO-aware tiling algorithm but significantly improves GPU utilization — by reducing non-matrix-multiply FLOPs and restructuring work distribution across thread blocks. Achieves roughly **2× the throughput** of Flash Attention 1 on A100.

---

## Intuition

Flash Attention 1 was IO-efficient but left GPU compute on the table. Tensor cores are optimized for GEMMs; any non-GEMM work (softmax rescaling, statistics updates) runs on slower CUDA cores. FA1 also distributed work across thread blocks only over batch and head dimensions — leaving the GPU underoccupied for long sequences with small batch sizes.

FA2 attacks both bottlenecks: it restructures the online softmax rescaling to minimize non-GEMM operations, and adds sequence-dimension parallelism so thread blocks stay busy even at batch size 1.

```{mermaid}
flowchart TD
    A([FA1 bottlenecks]) --> B["Excess non-GEMM FLOPs\nin softmax rescaling"]
    A --> C["Parallelism only over\nbatch × heads"]
    B --> D["FA2: fuse rescaling\ninto GEMM epilogue"]
    C --> E["FA2: also parallelize\nover sequence blocks"]
    D --> F([~2× throughput\non A100])
    E --> F
```

---

## Theory

> **Paper:** [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning — Dao (2023)](https://arxiv.org/abs/2307.08691)

The mathematical formula is unchanged from standard attention and FA1:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

**Key algorithmic changes over FA1:**

1. **Fewer non-GEMM FLOPs** — FA1 rescaled the output accumulator at every inner-loop K/V block. FA2 reorganizes the loop so rescaling happens once per Q block (not per K/V block), reducing non-GEMM work by roughly the number of K/V tiles.

2. **Sequence-dimension parallelism** — FA1 launched one thread block per (batch, head) pair. For long-context inference with batch=1 and few heads, many SMs sat idle. FA2 adds a third parallelism dimension: different thread blocks handle different Q blocks, keeping all SMs occupied.

3. **Better causal work balance** — FA1's causal masking caused work imbalance between thread blocks (early blocks do more masked-out work). FA2 fixes the tile assignment so all blocks handle similar amounts of live computation.

### Complexity

| Property | FA1 | FA2 |
|---|---|---|
| HBM IO | $O(T^2 d / M)$ | $O(T^2 d / M)$ — same |
| Memory | $O(T)$ | $O(T)$ — same |
| Non-GEMM FLOPs | Per K/V block | Per Q block (~$T_K / T_Q$ reduction) |
| Parallelism axes | Batch × Heads | Batch × Heads × **Sequence** |
| Throughput vs FA1 | — | ~2× on A100 (fp16/bf16) |

---

## PyTorch / Flash Attention 2 API

```{code-block} python
:linenos:
:emphasize-lines: 5,6,7,8,9,10

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_func

# q, k, v: (B, T, H, d_k)  dtype: fp16 or bf16
out = flash_attn_func(
    q, k, v,
    causal=False,
    dropout_p=0.0,
    softmax_scale=None,     # defaults to 1/√d_k
    window_size=(-1, -1),   # (-1,-1) = full; (-1, 0) = causal sliding window
)  # (B, T, H, d_k)

# GQA/MQA: k, v can have fewer heads than q (H_kv < H)
out = flash_attn_func(q, k, v)  # FA2 handles grouped-query natively
```

**KV cache for incremental decoding:**

```python
from flash_attn import flash_attn_with_kvcache

out = flash_attn_with_kvcache(
    q,           # (B, 1, H, d_k) — new query token
    k_cache,     # (B, max_seqlen, H_kv, d_k) — KV cache buffer
    v_cache,     # (B, max_seqlen, H_kv, d_k)
    k,           # (B, 1, H_kv, d_k) — new key to append
    v,           # (B, 1, H_kv, d_k) — new value to append
    cache_seqlens=current_lengths,  # (B,) tokens filled so far
    causal=True,
)  # (B, 1, H, d_k) — updates cache in-place
```

---

## Modern Usage

FA2 is what `pip install flash-attn` installs (≥ 2.0). It is also the kernel used by `F.scaled_dot_product_attention` on Ampere+ GPUs with fp16/bf16:

```python
# Force FA2 — disable math and memory-efficient fallbacks
with torch.backends.cuda.sdp_kernel(
    enable_flash=True, enable_math=False, enable_mem_efficient=False
):
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

Check the active kernel:
```python
import torch
print(torch.backends.cuda.flash_sdp_enabled())  # True if FA is available
```

---

```{warning} Common Pitfalls
- **FA2 ≥ 2.1 required for GQA**: Earlier FA2 versions don't support mismatched Q / K/V head counts natively.
- **`window_size` convention**: `(-1, 0)` means "all past + current token, no future" (causal). `(-1, -1)` is full bidirectional attention. `(w, 0)` is causal sliding window of width `w`.
- **KV cache length**: `cache_seqlens` must be set correctly or attention will include uninitialized cache entries.
- **ALiBi slopes**: Pass as `alibi_slopes=(H,)` tensor, not folded into `softmax_scale`.
- **Sequence length must be ≥ 1** for the `flash_attn_with_kvcache` path.
```

```{tip} Tips
- FA2 is integrated into HuggingFace Transformers, vLLM, LightLLM, and essentially every modern LLM serving framework.
- `flash_attn_with_kvcache` is the recommended decoding path — it updates the cache buffer in-place, avoiding Python-level concatenation.
- For H100/H800, prefer Flash Attention 3 (next page), which exploits Hopper-specific hardware for an additional 1.5–2× speedup.
- FA2 achieves ~40–70% of A100 peak BF16 throughput for attention, compared to ~30–35% for FA1.
- The sequence-parallelism improvement matters most at batch_size=1 or small-batch long-context inference.
```
