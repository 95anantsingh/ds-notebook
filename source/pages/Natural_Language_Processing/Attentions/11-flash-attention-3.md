# Flash Attention 3

The Hopper-GPU-specific revision of Flash Attention (2024) that exploits NVIDIA H100/H800 hardware features unavailable on earlier architectures. FA3 achieves **1.5–2× throughput over FA2** on H100 by overlapping GEMM and softmax operations using asynchronous warp-group pipelines, and adds native FP8 support.

---

## Intuition

FA2 runs its inner loop sequentially: load K/V tile → compute QK GEMM → update softmax statistics → accumulate into output. On H100, the new WGMMA (Warp Group Matrix Multiply-Accumulate) and TMA (Tensor Memory Accelerator) hardware supports truly asynchronous operations. FA3 restructures the pipeline so the GEMM for tile $i+1$ runs *while* the softmax rescaling for tile $i$ is still in flight, hiding the latency of both.

```{mermaid}
flowchart TD
    A([H100 Hardware]) --> B["WGMMA\nasync GEMM engine"]
    A --> C["TMA\nasync memory copy"]
    A --> D["FP8 Tensor Cores\n2× FLOPs vs BF16"]
    B --> E["FA3: Overlap tile N+1 GEMM\nwith tile N softmax rescaling"]
    C --> E
    D --> F["FA3 FP8 forward:\n~2× over BF16 FA2"]
    E --> G([1.5–2× over FA2\non H100/H800])
    F --> G
```

---

## Theory

> **Paper:** [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision — Shah et al. (2024)](https://arxiv.org/abs/2407.08608)

The math is unchanged from standard attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

**Key algorithmic changes over FA2:**

1. **Producer-consumer warp specialization** — Warps are split into producer groups (handle TMA data loads) and consumer groups (run WGMMA GEMMs). They communicate via shared memory ping-pong buffers, enabling true pipeline parallelism within a thread block.

2. **Overlapped GEMM and softmax** — The two GEMMs in attention (QK scores, then scores·V) and the softmax rescaling are staged across pipeline slots. Consumer warp group starts the next GEMM tile while the softmax of the current tile completes concurrently.

3. **FP8 forward pass** — H100 FP8 tensor cores offer 2× the throughput of BF16 for the GEMM steps. FA3 keeps the softmax statistics in FP32/BF16 for numerical stability while running the QK and score·V matmuls in FP8.

### Complexity

| Property | FA2 | FA3 |
|---|---|---|
| HBM IO | $O(T^2 d / M)$ | $O(T^2 d / M)$ — same |
| Memory | $O(T)$ | $O(T)$ — same |
| Pipeline | Sequential GEMM → softmax | Overlapped async GEMM + softmax |
| FP8 support | No | Yes (forward pass) |
| Throughput vs FA2 | — | ~1.5–2× on H100 |
| GPU requirement | Ampere+ | **Hopper only** (SM ≥ 90) |

---

## PyTorch / Flash Attention 3 API

FA3 uses a distinct import path from FA2. Confirm your GPU is Hopper (H100/H800) before using it.

```{code-block} python
:linenos:
:emphasize-lines: 4,5,6,7,8,9

import torch
from flash_attn.flash_attn_interface import flash_attn_func  # FA3 module path

# q, k, v: (B, T, H, d_k)  dtype: fp16 or bf16
out = flash_attn_func(
    q, k, v,
    causal=True,
    window_size=(-1, 0),   # causal; use (-1,-1) for bidirectional
)  # (B, T, H, d_k)
```

**FP8 forward pass** (H100 only):

```python
from flash_attn.flash_attn_interface import flash_attn_func

# e4m3fn: higher dynamic range for activations
q_fp8 = q.to(torch.float8_e4m3fn)
k_fp8 = k.to(torch.float8_e4m3fn)
v_fp8 = v.to(torch.float8_e4m3fn)

out = flash_attn_func(
    q_fp8, k_fp8, v_fp8,
    causal=True,
    descale_q=q_scale,   # float32 scale factors from quantization
    descale_k=k_scale,
    descale_v=v_scale,
)  # output is bf16
```

---

## Modern Usage

Guard FA3 usage with a device capability check — it will not run on Ampere:

```python
def get_flash_attn():
    major, _ = torch.cuda.get_device_capability()
    if major >= 9:  # SM90 = Hopper (H100, H800)
        from flash_attn.flash_attn_interface import flash_attn_func
    else:
        from flash_attn import flash_attn_func   # FA2 fallback
    return flash_attn_func

flash_fn = get_flash_attn()
out = flash_fn(q, k, v, causal=True)
```

**Combining FA3 + GQA + FP8** for maximum H100 inference throughput:

```python
# q: (B, T, H, d_k) fp8,  k/v: (B, T, G, d_k) fp8  (G < H, GQA)
out = flash_fn(q_fp8, k_fp8, v_fp8, causal=True,
               descale_q=sq, descale_k=sk, descale_v=sv)
# Output: (B, T, H, d_k) bf16
```

---

```{warning} Common Pitfalls
- **Hopper-only — hard failure on Ampere**: FA3 kernel requires SM ≥ 90. Always gate with `torch.cuda.get_device_capability()[0] >= 9` or it will raise a CUDA error.
- **FP8 accuracy**: FP8 e4m3fn clips at ±448. Large activations will saturate, causing accuracy loss. Always benchmark perplexity before deploying FP8 in production.
- **Backward pass is BF16**: Even with FP8 forward, FA3 backward runs in BF16. FP8 end-to-end training is not yet fully supported.
- **Import path changed**: FA3 uses `flash_attn.flash_attn_interface`, not `flash_attn`. Mixing up the import gives FA2 behavior silently.
- **Sequence length alignment**: For peak throughput, align `T` to multiples of 128. Misaligned lengths incur kernel padding overhead.
```

```{tip} Tips
- FA3 achieves ~75% of H100 BF16 peak throughput (vs ~70% for FA2) — very close to the hardware ceiling for an operation that is memory-bandwidth limited.
- The WGMMA + TMA pipeline is the same technique used in cuBLAS GEMM kernels — FA3 effectively brings attention to GEMM-level hardware utilization.
- Combining FA3 (for compute) + GQA (for KV cache size) + FP8 KV cache quantization is a common production recipe for long-context H100 inference.
- FA3 is used in production at inference providers running H100 fleets. HuggingFace and vLLM are adding support.
- For training, the backward pass determines overall throughput — FA3's forward speedup is partially offset if backward doesn't use equivalent hardware features.
```
