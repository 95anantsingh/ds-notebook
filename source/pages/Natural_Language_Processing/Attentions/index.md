# Attention Mechanisms

Attention is the core operation that lets a model decide *which parts of the input to focus on* when producing each output. The mechanism evolved from a narrow fix for seq2seq bottlenecks in 2015 into the central primitive of modern large language models.

<!-- TOC START -->

```{toctree}
:maxdepth: 2
:glob:

01-scaled-dot-product-attention
02-additive-attention
03-multi-head-attention
04-causal-attention
05-cross-attention
06-multi-query-attention
07-grouped-query-attention
08-sliding-window-attention
09-flash-attention
10-flash-attention-2
11-flash-attention-3
```

<!-- TOC END -->


## A Brief History

- **2015** — [Bahdanau et al.](https://arxiv.org/abs/1409.0473) introduced additive attention for neural machine translation, replacing the fixed encoder summary vector with a dynamic context computed from all source hidden states.
- **2017** — [Vaswani et al. — *Attention Is All You Need*](https://arxiv.org/abs/1706.03762) replaced recurrence entirely with scaled dot-product and multi-head self-attention, establishing the Transformer.
- **2019** — [Shazeer](https://arxiv.org/abs/1911.02150) proposed Multi-Query Attention to cut KV cache memory at inference by sharing a single K/V head across all query heads.
- **2020** — [Beltagy et al. — Longformer](https://arxiv.org/abs/2004.05150) introduced sliding window attention to bring $O(T^2)$ down to $O(T \cdot w)$ for long documents.
- **2022–2024** — Flash Attention [1](https://arxiv.org/abs/2205.14135) / [2](https://arxiv.org/abs/2307.08691) / [3](https://arxiv.org/abs/2407.08608) made standard attention IO-efficient on GPU via tiling and kernel fusion — same math, dramatically less memory traffic.
- **2023** — [Ainslie et al. — GQA](https://arxiv.org/abs/2305.13245) generalized MQA with grouped heads, now standard in LLaMA 2/3, Mistral, and Gemma.

## Recommended Reading Order

If you're new to attention, work through the pages in this order:

1. **Scaled Dot-Product** — the building block every other mechanism is built on
2. **Additive (Bahdanau)** — historical context; understand what dot-product replaced
3. **Multi-Head Attention** — the Transformer standard
4. **Causal Self-Attention** — add the autoregressive mask for decoder-only models
5. **Cross-Attention** — connect encoder and decoder
6. **MQA → GQA → Sliding Window** — inference efficiency variants
7. **Flash Attention 1 → 2 → 3** — algorithmic optimizations for the same math


## Attentions
