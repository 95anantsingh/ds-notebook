# Additive (Bahdanau) Attention

The first attention mechanism, introduced in [*Neural Machine Translation by Jointly Learning to Align and Translate*](arxiv:1409.0473) {bdg-info}`ICLR 2015`. It shattered the fixed-length encoding bottleneck in seq2seq models and established the core attention pattern -- dynamic context from learned alignment -- that every subsequent mechanism inherits.

---

## Intuition

:::::{grid} 1 1 1 2
:gutter: 3

::::{grid-item}
:columns: 12 12 12 7
:child-align: start

In a classic encoder-decoder model, the decoder receives one fixed summary vector from the encoder -- a severe bottleneck for long sentences. Additive attention replaces this with a **dynamic context vector**: at each decoding step, a small alignment network scores every encoder hidden state against the current decoder state, and the context is a weighted sum of encoder states based on those scores.

**Key insight: the decoder doesn't read a fixed summary -- it dynamically decides *how much* of each encoder state to use at every generation step.**

**What each component represents:**

- **Encoder states ($h_j$)** -- one vector per source token, available to the decoder throughout generation.
- **Decoder state ($s_i$)** -- what the decoder knows at step $i$; encodes everything generated so far.
- **Alignment score ($e_{ij}$)** -- how relevant encoder state $h_j$ is to decoder step $i$; output of a small MLP.
- **Context vector ($c_i$)** -- a soft weighted blend of encoder states; the dynamic summary fed to the decoder at step $i$.

:::{dropdown} Why "additive"?
:color: primary
:icon: info
:animate: fade-in-slide-down
The score is computed by *adding* the projected query and key -- $\mathbf{W}_1 \mathbf{h}_j + \mathbf{W}_2 \mathbf{s}_i$ -- before applying tanh and a linear projection. This contrasts with dot-product attention, which *multiplies* query and key directly. The additive form handles different query and key dimensions without requiring them to match, at the cost of introducing extra learnable parameters.
:::

:::{dropdown} Why tanh?
:color: primary
:icon: info
:animate: fade-in-slide-down
Tanh introduces non-linearity so the alignment network can model interactions between encoder and decoder states, not just linear combinations. Without it the score collapses to a purely linear function of $h_j$ and $s_i$ -- limiting expressivity. Tanh also bounds outputs to $(-1, 1)$, which helped gradient stability in the original RNN setting.
:::

::::

::::{grid-item}
:child-align: center
:columns: 12 12 12 5

```{mermaid}
flowchart TD
    H(["h₁ … hₙ<br>(encoder states)"]) --> E["alignment score<br>e_ij"]
    S(["sᵢ<br>(decoder state)"]) --> E
    E --> A["softmax<br>→ α_ij (weights)"]
    A --> C["weighted sum<br>→ cᵢ (context)"]
    C --> O([Decoder output])
```

::::
:::::

---

## Deep Dive

:::::{grid} 1 1 1 2
:gutter: 3

::::{grid-item}
:columns: 12 12 12 7

Let $\mathbf{h}_j \in \mathbb{R}^{d_h}$ be the $j$-th encoder hidden state and $\mathbf{s}_i \in \mathbb{R}^{d_s}$ the decoder state at step $i$.

$$
e_{ij} = \mathbf{v}^\top \tanh\!\bigl(\mathbf{W}_1\, \mathbf{h}_j + \mathbf{W}_2\, \mathbf{s}_i\bigr)
$$

$$
\alpha_{ij} = \text{softmax}_j(e_{ij}) = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
\quad \boldsymbol{\alpha}_i \in \mathbb{R}^{T_x}
$$

$$
\mathbf{c}_i = \sum_{j=1}^{T_x} \alpha_{ij}\, \mathbf{h}_j
\quad \mathbf{c}_i \in \mathbb{R}^{d_h}
$$

The parameters $\mathbf{W}_1$, $\mathbf{W}_2$, and $\mathbf{v}$ form the *alignment model* -- a small MLP trained end-to-end with the encoder and decoder.

Where,

:$\mathbf{h}_j$: $j$-th encoder hidden state
:$\mathbf{s}_i$: Decoder state at step $i$
:$e_{ij}$: Unnormalized alignment score -- how well $h_j$ matches decoder step $i$
:$\alpha_{ij}$: Attention weight over encoder position $j$ at decoder step $i$
:$\mathbf{c}_i$: Dynamic context vector fed to the decoder at step $i$
:$\mathbf{W}_1 \in \mathbb{R}^{d_a \times d_h}$: Projects encoder states into alignment space
:$\mathbf{W}_2 \in \mathbb{R}^{d_a \times d_s}$: Projects decoder state into alignment space
:$\mathbf{v} \in \mathbb{R}^{d_a}$: Scoring vector; collapses tanh output to a scalar
:$d_a$: Alignment (hidden) dimension -- a tunable hyperparameter
:$T_x$: Source sequence length; $T_y$: target sequence length

::::

::::{grid-item}
:columns: 12 12 12 5
:child-align: center

:::{container} w-xs
```{mermaid}
flowchart TD
    H(["h_j"]) --> E["$$e_{ij} = v^T \tanh(W_1 h_j + W_2 s_i)$$<br>(score)"]
    S(["s_i"]) --> E
    E --> A["softmax<br>(weights α_ij)"]
    A --> C["$$c_i = \sum_j \alpha_{ij} h_j$$<br>(context)"]
    C --> O([Output c_i])
```
:::

::::
:::::

### Complexity

::::{grid} 1 1 1 2
:gutter: 3

:::{grid-item}
:columns: 12 12 12 6
:child-align: center

:Time: $O(T_x \cdot T_y \cdot d_a)$ -- one MLP call per source-target pair
:Memory: $O(T_x \cdot T_y)$ -- alignment matrix stored in full
:Extra params: $\mathbf{W}_1$, $\mathbf{W}_2$, $\mathbf{v}$ -- the alignment network

:::

:::{grid-item}
:columns: 12 12 12 6
:child-align: start

:Parallelizable: Scores per decoder step; decoder itself is sequential
:Bottleneck: $T_x \cdot T_y$ MLP evaluations -- no batched matrix multiply
:vs. dot-product: Slower per token; handles $d_q \neq d_k$ natively

:::
::::

---

## Implementation

::::{tab-set}

:::{tab-item} PyTorch

```{code-block} python
:linenos:
:emphasize-lines: 20,21,22

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdditiveAttention(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, hidden_dim: int):
        super().__init__()
        self.W_query = nn.Linear(query_dim, hidden_dim, bias=False)
        self.W_key   = nn.Linear(key_dim,   hidden_dim, bias=False)
        self.v       = nn.Linear(hidden_dim, 1,          bias=False)

    def forward(
        self,
        query: torch.Tensor,  # (B, d_s)        — decoder state
        keys:  torch.Tensor,  # (B, T_x, d_h)   — encoder hidden states
        values: torch.Tensor, # (B, T_x, d_v)   — usually same as keys
        mask: torch.Tensor | None = None,  # (B, T_x) bool, True = keep
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.W_query(query).unsqueeze(1)           # (B, 1, d_a)
        k = self.W_key(keys)                           # (B, T_x, d_a)
        energy = self.v(torch.tanh(q + k)).squeeze(-1) # (B, T_x)

        if mask is not None:
            energy = energy.masked_fill(~mask, float("-inf"))

        weights = F.softmax(energy, dim=-1)             # (B, T_x)
        context = (weights.unsqueeze(1) @ values).squeeze(1)  # (B, d_v)
        return context, weights
```

:::

:::{tab-item} Modern Equivalent

For most new work, project to a shared dimension and use scaled dot-product attention instead:

```python
import torch.nn.functional as F

# Project to a shared dimension, then use dot-product
q = nn.Linear(query_dim, d_k)(query)   # (B, d_k)
k = nn.Linear(key_dim,   d_k)(keys)    # (B, T_x, d_k)

context = F.scaled_dot_product_attention(
    q.unsqueeze(1), k, keys
).squeeze(1)  # (B, d_v)
```

See {doc}`Scaled Dot-Product Attention <01-scaled-dot-product-attention>` for the full API.

:::

::::

---

## When to Use

Additive attention was **superseded by scaled dot-product attention** for most tasks -- dot-product is faster (matrix multiplication vs. per-pair MLP calls) and parallelizes trivially across positions.

**Still reach for additive attention when:**

- **Interpretable alignment** -- the weights $\alpha_{ij}$ form an alignment matrix; visualizing it as a heatmap shows which source tokens the model attends to at each decoding step.
- **Mismatched dimensions** -- key and query have different sizes ($d_q \neq d_k$) and you prefer not to add projection layers; $\mathbf{W}_1$ and $\mathbf{W}_2$ handle this naturally.
- **Lightweight CPU inference** -- on small encoder-decoder models where the MLP overhead is negligible compared to the RNN body.
- **Replicating classic architectures** -- reproducing Bahdanau or Luong (2015) results for research comparison.

---

:::{admonition} Pitfalls
:class: warning
- **Broadcasting bug**: `q + k` only works if shapes broadcast correctly. `query` must be expanded to `(B, 1, d_a)` before adding to `keys` of shape `(B, T_x, d_a)`.
- **Gradient saturation**: tanh can saturate for large pre-activations; keep `hidden_dim` moderate (32–128) and initialize $\mathbf{v}$ with a small norm.
- **Memory for long sequences**: The alignment matrix is $O(T_x \cdot T_y)$; for long documents this grows quickly even though per-step complexity is linear in $T_x$.
:::

```{tip}
- The alignment model introduces extra parameters ($\mathbf{W}_1$, $\mathbf{W}_2$, $\mathbf{v}$) unlike scaled dot-product attention, which is parameter-free.
- Bahdanau used a *bidirectional* LSTM encoder; $h_j$ is the concatenated forward and backward state -- so $d_h$ is twice the LSTM hidden size.
- Luong (2015) simplified the score to a bilinear form: $e_{ij} = \mathbf{s}_i^\top \mathbf{W}\, \mathbf{h}_j$ -- fewer parameters, similar performance.
- The alignment weights $\alpha_{ij}$ visualize as a heatmap showing the learned source-target correspondence.
```
