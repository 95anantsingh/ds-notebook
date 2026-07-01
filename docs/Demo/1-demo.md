---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Component Reference

A living reference for every component available when writing notes on this site.
Each block below shows the rendered output followed by the source that produced it.

---

## Typography

### Inline Formatting

**bold**, _italic_, `inline code`, \*escaped symbols\*

Strikethrough: ~~deprecated method~~ — enabled by the `strikethrough` extension.

Subscript and superscript: H{sub}`2`O and the 4{sup}`th` of July.

Inline attributes (`attrs_inline` extension) attach classes/attributes to inline elements — e.g. open a link in a new tab: [docs](https://example.com){target="_blank"}, or tag a span {.bg-warning}`flagged text`.

### Smartquotes and Replacements

The `smartquotes` and `replacements` extensions auto-convert common text:

| Source | Rendered |
|--------|----------|
| `'single'` and `"double"` | 'single' and "double" |
| `--` | -- |
| `---` | --- |
| `...` | ... |
| `+-` | +- |

### Line Breaks

Use `\` at end of a line to insert a `<br>` without starting a new paragraph:

**Fleas** \
Adam \
Had 'em.

### Lists

Unordered (use `-` or `*`):

- First item
  - Nested item (2-space indent)
  - Another nested item
- Second item

Ordered:

1. First step
2. Second step
   1. Sub-step
   2. Sub-step

Task list (`tasklist` extension). With `myst_enable_checkboxes = True` the boxes are interactive rather than read-only:

- [ ] Implement attention mechanism
- [x] Add positional encoding
- [ ] Train on GPU

### Comments

Lines starting with `%` are stripped from the rendered output:

% This comment will not appear in the HTML.

### Footnotes

Auto-numbered[^autoref] and manually-numbered[^3] footnotes are collected at the bottom of the page.

[^autoref]: This is the auto-numbered footnote definition.
[^3]: This footnote always renders with the number 3.

---

## Math

### Inline Math

Inline equations render mid-sentence: the softmax function is $\sigma(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$, and the dot-product attention score is $\text{score}(q, k) = \frac{q \cdot k}{\sqrt{d_k}}$.

### Block Equations

$$
\mathcal{L} = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)
$$

### Multi-line with Alignment

$$
\begin{aligned}
\mu &= \frac{1}{n} \sum_{i=1}^{n} x_i \\
\sigma^2 &= \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2 \\
\hat{x}_i &= \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
\end{aligned}
$$

### Transformer Attention

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

### Matrix Notation

$$
W = \begin{pmatrix} w_{11} & w_{12} & \cdots & w_{1n} \\ w_{21} & w_{22} & \cdots & w_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ w_{m1} & w_{m2} & \cdots & w_{mn} \end{pmatrix}
$$

### Using math block

This is much more user readable than other options, but might not be sufficient for complex equations.

```{math}
(a + b)^2 = a^2 + 2ab + b^2

(a + b)^2  &=  (a + b)(a + b) \\
           &=  a^2 + 2ab + b^2
```

---

## Code Blocks

All code blocks automatically get a copy button from `sphinx_copybutton`.

### Python

```python
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.d_k)
        q, k, v = qkv.unbind(dim=2)
        scale = self.d_k ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        return self.out((attn @ v).reshape(B, T, C))
```

### With Line Numbers and Highlights

```{code-block} python
:linenos:
:emphasize-lines: 3,8,9

def scaled_dot_product_attention(q, k, v, mask=None):
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)  # scale

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    weights = F.softmax(scores, dim=-1)  # normalize
    return torch.matmul(weights, v)      # weighted sum
```

### With Caption

Use `:caption:` on `{code-block}` to add a label above the block:

```{code-block} python
:caption: training_loop.py
:linenos:
:emphasize-lines: 4,5

for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()
        loss = criterion(model(batch.x), batch.y)
        loss.backward()
        optimizer.step()
```

### Including Code from Files

`{literalinclude}` embeds source from a file. Use `:start-after:` / `:end-before:` to extract a slice:

```{literalinclude} assets/example.py
:language: python
:start-after: start example
:end-before: end example
```

**All options:**

| Option | Description |
|--------|-------------|
| `language` | Syntax lexer |
| `linenos` | Show line numbers |
| `lineno-start` | Starting line number |
| `emphasize-lines` | Comma-separated lines to highlight |
| `caption` | Label above the block |
| `dedent` | Strip N leading spaces |
| `start-after` | Begin after line containing this string |
| `end-before` | End before line containing this string |
| `lines` | Explicit range, e.g. `5-10` |
| `pyobject` | Include a named Python class or function |

---

## Mermaid Diagrams

All diagrams are written in Markdown and rendered in the browser — no image files needed.

### Flowchart — Training Loop

:::{container} w-md
```{mermaid}
flowchart TD
    A([Start Training]) --> B[Load Batch]
    B --> C[Forward Pass]
    C --> D[Compute Loss]
    D --> E{Loss < threshold?}
    E -->|No| F[Backward Pass]
    F --> G[Clip Gradients]
    G --> H[Optimizer Step]
    H --> I[Zero Gradients]
    I --> B
    E -->|Yes| J([Save Checkpoint])
```
:::

### Line Breaks in Node Labels

`\n` is **not** supported inside Mermaid node text. Use `<br>` instead:

:::{container} w-sm
```{mermaid}
flowchart LR
    A["Input<br>Embedding"] --> B["Multi-Head<br>Attention"] --> C["Feed<br>Forward"]
```
:::

### Math in Diagrams

`$$...$$` (MathJax) is supported inside node labels:

:::{container} w-sm
```{mermaid}
flowchart LR
    A["$$Q, K, V$$"] --> B["$$\text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$"]
```
:::

### Edge Animations

Edges can be animated using the `e1@-->` syntax to assign an edge ID, then setting animation properties inline or via `classDef`.

**Shorthand — inline speed:**

:::{container} w-xs
```{mermaid}
flowchart LR
    A e1@--> B
    e1@{ animation: fast }
```
:::

Two speeds are supported: `fast` and `slow`. This is shorthand for `{ animate: true, animation: fast }`.

**Via classDef — full control:**

:::{container} w-xs
```{mermaid}
flowchart LR
    A e1@--> B
    classDef animate stroke-dasharray: 9\,5,stroke-dashoffset: 900,animation: dash 25s linear infinite;
    class e1 animate
```
:::

- `e1@-->` creates an edge with the ID `e1`
- `classDef` sets stroke dash pattern, offset, and CSS animation
- `class e1 animate` applies the class to that edge
- Commas inside `stroke-dasharray` must be escaped as `\,` since commas are Mermaid's delimiter

### Size Control

Diagrams render at their natural size by default. Wrap one in a `{container}` with a width class (`w-xs` … `w-full`) to cap its width — see [Containers & Width](#containers-width) for the full reference. For tall diagrams, fullscreen control is already enabled site-wide.

```markdown
:::{container} w-sm
```{mermaid}
flowchart LR
    A --> B --> C
```
:::
```

Height is auto-managed per the rendering.

---

## Containers & Width

`{container}` renders as a plain `<div class="…">` — use it instead of a raw `<div>` to avoid raw-HTML parsing issues, and to attach utility classes to any block of content.

The width utilities below are **general-purpose**: each caps `max-width` and centres the block, so they work on **anything** wrapped in a `{container}` — Mermaid diagrams, figures, images, Plotly plots, tables, etc.

| Class | Max width |
|---|---|
| `w-xs` | 250px |
| `w-sm` | 400px |
| `w-md` | 600px |
| `w-lg` | 850px |
| `w-full` | 100% |

````markdown
:::{container} w-md
![diagram](assets/owl.jpg)
:::
````

Combine the class with any inner content; the container only constrains width and centres — height is left to the content.

---

## Figures

```{figure} assets/owl.jpg
:width: 50%
:align: center
:alt: Owl artwork
:name: fig-owl

*Figure 1.* Use `{figure}` to add a caption and alt text to any image.
```

Cross-reference with `` {ref}`fig-logo` `` or `` {numref}`fig-logo` ``.

**All options:** `:alt:`, `:height:`, `:width:`, `:scale:`, `:align:` (`left`/`center`/`right`), `:target:`, `:name:`, `:class:`, `:figwidth:`, `:figclass:`.

---

## Images

Use `{image}` for a standalone image without a caption. Unlike `{figure}`, it does not produce a numbered label.

```{image} assets/owl.jpg
:width: 200px
:align: center
:alt: Owl artwork
```

Key options: `:width:` (px or %), `:height:`, `:align:` (`left` / `center` / `right`), `:alt:`, `:target:`, `:class:`.

---

## Admonitions

Use admonitions to call attention to important content.

```{note}
Use `{note}` for supplementary information that is helpful but not critical.
```

```{tip}
Use `{tip}` for shortcuts, best practices, or suggestions that improve understanding.
```

```{important}
Use `{important}` for information that must not be overlooked.
```

```{warning}
Use `{warning}` when a reader could make a costly mistake if they skip this.
```

```{caution}
Use `{caution}` for potential issues that are less severe than a warning.
```

```{danger}
Use `{danger}` for actions that can cause data loss, crashes, or irreversible damage.
```

```{attention}
Use `{attention}` as a general eye-catcher with no specific severity implied.
```

```{hint}
Use `{hint}` for small clues that help without giving away the full answer.
```

```{error}
Use `{error}` to document a known error condition or a common mistake.
```

Custom title and body:

:::{admonition} Mixed Precision Gotcha
:class: warning

When using `torch.autocast`, gradients for `float16` operations can underflow to zero.
Wrap the loss scaling step inside `torch.cuda.amp.GradScaler`.
:::

---

## Toggle Buttons

Use toggles to hide worked examples, derivations, or answers so the page stays scannable.

### Basic Toggle — Hidden by Default

```{toggle}
**Answer:** The gradient of the cross-entropy loss with respect to the softmax input $z_i$ is:

$$\frac{\partial \mathcal{L}}{\partial z_i} = \hat{y}_i - y_i$$

This is why softmax + cross-entropy is so convenient — the gradient is just the prediction error.
```

### Shown by Default — Use Dropdown Instead

For content that starts open and can be collapsed, use `{dropdown}` with `:open:` from sphinx-design. It renders open in the HTML itself without JavaScript.

::::{dropdown} Click to hide the derivation
:open:

**Derivation of Backprop through LayerNorm:**

Given $\hat{x} = \frac{x - \mu}{\sigma}$, the gradient with respect to $x$ requires the chain rule through both $\mu$ and $\sigma$:

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{1}{\sigma} \left( \frac{\partial \mathcal{L}}{\partial \hat{x}_i} - \frac{1}{n}\sum_j \frac{\partial \mathcal{L}}{\partial \hat{x}_j} - \frac{\hat{x}_i}{n} \sum_j \frac{\partial \mathcal{L}}{\partial \hat{x}_j} \hat{x}_j \right)$$
::::

### Toggle Inside an Admonition

:::{note}
Flash Attention rewrites the attention computation to be IO-aware.

```{toggle} Show memory complexity comparison
| Method          | Memory       | Passes over HBM |
|-----------------|-------------|-----------------|
| Standard Attn   | $O(N^2)$    | $O(N^2 / M)$    |
| Flash Attention | $O(N)$      | $O(N^2 / M)$    |
| Flash Attn 2    | $O(N)$      | $O(N^2 / M)$    |

Where $N$ = sequence length, $M$ = SRAM size.
```
:::

### Collapsible Elements

Any element with class `toggle` becomes collapsible — not just text blocks:

```{image} assets/owl.jpg
:class: toggle
```

---

## Tables

### Simple Table

| Optimiser  | Momentum | Adaptive LR | Weight Decay | Notes                        |
|-----------|----------|-------------|--------------|------------------------------|
| SGD       | ✓        | ✗           | L2           | Needs careful LR tuning      |
| Adam      | ✓        | ✓           | Decoupled*   | *use AdamW for correct WD    |
| AdamW     | ✓        | ✓           | ✓            | Default for transformer LM   |
| Adafactor | ✗        | ✓           | ✓            | Low memory; used in T5       |
| LION      | ✓        | ✗           | ✓            | Sign-based; memory efficient |

### Grid Table (complex spans)

```{list-table} Quantisation Format Comparison
:header-rows: 1
:widths: 15 10 10 15 30 20

* - Format
  - Bits
  - Type
  - Hardware
  - Typical Use
  - Accuracy Impact
* - FP32
  - 32
  - Float
  - All
  - Training baseline
  - None (reference)
* - BF16
  - 16
  - Float
  - A100, H100
  - Mixed-precision training
  - Negligible
* - FP8
  - 8
  - Float
  - H100
  - Training + inference
  - Small
* - INT8
  - 8
  - Int
  - All
  - Post-training inference
  - Small–medium
* - INT4
  - 4
  - Int
  - All
  - Compressed inference
  - Medium
* - GPTQ
  - 4
  - Int
  - All
  - LLM weight-only quant
  - Medium (depends on group size)
```

### Spanning Cells

Neither pipe tables nor `{list-table}` support `colspan`/`rowspan`. Use a raw HTML `<table>` block directly in the Markdown file. Use `class="docutils align-default"` to render it beautifully.

<table class="docutils align-default">
  <thead>
    <tr>
      <th rowspan="2">Format</th>
      <th rowspan="2">Bits</th>
      <th colspan="2">Supported On</th>
      <th rowspan="2">Notes</th>
    </tr>
    <tr>
      <th>Training</th>
      <th>Inference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="3"><strong>Float</strong></td>
      <td>32 (FP32)</td>
      <td>✓</td><td>✓</td>
      <td>Full precision baseline</td>
    </tr>
    <tr>
      <td>16 (BF16)</td>
      <td>✓</td><td>✓</td>
      <td>Same range as FP32; preferred on Ampere+</td>
    </tr>
    <tr>
      <td>8 (FP8)</td>
      <td>✓</td><td>✓</td>
      <td>H100+ only; two variants: E4M3 and E5M2</td>
    </tr>
    <tr>
      <td rowspan="3"><strong>Int</strong></td>
      <td>8 (INT8)</td>
      <td>—</td><td>✓</td>
      <td>Post-training quantisation</td>
    </tr>
    <tr>
      <td>4 (INT4)</td>
      <td>—</td><td>✓</td>
      <td>Weight-only; requires dequant at runtime</td>
    </tr>
    <tr>
      <td>2 (INT2)</td>
      <td>—</td><td>✓</td>
      <td>Experimental; high accuracy loss</td>
    </tr>
  </tbody>
</table>

---

## Block Quotes and Definitions

> "Attention is all you need."
> — Vaswani et al., 2017

Block quote with attribution (`attrs_block` extension):

{attribution="Vaswani et al., 2017"}
> We need no recurrence — attention alone suffices.

Definition list (term followed by indented definition):

Autoregressive
: Generates one token at a time, conditioning each output on all previous outputs.

KV Cache
: A memory buffer storing the key and value tensors from past decode steps to avoid recomputation.

**Perplexity**
: $\text{PPL} = \exp\!\left(-\frac{1}{T}\sum_{t=1}^{T} \log p(w_t \mid w_{<t})\right)$ — lower is better.

---

## Field Lists

Field lists are key-value mappings (`:key: value`), based on reStructuredText syntax. Enabled by the `fieldlist` extension.

### Basic Syntax

:name only:
:model: GPT-4
:*Nested syntax*: Both name and body support **bold**, `code`, and $math$.
:Multi-line: The second and subsequent lines only need to be indented
   to align with the body, not the colon.
:Blocks:

  Body can contain any block syntax:

  - item one
  - item two

  ```python
  print("hello")
  ```

### API Docstring

The canonical use case — document a function's parameters, return values, and exceptions:

```{py:function} scaled_dot_product(q, k, v, mask=None)

Compute scaled dot-product attention.

:param torch.Tensor q: Query matrix of shape ``(B, T, d_k)``.
:param torch.Tensor k: Key matrix of shape ``(B, T, d_k)``.
:param torch.Tensor v: Value matrix of shape ``(B, T, d_v)``.
:param mask: Optional boolean mask; masked positions are set to ``-1e9``.
:type mask: torch.Tensor or None
:return: Attention output of shape ``(B, T, d_v)``.
:rtype: torch.Tensor
:raises ValueError: if ``q`` and ``k`` have mismatched last dimensions.
```

---

## Cross-References

Because `autosectionlabel` is enabled with `autosectionlabel_prefix_document = True`, every heading gets an anchor prefixed by its document path.

- Link to a heading on this page: {ref}`pages/Demo/1-demo:Mermaid Diagrams`
- Link to a heading on this page with custom text: {ref}`jump to diagrams <pages/Demo/1-demo:Mermaid Diagrams>`
- Link to a heading in another file: `` {ref}`pages/Natural Language Processing/Attentions/index:Attentions` ``
- Link to another page by path: {doc}`../Natural_Language_Processing/Attentions/index`

### Heading Anchors (Markdown fragments)

With `myst_heading_anchors = 2`, headings (h1–h2) also get GitHub-style slug `id`s, so you can link with a plain Markdown fragment instead of the `{ref}` role:

- [Jump to Math](#math)
- [Jump to Code Blocks](#code-blocks)

The slug is the lower-cased heading text with spaces replaced by hyphens.

### External Links by Scheme

Custom `myst_url_schemes` give short link prefixes for common references:

- arXiv paper: [Attention Is All You Need](arxiv:1706.03762)
- DOI: [BERT](doi:10.18653/v1/N19-1423)
- Wikipedia: [Transformer](wiki:Transformer_(deep_learning_architecture))

### Intersphinx (external project docs)

With `sphinx.ext.intersphinx` mapped to Python, NumPy, and PyTorch, you can link straight to a symbol in their docs:

- Python: {py:func}`print`
- NumPy: {py:func}`numpy.matmul`
- PyTorch: {py:class}`torch.nn.MultiheadAttention`

### Hover Tooltips

`sphinx-tippy` adds hover-preview tooltips to internal cross-references automatically — hover over {ref}`pages/Demo/1-demo:Mermaid Diagrams` above to see a preview of the target section. No syntax needed; it applies to all internal links.

---

## Grids

Grids use a 12-column system that adapts to screen size. The argument is 1–4 integers for xs / sm / md / lg breakpoints.

### Responsive Columns

Resize the browser to see columns adapt (1 on mobile, 4 on large screens):

::::{grid} 1 2 3 4
:outline:
:gutter: 2

:::{grid-item-card} XS
1 column on mobile.
:::
:::{grid-item-card} SM
2 columns on small.
:::
:::{grid-item-card} MD
3 columns on medium.
:::
:::{grid-item-card} LG
4 columns on large.
:::
::::

### Gutter Control

`:gutter:` controls spacing. One number for all sizes; four numbers for xs / sm / md / lg:

::::{grid} 2
:gutter: 3 3 4 5

:::{grid-item-card}
Small gutter on mobile, larger on desktop.
:::
:::{grid-item-card}
Spacing adjusts responsively.
:::
::::

### Item-Level Column Width

Override how many columns (out of 12) a single item spans with `:columns:`:

::::{grid} 2

:::{grid-item-card}
:columns: 12 8 8 8

Spans full width on mobile, 8/12 on larger screens.
:::
:::{grid-item-card}
:columns: 12 4 4 4

Spans full width on mobile, 4/12 on larger screens.
:::
:::{grid-item-card}
:columns: auto

Auto width based on content.
:::
::::

### Nesting

Grids can be nested to create complex adaptive layouts:

::::::{grid} 1 1 2 2
:gutter: 2

:::::{grid-item}
::::{grid} 1 1 1 1
:gutter: 1

:::{grid-item-card} Item 1.1
Multi-line

content
:::
:::{grid-item-card} Item 1.2
Content
:::
::::
:::::

:::::{grid-item}
::::{grid} 1 1 1 1
:gutter: 1

:::{grid-item-card} Item 2.1
Content
:::
:::{grid-item-card} Item 2.2
Content
:::
:::{grid-item-card} Item 2.3
Content
:::
::::
:::::

::::::

### Options Reference

**`grid` options**

| Option | Values | Description |
|--------|--------|-------------|
| `gutter` | 0–5 or four values | Spacing between items |
| `margin` | 0–5 / `auto` | Outer margin |
| `padding` | 0–5 | Inner padding |
| `outline` | flag | Border around the grid |
| `reverse` | flag | Reverse item order |
| `class-container` | CSS class | Container element |
| `class-row` | CSS class | Row element |

**`grid-item` options**

| Option | Values | Description |
|--------|--------|-------------|
| `columns` | 1–12 / `auto` / four values | Column span |
| `margin` | 0–5 / `auto` | Outer margin |
| `padding` | 0–5 | Inner padding |
| `child-direction` | `column` / `row` | Direction of children |
| `child-align` | `start` / `end` / `center` / `justify` / `spaced` | Child alignment |
| `outline` | flag | Border around item |
| `class` | CSS class | Item element |

**`grid-item-card` options** — same as `grid-item` plus all `card` options below.

---

## Design Components

### Tabs

Use tabs to show the same concept across frameworks or to separate long alternatives.

::::{tab-set}

:::{tab-item} PyTorch
```python
import torch.nn.functional as F

output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=True,
)
```
:::

:::{tab-item} JAX
```python
import jax.numpy as jnp

def dot_product_attention(q, k, v, mask=None):
    d_k = q.shape[-1]
    scores = jnp.matmul(q, k.swapaxes(-2, -1)) / jnp.sqrt(d_k)
    if mask is not None:
        scores = jnp.where(mask, scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    return jnp.matmul(weights, v)
```
:::

:::{tab-item} NumPy
```python
import numpy as np

def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = np.exp(scores - scores.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    return weights @ V
```
:::

::::

### Synced Tabs

Add `:sync-group:` to each `tab-set` and `:sync:` to each `tab-item` — selecting a tab in one set syncs all sets with the same group:

::::{tab-set}
:sync-group: framework

:::{tab-item} PyTorch
:sync: pt
`torch.optim.AdamW`
:::
:::{tab-item} JAX
:sync: jax
`optax.adamw`
:::
::::

::::{tab-set}
:sync-group: framework

:::{tab-item} PyTorch
:sync: pt
`torch.nn.CrossEntropyLoss`
:::
:::{tab-item} JAX
:sync: jax
`optax.softmax_cross_entropy`
:::
::::

### tab-set-code

Shorthand for language-labelled, auto-synced code examples. Tabs are labelled and synced by language:

````{tab-set-code}
```python
import torch
loss = torch.nn.CrossEntropyLoss()(logits, labels)
```

```bash
pip install torch torchvision
```
````

**Tab options**

| Option | Description |
|--------|-------------|
| `tab-set` `:sync-group:` | Group name for synchronisation (default: `tab`) |
| `tab-set` `:class:` | CSS class on container |
| `tab-item` `:sync:` | Key for syncing across sets |
| `tab-item` `:selected:` | Pre-select this tab |
| `tab-item` `:name:` | Referenceable anchor |
| `tab-item` `:class-container:` `:class-label:` `:class-content:` | CSS overrides |

---

### Cards

Use cards to organise related concepts side-by-side.

::::{grid} 2

:::{grid-item-card} Encoder-only Models
**Examples:** BERT, RoBERTa, DeBERTa

Best for tasks that require understanding the full context simultaneously — classification, NER, extractive QA.

Trained with Masked Language Modelling (MLM).
:::

:::{grid-item-card} Decoder-only Models
**Examples:** GPT-4, LLaMA, Mistral

Best for generation tasks. Process tokens left-to-right with causal masking.

Trained with next-token prediction (CLM).
:::

:::{grid-item-card} Encoder-Decoder Models
**Examples:** T5, BART, mT5

Best for seq2seq tasks — translation, summarisation, generative QA.

Encoder builds full context; decoder attends to it via cross-attention.
:::

:::{grid-item-card} Mixture of Experts
**Examples:** Mixtral, Switch Transformer

Scales parameter count without scaling compute. A router sends each token to $k$ of $N$ expert FFN layers.

Active params ≪ total params.
:::

::::

### Card with Header and Footer

Anything before the first `^^^` is the header; anything after the last `+++` is the footer:

:::{card} Attention Score
Flash Attention v3
^^^
$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$
+++
H100 only · IO-aware · FP8
:::

### Card Alignment

Use `:text-align:` and `:margin: auto` to control card alignment:

::::{grid} 3

:::{grid-item-card} Left
:text-align: left

Default alignment.
:::

:::{grid-item-card} Centre
:text-align: center

Content centred.
:::

:::{grid-item-card} Right
:text-align: right

Content right-aligned.
:::

::::

### Card Carousel

A horizontally scrollable row of fixed-width cards. The argument is the number of cards visible at once:

::::{card-carousel} 3

:::{card} Transformer
Encoder-decoder. Attention is $O(N^2)$ in sequence length.
:::
:::{card} Flash Attention
IO-aware rewrite. Reduces HBM reads to $O(N^2/M)$.
:::
:::{card} RoPE
Rotary position encoding applied to Q/K directly.
:::
:::{card} GQA
Grouped Query Attention — shares K/V heads to cut KV cache size.
:::
:::{card} SWA
Sliding Window Attention — limits attention to a local window.
:::
::::

### Cards with Links

```{note}
For `:link-type: doc`, paths with spaces in directory names are not supported by sphinx-design.
Use `:link-type: url` with a relative `.html` path, or only link to docs whose full path has no spaces.
```

::::{grid} 2

:::{grid-item-card} Inference Notes
:link: ../Interview/Inference/1-inference
:link-type: doc

Click this card to navigate to a page. Add `:link:` (doc path) and `:link-type: doc` to any card to make the whole card clickable.
:::

:::{grid-item-card} NLP Index
:link: ../Interview/NLP/index
:link-type: doc

Cards can also link to section index pages, not just leaf documents.
:::

::::

**Card options**

| Option | Values | Description |
|--------|--------|-------------|
| `width` | `auto` `25%` `50%` `75%` `100%` | Card width |
| `margin` | 0–5 / `auto` | Outer margin |
| `text-align` | `left` `right` `center` `justify` | Text alignment |
| `img-background` | URI | Image behind content |
| `img-top` | URI | Image above content |
| `img-bottom` | URI | Image below content |
| `img-alt` | string | Alt text for image |
| `link` | URL or path | Makes entire card clickable |
| `link-type` | `url` `ref` `doc` `any` | Link resolution type |
| `link-alt` | string | Screen-reader link text |
| `shadow` | `none` `sm` `md` `lg` | Drop shadow size |
| `class-card` `class-header` `class-body` `class-title` `class-footer` | CSS class | CSS overrides |

---

### Dropdowns

::::{dropdown} When to use BF16 vs FP16
BF16 and FP16 both use 16 bits, but allocate them differently:

| Format | Exponent bits | Mantissa bits | Max value |
|--------|--------------|---------------|-----------|
| FP16   | 5            | 10            | 65504     |
| BF16   | 8            | 7             | ~3.4×10³⁸ |

**Use BF16** on Ampere+ GPUs (A100, H100, RTX 30xx+) — same dynamic range as FP32, avoids overflow during training.

**Use FP16** only when targeting older hardware (V100, T4) that lacks BF16 support. Requires loss scaling to prevent underflow.
::::

::::{dropdown} KV Cache — How it works
:open:

During autoregressive generation, every new token attends to all previous tokens. Without a cache, keys and values are recomputed from scratch at each step — $O(T^2)$ total work.

With a KV cache:
1. On the first forward pass (prefill), compute and **store** $K, V$ for the entire prompt.
2. On each decode step, compute $K, V$ for only the **new token** and append to the cache.
3. Attention uses the full cached $K, V$ — $O(T)$ work per step.

Memory cost: `2 × num_layers × num_heads × d_head × seq_len × dtype_bytes` per sequence.
::::

### Dropdown with Animation and Color

::::{dropdown} Flash Attention — Why it's faster
:color: info
:icon: info
:animate: fade-in-slide-down

Standard attention materialises the full $N \times N$ score matrix in HBM — that's the bottleneck, not FLOPs.

Flash Attention tiles Q, K, V into SRAM blocks and fuses softmax + matmul into a single kernel, never writing the full score matrix to HBM.
::::

**Dropdown options**

| Option | Values | Description |
|--------|--------|-------------|
| `open` | flag | Open by default |
| `color` | `primary` `secondary` `success` `danger` `warning` `info` `light` `dark` `muted` | Header color |
| `icon` | octicon name | Icon in header |
| `chevron` | `right-down` `down-up` | Chevron direction |
| `animate` | `fade-in` `fade-in-slide-down` | Open animation |
| `margin` | 0–5 / `auto` | Outer margin |
| `name` | string | Referenceable anchor |
| `class-container` `class-title` `class-body` | CSS class | CSS overrides |

---

### Badges

Use badges inline to tag content with status or category labels.

**Filled variants:**

{bdg}`plain` {bdg-primary}`primary` {bdg-secondary}`secondary` {bdg-success}`success` {bdg-info}`info` {bdg-warning}`warning` {bdg-danger}`danger` {bdg-light}`light` {bdg-muted}`muted` {bdg-dark}`dark` {bdg-white}`white` {bdg-black}`black`

**Outline variants:**

{bdg-primary-line}`primary` {bdg-secondary-line}`secondary` {bdg-success-line}`success` {bdg-info-line}`info` {bdg-warning-line}`warning` {bdg-danger-line}`danger` {bdg-light-line}`light` {bdg-muted-line}`muted` {bdg-dark-line}`dark` {bdg-white-line}`white` {bdg-black-line}`black`

**In context:** Flash Attention 2 {bdg-success}`stable` is the current default in HuggingFace Transformers {bdg-info}`≥4.36`. RoPE embeddings are {bdg-success-line}`production` while ALiBi is {bdg-warning-line}`legacy`.

**Link badges** — clickable, using `{bdg-link-*}`:

{bdg-link-primary}`https://example.com` {bdg-link-primary-line}`Documentation <https://example.com>`

---

## Buttons

Buttons let users jump to an external (`{button-link}`) or internal (`{button-ref}`) target with a single click.

```{button-link} https://sphinx-design.readthedocs.io
:color: primary
:shadow:
Sphinx Design docs
```

```{button-ref} 1-demo
:ref-type: doc
:color: info
:outline:
Back to this page
```

By default Sphinx renders the button content as **raw text** — so `**Bold text**` with `:ref-type: ref` shows the asterisks literally. With `:ref-type: myst`, the content is parsed as Markdown and renders properly:

```{button-ref} 1-demo
:ref-type: myst
:color: success
**Bold** button label
```

Use `:click-parent:` to make the button's parent container clickable too:

::::{card} Card with an expanded button
Click anywhere on this card.
+++
:::{button-ref} 1-demo
:ref-type: doc
:expand:
:color: secondary
:click-parent:
Open
:::
::::

**`{button-link}` / `{button-ref}` options**

| Option | Description |
|--------|-------------|
| `ref-type` | (`button-ref` only) Reference type: `any` (default), `ref`, `doc`, or `myst` |
| `color` | Semantic color: `primary` `secondary` `success` `danger` `warning` `info` `light` `dark` `muted` |
| `outline` | Outline color variant |
| `align` | `left` `right` `center` `justify` |
| `expand` | Expand to fit parent width |
| `click-parent` | Make parent container also clickable |
| `tooltip` | Tooltip on hover |
| `shadow` | Add shadow CSS |
| `class` | Additional CSS classes |

---

## Executable Code Cells

Unlike `{code-block}` (which only displays source), `{code-cell}` **runs** the code at build time and renders its output below. The page needs a jupytext frontmatter header (see the top of this file) so myst-nb treats it as a notebook.

```{code-cell} python
import math
[round(math.sqrt(n), 3) for n in range(1, 6)]
```

The last expression is auto-displayed, just like in a Jupyter notebook.

**Hide the input** with `:tags: [hide-input]` — the code becomes a collapsible toggle, output always shown:

```{code-cell} python
:tags: [hide-input]
total = sum(range(1, 101))
f"Sum of 1..100 = {total}"
```

**Show output only** with `:tags: [remove-input]` — the source is dropped entirely (used for the plots on this site):

```{code-cell} python
:tags: [remove-input]
print("Generated at build time — no input cell shown.")
```

**Cell tags**

| Tag | Effect |
|-----|--------|
| `hide-input` | Collapse the source into a toggle |
| `hide-output` | Collapse the output into a toggle |
| `remove-input` | Drop the source; keep output |
| `remove-output` | Drop the output; keep source |
| `remove-cell` | Drop the cell entirely (runs, shows nothing) |
| `raises-exception` | Allow the cell to error without failing the build |

```{note}
Cells execute top-to-bottom in a shared kernel, so later cells can use names defined earlier. For figures pulled from a *separate* notebook, use the `{glue}` approach in [Interactive Plots](#interactive-plots) instead.
```

---

## Interactive Plots

Interactive [Plotly](https://plotly.com/python/) figures embed via **myst-nb**'s `{glue}` mechanism: a code cell stores a figure under a key, and a `{glue}` directive pastes it anywhere on the page. Requires the jupytext header (top of this file).

### In-file glue

For a quick, one-off plot, build and glue it in a `remove-input` cell right on the page, then paste it by key:

```{code-cell} python
:tags: [remove-input]
import math
import plotly.graph_objects as go
from myst_nb import glue
from IPython.display import HTML

t = [i / 20 for i in range(200)]
fig = go.Figure(go.Scatter3d(
    x=[math.cos(4 * math.pi * v) for v in t],
    y=[math.sin(4 * math.pi * v) for v in t],
    z=t, mode='lines', line=dict(color=t, colorscale='Viridis', width=6),
))
fig.update_layout(
    scene=dict(xaxis_title='x', yaxis_title='y', zaxis_title='t'),
    paper_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0), height=350,
)
glue('demo_plotly', HTML(fig.to_html(include_plotlyjs='cdn', full_html=False)), display=False)
```

```{glue} demo_plotly
```

*Drag to rotate, scroll to zoom.* The source is a `:tags: [remove-input]` cell that ends with the `glue(...)` call, followed by a `{glue}` directive:

````md
```{code-cell} python
:tags: [remove-input]
# ...build fig...
glue('demo_plotly', HTML(fig.to_html(include_plotlyjs='cdn', full_html=False)), display=False)
```

```{glue} demo_plotly
```
````

### Cross-file glue

To reuse a figure across pages, or to keep a page tidy, put the builder in a **separate orphan notebook** and reference it with `:doc:`. Create e.g. `assets/my-plot.ipynb` with `"orphan": true` in its metadata and a single cell:

```python
import plotly.graph_objects as go
from myst_nb import glue
from IPython.display import HTML

fig = go.Figure(...)  # build your figure
glue('my_plot', HTML(fig.to_html(include_plotlyjs='cdn', full_html=False)), display=False)
```

Then paste it on any page, pointing `:doc:` at the notebook:

````md
```{glue} my_plot
:doc: assets/my-plot.ipynb
```
````

```{tip}
**Short helper code is fine inline** on the page (in-file glue). **Long or reused builders belong in an orphan notebook** (cross-file glue) — it keeps the prose readable and lets several pages share one figure.
```

```{important}
- Glue **an `HTML(fig.to_html(...))` object**, not the raw `fig`. `glue()` captures via `_repr_html_`; a bare Plotly figure renders through `_ipython_display_` side effects that `glue()` can't see, so it comes out blank.
- Do **not** add require.js to `conf.py` — its AMD loader steals Plotly's global and the CDN script fails to initialise.
```

### plotly_utils — Matrix Figures (2D & 3D)

`source/_lib/plotly_utils.py` is the reusable **Op model** behind the matrix-multiplication visualisations in the NLP notes — both the 3D bracket/ghost-plane figures and the 2D heatmap figures, static or animated. It is importable from any notebook or `{code-cell}` because `conf.py` prepends `source/_lib` to `PYTHONPATH` at build time (via `os.environ["PYTHONPATH"]`, which child kernel processes inherit).

**Op model** — an operation owns its result, its display glyph, and its animation schedule. Static vs animated and 3D vs 2D are *modes over the same op*, not separate functions: a static figure is just the terminal frame of the op's schedule.

```{list-table}
:header-rows: 1
:widths: 38 14 48

* - Call
  - Returns
  - Purpose
* - `Matrix(data, name, color=None, shape=None, planes=None)`
  - `Matrix`
  - Wrap a `(rows, cols)` or `(B, rows, cols)` array. Front slice drawn; `planes` ghost planes behind (3D). `color`/`shape` auto-fill if omitted.
* - `matmul(A, B, *, name, color, shape)`
  - `Op`
  - `A @ B` — owns the per-output-cell animation schedule.
* - `scale(A, by, *, glyph, name, color, shape)`
  - `Op`
  - Element-wise `A / by` — static by default.
* - `softmax(A, *, axis, prefix, name, color, shape)`
  - `Op`
  - Row-wise softmax — static by default.
* - `op(matrices, glyphs, *, prefix, states)`
  - `Op`
  - Escape hatch for arbitrary layouts or custom schedules.
* - `figure(op, *, style, animate, height, planes)`
  - `go.Figure`
  - Pure construction. `style='3d'` (brackets) / `'2d'` (heatmap); `animate=True` attaches frames.
* - `show(fig, *, div_id, controls, loop, height, steps, modebar)`
  - `HTML`
  - The **one** CDN-safe display boundary — always `include_plotlyjs='cdn'`. Use as a `glue()` value or a `{code-cell}` final expression.
```

Operands accept `Matrix` **or** `Op` (auto-unwrapped to `.result`), so ops chain like math:

```python
from plotly_utils import Matrix, matmul, scale, softmax, figure, show
from plotly_utils import PALETTE, PLOT_HEIGHT
import numpy as np

Q_m  = Matrix([[[1,0,2,1],[0,1,1,0],[2,1,0,1]]], "Q",  PALETTE[0], "(B, T, d_k)")
Kt_m = Matrix(np.array([[[1,0,1,0],[0,1,0,1],[1,1,0,0]]]).transpose(0,2,1),
              "Kᵀ", PALETTE[1], "(B, d_k, T)")

scores  = matmul(Q_m, Kt_m, name="scores", color=PALETTE[2])  # Q @ Kᵀ
scaled_ = scale(scores, by=2)                                  # chains — uses scores.result
W       = softmax(scaled_)

# static 3D  (bracket + ghost-plane style)
show(figure(scores))

# animated 3D — loads paused at the end frame; ▶⏸ toggle fades in on hover
show(figure(scores, animate=True), controls="hover", steps=True)

# animated 2D — heatmap, loops with a ~2 s pause on the completed equation
show(figure(scores, style="2d", animate=True), loop=True, steps=True)

# cross-file use from a notebook
from myst_nb import glue
glue("scores_plot", show(figure(scores, height=PLOT_HEIGHT)), display=False)
```

**Live example — static 3D scores figure:**

```{code-cell} python
:tags: [remove-input]
import numpy as np
from plotly_utils import Matrix, matmul, figure, show, PALETTE, PLOT_HEIGHT

Q_m  = Matrix([[[1,0,2,1],[0,1,1,0],[2,1,0,1]]],
              "Q", PALETTE[0], "(B, T, d<sub>k</sub>)")
Kt_m = Matrix(np.array([[[1,0,1,0],[0,1,0,1],[1,1,0,0]]]).transpose(0,2,1),
              "K<sup>T</sup>", PALETTE[1], "(B, d<sub>k</sub>, T)")
scores = matmul(Q_m, Kt_m, name="scores", color=PALETTE[2])
show(figure(scores, height=PLOT_HEIGHT))
```

**Live example — animated 2D heatmap (loops, with step text):**

```{code-cell} python
:tags: [remove-input]
import numpy as np
from plotly_utils import Matrix, matmul, figure, show, PALETTE

# 2D shows a single flat slice, so the shapes carry no batch B.
Q_m  = Matrix(np.array([[1,0,2,1],[0,1,1,0],[2,1,0,1]], float), "Q",  PALETTE[0], "(T, d<sub>k</sub>)")
Kt_m = Matrix(np.array([[1,0,1,0],[0,1,0,1],[1,1,0,0]], float).T, "Kᵀ", PALETTE[1], "(d<sub>k</sub>, T)")
scores2d = matmul(Q_m, Kt_m, name="S", color=PALETTE[2], shape="(T, T)")
show(figure(scores2d, style="2d", animate=True), loop=True, steps=True)
```

**`show()` options**

| Option | Values | Effect |
|---|---|---|
| `controls` | `"hover"` (default) / `"always"` / `None` | Play-pause toggle: fades in on hover / always visible / hidden. Ignored for static figures. |
| `loop` | `True` | Autoplay: jump to frame 0, play through, pause ~2 s on the terminal frame, repeat. |
| `steps` | `True` | Show the per-frame step text (bottom-centre, black). Hidden by default. |
| `modebar` | `True` | Show Plotly's top toolbar. Hidden by default. |
| `div_id` | string | Fixes the div id (else auto-generated). |
| `height` | int | Override figure height before serialising. |

**`figure()` options**

| Option | Values | Effect |
|---|---|---|
| `style` | `"3d"` (default) / `"2d"` | Bracket + ghost-plane 3D, or square-cell heatmap. |
| `animate` | `False` (default) / `True` | Terminal frame only, or attach all frames (loads paused at end). |
| `height` | int | Figure height in px (default `PLOT_HEIGHT = 250`). For 2D this is the constraint — cell size is derived from it. |
| `planes` | int | Override ghost-plane count for all matrices (3D). `0` = flat, no batch depth. |

**2D layout notes** — `style="2d"` derives a uniform cell size from `height` (the tallest matrix fills the plot area; shorter matrices are centre-aligned with the *same* cell size). Cells are square by default and widen only if a decimal value needs the room. The figure is fixed-width (for square cells) and centred on the page via a flex wrapper; hover tooltips and drag-zoom are disabled. Because a 2D heatmap shows a single flat slice, its shapes should omit the batch `B` (the 3D ghost planes are what represent the batch).

**Color palette** — 10 generic colours, auto-assigned by matrix position if `color` is omitted. Import `PALETTE` and index into it, or pass any hex string:

| Index | Hex | Swatch | | Index | Hex | Swatch |
|---|---|---|---|---|---|---|
| `PALETTE[0]` | `#2C5C8A` | blue | | `PALETTE[5]` | `#B5651D` | sienna |
| `PALETTE[1]` | `#4F7A1A` | green | | `PALETTE[6]` | `#1F6F8B` | steel-blue |
| `PALETTE[2]` | `#A86A12` | amber | | `PALETTE[7]` | `#8A2C5C` | wine |
| `PALETTE[3]` | `#7A4FA0` | purple | | `PALETTE[8]` | `#6B8A1A` | olive |
| `PALETTE[4]` | `#1F7A6B` | teal-green | | `PALETTE[9]` | `#5C1F7A` | deep-violet |

If `color` is omitted on a `Matrix`/op, the renderer assigns `PALETTE[position % 10]` at render time. Pass `color=PALETTE[n]` (or any hex) to pin a matrix to one colour across figures.


