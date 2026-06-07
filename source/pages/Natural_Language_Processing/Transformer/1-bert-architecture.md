(bert-architecture)=
# BERT Architecture

> **Depends on**: {ref}`Encoder Block <encoder-block>`
> **Interview Q&A**: {ref}`Transformer Variants Q&A <transformer-variants-qa>`

BERT (Bidirectional Encoder Representations from Transformers) is an encoder-only model. Every token attends to every other token — no causal mask. This bidirectionality makes it excellent for understanding tasks (classification, NER, QA extraction) but unsuitable for text generation.

```{mermaid}
---
title: BERT Architecture
---
flowchart TD
    INPUT["[CLS] token₁ … tokenₙ [SEP]"]

    subgraph embed["BERTEmbedding  (summed, not concatenated)"]
        TOK["token_embed  (B, T, d_model)"]
        SEG["segment_embed  0=sent A · 1=sent B"]
        POS["pos_embed  (learned absolute)"]
        SUM(["+ sum → LayerNorm → Dropout"])
    end

    subgraph encoder["N × EncoderBlock  (bidirectional — no causal mask)"]
        ENC["Multi-Head Self-Attention + FFN × N layers"]
        LN["LayerNorm"]
    end

    SEQ["sequence_output  (B, T, d_model)"]
    CLS["[:, 0, :]  CLS token"]
    POOL["Pooler  Linear + Tanh → (B, d_model)"]

    subgraph heads["Fine-Tuning Heads"]
        H1["Classification\nLinear → num_classes"]
        H2["Token Classification (NER)\nLinear → num_labels per token"]
        H3["Extractive QA\nLinear → start & end logits"]
    end

    INPUT --> TOK & SEG & POS
    TOK & SEG & POS --> SUM
    SUM --> ENC --> LN
    LN --> SEQ
    SEQ --> CLS --> POOL
    POOL --> H1
    SEQ --> H2 & H3
```

---

## BERT Embedding Layer

Three embeddings are **summed** (not concatenated):

```python
import torch
import torch.nn as nn


class BERTEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.seg_embed = nn.Embedding(2, d_model)   # segment 0 (A) or 1 (B)
        self.norm      = nn.LayerNorm(d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,     # (B, T) — token ids, [CLS]=101, [SEP]=102, [PAD]=0
        segment_ids: torch.Tensor,   # (B, T) — 0 for sentence A, 1 for sentence B
    ) -> torch.Tensor:
        T = input_ids.size(1)
        positions = torch.arange(T, device=input_ids.device)
        x = self.tok_embed(input_ids) + self.pos_embed(positions) + self.seg_embed(segment_ids)
        return self.dropout(self.norm(x))
```

---

## Full BERT Model

```python
from .encoder_block import EncoderBlock


class BERT(nn.Module):
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
        self.embedding = BERTEmbedding(vocab_size, d_model, max_len, dropout)
        self.layers    = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm   = nn.LayerNorm(d_model)
        self.pooler = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Tanh(),
        )

    def forward(
        self,
        input_ids: torch.Tensor,             # (B, T)
        segment_ids: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,   # (B, T) bool — True for PAD
    ) -> dict[str, torch.Tensor]:
        if segment_ids is None:
            segment_ids = torch.zeros_like(input_ids)

        x = self.embedding(input_ids, segment_ids)   # (B, T, d_model)

        attn_mask = None
        if padding_mask is not None:
            attn_mask = padding_mask[:, None, None, :]  # (B, 1, 1, T) broadcast

        for layer in self.layers:
            x = layer(x, mask=attn_mask)

        sequence_output = self.norm(x)               # (B, T, d_model) — per-token

        # [CLS] token is always at position 0
        cls_output = self.pooler(sequence_output[:, 0, :])   # (B, d_model)

        return {'sequence_output': sequence_output, 'pooled_output': cls_output}
```

---

## Fine-Tuning Heads

::::{tab-set}

:::{tab-item} Sequence Classification
```python
class BERTClassifier(nn.Module):
    def __init__(self, bert: BERT, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.bert    = bert
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(bert.pooler[0].out_features, num_classes)

    def forward(self, input_ids, segment_ids=None, padding_mask=None):
        out = self.bert(input_ids, segment_ids, padding_mask)
        return self.head(self.dropout(out['pooled_output']))   # (B, num_classes) logits
```
Uses the `[CLS]` pooled output — aggregated sentence representation.
:::

:::{tab-item} Token Classification (NER)
```python
class BERTTokenClassifier(nn.Module):
    def __init__(self, bert: BERT, num_labels: int, dropout: float = 0.1):
        super().__init__()
        self.bert    = bert
        self.dropout = nn.Dropout(dropout)
        self.head    = nn.Linear(bert.norm.normalized_shape[0], num_labels)

    def forward(self, input_ids, segment_ids=None, padding_mask=None):
        out = self.bert(input_ids, segment_ids, padding_mask)
        return self.head(self.dropout(out['sequence_output']))   # (B, T, num_labels)
```
Uses the full `sequence_output` — one prediction per token.
:::

:::{tab-item} Extractive QA (SQuAD)
```python
class BERTQuestionAnswering(nn.Module):
    def __init__(self, bert: BERT):
        super().__init__()
        self.bert      = bert
        d_model        = bert.norm.normalized_shape[0]
        self.qa_head   = nn.Linear(d_model, 2)   # predicts start and end logits

    def forward(self, input_ids, segment_ids=None, padding_mask=None):
        out    = self.bert(input_ids, segment_ids, padding_mask)
        logits = self.qa_head(out['sequence_output'])            # (B, T, 2)
        start_logits, end_logits = logits.split(1, dim=-1)       # (B, T, 1) each
        return start_logits.squeeze(-1), end_logits.squeeze(-1)  # (B, T)
```
Predicts start and end token indices of the answer span.
:::

::::

---

## Key Points

```{note}
- **No causal mask** — every position attends to every other position
- **`[CLS]` = token id 101**, always prepended; its hidden state represents the sequence
- **`[SEP]` = token id 102**, separates sentences and marks sequence end
- **`[PAD]` = token id 0**, used to pad sequences to the same length in a batch; must be masked in attention
- BERT-base {bdg-secondary}`110M`: 12 layers, 768 `d_model`, 12 heads
- BERT-large {bdg-secondary}`340M`: 24 layers, 1024 `d_model`, 16 heads
```
