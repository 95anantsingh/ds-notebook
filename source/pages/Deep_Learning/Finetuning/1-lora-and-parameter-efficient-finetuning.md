# LoRA and Parameter-Efficient Fine-Tuning

Parameter-efficient fine-tuning (PEFT) methods fine-tune a small fraction of parameters while keeping the pre-trained weights frozen. Essential for adapting large models on constrained hardware.

---

## LoRA (Low-Rank Adaptation)

**Key idea**: Freeze the pre-trained weight matrix $W_0 \in \mathbb{R}^{d \times k}$ and inject a low-rank update:

$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} \cdot B A$$

where $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times k}$, and $r \ll \min(d, k)$.

Only $A$ and $B$ are trained. At merge time, $\Delta W$ is added to $W_0$ with no inference overhead.

```{mermaid}
---
title: LoRA — Low-Rank Adaptation
---
flowchart TD
    X["Input  x"]

    subgraph frozen["Frozen — not trained"]
        W0["W₀ · x\npretrained weight"]
    end

    subgraph lora["Trainable — only A and B updated"]
        A["A · x\nrank-r projection  (r, in_features)\nr ≪ min(d, k)"]
        B["B · Ax\nexpand back  (out_features, r)"]
        SCALE["× α / r\nscaling factor"]
    end

    ADD(["+ add"])
    OUT["Output  y = W₀x + (α/r)·BAx"]

    X --> W0
    X --> A --> B --> SCALE

    W0 --> ADD
    SCALE --> ADD
    ADD --> OUT
```

---

## `LoRALinear` Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features:  int,
        out_features: int,
        rank:         int   = 4,
        alpha:        float = 1.0,   # scaling factor; common to set alpha = rank
        dropout:      float = 0.0,
    ):
        super().__init__()
        self.rank  = rank
        self.scale = alpha / rank

        # Pre-trained weights — frozen
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight)
        self.weight.requires_grad = False

        # LoRA matrices — trained
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))   # B=0 → ΔW=0 at init
        nn.init.kaiming_uniform_(self.lora_A)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base forward
        base_out = F.linear(x, self.weight)
        # LoRA forward: x → A → B → scale
        lora_out = F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scale
        return base_out + lora_out

    def merge_weights(self) -> nn.Linear:
        """Merge ΔW into W for zero-overhead inference."""
        merged = nn.Linear(self.weight.shape[1], self.weight.shape[0], bias=False)
        merged.weight = nn.Parameter(self.weight + self.scale * (self.lora_B @ self.lora_A))
        return merged
```

---

## Applying LoRA to a Transformer

Replace the Q and V projection matrices in each attention layer (common heuristic; K and FFN layers are less impactful):

```python
def add_lora_to_model(model: nn.Module, rank: int = 8, alpha: float = 8.0) -> None:
    """Replace nn.Linear modules named 'q_proj' or 'v_proj' with LoRALinear in-place."""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(k in name for k in ('q_proj', 'v_proj')):
            parent_name, child_name = name.rsplit('.', 1)
            parent = model.get_submodule(parent_name)
            lora = LoRALinear(
                module.in_features, module.out_features,
                rank=rank, alpha=alpha,
            )
            lora.weight = nn.Parameter(module.weight.data.clone(), requires_grad=False)
            setattr(parent, child_name, lora)
```

---

## Training Setup

```python
# 1. Add LoRA adapters
add_lora_to_model(model, rank=8)

# 2. Only optimize LoRA params (base model is frozen)
lora_params = [p for n, p in model.named_parameters() if 'lora' in n]
optimizer = torch.optim.AdamW(lora_params, lr=3e-4)

# 3. Verify parameter counts
total    = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable: {trainable/total*100:.2f}% of {total/1e6:.0f}M params")
```

---

## LoRA vs QLoRA

::::{tab-set}

:::{tab-item} LoRA
- Base model weights in **FP16 or BF16** — not quantized
- LoRA adapter matrices in same dtype
- Typical GPU requirement: full model in memory (~14 GB for 7B in FP16)
- When to use: you have enough GPU memory for the base model weights

```python
add_lora_to_model(model, rank=8)
```
:::

:::{tab-item} QLoRA {bdg-success}`24 GB GPU → 7B+`
Combines LoRA with 4-bit quantization of the base model:

1. **Base model in 4-bit NF4** — quantized and frozen (~4 GB for 7B)
2. **LoRA adapters in BF16** — trained normally
3. **Double quantization**: quantize the quantization constants for extra memory savings
4. **Paged optimizers**: page optimizer states to CPU when GPU memory is tight

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(
    'meta-llama/Llama-2-7b-hf', quantization_config=bnb_config
)

peft_config = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=['q_proj', 'v_proj'],
    lora_dropout=0.05,
)
model = get_peft_model(model, peft_config)
```
:::

::::

---

## Why LoRA Works

```{important}
The **low intrinsic dimensionality** hypothesis: fine-tuning updates tend to live in a low-dimensional subspace of the full weight space. LoRA explicitly parameterizes this subspace with rank-$r$ matrices.

Empirically, $r = 4$–$16$ captures most of the useful adaptation signal for most tasks. Using $r = 64$ or higher rarely improves results and defeats the purpose.
```

---

## Interview Q: "How would you fine-tune a 7B LLM on a single 24 GB GPU?"

::::{dropdown} Show answer
1. Load base model in **4-bit quantization** (bitsandbytes NF4) → ~4 GB for 7B params
2. Apply **LoRA** to Q/V projections in all attention layers, rank=8–16
3. Train only LoRA params (~0.1% of total) with AdamW in **BF16**
4. Use **gradient accumulation** (8–16 steps) to simulate larger batch size
5. Use **paged AdamW** if optimizer states threaten to overflow GPU memory
6. Keep sequence length ≤ 2048 for training — reduce if OOM

Result: ~7 GB total GPU usage; full fine-tuning of a 7B model at 7B × 4 bytes = 28 GB is impossible on 24 GB.
::::
