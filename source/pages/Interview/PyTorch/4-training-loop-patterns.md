(training-loop-patterns)=
# PyTorch Training Loop Patterns

> Reference implementation with full annotated code: {ref}`Training Loop Reference <training-loop-reference>`

```{mermaid}
---
title: PyTorch AMP Training Loop
---
flowchart TD
    START(["Start Epoch"])

    subgraph train["Training  —  model.train()"]
        BATCH["Load Batch → .to(device)"]
        ZERO["optimizer.zero_grad()"]
        FWD["Forward Pass\ninside autocast()"]
        LOSS["Compute Loss"]
        BWD["scaler.scale(loss).backward()"]
        UNSCALE["scaler.unscale_(optimizer)"]
        CLIP["clip_grad_norm_()"]
        STEP["scaler.step(optimizer)"]
        UPDATE["scaler.update()"]
        SCHED["scheduler.step()"]
        MORE{More batches?}
    end

    subgraph val["Validation  —  model.eval()"]
        VBATCH["Load Val Batch → .to(device)"]
        VNOINF["torch.no_grad()"]
        VFWD["Forward Pass\ninside autocast()"]
        VLOSS["Accumulate Loss"]
        VMORE{More val batches?}
        LOG["Log avg val loss"]
    end

    DONE(["End Epoch"])

    START --> BATCH
    BATCH --> ZERO --> FWD --> LOSS --> BWD
    BWD --> UNSCALE --> CLIP --> STEP --> UPDATE --> SCHED
    SCHED --> MORE
    MORE -->|yes| BATCH
    MORE -->|no| VBATCH

    VBATCH --> VNOINF --> VFWD --> VLOSS --> VMORE
    VMORE -->|yes| VBATCH
    VMORE -->|no| LOG --> DONE
```

---

## Canonical Training Loop

```python
model.train()
for epoch in range(num_epochs):
    for batch in dataloader:
        inputs, targets = batch

        optimizer.zero_grad()            # clear old gradients BEFORE forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clip before step
        optimizer.step()
        scheduler.step()                 # step after optimizer (most schedulers)
```

**Ordering matters**:
1. `zero_grad()` → before forward (not after `step()` — same effect but clearer intent)
2. `backward()` → computes gradients
3. `clip_grad_norm_` → before `step()` (clipping gradients that are about to be applied)
4. `optimizer.step()` → apply gradients
5. `scheduler.step()` → update LR

---

## Gradient Accumulation

Simulate a larger effective batch size when GPU memory is limited:

```python
accumulation_steps = 4   # effective batch = batch_size × 4
optimizer.zero_grad()

for i, (inputs, targets) in enumerate(dataloader):
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps   # scale loss
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
```

```{important}
Divide loss by `accumulation_steps` **before** `backward()`. If you don't, the accumulated gradients are `N×` larger than a single full-batch gradient — the effective learning rate is multiplied by N.
```

---

## Automatic Mixed Precision (AMP)

::::{tab-set}

:::{tab-item} FP16 + GradScaler {bdg-warning}`older hardware`
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()   # needed for FP16; scales loss to avoid underflow

model.train()
for inputs, targets in dataloader:
    optimizer.zero_grad()

    with autocast():                          # FP16 forward + backward
        outputs = model(inputs)
        loss = criterion(outputs, targets)

    scaler.scale(loss).backward()             # scale to avoid FP16 underflow
    scaler.unscale_(optimizer)                # unscale before clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)                    # skip step if gradients are inf/nan
    scaler.update()                           # adjust scale factor for next iter
```

FP16 has limited dynamic range (max ~65504). Use GradScaler to prevent gradient underflow. Required on V100, T4 and older hardware.
:::

:::{tab-item} BF16 {bdg-success}`Ampere+ recommended`
```python
with autocast(dtype=torch.bfloat16):   # wider dynamic range than FP16 → no NaN risk
    outputs = model(inputs)
    loss = criterion(outputs, targets)

loss.backward()                         # no scaler needed
optimizer.step()
```

BF16 has the same exponent range as FP32 — overflow/underflow is essentially impossible. No GradScaler required. Use on A100, H100, RTX 30xx/40xx and Apple Silicon.
:::

::::

```{note}
**When to use which**: BF16 is preferred on Ampere+ GPUs (A100, 4090) — same range as FP32, less precision. FP16 is faster on older hardware (V100, T4) but requires GradScaler to prevent underflow.
```

---

## Gradient Clipping

```python
# Clip by global norm — standard for transformers
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Returns the norm BEFORE clipping (useful for monitoring)
grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Why transformers need it**: Attention + deep residual networks can produce very large gradient norms, especially early in training. Clipping at 1.0 is the standard for most transformer pre-training.

---

## LR Scheduling: Warmup + Cosine Decay

```python
import math

def lr_lambda(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / warmup_steps                     # linear warmup
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * progress)) # cosine decay to 0

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda s: lr_lambda(s, 1000, 100000))
```

Or use HuggingFace `get_cosine_schedule_with_warmup` directly.

---

## Evaluation Loop

```python
model.eval()
total_loss = 0.0

with torch.no_grad():                  # disable gradient tracking entirely
    for inputs, targets in val_loader:
        with autocast():               # optional: same dtype as training
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        total_loss += loss.item()

avg_loss = total_loss / len(val_loader)
model.train()                          # switch back after evaluation
```

---

## Common Bugs

| Bug | Symptom | Fix |
|-----|---------|-----|
| Forgot `zero_grad()` | Gradients accumulate → wrong updates | Call before `backward()` each step |
| Forgot `scaler.update()` | Scale never adjusts, potential NaN spiral | Call at end of every step |
| `scheduler.step()` before `optimizer.step()` | PyTorch warning + incorrect LR | Always step optimizer first |
| `clip_grad_norm_` after `optimizer.step()` | Clipping has no effect | Must be before `step()` |
| Loss not divided in grad accumulation | Gradients N× too large | Divide by `accumulation_steps` |
| `model.eval()` never called | Dropout active at eval | Call before any evaluation |
