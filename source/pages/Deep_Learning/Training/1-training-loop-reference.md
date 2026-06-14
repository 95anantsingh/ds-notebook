(training-loop-reference)=
# Training Loop Reference

> **Quick-recall cheat sheet**: {ref}`Training Loop Patterns <training-loop-patterns>`

Complete, annotated PyTorch training loop with all production patterns in one place.

---

## Setup

Boilerplate that goes at the top of every training script.

```python
import os
import random
import numpy as np
import torch

# ── Device ────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Reproducibility ───────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # same result every run
    torch.backends.cudnn.benchmark     = False  # disable auto-tuner (it picks fastest algo per shape, non-deterministic)

set_seed(42)

# ── CUDA memory ───────────────────────────────────────────────────────────
torch.cuda.empty_cache()                    # release cached but unused memory back to the OS
torch.cuda.reset_max_memory_allocated()     # reset the peak memory stat so profiling starts fresh
```

```{note}
Set `cudnn.benchmark = True` (and `deterministic = False`) when input shapes are fixed and you want maximum speed. Use `deterministic = True` only when exact reproducibility is required — it adds a small overhead.
```

---

## Minimal Training Loop

The bare minimum — no AMP, no clipping, no scheduler. Use this as a mental anchor before layering in the production patterns below.

```python
import torch
import torch.nn as nn

model     = MyModel().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for epoch in range(num_epochs):
    # ── Training ──────────────────────────────────────────────────────
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(inputs), targets)
        loss.backward()
        optimizer.step()

    # ── Validation ────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            val_loss += nn.functional.cross_entropy(model(inputs), targets).item()
    print(f"Epoch {epoch+1}: val={val_loss/len(val_loader):.4f}")
```

---

## Data Loaders

```python
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data   = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset    = MyDataset(data, labels)
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True,  num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
```

```{tip}
`pin_memory=True` speeds up CPU→GPU transfers by using page-locked host memory. Always set it when training on CUDA. `num_workers` controls parallel data loading — 4–8 is a good default.
```

---

## Optimizer

```python
# Separate weight decay — do not apply it to biases or LayerNorm params
decay_params    = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]

optimizer = torch.optim.AdamW([
    {'params': decay_params,    'weight_decay': 0.1},
    {'params': no_decay_params, 'weight_decay': 0.0},
], lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
```

```{tip}
**Why exclude 1D params from weight decay?** Bias terms and LayerNorm scale/shift are 1D. Decaying them can destabilize training — they're not overfit risks. Always split AdamW param groups for transformer training.
```

---

## Scheduler

```python
import math

def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return current_step / max(1, warmup_steps)
        progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Usage
optimizer  = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
scheduler  = get_cosine_schedule_with_warmup(optimizer, warmup_steps=1000, total_steps=100_000)
```

---

## Standard Training Loop

```{note}
The loop below uses FP16 AMP with GradScaler — the safe default for any CUDA device. On Ampere+ hardware (A100, H100, RTX 30xx/40xx), switch to BF16 (drop the scaler entirely).
```

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    num_epochs: int,
    device: torch.device,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
):
    scaler = GradScaler(enabled=use_amp and device.type == 'cuda')  # no-op on CPU / BF16 paths

    for epoch in range(num_epochs):
        # ── Training ──────────────────────────────────────────────────────
        model.train()                                    # enable dropout, batchnorm in train mode
        train_loss = 0.0

        for step, (inputs, targets) in enumerate(train_loader):
            inputs  = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()                        # clear stale gradients from last step

            with autocast(enabled=use_amp):              # cast eligible ops to FP16 automatically
                logits = model(inputs)
                loss   = nn.functional.cross_entropy(logits, targets)

            scaler.scale(loss).backward()                # scale loss up → keep gradients in FP16 range
            scaler.unscale_(optimizer)                   # restore true magnitudes before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # prevent exploding gradients
            scaler.step(optimizer)                       # optimizer.step(); skipped if grads have inf/NaN
            scaler.update()                              # grow/shrink scale factor for next iteration
            scheduler.step()                             # advance LR schedule every step (not every epoch)

            train_loss += loss.item()                    # .item() detaches from graph, returns Python float

        # ── Validation ────────────────────────────────────────────────────
        model.eval()                                     # disable dropout, batchnorm uses running stats
        val_loss = 0.0

        with torch.no_grad():                            # disable grad tracking — saves memory and compute
            for inputs, targets in val_loader:
                inputs  = inputs.to(device)
                targets = targets.to(device)
                with autocast(enabled=use_amp):
                    logits = model(inputs)
                    loss   = nn.functional.cross_entropy(logits, targets)
                val_loss += loss.item()

        print(f"Epoch {epoch+1}: train={train_loss/len(train_loader):.4f}  val={val_loss/len(val_loader):.4f}")
```

---

## Gradient Accumulation

```python
accumulation_steps = 4   # effective_batch = batch_size × accumulation_steps
optimizer.zero_grad()

for step, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.to(device), targets.to(device)

    with autocast(enabled=use_amp):
        logits = model(inputs)
        # Divide loss BEFORE backward so accumulated gradients match a single large batch
        loss   = nn.functional.cross_entropy(logits, targets) / accumulation_steps

    scaler.scale(loss).backward()

    if (step + 1) % accumulation_steps == 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
```

---

## Checkpointing

### Save

```python
def save_checkpoint(path: str, model, optimizer, scheduler, step: int, loss: float):
    torch.save({
        'step':            step,
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'loss':            loss,
    }, path)
```

### Resume

```python
def load_checkpoint(path: str, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    return ckpt['step'], ckpt['loss']

# Usage
start_step, last_loss = load_checkpoint('ckpt.pt', model, optimizer, scheduler, device)
```

---

## Torch Compile

`torch.compile` traces the model into an optimized compute graph (via TorchInductor), fusing kernels and eliminating Python overhead. It helps both inference and training — training often benefits more because the same graph executes thousands of times.

```python
model = torch.compile(model)   # wrap once before the training loop; everything else stays the same
```

```{note}
The first 1–3 iterations are slow while compilation happens. Don't benchmark those steps.
```

**Key options:**

```python
torch.compile(model, mode="default")         # balanced compile time vs runtime
torch.compile(model, mode="reduce-overhead") # minimise Python overhead, good for small models
torch.compile(model, mode="max-autotune")    # slowest compile, fastest runtime — use for long runs
torch.compile(model, dynamic=True)           # handle variable shapes (sequence length, batch size)
torch.compile(model, fullgraph=False)        # default — falls back to eager on unsupported ops (safer)
```

**Compatibility:**

| Feature | Works with `torch.compile`? |
|---|---|
| `autocast` (AMP) | ✓ |
| `GradScaler` | ✓ |
| Gradient accumulation | ✓ |
| DDP | ✓ (compile before wrapping with DDP) |
| Custom CUDA ops | Sometimes — use `fullgraph=False` |

```{tip}
On Ampere+ GPUs (A100, H100, RTX 30xx/40xx), pair `torch.compile` with BF16 instead of FP16+GradScaler — BF16 compiles more cleanly and you can drop the scaler entirely.
```

---

## Distributed Data Parallel (quick reference)

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group(backend='nccl')
model = model.to(local_rank)
model = DDP(model, device_ids=[local_rank])

# Gradients sync automatically on backward. Access underlying model via:
model.module.state_dict()
```

---

## Full Training Loop

Everything from the sections above combined: AMP, gradient clipping, accumulation, warmup+cosine schedule, checkpointing, and `torch.compile`.

```python
import math
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(path, model, optimizer, scheduler, scaler, step, loss):
    torch.save({
        'step':            step,
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'scaler_state':    scaler.state_dict(),
        'loss':            loss,
    }, path)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int,
    device: torch.device,
    # optimizer hyperparams
    lr: float            = 3e-4,
    weight_decay: float  = 0.1,
    betas: tuple         = (0.9, 0.95),
    # schedule
    warmup_steps: int    = 1000,
    # regularisation
    max_grad_norm: float = 1.0,
    # AMP
    use_amp: bool        = True,
    # gradient accumulation
    accumulation_steps: int = 1,
    # compile
    use_compile: bool    = False,
    # checkpointing
    ckpt_path: str       = 'checkpoint.pt',
    ckpt_every: int      = 1,
):
    # ── Setup ─────────────────────────────────────────────────────────────
    decay_params    = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW([
        {'params': decay_params,    'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=lr, betas=betas, eps=1e-8)

    total_steps = len(train_loader) // accumulation_steps * num_epochs
    scheduler   = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler      = GradScaler(enabled=use_amp and device.type == 'cuda')

    if use_compile:
        model = torch.compile(model)                     # fuse kernels; first few iters are slow

    # ── Training loop ─────────────────────────────────────────────────────
    global_step = 0
    optimizer.zero_grad()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for step, (inputs, targets) in enumerate(train_loader):
            inputs  = inputs.to(device)
            targets = targets.to(device)

            with autocast(enabled=use_amp):
                logits = model(inputs)
                loss   = nn.functional.cross_entropy(logits, targets) / accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

            train_loss += loss.item() * accumulation_steps  # undo the division for logging

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs  = inputs.to(device)
                targets = targets.to(device)
                with autocast(enabled=use_amp):
                    logits = model(inputs)
                    val_loss += nn.functional.cross_entropy(logits, targets).item()

        train_loss /= len(train_loader)
        val_loss   /= len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}  step={global_step}  train={train_loss:.4f}  val={val_loss:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if (epoch + 1) % ckpt_every == 0:
            save_checkpoint(ckpt_path, model, optimizer, scheduler, scaler, global_step, val_loss)
```
