(training-loop-reference)=
# Training Loop Reference

> **Quick-recall cheat sheet**: {ref}`Training Loop Patterns <training-loop-patterns>`

Complete, annotated PyTorch training loop with all production patterns in one place.

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
    scaler = GradScaler(enabled=use_amp and device.type == 'cuda')

    for epoch in range(num_epochs):
        # ── Training ──────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0

        for step, (inputs, targets) in enumerate(train_loader):
            inputs  = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with autocast(enabled=use_amp):
                logits = model(inputs)
                loss   = nn.functional.cross_entropy(logits, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                   # unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
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

## Warmup + Cosine Decay Scheduler

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

## AdamW Configuration

```python
# Separate weight decay — do not apply it to biases or LayerNorm params
decay_params     = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
no_decay_params  = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]

optimizer = torch.optim.AdamW([
    {'params': decay_params,    'weight_decay': 0.1},
    {'params': no_decay_params, 'weight_decay': 0.0},
], lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
```

```{tip}
**Why exclude 1D params from weight decay?** Bias terms and LayerNorm scale/shift are 1D. Decaying them can destabilize training — they're not overfit risks. Always split AdamW param groups for transformer training.
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
