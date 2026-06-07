# PyTorch Gotchas and Coding Patterns

## `view` vs `reshape`

```python
# view — requires contiguous memory; errors if tensor is non-contiguous
x = x.view(B, T, D)

# reshape — makes a copy if needed; safe after transpose/permute
x = x.reshape(B, T, D)

# Safe pattern after transpose:
x = x.transpose(1, 2).contiguous().view(B, D, T)
# or simply:
x = x.transpose(1, 2).reshape(B, D, T)
```

```{tip}
**Default to `reshape`.** It only copies when required (non-contiguous memory), so it's identical to `view` in the common case but never errors. Use `view` only when you explicitly want to catch contiguity bugs.
```

**When you'll hit this**: `x.permute(...).view(...)` always fails. `x.transpose(1,2).view(...)` fails unless the underlying data is contiguous.

---

## In-Place Operations and Autograd

```python
# WRONG — in-place on a tensor that requires_grad breaks the computation graph
x += 1                   # same as x.__iadd__(1) — modifies in-place
x[0] = 0                 # in-place assignment

# CORRECT
x = x + 1
x = torch.cat([torch.zeros(1), x[1:]])

# EXCEPTION: optimizer.zero_grad() is fine — it's intentionally clearing grad
```

```{warning}
**Why it breaks**: PyTorch's autograd records operations on tensors. An in-place operation mutates the saved tensor that `backward()` needs to compute the gradient. The error (`RuntimeError: a leaf Variable that requires grad has been used in an in-place operation`) often appears far from the actual in-place line, making it hard to debug.
```

---

## `register_buffer` for Non-Parameter Tensors

```{important}
Use `register_buffer` for any tensor that is part of the model's state but is not a learnable parameter (e.g., causal mask, running stats). A plain `self.mask = mask` won't move to device with `.to()` and won't be saved in `state_dict`. This is a common bug when moving models between CPU and GPU.
```

Use `register_buffer` for any tensor that is:
- Part of the model's state (e.g., causal mask, running stats)
- Not a learnable parameter
- Should move with the model when calling `.to(device)` or `.cuda()`

```python
class MyModel(nn.Module):
    def __init__(self, max_len: int):
        super().__init__()
        mask = torch.triu(torch.ones(max_len, max_len, dtype=torch.bool), diagonal=1)
        self.register_buffer('causal_mask', mask)   # saved in state_dict, moves to device

    def forward(self, x):
        T = x.size(1)
        return x, self.causal_mask[:T, :T]   # always on the same device as x
```

**Contrast with**: plain `self.mask = mask` — won't move to device with `.to()`, won't be saved in `state_dict`.

---

## `einsum` Patterns for Attention

```python
# Scaled dot-product (batched, multi-head): q,k are (B,H,T,dk)
scores = torch.einsum('bhid,bhjd->bhij', q, k)        # same as q @ k.transpose(-2,-1)

# Weighted sum (batched, multi-head): a=(B,H,T,T), v=(B,H,T,dk)
out = torch.einsum('bhij,bhjd->bhid', attn, v)        # same as attn @ v

# Outer product for position bias
bias = torch.einsum('i,j->ij', pos_i, pos_j)
```

`einsum` is clearer for non-standard contractions but slightly slower than `@` for standard matmul.

---

## Common Shape-Debugging Pattern

```python
def dbg(name: str, x: torch.Tensor) -> torch.Tensor:
    """Print shape and return tensor unchanged — drop into any forward pass."""
    print(f"{name}: {tuple(x.shape)} {x.dtype} {x.device}")
    return x

# Usage
q = dbg("q after split_heads", q)
```

Alternatively, assert expected shapes at key checkpoints:

```python
assert q.shape == (B, num_heads, T, d_k), f"Expected {(B, num_heads, T, d_k)}, got {q.shape}"
```

---

## Reproducibility

```python
import torch
import random
import numpy as np

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False   # disable auto-tuner for reproducibility
```

**Note**: `cudnn.benchmark = True` auto-selects the fastest algorithm per input shape but is non-deterministic. Set to `False` when reproducing exact results.

---

## Useful Patterns

### Parameter counting

```python
total   = sum(p.numel() for p in model.parameters())
trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{trained/1e6:.1f}M trainable / {total/1e6:.1f}M total")
```

### Moving a batch to device

```python
def to_device(batch, device):
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(b, device) for b in batch)
    return batch
```

### Freezing / unfreezing layers

```python
# Freeze
for param in model.encoder.parameters():
    param.requires_grad = False

# Unfreeze specific module later (e.g., for fine-tuning)
for param in model.encoder.layers[-2:].parameters():
    param.requires_grad = True
```

### Gradient norm monitoring (without clipping)

```python
total_norm = sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
# or with built-in:
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))  # inf = no clip
```
