# PyTorch Fundamentals Cheat Sheet

## Tensor Creation & Device Placement

```python
import torch

# Creation
x = torch.zeros(B, T, D)
x = torch.ones(B, T, D)
x = torch.randn(B, T, D)           # N(0,1)
x = torch.arange(T).float()        # [0, 1, ..., T-1]
x = torch.full((B, T), fill_value=0.0)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x = x.to(device)
x = x.cuda()                       # shorthand

# Dtype
x = x.float()                      # fp32
x = x.half()                       # fp16
x = x.to(torch.bfloat16)
```

---

## Shape Manipulation

```python
x.shape                            # torch.Size([B, T, D])
x.size(0)                          # B

# Reshape (may copy)
x.view(B, T*D)                     # contiguous only — errors if not
x.reshape(B, T*D)                  # safe: copies if needed

# Axes
x.transpose(1, 2)                  # swap dims 1 and 2
x.permute(0, 2, 1)                 # arbitrary reorder
x.squeeze(dim=1)                   # remove size-1 dim
x.unsqueeze(dim=0)                 # add size-1 dim

# After transpose, view requires contiguous:
x.transpose(1, 2).contiguous().view(B, D, T)
```

### Broadcasting rules

```
(B, 1, T) op (B, H, T) → (B, H, T)   ✓
(B, T, 1) op (B, T, D) → (B, T, D)   ✓
(T,)      op (B, T, D) → error        ✗  (need unsqueeze)
```

---

## Autograd Lifecycle

```python
# Forward
loss = model(x)

# Backward
optimizer.zero_grad()              # clear old gradients first
loss.backward()                    # compute gradients
optimizer.step()                   # update parameters

# Inference — no gradient tracking
with torch.no_grad():
    out = model(x)

# Detach from graph (e.g., for logging)
value = loss.detach().item()
```

```{important}
**`zero_grad` order**: Always call **before** `loss.backward()`, not after `optimizer.step()`. Calling after `step()` discards the gradients you just computed and will cause the next step to error (no grad to clip).
```

---

## `nn.Module` Pattern

```python
class MyLayer(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)  # registered parameter
        self.norm   = nn.LayerNorm(d_model)
        self.register_buffer('mask', torch.ones(100, 100))  # non-param tensor, moves with .to()

    def forward(self, x):
        return self.norm(self.linear(x))

model = MyLayer(512).to(device)
model.train()   # enables dropout, batchnorm in training mode
model.eval()    # disables dropout, uses running stats for batchnorm
```

### Freezing layers

```python
for param in model.encoder.parameters():
    param.requires_grad = False
```

---

## Common Dtype/Device Bugs

| Bug | Symptom | Fix |
|-----|---------|-----|
| Mixed CPU/GPU tensors | RuntimeError: Expected all tensors on the same device | `.to(device)` before operation |
| Int tensor in float op | RuntimeError: expected scalar type Float | `.float()` |
| FP16 overflow | NaN loss | Use BF16 or GradScaler with FP16 |
| `model.half()` + batchnorm | Unstable training | Keep BN in FP32 |

---

## Useful One-Liners

```python
# Count parameters
sum(p.numel() for p in model.parameters() if p.requires_grad)

# Gradient norm (for monitoring)
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float('inf'))

# Clip gradient norm (before optimizer.step)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Save / load
torch.save(model.state_dict(), 'model.pt')
model.load_state_dict(torch.load('model.pt', map_location=device))

# Reproducibility
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
```

---

## Loss Functions

```python
# Classification — expects raw logits, NOT probabilities
criterion = nn.CrossEntropyLoss()
loss = criterion(logits, targets)      # logits: (B, C), targets: (B,) int

# Language modeling — flatten over sequence
loss = criterion(logits.view(-1, vocab_size), targets.view(-1))

# Binary
criterion = nn.BCEWithLogitsLoss()     # sigmoid + BCE, numerically stable
```

```{warning}
`nn.CrossEntropyLoss` applies softmax internally. **Never** pass `F.softmax(logits)` to it — that double-applies softmax and produces incorrect gradients that look plausible but are wrong.
```

---

## Common `nn` Layers

```python
nn.Linear(in, out, bias=True)
nn.Embedding(vocab_size, d_model)
nn.LayerNorm(d_model)
nn.Dropout(p=0.1)
nn.MultiheadAttention(d_model, num_heads, batch_first=True)   # PyTorch built-in
nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, batch_first=True)
```
