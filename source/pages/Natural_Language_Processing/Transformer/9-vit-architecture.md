(vit-architecture)=
# Vision Transformer (ViT)

> **Paper**: "An Image is Worth 16x16 Words" (Dosovitskiy et al., 2020)
> **Depends on**: {ref}`Encoder Block <encoder-block>`
> **Interview Q&A**: {ref}`Transformer Variants Q&A <transformer-variants-qa>`

ViT applies a standard transformer encoder to sequences of image patches, with minimal domain-specific modifications.

---

## Architecture Overview

```
Input image (B, C, H, W)
  → PatchEmbedding: (B, N, d_model)   where N = (H/P) × (W/P)
  → Prepend [CLS] token: (B, N+1, d_model)
  → Add position embedding: (B, N+1, d_model)
  → Dropout
  → N × EncoderBlock
  → LayerNorm
  → [CLS] position → MLP head → class logits
```

---

## Patch Embedding

The efficient implementation uses `nn.Conv2d` with `kernel_size=stride=patch_size` — this extracts non-overlapping patches and projects them in a single operation.

```python
import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels: int, d_model: int, patch_size: int, image_size: int):
        super().__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.num_patches = (image_size // patch_size) ** 2
        # Conv2d with kernel=stride=patch_size ≡ unfold + linear
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)                           # (B, d_model, H/P, W/P)
        x = x.flatten(2)                           # (B, d_model, N)
        x = x.transpose(1, 2)                      # (B, N, d_model)
        return x
```

---

## Full ViT

```python
from .encoder_block import EncoderBlock


class ViT(nn.Module):
    def __init__(
        self,
        image_size:  int,    # H = W (square image)
        patch_size:  int,    # P — must divide image_size
        in_channels: int,    # 3 for RGB
        num_classes: int,
        d_model:     int,
        num_heads:   int,
        num_layers:  int,
        d_ff:        int | None = None,
        dropout:     float = 0.1,
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, d_model, patch_size, image_size)
        N = self.patch_embed.num_patches

        # [CLS] token — learned parameter, broadcast across batch
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Learned 1D position embedding over N+1 positions (patches + CLS)
        self.pos_embed = nn.Parameter(torch.zeros(1, N + 1, d_model))
        self.dropout   = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

        # Classification head: applied to [CLS] token only
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B = x.size(0)

        patches = self.patch_embed(x)                          # (B, N, d_model)

        # Prepend [CLS] token — expand from (1,1,d) to (B,1,d)
        cls = self.cls_token.expand(B, -1, -1)                # (B, 1, d_model)
        x   = torch.cat([cls, patches], dim=1)                # (B, N+1, d_model)

        x   = self.dropout(x + self.pos_embed)                # add position embedding

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        cls_out = x[:, 0]                                      # (B, d_model) — [CLS] position
        return self.head(cls_out)                              # (B, num_classes)
```

---

## Instantiation Example

```python
# ViT-B/16 (ViT-Base, 16×16 patches, ImageNet)
model = ViT(
    image_size=224, patch_size=16, in_channels=3,
    num_classes=1000, d_model=768, num_heads=12, num_layers=12,
)
# N = (224/16)^2 = 196 patches + 1 CLS = 197 tokens per image
```

---

## Key Gotchas

```{warning}
**Patch divisibility**: `image_size % patch_size` must be 0 or the Conv2d stride produces a fractional output. Always assert this in `__init__`. ViT-B uses 16×16 patches on 224×224 images → 196 patches. Resizing to 256×256 gives 256 patches — valid. Resizing to 200×200 gives non-integer patches — invalid.
```

**Position embedding doesn't scale**: Learned 1D positional embeddings are tied to training resolution. Fine-tuning at higher resolution requires interpolating the position embeddings.

**No inductive bias**: A CNN's conv layers encode locality and translation equivariance. ViT must learn these from data. This means ViT typically needs large-scale pre-training or self-supervised objectives (MAE, DINO) to match CNN performance on smaller datasets.

**`[CLS]` expansion**: `self.cls_token` has shape `(1, 1, d_model)`.

```{tip}
Use `.expand(B, -1, -1)` not `.repeat(B, 1, 1)` to broadcast the `[CLS]` token across the batch. `expand` creates a view (zero-copy); `repeat` allocates new memory. For a batch of 256 sequences, `repeat` wastes ~750KB per layer.
```

---

## Swin Transformer (brief)

Swin adds two inductive biases back:
1. **Windowed attention**: attention computed within local windows (not global), reducing complexity from O(N²) to O(N)
2. **Hierarchical representation**: feature maps are downsampled, creating a CNN-like pyramid

Useful when you want transformer expressiveness but CNN-like efficiency on high-resolution images.
