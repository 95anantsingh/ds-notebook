import math
import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):

    def __init__(self, d_model, num_heads, use_gqa=False, num_kv_heads=None):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.use_gqa = use_gqa
        self.d_head = d_model // num_heads

        if not use_gqa:
            self.num_kv_heads = num_heads
        else:
            assert num_kv_heads is not None, "num_kv_heads must be provided when use_gqa=True"
            assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
            self.num_kv_heads = num_kv_heads

        self.q_proj   = nn.Linear(d_model, d_model, bias=False)
        self.k_proj   = nn.Linear(d_model, self.num_kv_heads * self.d_head, bias=False)
        self.v_proj   = nn.Linear(d_model, self.num_kv_heads * self.d_head, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, D = x.shape
        H    = self.num_heads
        Hkv  = self.num_kv_heads
        dh   = self.d_head

        # Project and reshape to (B, H, T, d_head)
        q = self.q_proj(x).view(B, T, H,   dh).transpose(1, 2)   # (B, H,   T, dh)
        k = self.k_proj(x).view(B, T, Hkv, dh).transpose(1, 2)   # (B, Hkv, T, dh)
        v = self.v_proj(x).view(B, T, Hkv, dh).transpose(1, 2)   # (B, Hkv, T, dh)

        if self.use_gqa:
            # Repeat each KV head for its group of query heads
            groups = H // Hkv
            k = k.repeat_interleave(groups, dim=1)   # (B, H, T, dh)
            v = v.repeat_interleave(groups, dim=1)   # (B, H, T, dh)

        # Scaled dot-product attention
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dh)   # (B, H, T, T)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        probs = torch.softmax(scores, dim=-1)   # softmax over keys

        out = probs @ v   # (B, H, T, dh)

        out = out.transpose(1, 2).contiguous().view(B, T, D)

        return self.out_proj(out)


def make_causal_mask(B, T, device):
    mask = torch.tril(torch.ones(T, T, device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(0).expand(B, 1, T, T)


def test_constructor_shapes_mha():
    D, H = 8, 2
    attn = CausalSelfAttention(d_model=D, num_heads=H, use_gqa=False)
    assert attn.d_model == D
    assert attn.num_heads == H
    assert attn.num_kv_heads == H
    assert attn.d_head == D // H
    assert attn.q_proj.weight.shape   == (D, D)
    assert attn.k_proj.weight.shape   == (D, D)
    assert attn.v_proj.weight.shape   == (D, D)
    assert attn.out_proj.weight.shape == (D, D)


def test_constructor_shapes_gqa():
    D, H, Hkv = 8, 4, 2
    d_head = D // H
    attn = CausalSelfAttention(d_model=D, num_heads=H, use_gqa=True, num_kv_heads=Hkv)
    assert attn.d_model == D
    assert attn.num_heads == H
    assert attn.num_kv_heads == Hkv
    assert attn.d_head == d_head
    assert attn.q_proj.weight.shape   == (D, D)
    assert attn.k_proj.weight.shape   == (Hkv * d_head, D)
    assert attn.v_proj.weight.shape   == (Hkv * d_head, D)
    assert attn.out_proj.weight.shape == (D, D)


def test_invalid_configs_raise():
    invalid_configs = [
        dict(d_model=10, num_heads=3),
        dict(d_model=8, num_heads=4, use_gqa=True, num_kv_heads=None),
        dict(d_model=8, num_heads=4, use_gqa=True, num_kv_heads=3),
    ]
    for kwargs in invalid_configs:
        try:
            CausalSelfAttention(**kwargs)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"Expected config to fail: {kwargs}")


def test_mha_smoke():
    torch.manual_seed(0)
    B, T, D, H = 2, 4, 8, 2
    x    = torch.randn(B, T, D)
    mask = make_causal_mask(B, T, x.device)
    attn = CausalSelfAttention(d_model=D, num_heads=H, use_gqa=False)
    y    = attn(x, mask)
    assert y.shape == (B, T, D)
    assert torch.isfinite(y).all()
    print("=====================")
    print("congrats on debuging :) now onward towards better inference")
    print("=====================")


def test_gqa_smoke():
    torch.manual_seed(0)
    B, T, D, H, Hkv = 2, 4, 8, 4, 2
    x    = torch.randn(B, T, D)
    mask = make_causal_mask(B, T, x.device)
    attn = CausalSelfAttention(d_model=D, num_heads=H, use_gqa=True, num_kv_heads=Hkv)
    y    = attn(x, mask)
    assert y.shape == (B, T, D)
    assert torch.isfinite(y).all()
    print("yay, good work on completing this!")


if __name__ == "__main__":
    test_constructor_shapes_mha()
    test_constructor_shapes_gqa()
    test_invalid_configs_raise()
    test_mha_smoke()
    test_gqa_smoke()
    print("All tests passed")
