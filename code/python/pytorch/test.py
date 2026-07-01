import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):
    """
    Implement causal self-attention with optional grouped-query attention.

    Input:
        x:    (B, T, D)
        mask: broadcastable to (B, H, T, T)

    Output:
        y:    (B, T, D)

    Tasks:
    1. Fill in the __init__ boilerplate.
    2. Debug the MHA path.
    3. Implement the GQA TODO.

    Notes:
    - B = batch size
    - T = sequence length
    - D = model dimension
    - H = number of query heads
    - Hkv = number of key/value heads
    - d_head = D // H

    Hints:
    - Attention scores should have shape (B, H, T, T)
    - Softmax should be over keys
    - Be careful after transpose before view
    - In GQA, q keeps H heads, but k/v use fewer heads and are shared across groups
    """

    def __init__(self, d_model, num_heads, use_gqa=False, num_kv_heads=None):
        super().__init__()

        # TODO:
        # 1. Validate that d_model is divisible by num_heads.
        # 2. Store d_model, num_heads, use_gqa.
        # 3. Set d_head.
        # 4. If use_gqa is False:
        #       num_kv_heads should equal num_heads.
        # 5. If use_gqa is True:
        #       num_kv_heads must be provided.
        #       num_heads must be divisible by num_kv_heads.
        #
        # 6. Create projection layers with bias=False:
        #       q_proj:   d_model -> d_model
        #       k_proj:   d_model -> num_kv_heads * d_head
        #       v_proj:   d_model -> num_kv_heads * d_head
        #       out_proj: d_model -> d_model

    def forward(self, x, mask=None):
        """
        Notes:
            - B = batch size
            - T = sequence length
            - D = model dimension
            - H = number of query heads
            - Hkv = number of key/value heads
            - d_head = D // H
        """
        B, T, D = x.shape

        H = self.num_heads
        Hkv = self.num_kv_heads
        d_head = self.d_head

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(B, T, H, d_head)
        k = k.view(B, T, Hkv, d_head)
        v = v.view(B, T, Hkv, d_head)

        if self.use_gqa:
            # TODO:
            # Expand k and v along the head dimension so they match q.
            #
            # Example:
            #   H = 8, Hkv = 2
            #   Each KV head should be shared by 4 query heads.
            pass

        # TODO:
        # Debug the MHA / shared attention path below.

        scores = q @ k.transpose(-1, -2)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        probs = torch.softmax(scores, dim=-2)

        out = probs @ v

        out = out.transpose(1, 2).view(B, T, D)

        y = self.out_proj(out)
        return y


def make_causal_mask(B, T, device):
    mask = torch.tril(torch.ones(T, T, device=device)).bool()
    return mask.unsqueeze(0).unsqueeze(0).expand(B, 1, T, T)


def test_constructor_shapes_mha():
    D, H = 8, 2

    attn = CausalSelfAttention(
        d_model=D,
        num_heads=H,
        use_gqa=False,
    )

    assert attn.d_model == D
    assert attn.num_heads == H
    assert attn.num_kv_heads == H
    assert attn.d_head == D // H

    assert attn.q_proj.weight.shape == (D, D)
    assert attn.k_proj.weight.shape == (D, D)
    assert attn.v_proj.weight.shape == (D, D)
    assert attn.out_proj.weight.shape == (D, D)


def test_constructor_shapes_gqa():
    D, H, Hkv = 8, 4, 2
    d_head = D // H

    attn = CausalSelfAttention(
        d_model=D,
        num_heads=H,
        use_gqa=True,
        num_kv_heads=Hkv,
    )

    assert attn.d_model == D
    assert attn.num_heads == H
    assert attn.num_kv_heads == Hkv
    assert attn.d_head == d_head

    assert attn.q_proj.weight.shape == (D, D)
    assert attn.k_proj.weight.shape == (Hkv * d_head, D)
    assert attn.v_proj.weight.shape == (Hkv * d_head, D)
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

    x = torch.randn(B, T, D)
    mask = make_causal_mask(B, T, x.device)

    attn = CausalSelfAttention(
        d_model=D,
        num_heads=H,
        use_gqa=False,
    )

    y = attn(x, mask)

    assert y.shape == (B, T, D)
    assert torch.isfinite(y).all()
    print("=====================")
    print("congrats on debuging :) now onward towards better inference")
    print("=====================")


def test_gqa_smoke():
    torch.manual_seed(0)

    B, T, D = 2, 4, 8
    H = 4
    Hkv = 2

    x = torch.randn(B, T, D)
    mask = make_causal_mask(B, T, x.device)

    attn = CausalSelfAttention(
        d_model=D,
        num_heads=H,
        use_gqa=True,
        num_kv_heads=Hkv,
    )

    y = attn(x, mask)

    assert y.shape == (B, T, D)
    assert torch.isfinite(y).all()
    print("yay, good work on completing this!")


if __name__ == "__main__":
    test_constructor_shapes_mha()
    test_constructor_shapes_gqa()
    test_invalid_configs_raise()
    test_mha_smoke()
    test_gqa_smoke()

    print("Visible tests passed")
