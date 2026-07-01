import torch
from torch import nn, math
from torch.nn import functional as F


class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.scale = d_model**0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(x) * self.scale


class SinusoidalPositionalEncoding(nn.Module):
    pe: torch.Tensor

    def __init__(self, d_model: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)  # max_len x d_model
        div = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000) / d_model))
        pos = torch.arange(max_len).unsqueeze(1).float()  # max_len x 1

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        pe = pe.unsqueeze(0)  # 1 x max_len x d_model

        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class AttentionHead(nn.Module):
    def __init__(self, d_model: int, d_k: int, dropout: float = 0.1):
        super().__init__()

        self.d_k = d_k

        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_k)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,  # (B, T_q, d_model)
        k: torch.Tensor,  # (B, T_k, d_model)
        v: torch.Tensor,  # (B, T_v, d_model)
        mask: torch.Tensor | None = None,
    ):
        Q = self.W_q(q)  # (B, T_q, d_k)
        K = self.W_k(k)  # (B, T_k, d_k)
        V = self.W_v(v)  # (B, T_v, d_k)

        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)  # (B, T_q, T_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        return attn @ V


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.heads = nn.ModuleList(
            [AttentionHead(d_model, self.d_k, dropout) for _ in range(n_heads)]
        )

        self.W_o = nn.Linear(d_model, d_model)

    def forward(
        self,
        q: torch.Tensor,  # (B, T_q, d_model)
        k: torch.Tensor,  # (B, T_k, d_model)
        v: torch.Tensor,  # (B, T_v, d_model)
        mask: torch.Tensor | None = None,
    ):
        head_outputs = [head(q, k, v, mask) for head in self.heads]  # H x (B, T_q, d_k)
        outputs = torch.cat(head_outputs, -1)  # (B, T_q, d_model)

        return self.W_o(outputs)


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        return self.attn(x, x, x, mask)


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor, mask: torch.Tensor | None = None
    ):
        return self.attn(x, context, context, mask)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        out = x + self.dropout(self.ff(self.norm2(x)))
        return out


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        cross_attn=True,
    ):
        super().__init__()
        self.self_attn = SelfAttention(d_model, n_heads, dropout)
        self.cross_attn = (
            CrossAttention(d_model, n_heads, dropout) if cross_attn else None
        )

        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model) if cross_attn else None
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ):
        x = x + self.dropout(self.self_attn(self.norm1(x), tgt_mask))
        if (
            context is not None
            and self.cross_attn is not None
            and self.norm2 is not None
        ):
            x = x + self.dropout(self.cross_attn(self.norm2(x), context, src_mask))
        out = x + self.dropout(self.ff(self.norm3(x)))

        return out


class Encoder(nn.Module):
    def __init__(
        self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        cross_attn=True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model, n_heads, d_ff, dropout, cross_attn)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ):
        for layer in self.layers:
            x = layer(x, context, src_mask, tgt_mask)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 512,
        n_heads: int = 8,
        d_ff: int = 2048,       # 4 * d_model
        n_layers: int = 6,
        max_len: int = 512,
        dropout: float = 0.1,
        mode: str = "encoder-decoder",  # "encoder-only" | "decoder-only" | "encoder-decoder"
        shared_vocab: bool = False,
    ):
        super().__init__()
        self.mode = mode
        self.src_embed = TokenEmbeddings(src_vocab, d_model)
        self.tgt_embed = self.src_embed if shared_vocab else TokenEmbeddings(tgt_vocab, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        self.encoder = (
            Encoder(n_layers, d_model, n_heads, d_ff, dropout)
            if mode != "decoder-only"
            else None
        )
        self.decoder = (
            Decoder(
                n_layers,
                d_model,
                n_heads,
                d_ff,
                dropout,
                cross_attn=(mode == "encoder-decoder"),
            )
            if mode != "encoder-only"
            else None
        )
        self.out = nn.Linear(d_model, tgt_vocab)

    def forward(
        self,
        src: torch.Tensor,  # (B, T_src)
        tgt: torch.Tensor | None = None,  # (B, T_tgt) — not used in encoder-only
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.encoder is not None:
            src = self.pos_enc(self.src_embed(src))  # (B, T_src, d_model)
            context = self.encoder(src, src_mask)  # (B, T_src, d_model)
        else:
            context = None

        if self.decoder is not None:
            assert tgt is not None, "tgt is required for decoder"
            tgt = self.pos_enc(self.tgt_embed(tgt))  # (B, T_tgt, d_model)
            out = self.decoder(tgt, context, src_mask, tgt_mask)
        else:
            out = context  # encoder-only output

        return self.out(out)  # (B, T, tgt_vocab)
