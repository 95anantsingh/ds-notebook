import torch
from torch import nn


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
        mode: str = "encoder-decoder",  # "encoder-only" | "decoder-only" | "encoder-decoder"
        shared_vocab: bool = False,
    ):
        super().__init__()
        self.mode = mode

        self.src_embed = nn.Embedding(src_vocab, d_model)
        self.tgt_embed = self.src_embed if shared_vocab else nn.Embedding(tgt_vocab, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)

        if mode == "encoder-only":
            encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model))
            self.decoder = None
        elif mode == "decoder-only":
            decoder_layer = nn.TransformerDecoderLayer(d_model, n_heads, d_ff, dropout, batch_first=True)
            self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers, norm=nn.LayerNorm(d_model))
            self.encoder = None
        else:  # encoder-decoder
            self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=n_heads,
                num_encoder_layers=n_layers,
                num_decoder_layers=n_layers,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = None
            self.decoder = None

        self.out = nn.Linear(d_model, tgt_vocab)

    def _embed(self, x: torch.Tensor, embed: nn.Embedding) -> torch.Tensor:
        T = x.size(1)
        pos = torch.arange(T, device=x.device)
        return embed(x) + self.pos_embed(pos)

    def forward(
        self,
        src: torch.Tensor,                          # (B, T_src)
        tgt: torch.Tensor | None = None,            # (B, T_tgt) — decoder modes only
        src_key_padding_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.mode == "encoder-only":
            assert self.encoder is not None
            x = self._embed(src, self.src_embed)
            out = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        elif self.mode == "decoder-only":
            assert self.decoder is not None
            assert tgt is not None, "tgt is required for decoder-only"
            x = self._embed(tgt, self.tgt_embed)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(x.size(1), device=x.device)
            out = self.decoder(x, x, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        else:  # encoder-decoder
            assert tgt is not None, "tgt is required for encoder-decoder"
            src = self._embed(src, self.src_embed)
            tgt = self._embed(tgt, self.tgt_embed)
            if tgt_mask is None:
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1), device=tgt.device)
            out = self.transformer(
                src, tgt,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )

        return self.out(out)                        # (B, T, tgt_vocab)
