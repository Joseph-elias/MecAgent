from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(slots=True)
class ModelConfig:
    vocab_size: int
    max_length: int = 512
    image_size: int = 224
    patch_size: int = 16
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    ff_mult: int = 4
    dropout: float = 0.1


class PatchImageEncoder(nn.Module):
    """Lightweight ViT-style encoder implemented with plain PyTorch blocks."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if config.image_size % config.patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=config.d_model,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        n_patches = (config.image_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, config.d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * config.ff_mult,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, : x.shape[1], :]
        x = self.encoder(x)
        return self.norm(x)


class CadQuerySOTAModel(nn.Module):
    """Image-conditioned Transformer decoder for CadQuery token generation."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = PatchImageEncoder(config)

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.token_positional = nn.Parameter(torch.zeros(1, config.max_length, config.d_model))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * config.ff_mult,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.n_layers)
        self.output = nn.Linear(config.d_model, config.vocab_size)

    def _causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((size, size), float("-inf"), device=device), diagonal=1)

    def forward(self, images: torch.Tensor, tokens_in: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(images)
        token_emb = self.token_embedding(tokens_in)
        token_emb = token_emb + self.token_positional[:, : token_emb.size(1), :]

        mask = self._causal_mask(token_emb.size(1), token_emb.device)
        decoded = self.decoder(tgt=token_emb, memory=memory, tgt_mask=mask)
        return self.output(decoded)
