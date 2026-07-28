"""Backbone AST (Gong et al., 2021) y pirámide multiescala al estilo ViTDet (Li et al., 2022)."""

import torch
from torch import Tensor, nn


class PatchEmbed(nn.Module):
    def __init__(self, in_channels: int = 1, embed_dim: int = 256, patch_size: int = 16) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.projection(x).flatten(2).transpose(1, 2)


class ASTBackbone(nn.Module):
    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 6,
        num_heads: int = 8,
        patch_size: int = 16,
        max_tokens: int = 4096,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(1, embed_dim, patch_size)

        self.pos_embed = nn.Parameter(torch.zeros(1, max_tokens, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        height = x.shape[-2] // self.patch_size
        width = x.shape[-1] // self.patch_size

        tokens = self.patch_embed(x)
        tokens = tokens + self.pos_embed[:, : tokens.shape[1]]
        tokens = self.norm(self.encoder(tokens))

        return tokens.transpose(1, 2).unflatten(-1, (height, width))


class MultiScalePyramid(nn.Module):
    def __init__(self, dim: int = 256, num_groups: int = 8) -> None:
        super().__init__()
        self.upsample_4x = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
            nn.GroupNorm(num_groups, dim),
            nn.GELU(),
            nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2),
        )
        self.upsample_2x = nn.ConvTranspose2d(dim, dim, kernel_size=2, stride=2)
        self.identity = nn.Identity()
        self.downsample_2x = nn.Conv2d(dim, dim, kernel_size=2, stride=2)

    def forward(self, features: Tensor) -> list[Tensor]:
        return [
            self.upsample_4x(features),
            self.upsample_2x(features),
            self.identity(features),
            self.downsample_2x(features),
        ]
