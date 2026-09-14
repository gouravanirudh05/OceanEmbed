"""Satellite embedding engine.

Each encoder maps the stack of surface observations to a dense per-pixel latent
representation - the *satellite embedding* - of shape ``(B, embed_dim, H, W)``.
Everything downstream (profile decoder, auxiliary heads, surface reconstruction)
reads only that embedding, so encoders are interchangeable and can be compared
under identical training conditions.

Available encoders
------------------
``cnn``
    Dilated residual CNN.  Fully convolutional, so it accepts any input size and
    is the cheapest option.  Receptive field ~45 cells (11deg) after four blocks.
``cnn_vit``
    CNN stem for local texture, then global self-attention over 4x4-cell tokens.
    The attention lets a grid point condition on the whole window, which is what
    captures eddy-scale and basin-scale context that a local CNN cannot see.
    This is the default.
``unet``
    Multi-scale encoder-decoder with skip connections; wide context through
    downsampling rather than attention.
``autoencoder``
    Same stem, bottlenecked hard and trained primarily through the surface
    reconstruction head - used for embedding-only (self-supervised) pretraining.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(c: int) -> nn.Module:
    """GroupNorm rather than BatchNorm: batches are small and spatially
    correlated, which makes BatchNorm statistics unreliable here."""
    return nn.GroupNorm(num_groups=min(8, c), num_channels=c)


class ResBlock(nn.Module):
    """Pre-activation residual block with an optional dilation."""

    def __init__(self, c: int, dilation: int = 1, dropout: float = 0.0):
        super().__init__()
        self.n1, self.n2 = _norm(c), _norm(c)
        self.c1 = nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation)
        self.c2 = nn.Conv2d(c, c, 3, padding=dilation, dilation=dilation)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.gelu(self.n1(x)))
        h = self.c2(self.drop(F.gelu(self.n2(h))))
        return x + h


class ConvStem(nn.Module):
    """Lift the input channels to ``width`` and build local context."""

    def __init__(self, in_ch: int, width: int, dropout: float = 0.0):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, width, 3, padding=1)
        # Dilations 1,2,4,8 give an exponentially growing receptive field
        # without downsampling, so the 0.25deg resolution is preserved.
        self.blocks = nn.Sequential(*[ResBlock(width, d, dropout) for d in (1, 2, 4, 8)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.proj(x))


class CNNEncoder(nn.Module):
    def __init__(self, in_ch: int, width: int, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        self.stem = ConvStem(in_ch, width, dropout)
        self.extra = nn.Sequential(*[ResBlock(width, d, dropout) for d in (1, 2)])
        self.out = nn.Conv2d(width, embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.extra(self.stem(x)))


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0, "embed_dim must be divisible by n_heads"
        self.h, self.dk = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        o = F.scaled_dot_product_attention(q, k, v,
                                           dropout_p=self.drop if self.training else 0.0)
        return self.proj(o.transpose(1, 2).reshape(B, N, C))


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class CNNViTEncoder(nn.Module):
    """CNN stem -> token self-attention -> per-pixel embedding.

    The positional embedding is stored for a reference token grid and bilinearly
    resized whenever the input size differs, so a model trained on 32x32 patches
    can also be run on the whole 101x241 basin in one pass.
    """

    def __init__(self, in_ch: int, width: int, embed_dim: int, patch_size: int = 4,
                 depth: int = 4, heads: int = 4, dropout: float = 0.0,
                 ref_tokens: int = 8):
        super().__init__()
        self.p = patch_size
        self.stem = ConvStem(in_ch, width, dropout)
        self.to_tokens = nn.Conv2d(width, embed_dim, patch_size, stride=patch_size)
        self.pos = nn.Parameter(torch.zeros(1, embed_dim, ref_tokens, ref_tokens))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, heads, dropout=dropout)
                                     for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        # Fuse global (attention) and local (stem) information at full resolution.
        self.fuse = nn.Conv2d(embed_dim + width, embed_dim, 3, padding=1)

    def _pos_for(self, h: int, w: int) -> torch.Tensor:
        if self.pos.shape[-2:] == (h, w):
            return self.pos
        return F.interpolate(self.pos, size=(h, w), mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        local = self.stem(x)
        # Pad up to a whole number of tokens, then crop back after upsampling.
        H, W = local.shape[-2:]
        ph, pw = (-H) % self.p, (-W) % self.p
        padded = F.pad(local, (0, pw, 0, ph), mode="replicate") if (ph or pw) else local

        tok = self.to_tokens(padded)
        tok = tok + self._pos_for(*tok.shape[-2:])
        B, C, th, tw = tok.shape
        seq = tok.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            seq = blk(seq)
        seq = self.norm(seq)
        glob = seq.transpose(1, 2).reshape(B, C, th, tw)

        glob = F.interpolate(glob, size=padded.shape[-2:], mode="bilinear", align_corners=False)
        glob = glob[..., :H, :W]
        return self.fuse(torch.cat([glob, local], dim=1))


class UNetEncoder(nn.Module):
    """Three-level U-Net; context comes from downsampling instead of attention."""

    def __init__(self, in_ch: int, width: int, embed_dim: int, dropout: float = 0.0):
        super().__init__()
        w1, w2, w3 = width, width * 2, width * 4
        self.e1 = nn.Sequential(nn.Conv2d(in_ch, w1, 3, padding=1), ResBlock(w1, 1, dropout))
        self.e2 = nn.Sequential(nn.Conv2d(w1, w2, 3, stride=2, padding=1), ResBlock(w2, 1, dropout))
        self.e3 = nn.Sequential(nn.Conv2d(w2, w3, 3, stride=2, padding=1), ResBlock(w3, 1, dropout))
        self.d2 = nn.Sequential(nn.Conv2d(w3 + w2, w2, 3, padding=1), ResBlock(w2, 1, dropout))
        self.d1 = nn.Sequential(nn.Conv2d(w2 + w1, w1, 3, padding=1), ResBlock(w1, 1, dropout))
        self.out = nn.Conv2d(w1, embed_dim, 1)

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.e1(x)
        h2 = self.e2(h1)
        h3 = self.e3(h2)
        u2 = self.d2(torch.cat([self._up(h3, h2), h2], 1))
        u1 = self.d1(torch.cat([self._up(u2, h1), h1], 1))
        return self.out(u1)


class AutoencoderEncoder(nn.Module):
    """Hard-bottlenecked encoder for self-supervised embedding pretraining."""

    def __init__(self, in_ch: int, width: int, embed_dim: int, dropout: float = 0.0,
                 bottleneck: int = 16):
        super().__init__()
        self.stem = ConvStem(in_ch, width, dropout)
        self.down = nn.Conv2d(width, bottleneck, 1)
        self.up = nn.Sequential(nn.Conv2d(bottleneck, width, 3, padding=1), nn.GELU(),
                                nn.Conv2d(width, embed_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(self.stem(x)))


ENCODERS = {"cnn": CNNEncoder, "cnn_vit": CNNViTEncoder,
            "unet": UNetEncoder, "autoencoder": AutoencoderEncoder}


def build_encoder(name: str, in_ch: int, cfg) -> nn.Module:
    if name not in ENCODERS:
        raise KeyError(f"unknown encoder {name!r}; have {sorted(ENCODERS)}")
    kw = dict(in_ch=in_ch, width=cfg.stem_width, embed_dim=cfg.embed_dim, dropout=cfg.dropout)
    if name == "cnn_vit":
        kw.update(patch_size=cfg.patch_size, depth=cfg.depth_blocks, heads=cfg.n_heads)
    return ENCODERS[name](**kw)
