"""
src/models/cellmorphnet.py — Implementation of CellMorphNet (Xu et al., 2024).

Key Components:
1. Cellular Deconvolution: Separates H&E/D stains using fixed optical density matrix.
2. Hierarchical Cellular Routing Attention (HCRA): LOCAL window-based attention —
   NOT global self-attention. Each token attends only within a (window_size×window_size)
   local region. This is the crucial memory-saving mechanism from the paper.
3. 4-Stage Pyramid Backbone: Hierarchical feature extraction with spatial downsampling.
4. Task Heads: Supports both Classification (TMB-H/L) and Regression (TMB count).

Memory budget (11 GB GPU, batch_size=4, max_patches=8, 224×224 tiles):
  - Tiles in one forward: 4 × 8 = 32
  - Stage-1 feature map: (32, 56, 56, 64)  window_size=7 → 64 windows × 49 tokens
  - Attention matrix (per head): 64 × 49 × 49 = 153 k floats  ← manageable
  - vs global: 1 × 3136 × 3136 = 9.8 M floats per tile  ← OOM

Reference:
Xu et al., "Advanced Deep Learning Framework for Cancer Cell Morphological Analysis
and Tumor Mutational Burden Prediction", IEEE JBHI, 2024.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DotDict
from src.models.image import _TorchBase


# ─────────────────────────────────────────────────────────────────────────
# 1. Cellular Deconvolution Module (Eq. 6 in paper)
# ─────────────────────────────────────────────────────────────────────────
class CellularDeconvolution(nn.Module):
    """
    Separates RGB image into stain channels (H, E, D) using Beer-Lambert law.
    Implements the fixed deconvolution matrix Δ from Xu et al. (Eq. 6).
    """
    def __init__(self):
        super().__init__()
        # Fixed matrix from paper (H, E, D coefficients) - shape: (3, 3)
        self.register_buffer('delta', torch.tensor([
            [1.92, -0.05, -0.63],
            [-1.05,  1.16, -0.51],
            [-0.58, -0.11,  1.62]
        ], dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) RGB image in range [0, 1]
        Returns:
            (B, 3, H, W) Deconvolved stain density map
        """
        x_clamped = x.clamp(min=1e-5, max=1.0)
        od = -torch.log(x_clamped)          # Optical Density: (B, 3, H, W)
        od = od.permute(0, 2, 3, 1)         # (B, H, W, 3)
        stains = torch.matmul(od, self.delta.T)  # (B, H, W, 3)
        return stains.permute(0, 3, 1, 2)   # (B, 3, H, W)


# ─────────────────────────────────────────────────────────────────────────
# 2. Stem Patch Embedding (Stage-1 entry)
# ─────────────────────────────────────────────────────────────────────────
class _StemEmbed(nn.Module):
    """
    7×7 strided convolution to produce initial feature map, followed by
    LayerNorm on the channel (last) axis.
    Output layout: (B, H, W, C)  i.e. channels-last, matching window attention.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=7, stride=4, padding=3)
        self.norm = nn.LayerNorm(out_ch, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)              # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)    # (B, H, W, C)
        return self.norm(x)           # (B, H, W, C)


# ─────────────────────────────────────────────────────────────────────────
# 3. Morphological Pooling (Learnable downsampling between stages)
# ─────────────────────────────────────────────────────────────────────────
class MorphologicalPooling(nn.Module):
    """Strided 3×3 conv downsampling + LayerNorm. Input/output: (B, H, W, C)."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 1, 2)   # (B, C, H, W)
        x = self.conv(x)             # (B, out_dim, H/2, W/2)
        x = x.permute(0, 2, 3, 1)   # (B, H/2, W/2, out_dim)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────
# 4. Local Window Attention (HCRA core) — the key memory fix
# ─────────────────────────────────────────────────────────────────────────
def _window_partition(x: torch.Tensor, window_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Partition (B, H, W, C) into non-overlapping windows of shape
    (num_windows*B, window_size, window_size, C).
    Pads H and W if not divisible by window_size.
    """
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))  # pad last two spatial dims
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, nH, nW, ws, ws, C)
    windows = x.view(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def _window_reverse(windows: torch.Tensor, window_size: int,
                    Hp: int, Wp: int, H: int, W: int) -> torch.Tensor:
    """Reverse _window_partition back to (B, H, W, C)."""
    B = int(windows.shape[0] / (Hp * Wp / window_size / window_size))
    x = windows.view(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, Hp, Wp, C)
    x = x.view(B, Hp, Wp, -1)
    # Remove padding
    if Hp != H or Wp != W:
        x = x[:, :H, :W, :].contiguous()
    return x


# ─────────────────────────────────────────────────────────────────────────
# 5. CellMorph Block with Local Window Attention (Paper HCRA)
# ─────────────────────────────────────────────────────────────────────────
class CellMorphBlock(nn.Module):
    """
    Transformer block implementing HCRA-style LOCAL window attention.

    Instead of attending across all H*W spatial positions (which is O(H²W²) memory),
    this partitions the feature map into non-overlapping windows of size
    (window_size × window_size) and runs attention independently within each window.

    Memory: O(window_size^4 × C) per stage — constant w.r.t. image resolution.
    This is the canonical Swin-Transformer approach that CellMorphNet (Xu et al.)
    adopts for their HCRA mechanism.
    """
    def __init__(self, dim: int, num_heads: int, window_size: int = 7,
                 mlp_ratio: float = 4.0, drop: float = 0.1):
        super().__init__()
        self.window_size = window_size
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, batch_first=True, dropout=drop
        )
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C) channels-last feature map
        Returns:
            x: (B, H, W, C)
        """
        if x.dim() == 3:
            # (B, N, C) — shouldn't happen in normal flow but handle gracefully
            raise ValueError("CellMorphBlock expects 4-D (B,H,W,C) input")

        B, H, W, C = x.shape
        shortcut = x

        # ── Window partition ─────────────────────────────────────────────
        x_norm = self.norm1(x)
        windows, (Hp, Wp) = _window_partition(x_norm, self.window_size)
        # windows: (num_wins*B, ws, ws, C) → flatten spatial for attn
        nw = windows.shape[0]
        ws = self.window_size
        windows_flat = windows.view(nw, ws * ws, C)  # (nW, ws², C)

        # ── Local self-attention ─────────────────────────────────────────
        attn_out, _ = self.attn(windows_flat, windows_flat, windows_flat)

        # ── Reverse window partition ─────────────────────────────────────
        attn_out = attn_out.view(nw, ws, ws, C)
        x = _window_reverse(attn_out, self.window_size, Hp, Wp, H, W)  # (B, H, W, C)
        x = shortcut + x

        # ── MLP with pre-norm ─────────────────────────────────────────────
        x = x + self.mlp(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────
# 6. CellMorphNet Backbone (4-Stage Pyramid)
# ─────────────────────────────────────────────────────────────────────────
class CellMorphNet(nn.Module):
    """
    4-Stage Hierarchical Pyramid with Cellular Deconvolution + Window Attention.

    Stage  | Input       | Output      | embed_dim | depth | window
    -------|-------------|-------------|-----------|-------|-------
      1    | 224×224     | 56×56       |    64     |   2   |   7
      2    | 56×56       | 28×28       |   128     |   2   |   7
      3    | 28×28       | 14×14       |   256     |   6   |   7
      4    | 14×14       | 7×7         |   512     |   2   |   7

    Window size 7 matches the paper's design (56/7=8, 28/7=4, 14/7=2 windows per axis).
    """
    def __init__(
        self,
        img_size: int = 224,
        in_chans: int = 3,
        embed_dims: list | None = None,
        depths: list | None = None,
        num_heads: list | None = None,
        window_size: int = 7,
        drop_rate: float = 0.1,
    ):
        super().__init__()
        if embed_dims is None: embed_dims = [64, 128, 256, 512]
        if depths    is None: depths     = [2, 2, 6, 2]
        if num_heads is None: num_heads  = [2, 4, 8, 16]

        self.deconv = CellularDeconvolution()

        # Stage 1
        self.patch_embed1 = _StemEmbed(3, embed_dims[0])
        self.stage1 = nn.ModuleList([
            CellMorphBlock(embed_dims[0], num_heads[0], window_size=window_size, drop=drop_rate)
            for _ in range(depths[0])
        ])

        # Stage 2
        self.patch_embed2 = MorphologicalPooling(embed_dims[0], embed_dims[1])
        self.stage2 = nn.ModuleList([
            CellMorphBlock(embed_dims[1], num_heads[1], window_size=window_size, drop=drop_rate)
            for _ in range(depths[1])
        ])

        # Stage 3
        self.patch_embed3 = MorphologicalPooling(embed_dims[1], embed_dims[2])
        self.stage3 = nn.ModuleList([
            CellMorphBlock(embed_dims[2], num_heads[2], window_size=window_size, drop=drop_rate)
            for _ in range(depths[2])
        ])

        # Stage 4
        self.patch_embed4 = MorphologicalPooling(embed_dims[2], embed_dims[3])
        self.stage4 = nn.ModuleList([
            CellMorphBlock(embed_dims[3], num_heads[3], window_size=window_size, drop=drop_rate)
            for _ in range(depths[3])
        ])

        self.norm = nn.LayerNorm(embed_dims[-1])

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) — RGB tiles, values in [0, 1]
        Returns:
            (B, 512) slide-level feature vector
        """
        x = self.deconv(x)            # (B, 3, H, W)  stain-separated

        # Stage 1: 224→56
        x = self.patch_embed1(x)      # (B, 56, 56, 64)
        for blk in self.stage1:
            x = blk(x)

        # Stage 2: 56→28
        x = self.patch_embed2(x)      # (B, 28, 28, 128)
        for blk in self.stage2:
            x = blk(x)

        # Stage 3: 28→14
        x = self.patch_embed3(x)      # (B, 14, 14, 256)
        for blk in self.stage3:
            x = blk(x)

        # Stage 4: 14→7
        x = self.patch_embed4(x)      # (B, 7, 7, 512)
        for blk in self.stage4:
            x = blk(x)

        # Global Average Pool: (B, 7, 7, 512) → (B, 512)
        x = x.permute(0, 3, 1, 2)               # (B, 512, 7, 7)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B, 512)
        return self.norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_features(x)


# ─────────────────────────────────────────────────────────────────────────
# 7. Pipeline Wrapper: CellMorphNetModel
# ─────────────────────────────────────────────────────────────────────────
class CellMorphNetModel(_TorchBase):
    """
    Pipeline wrapper integrating CellMorphNet with the image_only training loop.

    Input shape: (B, N, C, H, W) — B slides, N tiles each, C=3 channels.
    Output shape: (B,) — one TMB prediction per slide (regression) or
                         probability in [0,1] (classification).

    Tile processing strategy (matches paper's MIL-style approach):
      1. Flatten tiles: (B, N, 3, H, W) → (B*N, 3, H, W)
      2. Pass all tiles through the shared CellMorphNet backbone
      3. Mean-pool tile features: (B, N, 512) → (B, 512)
      4. Task head: (B, 512) → (B,)
    """
    _category_label: str = "image_only"

    def _build_estimator(self, cfg: DotDict) -> Any:
        model_cfg  = cfg.get("model") or {}
        task_mode  = model_cfg.get("task_mode", "regression")
        dropout    = float(model_cfg.get("dropout", 0.1))
        img_size   = cfg.get("dataset", {}).get("image_size", 224)
        window_size = int(model_cfg.get("window_size", 7))

        backbone = CellMorphNet(
            img_size=img_size,
            in_chans=3,
            embed_dims=[64, 128, 256, 512],
            depths=[2, 2, 6, 2],
            num_heads=[2, 4, 8, 16],
            window_size=window_size,
            drop_rate=dropout,
        )

        embed_dim = 512
        if task_mode == "classification":
            head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embed_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
                nn.Sigmoid(),
            )
        else:  # regression
            head = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(embed_dim, 256),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
            )

        class CellMorphNetWrapper(nn.Module):
            def __init__(self, backbone: nn.Module, head: nn.Module):
                super().__init__()
                self.backbone = backbone
                self.head = head

            def forward(self, image: torch.Tensor,
                        tabular: torch.Tensor | None = None) -> torch.Tensor:
                """
                Args:
                    image:   (B, N, C, H, W) — tiles per slide
                    tabular: ignored (image_only model)
                Returns:
                    (B,) predictions
                """
                if image is None:
                    raise ValueError("image input required for image_only model")

                B, N, C, H, W = image.shape

                # Flatten tiles for batch processing
                tiles = image.view(B * N, C, H, W)

                # Clamp to [0,1] as CellularDeconvolution uses log(I)
                tiles = tiles.clamp(0.0, 1.0)

                # Backbone: (B*N, 3, H, W) → (B*N, 512)
                tile_feats = self.backbone(tiles)

                # Reshape and mean-pool across tiles (MIL aggregation)
                tile_feats = tile_feats.view(B, N, -1)    # (B, N, 512)
                slide_emb  = tile_feats.mean(dim=1)        # (B, 512)

                # Task head: (B, 512) → (B,)
                return self.head(slide_emb).squeeze(-1)

        return CellMorphNetWrapper(backbone, head)