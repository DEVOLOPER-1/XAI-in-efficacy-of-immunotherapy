"""
src/models/shimada_inception_swi.py — Shimada et al. (2021) InceptionV3 WSI predictor.

Implements the two-stage approach:
1. Tile-level feature extraction via pretrained InceptionV3
2. Slide-level aggregation via masked mean-pooling over valid (non-blank) tiles

Compatible with:
- image_only modality (tabular=None)
- regression or classification task_mode
- frozen or fine-tuned backbone

Key design choices:
- Robust valid_mask: detects zero-padded patches via norm threshold
- Numerical stability: eps in pooling, output clamping, grad clipping support
- Memory efficiency: processes tiles in chunks if needed (future extension)
"""
from __future__ import annotations
from typing import Any
import torch
import torch.nn as nn
import torchvision.models as models
from src.config import DotDict
from src.models.image import _TorchBase


class ShimadaInceptionWSI(_TorchBase):
    """
    InceptionV3-based WSI predictor for TMB prediction.
    
    Input: image tensor of shape (B, N, C, H, W) where:
        B = batch size, N = num patches (≤ max_patches), C=3, H=W=image_size
    Output: slide-level prediction of shape (B,) for regression or (B,1) for classification.
    
    Handles:
    - Variable patch counts per slide (via zero-padding + masking)
    - Frozen or trainable InceptionV3 backbone
    - Regression (continuous TMB) or classification (TMB-H vs TMB-L)
    """
    _category_label: str = "image_only"

    def _build_estimator(self, cfg: "DotDict") -> Any:
        model_cfg = cfg.get("model") or DotDict({})
        freeze_backbone = bool(model_cfg.get("freeze_backbone", True))
        image_size = cfg.get("dataset", {}).get("image_size", 299)

        class InnerInception(nn.Module):
            def __init__(
                self,
                dropout_rate: float = 0.5,
                mode: str = "regression",
                freeze: bool = True,
                tile_chunk_size: int = 32,
            ):
                super().__init__()
                self.freeze_backbone = freeze
                self.mode = mode
                self.tile_chunk_size = max(1, int(tile_chunk_size))

                # ── InceptionV3 backbone ─────────────────────────────────────
                # torchvision requires aux_logits=True when loading pretrained weights
                self.cnn = models.inception_v3(
                    weights=models.Inception_V3_Weights.DEFAULT,
                    aux_logits=True,
                )
                # Disable auxiliary classifier for clean single-tensor output
                self.cnn.aux_logits = False
                if hasattr(self.cnn, "AuxLogits"):
                    self.cnn.AuxLogits = None
                # Replace final FC with identity to get 2048-d features
                self.cnn.fc = nn.Identity()

                # Freeze backbone if requested
                if freeze:
                    for param in self.cnn.parameters():
                        param.requires_grad = False
                    self.cnn.eval()  # Important: BN/Dropout behave differently in eval

                # ── Task-specific head ────────────────────────────────────────
                in_dim = 2048  # InceptionV3 feature dimension
                if mode == "classification":
                    self.head = nn.Sequential(
                        nn.Dropout(dropout_rate),
                        nn.Linear(in_dim, 256),
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout_rate),
                        nn.Linear(256, 1),
                        nn.Sigmoid(),  # Output in [0, 1] for binary classification
                    )
                else:  # regression
                    self.head = nn.Sequential(
                        nn.Dropout(dropout_rate),
                        nn.Linear(in_dim, 512),
                        nn.ReLU(inplace=True),
                        nn.Dropout(dropout_rate),
                        nn.Linear(512, 1),
                        # No activation: output is unbounded for regression
                    )

            def train(self, mode: bool = True):
                super().train(mode)
                if self.freeze_backbone:
                    self.cnn.eval()
                return self

            def _encode_tiles(self, tiles: torch.Tensor) -> torch.Tensor:
                """Encode tiles in smaller chunks to avoid GPU OOM on large bags."""
                if tiles.numel() == 0:
                    return tiles.new_zeros((0, 2048))

                features: list[torch.Tensor] = []
                for start in range(0, tiles.shape[0], self.tile_chunk_size):
                    chunk = tiles[start : start + self.tile_chunk_size]
                    if self.freeze_backbone:
                        with torch.no_grad():
                            feats = self.cnn(chunk)
                    else:
                        feats = self.cnn(chunk)
                    features.append(feats)

                return torch.cat(features, dim=0)

            def _compute_valid_mask(self, patches: torch.Tensor) -> torch.Tensor:
                """
                Detect which patches contain actual tissue vs zero-padding.
                
                Strategy: After ImageNet normalization, a truly blank patch
                has pixel values very close to the normalization baseline.
                We compute the L2 norm per patch and threshold it.
                
                Args:
                    patches: (B, N, C, H, W) normalized image tensor
                Returns:
                    valid_mask: (B, N) float tensor with 1.0 for valid patches
                """
                B, N, C, H, W = patches.shape
                # Reshape to (B*N, C, H, W) for per-patch computation
                patches_flat = patches.view(B * N, C, H, W)
                
                # Compute L2 norm per patch (across C, H, W dimensions)
                patch_norms = patches_flat.norm(p=2, dim=[1, 2, 3])  # (B*N,)
                
                # Threshold: patches with norm < 1e-2 are considered blank
                # (adjust if your padding strategy uses non-zero values)
                valid = (patch_norms > 1e-2).float()
                
                return valid.view(B, N)  # (B, N)

            def forward(self, image: torch.Tensor, tabular: torch.Tensor | None = None):
                """
                Forward pass for slide-level TMB prediction.
                
                Args:
                    image: (B, N, C, H, W) tensor of normalized patches
                    tabular: ignored for image_only models (kept for protocol compatibility)
                Returns:
                    predictions: (B,) tensor of slide-level predictions
                """
                if image is None:
                    raise ValueError("image input is required for image_only model")
                
                B, N, C, H, W = image.shape
                
                # ── Tile-level feature extraction ───────────────────────────
                # Reshape to (B*N, C, H, W) for batched CNN inference
                tiles = image.view(B * N, C, H, W)

                tile_features = self._encode_tiles(tiles)  # (B*N, 2048)

                # Reshape back to (B, N, 2048)
                tile_features = tile_features.view(B, N, -1)
                
                # ── Mask invalid (zero-padded) patches ──────────────────────
                valid_mask = self._compute_valid_mask(image)  # (B, N)
                
                # Apply mask: zero out features of invalid patches
                tile_features = tile_features * valid_mask.unsqueeze(-1)  # (B, N, 2048)
                
                # ── Masked mean pooling ─────────────────────────────────────
                # Sum features over valid patches, then divide by count
                n_valid = valid_mask.sum(dim=1, keepdim=True)  # (B, 1)
                # Prevent division by zero: clamp to at least 1
                n_valid = n_valid.clamp(min=1.0)
                
                slide_emb = tile_features.sum(dim=1) / n_valid  # (B, 2048)
                
                # ── Task head ───────────────────────────────────────────────
                logits = self.head(slide_emb).squeeze(-1)  # (B,)
                
                # Optional: clamp outputs for numerical stability
                if self.mode == "regression":
                    # Clamp to reasonable TMB range [0, 500] to prevent explosion
                    logits = logits.clamp(min=0.0, max=500.0)
                # For classification, sigmoid already bounds to [0, 1]
                
                return logits

        # ── Instantiate inner model ─────────────────────────────────────────
        mode = model_cfg.get("task_mode", "regression")
        dropout = float(model_cfg.get("dropout", 0.5))
        freeze = bool(model_cfg.get("freeze_backbone", True))
        tile_chunk_size = int(model_cfg.get("tile_chunk_size", 32))

        return InnerInception(
            dropout_rate=dropout,
            mode=mode,
            freeze=freeze,
            tile_chunk_size=tile_chunk_size,
        )
