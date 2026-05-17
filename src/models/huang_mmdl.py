"""
src/models/huang_mmdl.py — Implementation of Huang et al. (2022) MMDL model.

Reference:
Huang et al., "Predicting colorectal cancer tumor mutational burden from
histopathological images and clinical information using multi-modal deep learning",
Bioinformatics, 2022. https://doi.org/10.1093/bioinformatics/btac641

Key modifications for this pipeline:
- Output changed from binary classification → continuous regression (TMB value)
- Integrated feature selection for clinical/tabular inputs
- Compatible with existing data_loader.py and train.py modality handling
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from src.config import DotDict
from src.models.image import _TorchBase


# ─────────────────────────────────────────────────────────────────────────
# 1. Compact Bilinear Pooling (MCB) — Faithful reproduction of paper Eq. 4-5
# ─────────────────────────────────────────────────────────────────────────
class CompactBilinearPooling(nn.Module):
    """
    Multi-Modal Compact Bilinear (MCB) pooling.

    Approximates the outer product of two vectors in a compact low-dimensional space
    using Count Sketch projection + FFT-based convolution.

    Reference: Fukui et al., "Multimodal Compact Bilinear Pooling for Visual Question
    Answering and Visual Grounding", EMNLP 2016.

    Args:
        input_dim1: Dimension of first input vector (image features)
        input_dim2: Dimension of second input vector (clinical features)
        output_dim: Output dimension after compact bilinear pooling (paper uses 8000)
    """

    def __init__(self, input_dim1: int, input_dim2: int, output_dim: int = 8000):
        super().__init__()
        self.output_dim = output_dim

        # Count Sketch projection matrices (fixed, not learned)
        # h: random hash functions mapping indices to [0, output_dim)
        # s: random sign functions mapping to {+1, -1}
        self.register_buffer("h1", torch.randint(0, output_dim, (input_dim1,)))
        self.register_buffer("s1", torch.randint(0, 2, (input_dim1,)) * 2 - 1)
        self.register_buffer("h2", torch.randint(0, output_dim, (input_dim2,)))
        self.register_buffer("s2", torch.randint(0, 2, (input_dim2,)) * 2 - 1)

        # Post-pooling projection + normalization (learnable)
        self.proj = nn.Linear(output_dim, output_dim)
        self.sqrt_norm = nn.Parameter(torch.tensor(1.0 / (output_dim**0.5)))

    def _count_sketch(
        self, x: torch.Tensor, h: torch.Tensor, s: torch.Tensor
    ) -> torch.Tensor:
        """Apply Count Sketch projection to input vector."""
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        B = x.shape[0]
        # Initialize sketch output
        sketch = torch.zeros(B, self.output_dim, device=x.device, dtype=x.dtype)
        # Scatter-add with signed hashing
        sketch.scatter_add_(
            1, h.unsqueeze(0).expand(B, -1), x * s.unsqueeze(0).expand(B, -1)
        )
        return sketch

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1: (B, input_dim1) — image features
            x2: (B, input_dim2) — clinical features
        Returns:
            (B, output_dim) — compact bilinear pooled representation
        """
        # Apply Count Sketch to each modality
        sketch1 = self._count_sketch(x1, self.h1, self.s1)  # (B, output_dim)
        sketch2 = self._count_sketch(x2, self.h2, self.s2)  # (B, output_dim)

        # FFT-based convolution (equivalent to outer product in compact space)
        # Always upcast to float32 here: cuFFT half-precision only supports
        # power-of-two signal sizes, and this layer uses a fixed 8000-d sketch.
        fft1 = torch.fft.rfft(sketch1.float(), dim=1)
        fft2 = torch.fft.rfft(sketch2.float(), dim=1)

        # Complex multiplication in frequency domain
        fft_out = fft1 * fft2

        # Inverse FFT to get compact bilinear result
        pooled = torch.fft.irfft(fft_out, n=self.output_dim, dim=1)

        # Signed square root normalization + L2 normalization (paper Eq. 5)
        pooled = torch.sign(pooled) * torch.sqrt(torch.abs(pooled) + 1e-6)
        pooled = F.normalize(pooled, p=2, dim=1)

        # Learnable projection
        return self.proj(pooled)


# ─────────────────────────────────────────────────────────────────────────
# 2. Feature Selection Module for Clinical Features
# ─────────────────────────────────────────────────────────────────────────
class ClinicalFeatureSelector(nn.Module):
    """
    Learnable feature selection for clinical/tabular inputs.

    Implements soft feature masking with sparsity regularization.
    Features with low importance scores are down-weighted during training.

    Args:
        input_dim: Number of clinical features
        k: Target number of features to select (soft constraint via regularization)
    """

    def __init__(self, input_dim: int | None = None, k: int | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.k = k  # Target sparsity (None = no constraint)

        # Learnable importance scores (one per feature). If input_dim is not
        # known yet, initialize lazily from the first batch seen at runtime.
        if input_dim is None:
            self.register_parameter("importance", None)
        else:
            self.importance = nn.Parameter(torch.randn(input_dim) * 0.01)

        # Temperature for soft masking (annealed during training)
        self.register_buffer("temperature", torch.tensor(1.0))

    def _ensure_initialized(
        self, input_dim: int, device: torch.device, dtype: torch.dtype
    ) -> None:
        if self.importance is None:
            self.input_dim = input_dim
            self.importance = nn.Parameter(
                torch.randn(input_dim, device=device, dtype=dtype) * 0.01
            )
        elif self.importance.numel() != input_dim:
            raise ValueError(
                f"ClinicalFeatureSelector expected {self.importance.numel()} features, "
                f"but received {input_dim}. Check cfg.dataset/tabular preprocessing."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) clinical feature tensor
        Returns:
            (B, input_dim) masked features
        """
        self._ensure_initialized(x.size(-1), x.device, x.dtype)

        # Compute soft mask via sigmoid with temperature
        mask = torch.sigmoid(self.importance / self.temperature)  # (input_dim,)

        # Apply mask to features
        return x * mask.unsqueeze(0)  # (B, input_dim)

    def get_selected_features(self, threshold: float = 0.5) -> list[int]:
        """Return indices of features with importance > threshold."""
        if self.importance is None:
            return []
        mask = torch.sigmoid(self.importance)
        return (mask > threshold).nonzero(as_tuple=True)[0].tolist()

    def sparsity_loss(self) -> torch.Tensor:
        """L1 regularization to encourage feature selection sparsity."""
        if self.importance is None:
            return torch.tensor(0.0, device=self.temperature.device)
        if self.k is None:
            return torch.tensor(0.0, device=self.importance.device)
        # Encourage exactly k features to be selected
        mask = torch.sigmoid(self.importance)
        return torch.abs(mask.sum() - self.k)


# ─────────────────────────────────────────────────────────────────────────
# 3. Huang MMDL Model — Multi-Modal Regression for Continuous TMB
# ─────────────────────────────────────────────────────────────────────────
class HuangMMDL(_TorchBase):
    """
    Multi-modal deep learning model for continuous TMB prediction.

    Architecture:
    1. Image branch: ResNet18 backbone → 512-d features
    2. Clinical branch: Feature selector → MLP projection → 128-d features
    3. Fusion: Compact Bilinear Pooling (MCB) → 8000-d → projection
    4. Regression head: MLP → continuous TMB prediction

    Modified from Huang et al. (2022):
    - Output: regression (continuous TMB) instead of binary classification
    - Loss: HuberLoss instead of cross-entropy
    - Added: learnable clinical feature selection
    """

    _category_label: str = "fusion"  # Requires both image and tabular inputs

    def _build_estimator(self, cfg: DotDict) -> Any:
        model_cfg = cfg.get("model") or DotDict({})

        # ── Hyperparameters from paper + pipeline defaults ─────────────
        freeze_backbone = bool(model_cfg.get("freeze_backbone", True))
        dropout_rate = float(model_cfg.get("dropout", 0.2))
        mcb_output_dim = int(model_cfg.get("mcb_output_dim", 8000))
        clinical_embed_dim = int(model_cfg.get("clinical_embed_dim", 128))
        fusion_embed_dim = int(model_cfg.get("fusion_embed_dim", 512))

        # Feature selection config
        select_features = bool(model_cfg.get("select_clinical_features", True))
        target_k_features = model_cfg.get("target_k_features", None)  # int or None
        sparsity_weight = float(model_cfg.get("feature_selection_weight", 0.01))

        # ── Image branch: ResNet18 backbone ─────────────────────────────
        class ImageEncoder(nn.Module):
            def __init__(self, freeze: bool = True):
                super().__init__()
                self.freeze = freeze
                # Pretrained ResNet18
                self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                # Remove final FC, keep avgpool output (512-d)
                self.backbone.fc = nn.Identity()

                if freeze:
                    for param in self.backbone.parameters():
                        param.requires_grad = False
                    self.backbone.eval()

            def forward(self, tiles: torch.Tensor) -> torch.Tensor:
                """
                Args:
                    tiles: (B*N, 3, H, W) batch of image tiles
                Returns:
                    (B*N, 512) ResNet18 features
                """
                if self.freeze:
                    with torch.no_grad():
                        return self.backbone(tiles)
                return self.backbone(tiles)

        # ── Clinical branch: Feature selection + MLP projection ─────────
        class ClinicalEncoder(nn.Module):
            def __init__(
                self,
                input_dim: int | None,
                embed_dim: int,
                select: bool = True,
                k: int | None = None,
            ):
                super().__init__()
                self.selector = (
                    ClinicalFeatureSelector(input_dim, k) if select else None
                )
                self.project = nn.Sequential(
                    nn.LazyLinear(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate),
                    nn.Linear(256, embed_dim),
                )
                self.select = select

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) #imputing null values by zeros
                if self.select and self.selector is not None:
                    x = self.selector(x)
                return self.project(x)

            def get_selection_loss(self) -> torch.Tensor:
                if self.select and self.selector is not None:
                    return self.selector.sparsity_loss()
                return torch.tensor(0.0, device=self.project[0].weight.device)

        # ── Full MMDL model ─────────────────────────────────────────────
        class InnerMMDL(nn.Module):
            def __init__(self, clinical_input_dim: int | None):
                super().__init__()

                # Branches
                self.image_encoder = ImageEncoder(freeze=freeze_backbone)
                self.clinical_encoder = ClinicalEncoder(
                    clinical_input_dim,
                    clinical_embed_dim,
                    select=select_features,
                    k=target_k_features,
                )

                # Fusion
                self.mcb = CompactBilinearPooling(
                    input_dim1=512,  # ResNet18 output
                    input_dim2=clinical_embed_dim,
                    output_dim=mcb_output_dim,
                )
                self.fusion_proj = nn.Sequential(
                    nn.Linear(mcb_output_dim, fusion_embed_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate),
                )

                # Regression head (MODIFIED: continuous output)
                self.regression_head = nn.Sequential(
                    nn.Linear(fusion_embed_dim, 256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(dropout_rate),
                    nn.Linear(256, 1),  # Single continuous output
                    # No activation: output is unbounded for regression
                )

                # Store for feature selection reporting
                self._clinical_input_dim = clinical_input_dim
                self._sparsity_weight = sparsity_weight

            def forward(self, image: torch.Tensor, tabular: torch.Tensor | None = None):
                """
                Args:
                    image: (B, N, 3, H, W) batch of tile batches
                    tabular: (B, F) clinical features (required for fusion)
                Returns:
                    (B,) continuous TMB predictions
                """
                if image is None or tabular is None:
                    raise ValueError("HuangMMDL requires both image and tabular inputs")

                B, N, C, H, W = image.shape

                # ── Image branch: process tiles → slide-level features ───
                # Reshape tiles: (B, N, C, H, W) → (B*N, C, H, W)
                tiles = image.view(B * N, C, H, W)
                tile_features = self.image_encoder(tiles)  # (B*N, 512)

                # Aggregate tiles via mean pooling (paper uses simple averaging)
                slide_image_features = tile_features.view(B, N, -1).mean(
                    dim=1
                )  # (B, 512)

                # ── Clinical branch: feature selection + projection ─────
                clinical_features = self.clinical_encoder(tabular)  # (B, embed_dim)

                # ── Fusion: Compact Bilinear Pooling ────────────────────
                fused = self.mcb(slide_image_features, clinical_features)  # (B, 8000)
                fused = self.fusion_proj(fused)  # (B, fusion_embed_dim)

                # ── Regression head ─────────────────────────────────────
                return self.regression_head(fused).squeeze(-1)  # (B,)

            def get_feature_selection_report(
                self,
                feature_names: list[str] | None = None,
            ) -> dict:
                """Return selected clinical features and importance scores."""
                if (
                    not hasattr(self.clinical_encoder, "selector")
                    or self.clinical_encoder.selector is None
                ):
                    return {
                        "selected_indices": list(range(self._clinical_input_dim))
                        if self._clinical_input_dim is not None
                        else [],
                        "importance_scores": [],
                        "n_selected": 0,
                    }

                selector = self.clinical_encoder.selector
                if selector.importance is None:
                    return {
                        "selected_indices": [],
                        "importance_scores": [],
                        "n_selected": 0,
                    }
                mask = torch.sigmoid(selector.importance).detach().cpu().numpy()
                selected = (mask > 0.5).nonzero()[0].tolist()

                report = {
                    "selected_indices": selected,
                    "importance_scores": mask.tolist(),
                    "n_selected": len(selected),
                }
                if feature_names:
                    report["selected_names"] = [
                        feature_names[i] for i in selected if i < len(feature_names)
                    ]
                return report

        # ── Instantiate with clinical input dimension from config ───────
        # Note: if clinical_input_dim is omitted, the selector and projection
        # layers infer the actual tabular width from the first batch at runtime.
        return InnerMMDL(
            clinical_input_dim=cfg.get("model", {}).get("clinical_input_dim")
        )

    def get_param_groups(self, head_lr: float, backbone_lr_factor: float = 0.1):
        """
        Return parameter groups for differential learning rates.
        Backbone gets lower LR to preserve pretrained weights.
        """
        if not hasattr(self._est, "image_encoder"):
            # Fallback for non-fusion models
            return filter(lambda p: p.requires_grad, self._est.parameters())

        groups = []

        # Backbone parameters (lower LR)
        backbone_params = [
            p for p in self._est.image_encoder.backbone.parameters() if p.requires_grad
        ]
        if backbone_params:
            groups.append(
                {
                    "params": backbone_params,
                    "lr": head_lr * backbone_lr_factor,
                    "name": "backbone",
                }
            )

        # Head/fusion parameters (full LR)
        head_params = [
            p
            for name, p in self._est.named_parameters()
            if "image_encoder.backbone" not in name and p.requires_grad
        ]
        if head_params:
            groups.append({"params": head_params, "lr": head_lr, "name": "head"})

        return (
            groups
            if groups
            else filter(lambda p: p.requires_grad, self._est.parameters())
        )
