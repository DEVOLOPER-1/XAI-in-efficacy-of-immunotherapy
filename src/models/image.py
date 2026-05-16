import torch
import torch.nn as nn
import torchvision.models as models
from src.config import DotDict
from typing import Any


class _TorchBase:
    """Shared scaffold for native PyTorch models to match the _est proxy pattern."""

    def __init__(self, cfg):
        self._cfg = cfg
        self._est = self._build_estimator(cfg)

    def _build_estimator(self, cfg) -> Any:
        raise NotImplementedError

    def parameters(self):
        return self._est.parameters()

    def state_dict(self):
        """Delegate to inner nn.Module for torch checkpoint compatibility."""
        return self._est.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        """Delegate to inner nn.Module for torch checkpoint compatibility."""
        return self._est.load_state_dict(state_dict, strict=strict)

    def train(self, mode: bool = True):
        self._est.train(mode)
        return self

    def eval(self):
        self._est.eval()
        return self

    def to(self, device):
        self._est.to(device)
        return self

    def __call__(self, image: torch.Tensor | None, tabular: torch.Tensor | None):
        return self._est(image, tabular)

class ResNetEncoder:
    pass


class GoogLeNetWSI(_TorchBase):
    _category_label: str = "image_only"

    def _build_estimator(self, cfg: "DotDict") -> Any:
        # Define the inner PyTorch network
        class InnerGoogLeNet(nn.Module):
            def __init__(self, dropout_rate=0.3):
                super().__init__()
                # 1. Load pre-trained GoogLeNet
                self.cnn = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
                self.cnn.fc = nn.Identity()

                # 2. FREEZE THE BACKBONE (Crucial for small medical datasets)
                for param in self.cnn.parameters():
                    param.requires_grad = False

                # 3. Attention Gating Network (The ABMIL Head)
                # GoogLeNet outputs 1024D. We map to a hidden 256D layer to score each patch.
                self.attention = nn.Sequential(
                    nn.Linear(1024, 256), nn.Tanh(), nn.Linear(256, 1)
                )

                # 4. Final Regression Head
                self.head = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(1024, 1))

            def forward(
                self, image, tabular=None
            ):  # Added tabular=None for pipeline compatibility
                B, N, C, H, W = image.shape
                x = image.view(B * N, C, H, W)

                # Extract features from frozen backbone
                features = self.cnn(x)  # Shape: (B*N, 1024)

                # Calculate Attention weights for each patch
                A = self.attention(features)  # Shape: (B*N, 1)
                A = A.view(B, N, 1)  # Reshape to (Batch, Patches, 1)
                A = torch.softmax(A, dim=1)  # Normalize so the 16 weights sum to 1.0

                features = features.view(B, N, -1)  # Shape: (B, N, 1024)

                # Weighted Average: Multiply features by their attention scores
                slide_embedding = torch.sum(features * A, dim=1)  # Shape: (B, 1024)

                return self.head(slide_embedding).squeeze(-1)

        droprate = cfg.model.get("dropout", 0.0)
        return InnerGoogLeNet(dropout_rate=droprate)