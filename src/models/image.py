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
            def __init__(self, dropout_rate):
                super().__init__()
                self.cnn = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
                self.cnn.fc = nn.Identity()
                self.regression_head = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(1024, 1)
                )

            def forward(self, image, tabular=None):
                B, N, C, H, W = image.shape
                x = image.view(B * N, C, H, W)
                features = self.cnn(x).view(B, N, -1)
                slide_embedding = features.mean(dim=1)
                return self.regression_head(slide_embedding)

        droprate = cfg.model.get("dropout", 0.0)
        return InnerGoogLeNet(dropout_rate=droprate)