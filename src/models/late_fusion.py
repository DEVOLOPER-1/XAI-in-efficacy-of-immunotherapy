from src.models.tabular import TMBNet
import torch
import torch.nn as nn
from typing import Any
from src.config import DotDict


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

    def state_dict(self, *args, **kwargs):
        return self._est.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self._est.load_state_dict(state_dict, strict=strict)


# =====================================================================
# ABMIL Head (Recreated to process the 16x1024 embeddings safely)
# =====================================================================
class ABMIL_Expert(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(1024, 256), nn.Tanh(), nn.Linear(256, 1)
        )
        self.head = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(1024, 1))

    def forward(self, features):
        if features is None:
            return None
        # features shape: [B, 16, 1024]
        B, N, _ = features.shape
        A = self.attention(features)  # [B, N, 1]
        A = torch.softmax(A, dim=1)  # Normalize weights across patches

        slide_embedding = torch.sum(features * A, dim=1)  # [B, 1024]
        return self.head(slide_embedding)


# =====================================================================
# Late Fusion Network (Decision-Level Meta Learner)
# =====================================================================
class LateFusionNet(_TorchBase):
    _category_label: str = "fusion"

    def _build_estimator(self, cfg: DotDict) -> Any:
        class InnerLateFusion(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                dropout_rate = cfg.model.get("droprate", 0.3)

                # --- 1. Load the RNA Expert ---
                self.tabular_net = TMBNet(input_dim=21, p=dropout_rate)
                tabular_weights_path = cfg.model.get(
                    "tabular_model_path",
                    "freezed-models/runs/checkpoints/best_dnn_model.pth",
                )

                # Load and clean RNA weights
                tab_state_dict = torch.load(tabular_weights_path, weights_only=False)
                clean_tab_dict = {
                    k.replace("_est.", ""): v for k, v in tab_state_dict.items()
                }
                self.tabular_net.load_state_dict(clean_tab_dict)

                # --- 2. Load the WSI ABMIL Expert ---
                self.wsi_net = ABMIL_Expert(dropout_rate=dropout_rate)
                wsi_weights_path = cfg.model.get(
                    "wsi_model_path",
                    "freezed-models/runs/checkpoints/best_wsi_model.pth",
                )

                # Load and filter ONLY the attention/head weights (ignoring the CNN backbone)
                wsi_state_dict = torch.load(wsi_weights_path, weights_only=True)
                clean_wsi_dict = {}
                for k, v in wsi_state_dict.items():
                    if "attention" in k or "head" in k:
                        clean_wsi_dict[k.replace("_est.", "")] = v
                self.wsi_net.load_state_dict(clean_wsi_dict, strict=False)

                # --- 3. FREEZE BOTH EXPERTS (Crucial for Late Fusion) ---
                for param in self.tabular_net.parameters():
                    param.requires_grad = False
                for param in self.wsi_net.parameters():
                    param.requires_grad = False

                # --- 4. The Meta-Learner ---
                # Takes 2 inputs (RNA prediction, WSI prediction) and outputs 1 Final Prediction
                self.meta_learner = nn.Linear(2, 1)

                # Initialize weights to [0.5, 0.5] so it starts as a perfect average
                nn.init.constant_(self.meta_learner.weight, 0.5)
                nn.init.constant_(self.meta_learner.bias, 0.0)

            def forward(self, image, tabular):
                B = tabular.size(0)
                device = tabular.device

                # 1. Get WSI Expert Prediction
                wsi_pred = self.wsi_net(image)  # [B, 1]
                if wsi_pred is None:
                    wsi_pred = torch.zeros(B, 1, dtype=torch.float32, device=device)

                # 2. Get RNA Expert Prediction
                # Explicitly name the arguments so the pipeline signature doesn't break!
                rna_pred = self.tabular_net(image=None, tabular=tabular)

                # Pro-tip safety check: ensure it's exactly [B, 1] before concatenating
                if rna_pred.dim() == 1:
                    rna_pred = rna_pred.view(-1, 1)
                # 3. Concatenate the predictions: [B, 2]
                combined_preds = torch.cat([wsi_pred, rna_pred], dim=1)

                # 4. Meta-Learner calculates final weighted score
                final_pred = self.meta_learner(combined_preds)

                return final_pred.squeeze(-1)

        return InnerLateFusion(cfg)
