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

                # 1. Define the Architectures
                self.tabular_net = TMBNet(input_dim=21, p=dropout_rate)
                self.wsi_net = ABMIL_Expert(dropout_rate=dropout_rate)

                # 2. Define the Non-Linear Meta-Learner (from your Stage 1 win)
                self.meta_learner = nn.Sequential(
                    nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1)
                )

                # =================================================================
                # 3. LOAD STAGE 1 WEIGHTS (The crucial step for Stage 2)
                # =================================================================
                stage1_weights_path = cfg.model.get("stage1_weights_path", None)

                if stage1_weights_path:
                    # We are in Stage 2! Load the entire fused network state
                    print(f"Loading Stage 1 Fusion weights from: {stage1_weights_path}")
                    state_dict = torch.load(stage1_weights_path, weights_only=False)
                    # Clean the '_est.' prefix if it exists from your _TorchBase wrapper
                    clean_dict = {
                        k.replace("_est.", ""): v for k, v in state_dict.items()
                    }
                    self.load_state_dict(clean_dict)
                else:
                    # Fallback: If you ever need to restart Stage 1, load individual experts here
                    print("No Stage 1 weights provided. Starting from frozen experts.")
                    # (You can paste your old individual loading code here if desired)

                # =================================================================
                # 4. GENTLE UNFREEZING LOGIC
                # =================================================================
                # First, lock down absolutely everything to prevent accidents.
                for param in self.parameters():
                    param.requires_grad = False

                # A. Unfreeze the Meta-Learner
                for param in self.meta_learner.parameters():
                    param.requires_grad = True

                # B. Unfreeze the top of the WSI Expert (ABMIL Attention & Head)
                for name, param in self.wsi_net.named_parameters():
                    if "attention" in name or "head" in name:
                        param.requires_grad = True

                # C. Unfreeze the top of the RNA Expert (Final Predictor Layer)
                # Note: Adjust 'net.6' if your final TMBNet layer has a different name
                for name, param in self.tabular_net.named_parameters():
                    if "net.6" in name or "predictor" in name:
                        param.requires_grad = True

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
