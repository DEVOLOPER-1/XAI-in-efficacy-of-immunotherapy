from src.models.tabular import TMBNet
import torch
import torch.nn as nn
import torchvision.models as models
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

    def load_state_dict(self, state_dict, strict=False):
        return self._est.load_state_dict(state_dict, strict=strict)


# =====================================================================
# ABMIL Head (Dynamically attaches CNN for Explainability)
# =====================================================================
class ABMIL_Expert(nn.Module):
    def __init__(self, dropout_rate=0.3, use_preextracted=True, wsi_model_path=None):
        super().__init__()
        self.use_preextracted = use_preextracted

        # --- DYNAMIC CNN LOADING ---
        if not self.use_preextracted:
            print(
                "LateFusion: preextracted=False detected. Attaching GoogLeNet backbone..."
            )
            self.cnn = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
            self.cnn.fc = nn.Identity()

            # Load the pre-trained WSI weights into the CNN
            if wsi_model_path:
                print(f"LateFusion: Loading CNN backbone weights from {wsi_model_path}")
                state_dict = torch.load(
                    wsi_model_path, map_location="cpu", weights_only=True
                )
                clean_dict = {k.replace("_est.", ""): v for k, v in state_dict.items()}
                # strict=False allows us to load just the CNN (and ignore the old regression head)
                self.load_state_dict(clean_dict, strict=False)
            else:
                raise FileNotFoundError

            # Freeze the CNN so gradients don't destroy it during XAI passes
            for param in self.cnn.parameters():
                param.requires_grad = False
        else:
            self.cnn = None

        self.attention = nn.Sequential(
            nn.Linear(1024, 256), nn.Tanh(), nn.Linear(256, 1)
        )
        self.head = nn.Sequential(nn.Dropout(dropout_rate), nn.Linear(1024, 1))

    def forward(self, features):
        if features is None:
            return None

        # 1. If raw images: [Batch, Patches, Channels, Height, Width]
        if not self.use_preextracted and features.ndim == 5:
            B, N, C, H, W = features.shape
            x = features.view(B * N, C, H, W)
            cnn_feats = self.cnn(x)  # Extract to [B*N, 1024]
            features = cnn_feats.view(B, N, 1024)

        # 2. If pre-extracted features: [Batch, Patches, 1024]
        elif features.ndim == 3:
            B, N, _ = features.shape
        else:
            raise ValueError(f"Unexpected features shape: {features.shape}")

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
                use_preextracted = cfg.dataset.get("use_preextracted", True)
                wsi_model_path = cfg.model.get(
                    "wsi_model_path",
                    "freezed-models/runs/checkpoints/best_wsi_model.pth",
                )

                # 1. Define the Architectures
                self.tabular_net = TMBNet(input_dim=21, p=dropout_rate)
                self.wsi_net = ABMIL_Expert(
                    dropout_rate=dropout_rate,
                    use_preextracted=use_preextracted,
                    wsi_model_path=wsi_model_path,
                )

                # 2. Define the Non-Linear Meta-Learner (from your Stage 1 win)
                self.meta_learner = nn.Sequential(
                    nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1)
                )

                # =================================================================
                # 3. LOAD STAGE 1 / PHASE 2 WEIGHTS
                # =================================================================
                stage1_weights_path = cfg.model.get("stage1_weights_path", None)

                if stage1_weights_path:
                    print(
                        f"Loading Stage 1/2 Fusion weights from: {stage1_weights_path}"
                    )
                    state_dict = torch.load(
                        stage1_weights_path, weights_only=False, map_location="cpu"
                    )
                    clean_dict = {
                        k.replace("_est.", ""): v for k, v in state_dict.items()
                    }
                    # CRITICAL: strict=False. Phase 2 weights won't contain the CNN backbone
                    # if it was trained on pre-extracted features. This safely ignores the mismatch!
                    self.load_state_dict(clean_dict, strict=False)

                # =================================================================
                # 4. GENTLE UNFREEZING LOGIC
                # =================================================================
                for param in self.parameters():
                    param.requires_grad = False

                for param in self.meta_learner.parameters():
                    param.requires_grad = True

                for name, param in self.wsi_net.named_parameters():
                    if "attention" in name or "head" in name:
                        param.requires_grad = True

                for name, param in self.tabular_net.named_parameters():
                    if "net.6" in name or "predictor" in name:
                        param.requires_grad = True

            # --- DYNAMIC GRAD-CAM SUPPORT ---
            # explainability.py looks for `model.cnn` to attach the Grad-CAM hook.
            # This property safely exposes GoogLeNet to the pipeline.
            @property
            def cnn(self):
                return self.wsi_net.cnn

            def forward(self, image, tabular):
                B = tabular.size(0)
                device = tabular.device

                # 1. Get WSI Expert Prediction
                wsi_pred = self.wsi_net(image)  # [B, 1]
                if wsi_pred is None:
                    wsi_pred = torch.zeros(B, 1, dtype=torch.float32, device=device)

                # 2. Get RNA Expert Prediction
                rna_pred = self.tabular_net(image=None, tabular=tabular)

                if rna_pred.dim() == 1:
                    rna_pred = rna_pred.view(-1, 1)

                # 3. Concatenate the predictions: [B, 2]
                combined_preds = torch.cat([wsi_pred, rna_pred], dim=1)

                # 4. Meta-Learner calculates final weighted score
                final_pred = self.meta_learner(combined_preds)

                return final_pred.squeeze(-1)

        return InnerLateFusion(cfg)
