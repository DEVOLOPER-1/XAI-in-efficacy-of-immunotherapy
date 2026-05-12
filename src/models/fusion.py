from src.models.tabular import TMBNet  # Update this to where your TMBNet class lives
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
        """Proxy to expose the PyTorch network's state_dict to the saver."""
        return self._est.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        """Proxy to allow loading weights directly into the PyTorch network."""
        return self._est.load_state_dict(state_dict, strict=strict)

# =====================================================================
# 1. The MCB Pooling Layer
# =====================================================================
class MCBLayer(nn.Module):
    """Multimodal Compact Bilinear Pooling via Count Sketch and FFT."""

    def __init__(self, dim1: int, dim2: int, out_dim: int):
        super().__init__()
        self.out_dim = out_dim

        # Sketch parameters (registered as buffers to sit on the GPU)
        self.register_buffer('h1', torch.randint(0, out_dim, (dim1,)))
        self.register_buffer('s1', torch.randint(0, 2, (dim1,)) * 2 - 1)

        self.register_buffer('h2', torch.randint(0, out_dim, (dim2,)))
        self.register_buffer('s2', torch.randint(0, 2, (dim2,)) * 2 - 1)

    def count_sketch(self, x, h, s):
        B = x.size(0)
        x_signed = x * s.unsqueeze(0)
        out = torch.zeros(B, self.out_dim, device=x.device, dtype=x.dtype)
        out.scatter_add_(1, h.unsqueeze(0).expand(B, -1), x_signed)
        return out

    def forward(self, x1, x2):
        sketch1 = self.count_sketch(x1, self.h1, self.s1)
        sketch2 = self.count_sketch(x2, self.h2, self.s2)

        fft1 = torch.fft.fft(sketch1)
        fft2 = torch.fft.fft(sketch2)

        fft_prod = fft1 * fft2
        return torch.fft.ifft(fft_prod).real


# =====================================================================
# 2. The MCB Fusion Network
# =====================================================================
class MCBFusionNet(_TorchBase):
    _category_label: str = "fusion"

    def _build_estimator(self, cfg: DotDict) -> Any:
        class InnerMCBFusion(nn.Module):
            def __init__(self, cfg):
                super().__init__()
                dropout_rate = cfg.model.get("droprate", 0.3)
                mcb_out = cfg.model.get("mcb_out_dim", 512)

                # --- 1. Load the FULL RNA Network ---
                self.tabular_net = TMBNet(input_dim=21, p=dropout_rate)

                # --- 1. Load the FULL RNA Network ---
                self.tabular_net = TMBNet(input_dim=21, p=dropout_rate)
                tabular_weights_path = cfg.model.get("tabular_model_path",
                                                     "freezed-models/runs/checkpoints/best_dnn_model.pth")

                # MINIMUM EDIT: Replace the pickle hack with standard torch.load
                state_dict = torch.load(tabular_weights_path, weights_only=False)

                cleaned_state_dict = {k.replace('_est.', ''): v for k, v in state_dict.items()}
                self.tabular_net.load_state_dict(cleaned_state_dict)

                # By setting this to True, the RNA weights will actively
                # shift to co-adapt with the fixed 1024D ImageNet features!
                for param in self.tabular_net.parameters():
                    param.requires_grad = True

                # --- 2. Split the RNA Network for the Forward Pass ---
                # We need the 17D features (layers 0-5) for fusion, and the final layer (6) for the baseline prediction
                self.tab_feat_extractor = nn.Sequential(*list(self.tabular_net.net.children())[:6])
                self.tab_predictor = list(self.tabular_net.net.children())[6]

                self.tabular_dim = 17
                self.image_dim = cfg.model.get("image_dim", 1024)

                # --- 3. The MCB WSI Branch ---
                self.mcb = MCBLayer(self.image_dim, self.tabular_dim, out_dim=mcb_out)
                self.norm = nn.LayerNorm(mcb_out)

                # --- 4. The Fusion Offset Head ---
                self.fusion_head = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(mcb_out, 128),
                    nn.ReLU(),
                    nn.Dropout(dropout_rate / 2),
                    nn.Linear(128, 1)
                )

                # Initialize the final layer to zero!
                # This guarantees the WSI offset starts at 0, preserving your 0.80 baseline on Epoch 1.
                nn.init.zeros_(self.fusion_head[-1].weight)
                nn.init.zeros_(self.fusion_head[-1].bias)

            def forward(self, image, tabular):
                B = tabular.size(0)
                device = tabular.device

                # 1. WSI Embeddings
                if image is not None:
                    img_feat = image.mean(dim=1)
                else:
                    img_feat = torch.zeros(B, self.image_dim, dtype=torch.float32, device=device)

                # 2. RNA Features & Baseline Prediction
                tab_feat = self.tab_feat_extractor(tabular)  # [B, 17]
                base_pred = self.tab_predictor(tab_feat)  # [B, 1]

                # 3. MCB Fusion (Find cross-modal interactions)
                fused = self.mcb(img_feat, tab_feat)
                fused = self.norm(fused)

                # 4. Calculate the WSI "Correction Offset"
                fusion_offset = self.fusion_head(fused)  # [B, 1]

                # 5. Final Boosting Prediction
                final_pred = base_pred + fusion_offset
                return final_pred.squeeze(-1)

        return InnerMCBFusion(cfg)