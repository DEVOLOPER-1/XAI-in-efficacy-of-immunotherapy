"""
src/models/__init__.py — Multimodal model factory / registry for Zerone Cancer Research.

Three model categories
──────────────────────
  TabularOnly  — receives only the tabular feature vector.
                 Suitable for: XGBoost, CatBoost, MLP.
                 Config: model.category: tabular_only

  ImageOnly    — receives only the image/patch array.
                 Suitable for: ResNet, ViT, ABMIL patch aggregators.
                 Config: model.category: image_only

  Fusion       — receives both modalities and merges them.
                 Two sub-strategies controlled by model.fusion_strategy:
                   early  — concatenate features before the final head
                   late   — independent heads, combine predictions
                 Config: model.category: fusion
                         model.fusion_strategy: early | late

How to register a new model
────────────────────────────
1. Add src/models/<your_model>.py. Export one class that inherits from
   nn.Module (or a sklearn-compatible API for tree models) and accepts a
   DotDict config in __init__.

2. Import it lazily in the relevant _load_* helper below.

3. Add one entry to the appropriate sub-registry (_TABULAR_REGISTRY,
   _IMAGE_REGISTRY, or _FUSION_REGISTRY).

4. Open a PR — per STANDARDS §5, registry changes require team review.

Usage:
    cfg     = load_config("configs/experiments/fusion_early.yaml")
    model   = get_model(cfg)
    outputs = model(batch["image"], batch["tabular"])   # Fusion / ImageOnly / TabularOnly
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.config import DotDict


# ---------------------------------------------------------------------------
# Unified model protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class CancerModelProtocol(Protocol):
    """Interface every cancer research model must satisfy.

    forward() / __call__() accepts both modalities; each may be None.
    Models must handle None gracefully (ignore, mask, or raise a clear error).

    For tree-based models (XGBoost / CatBoost) there is no .forward(); they
    implement .predict() instead. The training loop in train.py dispatches
    to the correct call style based on cfg.model.backend.
    """

    def __call__(
        self,
        image:   "Any | None",   # (B, N, C, H, W) tensor or ndarray, or None
        tabular: "Any | None",   # (B, F) tensor or ndarray, or None
    ) -> "Any":                   # (B, 1) or (B,) predictions
        ...


# ---------------------------------------------------------------------------
# Lazy import helpers — DL stack is optional for tabular-only members
# ---------------------------------------------------------------------------

# ── Tabular ────────────────────────────────────────────────────────────────

def _load_dicision_tree(cfg: DotDict) -> Any:
    try:
        from src.models.tabular import DecisionTreeRegressor  # type: ignore[import]
        return DecisionTreeRegressor(cfg)
    except ImportError as exc:
        raise ImportError(
            "Decision Tree model requires: uv add scikit-learn"
        ) from exc


def _load_catboost(cfg: DotDict) -> Any:
    try:
        from src.models.tabular import CatBoostRegressor  # type: ignore[import]
        return CatBoostRegressor(cfg)
    except ImportError as exc:
        raise ImportError(
            "CatBoost model requires: uv pip install catboost"
        ) from exc


def _load_tabular_mlp(cfg: DotDict) -> Any:
    try:
        from src.models.tabular import TabularMLP  # type: ignore[import]
        return TabularMLP(cfg)
    except ImportError as exc:
        raise ImportError(
            "TabularMLP requires PyTorch: uv pip install torch"
        ) from exc


# ── Image ──────────────────────────────────────────────────────────────────

def _load_resnet(cfg: DotDict) -> Any:
    """ResNet-based patch encoder + mean-pooling aggregator."""
    try:
        from src.models.image import ResNetEncoder  # type: ignore[import]
        return ResNetEncoder(cfg)
    except ImportError as exc:
        raise ImportError(
            "ResNet model requires: uv pip install torch torchvision"
        ) from exc


def _load_vit(cfg: DotDict) -> Any:
    """Vision Transformer patch encoder (timm-backed)."""
    try:
        from src.models.image import ViTEncoder  # type: ignore[import]
        return ViTEncoder(cfg)
    except ImportError as exc:
        raise ImportError(
            "ViT model requires: uv pip install torch timm"
        ) from exc


def _load_abmil(cfg: DotDict) -> Any:
    """Attention-Based Multiple Instance Learning aggregator.

    Operates on pre-extracted patch features — no CNN backbone needed.
    Best paired with cfg.dataset.use_preextracted: true.
    Reference: Ilse et al. (2018) — https://arxiv.org/abs/1802.04712
    """
    try:
        from src.models.image import ABMILModel  # type: ignore[import]
        return ABMILModel(cfg)
    except ImportError as exc:
        raise ImportError(
            "ABMIL requires PyTorch: uv pip install torch"
        ) from exc


# ── Fusion ─────────────────────────────────────────────────────────────────

def _load_early_fusion(cfg: DotDict) -> Any:
    """Early fusion: concatenate image + tabular embeddings → shared MLP head.

    Image branch: configurable encoder (default: ABMIL on preextracted features).
    Tabular branch: linear projection.
    Head: MLP over [image_emb ‖ tabular_emb].
    """
    try:
        from src.models.fusion import EarlyFusionModel  # type: ignore[import]
        return EarlyFusionModel(cfg)
    except ImportError as exc:
        raise ImportError(
            "EarlyFusionModel requires PyTorch: uv pip install torch"
        ) from exc


def _load_late_fusion(cfg: DotDict) -> Any:
    """Late fusion: independent image + tabular heads, combine predictions.

    Each modality produces a scalar prediction; the final output is a
    weighted sum (weights are learnable or configured via model.fusion_weights).
    Gracefully degrades when one modality is None — uses the other alone.
    """
    try:
        from src.models.fusion import LateFusionModel  # type: ignore[import]
        return LateFusionModel(cfg)
    except ImportError as exc:
        raise ImportError(
            "LateFusionModel requires PyTorch: uv pip install torch"
        ) from exc


# ---------------------------------------------------------------------------
# Sub-registries
# ─────────────────────────────────────────────────────────────────────────
# Key   = cfg.model.type string
# Value = callable (cfg) → model
#
# ⚠ Per STANDARDS §5: registry changes require a PR.
# ---------------------------------------------------------------------------

_TABULAR_REGISTRY: dict[str, Any] = {
    "decision_tree":     _load_dicision_tree,
    "catboost":    _load_catboost,
    "tabular_mlp": _load_tabular_mlp,
}

_IMAGE_REGISTRY: dict[str, Any] = {
    "resnet": _load_resnet,
    "vit":    _load_vit,
    "abmil":  _load_abmil,
}

_FUSION_REGISTRY: dict[str, Any] = {
    "early": _load_early_fusion,
    "late":  _load_late_fusion,
}

# Flat view for listing + validation
_ALL_REGISTRIES: dict[str, dict[str, Any]] = {
    "tabular_only": _TABULAR_REGISTRY,
    "image_only":   _IMAGE_REGISTRY,
    "fusion":       _FUSION_REGISTRY,
}


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_model(cfg: DotDict) -> CancerModelProtocol:
    """Instantiate and return the model specified in the config.

    Reads:
        cfg.model.category        — "tabular_only" | "image_only" | "fusion"
        cfg.model.type            — e.g. "xgboost", "abmil", "resnet", …
        cfg.model.fusion_strategy — "early" | "late"  (fusion category only)

    Args:
        cfg: Merged experiment config (from src.config.load_config).

    Returns:
        An instantiated model satisfying CancerModelProtocol.

    Raises:
        AttributeError: If required config keys are missing.
        ValueError:     If the category or type is not recognised.

    Example YAML:
        model:
          category: fusion
          type: resnet              # image branch backbone
          fusion_strategy: early
          tabular_dim: 128          # passed to the fusion model
          image_dim: 512

        model:
          category: tabular_only
          type: decision_tree

        model:
          category: image_only
          type: abmil
    """
    model_cfg = cfg.get("model") or DotDict({})
    category  = (model_cfg.get("category") or "").lower().strip()
    mod_cfg   = cfg.get("modalities") or DotDict({})

    # Validate category
    if category not in _ALL_REGISTRIES:
        available = ", ".join(sorted(_ALL_REGISTRIES))
        raise ValueError(
            f"Unknown model.category '{category}'. "
            f"Choose from: [{available}]. "
            "Update your experiment YAML."
        )

    registry = _ALL_REGISTRIES[category]

    # -- Resolve the registry key -------------------------------------------
    if category == "fusion":
        # For fusion, the key is the fusion_strategy (early / late)
        strategy = (model_cfg.get("fusion_strategy") or "early").lower().strip()
        if strategy not in registry:
            raise ValueError(
                f"Unknown model.fusion_strategy '{strategy}'. "
                f"Choose from: [{', '.join(sorted(registry))}]."
            )
        key = strategy
    else:
        # For tabular_only / image_only, the key is model.type
        key = (model_cfg.get("type") or "").lower().strip()
        if not key:
            raise AttributeError(
                "cfg.model.type is required for tabular_only and image_only categories."
            )
        if key not in registry:
            available = ", ".join(sorted(registry))
            raise ValueError(
                f"Unknown model.type '{key}' for category '{category}'. "
                f"Available: [{available}]."
            )

    # -- Check modality flags are consistent --------------------------------
    _warn_modality_mismatch(category, mod_cfg)

    # -- Instantiate ---------------------------------------------------------
    constructor = registry[key]
    model = constructor(cfg)

    return model


def list_models() -> dict[str, list[str]]:
    """Return all registered model types, grouped by category."""
    return {cat: sorted(reg.keys()) for cat, reg in _ALL_REGISTRIES.items()}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _warn_modality_mismatch(category: str, mod_cfg: DotDict) -> None:
    """Log a warning if the model category conflicts with enabled modalities."""
    use_image   = mod_cfg.get("image",   True)
    use_tabular = mod_cfg.get("tabular", True)

    if category == "tabular_only" and use_image:
        log.warning(
            "model.category is 'tabular_only' but cfg.modalities.image is true. "
            "The image branch will be ignored. Set modalities.image: false to suppress this."
        )
    if category == "image_only" and use_tabular:
        log.warning(
            "model.category is 'image_only' but cfg.modalities.tabular is true. "
            "The tabular branch will be ignored. Set modalities.tabular: false to suppress this."
        )
    if category == "fusion" and not (use_image and use_tabular):
        log.warning(
            "model.category is 'fusion' but not all modalities are enabled "
            "(image=%s, tabular=%s). Fusion model will receive None for the "
            "disabled modality — ensure your fusion model handles this.",
            use_image, use_tabular,
        )


import logging
log = logging.getLogger(__name__)