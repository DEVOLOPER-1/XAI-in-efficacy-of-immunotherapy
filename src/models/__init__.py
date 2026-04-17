"""
src/models/__init__.py — Model factory / registry for AIC-4 Zerone.

How to register a new model
────────────────────────────
1. Add a new file under src/models/, e.g. src/models/my_tracker.py.
   Export one callable (class or function) that accepts a DotDict config
   and returns a tracker object.

2. Import it below and add one line to _REGISTRY:
       "my_tracker": lambda cfg: MyTracker(cfg),

Usage:
    cfg = load_config("configs/experiments/siamfc_mobile.yaml")
    tracker = get_model(cfg)
    tracker.init(frame, bbox)
    pred_bbox = tracker.update(next_frame)
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from src.config import DotDict

# ---------------------------------------------------------------------------
# Tracker protocol — every model must satisfy this interface
# ---------------------------------------------------------------------------

@runtime_checkable
class TrackerProtocol(Protocol):
    """Minimal interface all trackers must implement.

    init()   — called once with frame 0 and its ground-truth bbox.
    update() — called for every subsequent frame; returns predicted bbox.
    """

    def init(self, frame: Any, bbox: tuple[int, int, int, int]) -> None:
        """Initialise the tracker on the first frame.

        Args:
            frame: BGR numpy array (H, W, 3).
            bbox:  (x, y, w, h) ground-truth bounding box for frame 0.
        """
        ...

    def update(self, frame: Any) -> tuple[int, int, int, int]:
        """Process the next frame and return the predicted bbox.

        Args:
            frame: BGR numpy array (H, W, 3).

        Returns:
            (x, y, w, h) predicted bounding box.
        """
        ...


# ---------------------------------------------------------------------------
# Lazy import helpers — keep DL stack optional for classical-CV members
# ---------------------------------------------------------------------------

def _load_siamfc(cfg: DotDict) -> Any:
    """Import SiamFC only when actually needed (avoids torch import at top level)."""
    try:
        from src.models.baselines import SiamFCTracker  # type: ignore[import]
        return SiamFCTracker(cfg)
    except ImportError as exc:
        raise ImportError(
            "SiamFC requires the deep-learning stack. "
            "Run 'make setup-dl' to install torch + timm."
        ) from exc


def _load_csrt(cfg: DotDict) -> Any:
    """Load OpenCV CSRT — available without the DL stack."""
    try:
        from src.models.baselines import CSRTTracker  # type: ignore[import]
        return CSRTTracker(cfg)
    except ImportError as exc:
        raise ImportError(
            "CSRT requires opencv-contrib-python. "
            "Run: uv pip install opencv-contrib-python"
        ) from exc


def _load_kcf(cfg: DotDict) -> Any:
    from src.models.baselines import KCFTracker  # type: ignore[import]
    return KCFTracker(cfg)


def _load_mosse(cfg: DotDict) -> Any:
    from src.models.baselines import MOSSETracker  # type: ignore[import]
    return MOSSETracker(cfg)


# ---------------------------------------------------------------------------
# Registry
# ─────────────────────────────────────────────────────────────────────────
# Key   = the string you put in your YAML under `model.type`
# Value = a callable (cfg) -> TrackerProtocol
#
# ⚠ Per STANDARDS §5: adding / removing entries here requires a PR.
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Any] = {
    # ── Neural trackers ────────────────────────────────────────────────
    "siamfc": _load_siamfc,

    # ── Classical CV trackers ──────────────────────────────────────────
    "csrt":  _load_csrt,
    "kcf":   _load_kcf,
    "mosse": _load_mosse,
}


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------

def get_model(cfg: DotDict) -> TrackerProtocol:
    """Instantiate and return the tracker specified in the config.

    Reads ``cfg.model.type`` to look up the correct constructor in the
    registry. Raises a clear error if the type is unknown.

    Args:
        cfg: Merged experiment config (from src.config.load_config).

    Returns:
        An initialised tracker that satisfies TrackerProtocol.

    Raises:
        AttributeError: If ``cfg.model`` or ``cfg.model.type`` is missing.
        ValueError:     If ``cfg.model.type`` is not in the registry.
    """
    model_type: str = cfg.model.type.lower().strip()

    if model_type not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(
            f"Unknown model type '{model_type}'. "
            f"Available: [{available}]. "
            "To add a new model, see the instructions at the top of src/models/__init__.py."
        )

    constructor = _REGISTRY[model_type]
    tracker = constructor(cfg)

    # Runtime sanity-check: verify the returned object satisfies the protocol.
    if not isinstance(tracker, TrackerProtocol):
        raise TypeError(
            f"Model '{model_type}' does not satisfy TrackerProtocol. "
            "Ensure it implements init(frame, bbox) and update(frame)."
        )

    return tracker


def list_models() -> list[str]:
    """Return a sorted list of all registered model type strings."""
    return sorted(_REGISTRY.keys())