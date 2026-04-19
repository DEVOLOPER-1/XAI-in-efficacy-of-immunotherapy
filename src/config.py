"""
src/config.py — Hierarchical YAML config loader for AIC-4 Zerone.

Design:
  1. Always load configs/_base.yaml first (shared team defaults).
  2. Deep-merge the experiment YAML on top — experiment keys win.
  3. Return a DotDict so callers can write cfg.model.type instead of cfg["model"]["type"].

Usage:
    cfg = load_config("configs/experiments/random_forest.yaml")
    print(cfg.model.type)          # "siamfc"
    print(cfg.training.lr)         # 0.001
    print(cfg.dataset.train_root)  # "data/train"
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# DotDict — lightweight dot-access wrapper around a plain dict
# ---------------------------------------------------------------------------

class DotDict:
    """Recursive dot-access wrapper.

    Converts any nested dict into an object whose keys are reachable as
    attributes. Preserves lists (elements are converted if they are dicts).

    Example:
        d = DotDict({"model": {"type": "siamfc", "backbone": "mobilenetv3"}})
        d.model.type      # "siamfc"
        d.model.backbone  # "mobilenetv3"
    """

    def __init__(self, data: dict[str, Any]) -> None:
        for key, value in data.items():
            setattr(self, key, self._wrap(value))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _wrap(value: Any) -> Any:
        """Recursively wrap nested dicts; leave everything else alone."""
        if isinstance(value, dict):
            return DotDict(value)
        if isinstance(value, list):
            return [DotDict._wrap(item) for item in value]
        return value

    # ------------------------------------------------------------------
    # Dict-style access helpers (for compatibility with **cfg-style calls)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert back to a plain nested dict (useful for W&B logging)."""
        result: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DotDict):
                result[key] = value.to_dict()
            elif isinstance(value, list):
                result[key] = [
                    item.to_dict() if isinstance(item, DotDict) else item
                    for item in value
                ]
            else:
                result[key] = value
        return result

    def get(self, key: str, default: Any = None) -> Any:
        """Safe attribute access with a fallback — mirrors dict.get()."""
        return getattr(self, key, default)

    def __repr__(self) -> str:
        return f"DotDict({self.to_dict()})"

    def __contains__(self, key: str) -> bool:
        return hasattr(self, key)


# ---------------------------------------------------------------------------
# Deep-merge utility
# ---------------------------------------------------------------------------

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict.

    Rules:
    - If both values are dicts, recurse.
    - Otherwise the override value wins (last-write-wins semantics).
    - Neither input dict is mutated.
    """
    merged = copy.deepcopy(base)
    for key, override_val in override.items():
        base_val = merged.get(key)
        if isinstance(base_val, dict) and isinstance(override_val, dict):
            merged[key] = _deep_merge(base_val, override_val)
        else:
            merged[key] = copy.deepcopy(override_val)
    return merged


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Path to the shared base config, relative to the project root.
_BASE_CONFIG = Path("configs/_base.yaml")


def load_config(experiment_path: str | Path) -> DotDict:
    """Load and return the merged experiment config as a DotDict.

    Args:
        experiment_path: Path to the experiment YAML, e.g.
                         "configs/experiments/random_forest.yaml".

    Returns:
        A DotDict that is the result of deep-merging the base config with
        the experiment config (experiment values win on conflicts).

    Raises:
        FileNotFoundError: If either the base or experiment config is missing.
        yaml.YAMLError: If either file contains invalid YAML.
    """
    experiment_path = Path(experiment_path)

    # -- 1. Load base config ------------------------------------------------
    if not _BASE_CONFIG.exists():
        raise FileNotFoundError(
            f"Base config not found at '{_BASE_CONFIG}'. "
            "Make sure you are running from the project root."
        )
    with _BASE_CONFIG.open("r") as fh:
        base_dict: dict[str, Any] = yaml.safe_load(fh) or {}

    # -- 2. Load experiment config ------------------------------------------
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment config not found: '{experiment_path}'")
    with experiment_path.open("r") as fh:
        exp_dict: dict[str, Any] = yaml.safe_load(fh) or {}

    # -- 3. Merge and wrap --------------------------------------------------
    merged = _deep_merge(base_dict, exp_dict)
    return DotDict(merged)
