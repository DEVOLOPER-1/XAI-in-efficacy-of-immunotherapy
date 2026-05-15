"""src/explainability.py — multimodal explainability orchestration.

This module is the new home for the project-level explanation pipeline.
It consumes the shared `src.data_loader` outputs and produces saved artefacts
for the supported explainability families:

  - SHAP
  - LIME
  - Grad-CAM
  - PDP / ICE / ACE (implemented as perturbation curves)

The code is intentionally defensive:
  - all heavy imports are lazy;
  - unsupported model/modality combinations are skipped with warnings;
  - outputs are always written to disk so runs stay auditable.

For multimodal models, tabular and image explainers are run as separate
instances when both modalities are available.
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np

from src.config import DotDict
from src.data_loader import PatientSample, build_dataloaders
from src.models import get_model
from src.utils import as_numpy, load_checkpoint, save_json

log = logging.getLogger(__name__)


DEFAULT_METHODS = ("shap", "lime", "gradcam", "pdp", "ice", "ace")
_IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Public report objects
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ExplainabilityArtifact:
    """One saved explanation artefact on disk."""

    patient_id: str
    modality: str
    method: str
    path: Path
    note: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "modality": self.modality,
            "method": self.method,
            "path": str(self.path),
            "note": self.note,
            "metadata": _json_safe(self.metadata),
        }


@dataclass(slots=True)
class ExplainabilityReport:
    """Summary returned by `run_explainability`."""

    split: str
    output_dir: Path
    experiment_name: str
    model_type: str
    artifacts: list[ExplainabilityArtifact] = field(default_factory=list)
    summary: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "output_dir": str(self.output_dir),
            "experiment_name": self.experiment_name,
            "model_type": self.model_type,
            "summary": _json_safe(self.summary),
            "artifacts": [item.to_dict() for item in self.artifacts],
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_explainability(
    cfg: DotDict,
    split: str = "val",
    output_dir: str | Path | None = None,
    methods: Iterable[str] | None = None,
    max_patients: int | None = None,
) -> ExplainabilityReport:
    """Run the explainability pipeline for one split and save artefacts.

    The split is resolved through `src.data_loader.build_dataloaders`. The
    returned report contains a manifest of every saved plot plus a compact
    numeric summary that can be printed in the CLI.
    """
    training_cfg = cfg.get("training") or DotDict({})
    exp_cfg = cfg.get("explainability") or DotDict({})
    model_cfg = cfg.get("model") or DotDict({})

    methods_set = {
        m.lower().strip() for m in (methods or exp_cfg.get("methods", DEFAULT_METHODS))
    }
    max_patients = int(max_patients or exp_cfg.get("max_patients", 3))
    background_size = int(exp_cfg.get("background_size", 32))
    seed = int(exp_cfg.get("seed", cfg.get("dataset", DotDict({})).get("seed", 42)))

    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    split_loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    if split not in split_loaders or split_loaders[split] is None:
        available = ", ".join(k for k, v in split_loaders.items() if v is not None)
        raise ValueError(
            f"Unknown or unavailable split '{split}'. Available: [{available}]."
        )

    selected_loader = split_loaders[split]
    assert selected_loader is not None

    model = _load_model(cfg)

    split_samples = _collect_samples(selected_loader)
    train_samples = _collect_samples(train_loader)

    experiment_name = (
        cfg.get("experiment_name")
        or (cfg.get("wandb") or DotDict({})).get("run_name")
        or model_cfg.get("type")
        or "experiment"
    )

    output_dir = Path(
        output_dir
        or exp_cfg.get("output_dir")
        or Path(training_cfg.get("save_dir", "logs/runs/checkpoints"))
        / f"{experiment_name}_explainability"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    report = ExplainabilityReport(
        split=split,
        output_dir=output_dir,
        experiment_name=str(experiment_name),
        model_type=str(model_cfg.get("type") or "unknown"),
    )

    use_tabular = bool((cfg.get("modalities") or DotDict({})).get("tabular", True))
    use_image = bool((cfg.get("modalities") or DotDict({})).get("image", False))

    # Prefer a patient with both modalities for multimodal runs.
    paired_sample = next(
        (
            sample
            for sample in split_samples
            if sample.tabular is not None and sample.image is not None
        ),
        None,
    )
    tabular_reference = next(
        (s for s in split_samples if s.tabular is not None), paired_sample
    )
    image_reference = next(
        (s for s in split_samples if s.image is not None), paired_sample
    )

    # Separate explainability instances for each modality.
    if use_tabular and tabular_reference is not None:
        tabular_artifacts = _explain_tabular_branch(
            model=model,
            train_samples=train_samples,
            split_samples=split_samples,
            reference_sample=tabular_reference,
            methods=methods_set,
            background_size=background_size,
            max_patients=max_patients,
            seed=seed,
            output_dir=output_dir / "tabular",
        )
        report.artifacts.extend(tabular_artifacts)

    if use_image and image_reference is not None:
        image_artifacts = _explain_image_branch(
            model=model,
            train_samples=train_samples,
            split_samples=split_samples,
            reference_sample=image_reference,
            methods=methods_set,
            background_size=background_size,
            max_patients=max_patients,
            seed=seed,
            output_dir=output_dir / "image",
        )
        report.artifacts.extend(image_artifacts)

    report.summary = _build_summary(report.artifacts)

    save_json(report.to_dict(), output_dir / "report.json")
    _write_report_csv(report, output_dir / "report.csv")

    log.info(
        "Explainability run complete — split=%s | artefacts=%d | output=%s",
        split,
        len(report.artifacts),
        output_dir,
    )
    return report


def evaluate(cfg: DotDict, split: str = "val") -> dict[str, float]:
    """Compatibility wrapper returning the numeric summary for CLI output."""
    report = run_explainability(cfg, split=split)
    return report.summary


def predict(
    cfg: DotDict,
    split: str = "val",
    output_path: str | Path = "submission.csv",
) -> Path:
    """Compatibility wrapper that writes a CSV explanation manifest.

    The old tracker-style API expected a CSV path. We keep that contract by
    writing a flat artefact summary table to the requested path.
    """
    output_path = Path(output_path)
    report = run_explainability(cfg, split=split, output_dir=output_path.parent)
    _write_report_csv(report, output_path)
    return output_path


# ---------------------------------------------------------------------------
# Branch orchestration
# ---------------------------------------------------------------------------


def _explain_tabular_branch(
    model: Any,
    train_samples: list[PatientSample],
    split_samples: list[PatientSample],
    reference_sample: PatientSample,
    methods: set[str],
    background_size: int,
    max_patients: int,
    seed: int,
    output_dir: Path,
) -> list[ExplainabilityArtifact]:
    output_dir.mkdir(parents=True, exist_ok=True)

    background = _collect_tabular_matrix(train_samples, max_rows=background_size)
    eval_rows = _collect_tabular_matrix(split_samples, max_rows=max_patients)
    if background.size == 0 or eval_rows.size == 0:
        log.warning(
            "No tabular data available for explainability; skipping tabular branch."
        )
        return []

    feature_names = _infer_tabular_feature_names(split_samples)
    reference_image = (
        reference_sample.image if reference_sample.image is not None else None
    )
    artifacts: list[ExplainabilityArtifact] = []

    if "shap" in methods:
        path = _tabular_shap_plot(
            model=model,
            background=background,
            explain_rows=eval_rows,
            feature_names=feature_names,
            reference_image=reference_image,
            reference_tabular=reference_sample.tabular,
            output_path=output_dir / "tabular_shap.png",
        )
        if path is not None:
            artifacts.append(
                ExplainabilityArtifact(
                    patient_id=reference_sample.patient_id,
                    modality="tabular",
                    method="shap",
                    path=path,
                    note="Tabular SHAP summary",
                    metadata={"feature_count": int(background.shape[1])},
                )
            )

    if "lime" in methods:
        path = _tabular_lime_plot(
            model=model,
            background=background,
            explain_row=eval_rows[0],
            feature_names=feature_names,
            reference_image=reference_image,
            reference_tabular=reference_sample.tabular,
            output_path=output_dir / "tabular_lime.png",
            seed=seed,
        )
        if path is not None:
            artifacts.append(
                ExplainabilityArtifact(
                    patient_id=reference_sample.patient_id,
                    modality="tabular",
                    method="lime",
                    path=path,
                    note="Tabular local explanation",
                    metadata={"feature_count": int(background.shape[1])},
                )
            )

    if methods.intersection({"pdp", "ice", "ace"}):
        path = _tabular_effect_curves(
            model=model,
            background=background,
            explain_rows=eval_rows,
            feature_names=feature_names,
            reference_image=reference_image,
            reference_tabular=reference_sample.tabular,
            output_path=output_dir / "tabular_effects.png",
            seed=seed,
        )
        if path is not None:
            artifacts.append(
                ExplainabilityArtifact(
                    patient_id=reference_sample.patient_id,
                    modality="tabular",
                    method="pdp_ice_ace",
                    path=path,
                    note="PDP / ICE / ACE curves",
                    metadata={"feature_count": int(background.shape[1])},
                )
            )

    return artifacts


def _explain_image_branch(
    model: Any,
    train_samples: list[PatientSample],
    split_samples: list[PatientSample],
    reference_sample: PatientSample,
    methods: set[str],
    background_size: int,
    max_patients: int,
    seed: int,
    output_dir: Path,
) -> list[ExplainabilityArtifact]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[ExplainabilityArtifact] = []

    # Two modes: raw WSI tiles or pre-extracted patch features.
    raw_train = [s for s in train_samples if s.image is not None and s.image.ndim >= 4]
    raw_eval = [s for s in split_samples if s.image is not None and s.image.ndim >= 4]
    feat_train = [s for s in train_samples if s.image is not None and s.image.ndim == 2]
    feat_eval = [s for s in split_samples if s.image is not None and s.image.ndim == 2]

    if raw_eval:
        artifacts.extend(
            _explain_raw_image_branch(
                model=model,
                train_samples=raw_train,
                split_samples=raw_eval,
                reference_sample=reference_sample,
                methods=methods,
                background_size=background_size,
                max_patients=max_patients,
                seed=seed,
                output_dir=output_dir / "raw",
            )
        )
    elif feat_eval:
        artifacts.extend(
            _explain_image_feature_branch(
                model=model,
                train_samples=feat_train,
                split_samples=feat_eval,
                reference_sample=reference_sample,
                methods=methods,
                background_size=background_size,
                max_patients=max_patients,
                seed=seed,
                output_dir=output_dir / "features",
            )
        )
    else:
        log.warning(
            "No image data available for explainability; skipping image branch."
        )

    return artifacts


# ---------------------------------------------------------------------------
# Tabular explainers
# ---------------------------------------------------------------------------


def _tabular_shap_plot(
    model: Any,
    background: np.ndarray,
    explain_rows: np.ndarray,
    feature_names: list[str],
    reference_image: np.ndarray | None,
    reference_tabular: np.ndarray | None,
    output_path: Path,
) -> Path | None:
    try:
        import shap
    except ImportError:
        log.warning("shap is not installed; skipping tabular SHAP.")
        return None

    try:
        predict_fn = lambda x: _predict_with_model(
            model,
            image=_repeat_optional_image(reference_image, len(x)),
            tabular=x,
            reference_tabular=reference_tabular,
        )

        explainer = shap.Explainer(predict_fn, background)
        explanation = explainer(explain_rows)
        values = np.asarray(getattr(explanation, "values", explanation))

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            values,
            features=explain_rows,
            feature_names=feature_names,
            show=False,
            max_display=min(15, len(feature_names)),
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return output_path
    except Exception as exc:  # pragma: no cover - best effort on optional deps
        log.warning("Tabular SHAP failed: %s", exc)
        return None


def _tabular_lime_plot(
    model: Any,
    background: np.ndarray,
    explain_row: np.ndarray,
    feature_names: list[str],
    reference_image: np.ndarray | None,
    reference_tabular: np.ndarray | None,
    output_path: Path,
    seed: int,
) -> Path | None:
    try:
        from lime import lime_tabular
    except ImportError:
        log.warning("lime is not installed; skipping tabular LIME.")
        return None

    try:
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=background,
            feature_names=feature_names,
            mode="regression",
            discretize_continuous=True,
            random_state=seed,
        )

        predict_fn = lambda x: _predict_with_model(
            model,
            image=_repeat_optional_image(reference_image, len(x)),
            tabular=x,
            reference_tabular=reference_tabular,
        )

        exp = explainer.explain_instance(
            explain_row,
            predict_fn,
            num_features=min(len(feature_names), 12),
        )
        fig = exp.as_pyplot_figure()
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_path
    except Exception as exc:  # pragma: no cover - best effort on optional deps
        log.warning("Tabular LIME failed: %s", exc)
        return None


def _tabular_effect_curves(
    model: Any,
    background: np.ndarray,
    explain_rows: np.ndarray,
    feature_names: list[str],
    reference_image: np.ndarray | None,
    reference_tabular: np.ndarray | None,
    output_path: Path,
    seed: int,
) -> Path | None:
    if background.size == 0:
        return None

    rng = np.random.default_rng(seed)
    n_features = background.shape[1]
    # Focus on the most variable features so the plots remain readable.
    feature_order = np.argsort(np.nanvar(background, axis=0))[::-1][
        : min(3, n_features)
    ]

    fig, axes = plt.subplots(
        len(feature_order), 1, figsize=(10, 4 * len(feature_order))
    )
    if len(feature_order) == 1:
        axes = [axes]

    for ax, feature_idx in zip(axes, feature_order, strict=False):
        column = background[:, feature_idx]
        finite = column[np.isfinite(column)]
        if finite.size == 0:
            continue

        grid = np.linspace(np.quantile(finite, 0.05), np.quantile(finite, 0.95), 20)
        pdp_values: list[float] = []
        ice_rows = explain_rows[: min(len(explain_rows), 5)]
        ice_curves: list[np.ndarray] = []

        for value in grid:
            modified = background.copy()
            modified[:, feature_idx] = value
            preds = _predict_with_model(
                model,
                image=_repeat_optional_image(reference_image, len(modified)),
                tabular=modified,
                reference_tabular=reference_tabular,
            )
            pdp_values.append(float(np.mean(preds)))

        for row in ice_rows:
            row_curve: list[float] = []
            for value in grid:
                modified_row = row.copy()[None, :]
                modified_row[0, feature_idx] = value
                pred = _predict_with_model(
                    model,
                    image=_repeat_optional_image(reference_image, 1),
                    tabular=modified_row,
                    reference_tabular=reference_tabular,
                )
                row_curve.append(float(pred[0]))
            ice_curves.append(np.asarray(row_curve, dtype=np.float32))

        ace_curve = np.cumsum(np.asarray(pdp_values) - np.mean(pdp_values)) / np.arange(
            1, len(grid) + 1
        )

        for curve in ice_curves:
            ax.plot(grid, curve, color="tab:blue", alpha=0.25, linewidth=1)
        ax.plot(grid, pdp_values, color="tab:orange", linewidth=2.5, label="PDP")
        ax.plot(grid, ace_curve, color="tab:green", linewidth=2, label="ACE")
        ax.set_title(f"Feature effect: {feature_names[feature_idx]}")
        ax.set_xlabel(feature_names[feature_idx])
        ax.set_ylabel("Predicted score")
        ax.grid(True, alpha=0.25)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Raw image explainers
# ---------------------------------------------------------------------------


def _explain_raw_image_branch(
    model: Any,
    train_samples: list[PatientSample],
    split_samples: list[PatientSample],
    reference_sample: PatientSample,
    methods: set[str],
    background_size: int,
    max_patients: int,
    seed: int,
    output_dir: Path,
) -> list[ExplainabilityArtifact]:
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[ExplainabilityArtifact] = []

    background_tiles = _collect_raw_image_tiles(train_samples, max_rows=background_size)
    explain_tiles = _collect_raw_image_tiles(split_samples, max_rows=max_patients)
    if background_tiles.size == 0 or explain_tiles.size == 0:
        log.warning("No raw image tiles available; skipping raw image explainers.")
        return []

    reference_tabular = reference_sample.tabular
    tile = explain_tiles[0]
    display_tile = _to_display_tile(tile)

    if "shap" in methods:
        path = _raw_image_shap_plot(
            model=model,
            background=background_tiles,
            explain_tiles=explain_tiles[: min(len(explain_tiles), 3)],
            reference_tabular=reference_tabular,
            output_path=output_dir / "image_shap.png",
        )
        if path is not None:
            artifacts.append(
                ExplainabilityArtifact(
                    patient_id=reference_sample.patient_id,
                    modality="image",
                    method="shap",
                    path=path,
                    note="Raw-image gradient SHAP",
                    metadata={"tile_shape": list(tile.shape)},
                )
            )

    if "lime" in methods:
        path = _raw_image_lime_plot(
            model=model,
            display_tile=display_tile,
            reference_tabular=reference_tabular,
            output_path=output_dir / "image_lime.png",
            seed=seed,
        )
        if path is not None:
            artifacts.append(
                ExplainabilityArtifact(
                    patient_id=reference_sample.patient_id,
                    modality="image",
                    method="lime",
                    path=path,
                    note="Raw-image local explanation",
                    metadata={"tile_shape": list(tile.shape)},
                )
            )

    if "gradcam" in methods:
        path = _raw_image_gradcam_plot(
            model=model,
            tile=tile,
            reference_tabular=reference_tabular,
            output_path=output_dir / "image_gradcam.png",
        )
        if path is not None:
            artifacts.append(
                ExplainabilityArtifact(
                    patient_id=reference_sample.patient_id,
                    modality="image",
                    method="gradcam",
                    path=path,
                    note="Raw-image gradient CAM",
                    metadata={"tile_shape": list(tile.shape)},
                )
            )

    if methods.intersection({"pdp", "ice", "ace"}):
        path = _image_effect_curves(
            model=model,
            background_tiles=background_tiles,
            explain_tiles=explain_tiles,
            reference_tabular=reference_tabular,
            output_path=output_dir / "image_effects.png",
            seed=seed,
        )
        if path is not None:
            artifacts.append(
                ExplainabilityArtifact(
                    patient_id=reference_sample.patient_id,
                    modality="image",
                    method="pdp_ice_ace",
                    path=path,
                    note="Image perturbation curves",
                    metadata={"tile_shape": list(tile.shape)},
                )
            )

    return artifacts


def _raw_image_shap_plot(
    model: Any,
    background: np.ndarray,
    explain_tiles: np.ndarray,
    reference_tabular: np.ndarray | None,
    output_path: Path,
) -> Path | None:
    try:
        import shap
        import torch
    except ImportError:
        log.warning("shap/torch not available; skipping raw image SHAP.")
        return None

    try:
        background_tensor = torch.from_numpy(background).float()
        explain_tensor = torch.from_numpy(explain_tiles).float()
        background_input: Any = background_tensor
        explain_input: Any = explain_tensor

        class _ImageWrapper(torch.nn.Module):
            def __init__(
                self, base_model: Any, fixed_tabular: np.ndarray | None
            ) -> None:
                super().__init__()
                self.base_model = base_model
                self.fixed_tabular = fixed_tabular

            def forward(self, image: Any) -> Any:
                if image.ndim == 4:
                    image = image.unsqueeze(1)
                tab = None
                if self.fixed_tabular is not None:
                    tab = torch.from_numpy(self.fixed_tabular).float().to(image.device)
                    tab = tab.expand(image.size(0), -1)
                return self.base_model(image, tab).reshape(-1)

        wrapper = _ImageWrapper(model, reference_tabular)
        wrapper.eval()
        explainer = shap.GradientExplainer(wrapper, background_input)
        shap_values = explainer.shap_values(explain_input)
        values = as_numpy(shap_values)
        if isinstance(values, list):
            values = values[0]

        # Save a single representative heatmap.
        heatmap = np.mean(np.abs(values[0]), axis=0)
        display = _to_display_tile(explain_tiles[0])

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(display)
        ax.imshow(heatmap, cmap="magma", alpha=0.45)
        ax.set_title("Gradient SHAP (image)")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_path
    except Exception as exc:  # pragma: no cover - best effort on optional deps
        log.warning("Raw image SHAP failed: %s", exc)
        return None


def _raw_image_lime_plot(
    model: Any,
    display_tile: np.ndarray,
    reference_tabular: np.ndarray | None,
    output_path: Path,
    seed: int,
) -> Path | None:
    try:
        from lime import lime_image
    except ImportError:
        log.warning("lime is not installed; skipping raw image LIME.")
        return None

    try:
        explainer = lime_image.LimeImageExplainer(random_state=seed)

        def predict_fn(images: np.ndarray) -> np.ndarray:
            batch = np.stack(
                [_display_tile_to_model_tile(img) for img in images], axis=0
            )
            preds = _predict_with_model(
                model,
                image=batch[:, None, ...],
                tabular=_repeat_optional_tabular(reference_tabular, len(batch)),
                reference_tabular=reference_tabular,
            )
            return _scores_to_two_class_probs(preds)

        exp = explainer.explain_instance(
            display_tile.astype(np.float64),
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=300,
        )
        temp, mask = exp.get_image_and_mask(
            exp.top_labels[0], positive_only=True, num_features=10, hide_rest=False
        )
        overlay = _blend_mask(temp / 255.0 if temp.max() > 1.5 else temp, mask)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(display_tile)
        axes[0].set_title("Original tile")
        axes[0].axis("off")
        axes[1].imshow(overlay)
        axes[1].set_title("LIME positive regions")
        axes[1].axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return output_path
    except Exception as exc:  # pragma: no cover - best effort on optional deps
        log.warning("Raw image LIME failed: %s", exc)
        return None


def _raw_image_gradcam_plot(
    model: Any,
    tile: np.ndarray,
    reference_tabular: np.ndarray | None,
    output_path: Path,
) -> Path | None:
    try:
        import torch
    except ImportError:
        log.warning("torch not available; skipping Grad-CAM.")
        return None

    try:
        model_module = getattr(model, "_est", model)
        target_layer = _find_last_conv_layer(model_module)
        if target_layer is None:
            log.warning("No Conv2d layer found; Grad-CAM is not applicable.")
            return None

        activations: dict[str, Any] = {}
        gradients: dict[str, Any] = {}

        def _forward_hook(_: Any, _inp: Any, out: Any) -> None:
            activations["value"] = out.detach()

        def _backward_hook(_: Any, _grad_in: Any, grad_out: Any) -> None:
            gradients["value"] = grad_out[0].detach()

        handle_fwd = target_layer.register_forward_hook(_forward_hook)
        handle_bwd = target_layer.register_full_backward_hook(_backward_hook)

        try:
            image_tensor = torch.from_numpy(tile).float().unsqueeze(0)
            image_tensor = image_tensor.unsqueeze(1)
            tab = None
            if reference_tabular is not None:
                tab = torch.from_numpy(reference_tabular).float().unsqueeze(0)
            output = model(image_tensor, tab).reshape(-1)
            score = output[0]
            model_module.zero_grad(set_to_none=True) if hasattr(
                model_module, "zero_grad"
            ) else None
            score.backward()

            if "value" not in activations or "value" not in gradients:
                return None

            cam = (
                gradients["value"].mean(dim=(2, 3), keepdim=True) * activations["value"]
            )
            cam = cam.mean(dim=1)[0].cpu().numpy()
            cam = np.maximum(cam, 0)
            if np.max(cam) > 0:
                cam = cam / np.max(cam)
            cam = plt.imshow  # silence type checker; replaced immediately below
            del cam

            cam = (
                gradients["value"].mean(dim=(2, 3), keepdim=True) * activations["value"]
            )
            cam = cam.mean(dim=1)[0].cpu().numpy()
            cam = np.maximum(cam, 0)
            if np.max(cam) > 0:
                cam = cam / np.max(cam)

            heatmap = np.interp(
                np.linspace(0, cam.shape[0] - 1, tile.shape[1]),
                np.arange(cam.shape[0]),
                cam.mean(axis=1),
            )
            heatmap = np.tile(heatmap[:, None], (1, tile.shape[2]))
            display = _to_display_tile(tile)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(display)
            ax.imshow(heatmap, cmap="jet", alpha=0.40)
            ax.set_title("Grad-CAM")
            ax.axis("off")
            fig.tight_layout()
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            return output_path
        finally:
            handle_fwd.remove()
            handle_bwd.remove()
    except Exception as exc:  # pragma: no cover - best effort on optional deps
        log.warning("Grad-CAM failed: %s", exc)
        return None


def _image_effect_curves(
    model: Any,
    background_tiles: np.ndarray,
    explain_tiles: np.ndarray,
    reference_tabular: np.ndarray | None,
    output_path: Path,
    seed: int,
) -> Path | None:
    if background_tiles.size == 0:
        return None

    rng = np.random.default_rng(seed)
    scale_grid = np.linspace(0.6, 1.4, 21)
    chosen_tiles = explain_tiles[: min(len(explain_tiles), 5)]

    pdp_values: list[float] = []
    ice_curves: list[np.ndarray] = []

    for scale in scale_grid:
        modified = background_tiles * scale
        preds = _predict_with_model(
            model,
            image=modified[:, None, ...],
            tabular=_repeat_optional_tabular(reference_tabular, len(modified)),
            reference_tabular=reference_tabular,
        )
        pdp_values.append(float(np.mean(preds)))

    for tile in chosen_tiles:
        curve: list[float] = []
        for scale in scale_grid:
            pred = _predict_with_model(
                model,
                image=(tile * scale)[None, None, ...],
                tabular=_repeat_optional_tabular(reference_tabular, 1),
                reference_tabular=reference_tabular,
            )
            curve.append(float(pred[0]))
        ice_curves.append(np.asarray(curve, dtype=np.float32))

    ace_curve = np.cumsum(np.asarray(pdp_values) - np.mean(pdp_values)) / np.arange(
        1, len(scale_grid) + 1
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    for curve in ice_curves:
        ax.plot(scale_grid, curve, color="tab:blue", alpha=0.25, linewidth=1)
    ax.plot(scale_grid, pdp_values, color="tab:orange", linewidth=2.5, label="PDP")
    ax.plot(scale_grid, ace_curve, color="tab:green", linewidth=2, label="ACE")
    ax.set_xlabel("Image intensity scale")
    ax.set_ylabel("Predicted score")
    ax.set_title("Image PDP / ICE / ACE")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Image-feature explainers (pre-extracted embeddings)
# ---------------------------------------------------------------------------


def _explain_image_feature_branch(
    model: Any,
    train_samples: list[PatientSample],
    split_samples: list[PatientSample],
    reference_sample: PatientSample,
    methods: set[str],
    background_size: int,
    max_patients: int,
    seed: int,
    output_dir: Path,
) -> list[ExplainabilityArtifact]:
    # Reuse the tabular explainers on slide-level embeddings.
    output_dir.mkdir(parents=True, exist_ok=True)
    background = _collect_image_feature_matrix(train_samples, max_rows=background_size)
    explain_rows = _collect_image_feature_matrix(split_samples, max_rows=max_patients)
    if background.size == 0 or explain_rows.size == 0:
        log.warning(
            "No image feature vectors available; skipping image-feature branch."
        )
        return []

    feature_names = [f"image_feature_{i}" for i in range(background.shape[1])]
    reference_image = (
        reference_sample.image if reference_sample.image is not None else None
    )
    artifacts: list[ExplainabilityArtifact] = []

    if "shap" in methods:
        path = _tabular_shap_plot(
            model=model,
            background=background,
            explain_rows=explain_rows,
            feature_names=feature_names,
            reference_image=reference_image,
            reference_tabular=reference_sample.tabular,
            output_path=output_dir / "image_features_shap.png",
        )
        if path is not None:
            artifacts.append(
                ExplainabilityArtifact(
                    patient_id=reference_sample.patient_id,
                    modality="image_features",
                    method="shap",
                    path=path,
                    note="SHAP over pre-extracted image features",
                    metadata={"feature_count": int(background.shape[1])},
                )
            )

    if "lime" in methods:
        path = _tabular_lime_plot(
            model=model,
            background=background,
            explain_row=explain_rows[0],
            feature_names=feature_names,
            reference_image=reference_image,
            reference_tabular=reference_sample.tabular,
            output_path=output_dir / "image_features_lime.png",
            seed=seed,
        )
        if path is not None:
            artifacts.append(
                ExplainabilityArtifact(
                    patient_id=reference_sample.patient_id,
                    modality="image_features",
                    method="lime",
                    path=path,
                    note="LIME over pre-extracted image features",
                    metadata={"feature_count": int(background.shape[1])},
                )
            )

    if methods.intersection({"pdp", "ice", "ace"}):
        path = _tabular_effect_curves(
            model=model,
            background=background,
            explain_rows=explain_rows,
            feature_names=feature_names,
            reference_image=reference_image,
            reference_tabular=reference_sample.tabular,
            output_path=output_dir / "image_features_effects.png",
            seed=seed,
        )
        if path is not None:
            artifacts.append(
                ExplainabilityArtifact(
                    patient_id=reference_sample.patient_id,
                    modality="image_features",
                    method="pdp_ice_ace",
                    path=path,
                    note="PDP / ICE / ACE over image features",
                    metadata={"feature_count": int(background.shape[1])},
                )
            )

    return artifacts


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _load_model(cfg: DotDict) -> Any:
    model = get_model(cfg)
    training_cfg = cfg.get("training") or DotDict({})
    save_dir = Path(training_cfg.get("save_dir", "logs/runs/checkpoints"))
    experiment_name = (
        cfg.get("experiment_name")
        or (cfg.get("wandb") or DotDict({})).get("run_name")
        or (cfg.get("model") or DotDict({})).get("type")
        or "experiment"
    )

    candidates = [
        save_dir / f"{experiment_name}_weights.pth",
        save_dir / f"{experiment_name}_weights.pkl",
        save_dir / f"{experiment_name}_best.pth",
        save_dir / f"{experiment_name}_best.pkl",
    ]

    for path in candidates:
        if not path.exists():
            continue
        loaded = load_checkpoint(model, path)
        model = loaded if loaded is not None else model
        log.info("Loaded checkpoint for explainability: %s", path)
        break
    else:
        log.warning(
            "No checkpoint found; explainability will use the current model weights."
        )

    return model


def _collect_samples(loader: Any) -> list[PatientSample]:
    try:
        dataset = loader._dataset
    except AttributeError as exc:  # pragma: no cover - defensive path
        raise RuntimeError(
            "Batch loader does not expose its dataset; cannot build explanations."
        ) from exc
    if dataset is None:
        raise RuntimeError(
            "Batch loader does not expose its dataset; cannot build explanations."
        )
    dataset_size = len(dataset)
    return [dataset[i] for i in range(dataset_size)]


def _collect_tabular_matrix(samples: list[PatientSample], max_rows: int) -> np.ndarray:
    rows = [s.tabular for s in samples if s.tabular is not None]
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    rows = rows[:max_rows]
    return np.asarray(rows, dtype=np.float32)


def _collect_image_feature_matrix(
    samples: list[PatientSample], max_rows: int
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for sample in samples:
        if sample.image is None or sample.image.ndim != 2:
            continue
        rows.append(np.asarray(sample.image, dtype=np.float32).mean(axis=0))
    if not rows:
        return np.empty((0, 0), dtype=np.float32)
    return np.asarray(rows[:max_rows], dtype=np.float32)


def _collect_raw_image_tiles(samples: list[PatientSample], max_rows: int) -> np.ndarray:
    tiles: list[np.ndarray] = []
    for sample in samples:
        if sample.image is None or sample.image.ndim < 4:
            continue
        tiles.append(np.asarray(sample.image[0], dtype=np.float32))
    if not tiles:
        return np.empty((0, 0, 0, 0), dtype=np.float32)
    return np.asarray(tiles[:max_rows], dtype=np.float32)


def _infer_tabular_feature_names(samples: list[PatientSample]) -> list[str]:
    for sample in samples:
        if sample.tabular is not None:
            return [f"feature_{i}" for i in range(len(sample.tabular))]
    return []


def _build_summary(artifacts: list[ExplainabilityArtifact]) -> dict[str, float]:
    modality_counts: dict[str, int] = {}
    method_counts: dict[str, int] = {}
    for artifact in artifacts:
        modality_counts[artifact.modality] = (
            modality_counts.get(artifact.modality, 0) + 1
        )
        method_counts[artifact.method] = method_counts.get(artifact.method, 0) + 1

    summary: dict[str, float] = {
        "artifacts_total": float(len(artifacts)),
        "patients_covered": float(len({a.patient_id for a in artifacts})),
    }
    for key, value in modality_counts.items():
        summary[f"modality_{key}"] = float(value)
    for key, value in method_counts.items():
        summary[f"method_{key}"] = float(value)
    return summary


def _write_report_csv(report: ExplainabilityReport, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["patient_id", "modality", "method", "path", "note"],
        )
        writer.writeheader()
        for artifact in report.artifacts:
            writer.writerow(
                {
                    "patient_id": artifact.patient_id,
                    "modality": artifact.modality,
                    "method": artifact.method,
                    "path": str(artifact.path),
                    "note": artifact.note,
                }
            )
    return output_path


def _predict_with_model(
    model: Any,
    image: np.ndarray | None,
    tabular: np.ndarray | None,
    reference_tabular: np.ndarray | None = None,
) -> np.ndarray:
    """Predict using either sklearn-style or torch-style models."""
    # Fast path for classical tabular models.
    if tabular is not None and hasattr(model, "predict") and image is None:
        return np.asarray(
            model.predict(np.asarray(tabular, dtype=np.float32)), dtype=np.float32
        ).reshape(-1)

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "PyTorch is required for image or multimodal explainability."
        ) from exc

    device = _infer_device(model)
    image_tensor = _to_torch(image, device) if image is not None else None
    tabular_tensor = _to_torch(tabular, device) if tabular is not None else None

    if tabular_tensor is None and reference_tabular is not None:
        tabular_tensor = (
            torch.from_numpy(np.asarray(reference_tabular, dtype=np.float32))
            .float()
            .to(device)
        )
        batch_size = (
            image_tensor.shape[0]
            if image_tensor is not None
            else tabular_tensor.shape[0]
        )
        tabular_tensor = tabular_tensor.expand(batch_size, -1)

    if image_tensor is not None and image_tensor.ndim == 4:
        image_tensor = image_tensor.unsqueeze(1)

    model_eval = model
    if hasattr(model_eval, "eval"):
        model_eval.eval()

    with torch.no_grad():
        preds = model_eval(image_tensor, tabular_tensor)
    return np.asarray(as_numpy(preds), dtype=np.float32).reshape(-1)


def _to_torch(array: np.ndarray, device: Any) -> Any:
    import torch

    return torch.from_numpy(np.asarray(array, dtype=np.float32)).float().to(device)


def _infer_device(model: Any) -> Any:
    try:
        import torch
    except ImportError:
        return "cpu"

    module = getattr(model, "_est", model)
    if hasattr(module, "parameters"):
        try:
            first = next(module.parameters())
            return first.device
        except StopIteration:
            pass
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _repeat_optional_image(
    image: np.ndarray | None, batch_size: int
) -> np.ndarray | None:
    if image is None:
        return None
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 3:
        image = image[None, ...]
    return np.repeat(image[None, ...], batch_size, axis=0)


def _repeat_optional_tabular(
    tabular: np.ndarray | None, batch_size: int
) -> np.ndarray | None:
    if tabular is None:
        return None
    tabular = np.asarray(tabular, dtype=np.float32)
    if tabular.ndim == 1:
        tabular = tabular[None, ...]
    return np.repeat(tabular, batch_size, axis=0)


def _display_tile_to_model_tile(tile_hwc: np.ndarray) -> np.ndarray:
    tile = np.asarray(tile_hwc, dtype=np.float32)
    if tile.max() > 1.5:
        tile = tile / 255.0
    tile = np.clip(tile, 0.0, 1.0)
    tile = (tile - _IMAGE_MEAN) / _IMAGE_STD
    return tile.transpose(2, 0, 1)


def _to_display_tile(tile_chw: np.ndarray) -> np.ndarray:
    tile = np.asarray(tile_chw, dtype=np.float32)
    if tile.ndim != 3:
        raise ValueError(f"Expected CHW tile, got shape {tile.shape}")
    display = tile.transpose(1, 2, 0)
    # If the input already looks normalised, undo the ImageNet transform.
    if display.min() < 0.0 or display.max() > 1.5:
        display = display * _IMAGE_STD + _IMAGE_MEAN
    return np.clip(display, 0.0, 1.0)


def _scores_to_two_class_probs(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    centered = scores - np.median(scores)
    scale = float(np.std(centered) or 1.0)
    positive = 1.0 / (1.0 + np.exp(-(centered / scale)))
    return np.column_stack([1.0 - positive, positive]).astype(np.float32)


def _blend_mask(display_tile: np.ndarray, mask: np.ndarray) -> np.ndarray:
    display_tile = np.asarray(display_tile, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    overlay = display_tile.copy()
    overlay[mask, 0] = np.clip(overlay[mask, 0] + 0.35, 0.0, 1.0)
    overlay[mask, 1:] = overlay[mask, 1:] * 0.65
    return overlay


def _find_last_conv_layer(module: Any) -> Any | None:
    import torch.nn as nn

    last_conv = None
    for submodule in module.modules():
        if isinstance(submodule, nn.Conv2d):
            last_conv = submodule
    return last_conv


# ---------------------------------------------------------------------------
# JSON helper
# ---------------------------------------------------------------------------


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value
