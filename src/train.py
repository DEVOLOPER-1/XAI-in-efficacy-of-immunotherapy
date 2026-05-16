"""
src/train.py — Multimodal Cancer Research training loop for WhiteBox.

This file handles:
- Model instantiation via registry (src.models.get_model)
- Training loops for neural (PyTorch) and tree-based (sklearn/xgboost) models
- Mixed-modality support: models receive (image, tabular) where either may be None
- W&B logging: config at init, metrics at validation steps, summary at end
- Checkpointing: save best model by validation Huber loss

Design principles:
- Modality-agnostic: all tensor conversions guard against None inputs
- Numerically stable: target scaling, gradient clipping, NaN guards
- Reproducible: seeds, deterministic splits, config-driven hyperparameters
"""

from __future__ import annotations

from datetime import datetime
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.config import DotDict
from src.models import get_model
from src.utils import compute_all_metrics, save_checkpoint

log = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────────────
# W&B initialisation
# ───────────────────────────────────────────────────────────────────────────
def _init_wandb(cfg: DotDict) -> Any:
    """Initialise W&B run and log the full merged config as hyperparameters."""
    try:
        import wandb  # type: ignore[import]
    except ImportError:
        log.warning(
            "wandb not installed — metrics will only print to stdout. "
            "Install with: uv pip install wandb"
        )
        return None

    wandb_cfg = cfg.get("wandb") or DotDict({})
    model_cfg = cfg.get("model") or DotDict({})
    project_name = wandb_cfg.get("project") or wandb_cfg.get(
        "project_name", "TMB-prediction"
    )

    run = wandb.init(
        project=project_name,
        entity=wandb_cfg.get("entity", "mohamed-mourad-zewail-city"),
        name=wandb_cfg.get(
            "run_name", f"{datetime.now()}-{model_cfg.get('type', 'model')}"
        ),
        tags=wandb_cfg.get("tags", []) + ["success"],
        config=cfg.to_dict(),  # full merged config as hyperparameters — reproducible
    )
    log.info("W&B run initialised: %s", run.url)
    return run


# ───────────────────────────────────────────────────────────────────────────
# Backend detection helper
# ───────────────────────────────────────────────────────────────────────────
def _is_tree_model(cfg: DotDict) -> bool:
    """Return True for tree-based models that use fit() instead of forward()."""
    tree_types = {
        "xgboost",
        "catboost",
        "decision_tree",
        "random_forest",
        "lasso_regressor",
        "gradient_boosted",
    }
    model_cfg = cfg.get("model") or DotDict({})
    return (model_cfg.get("type") or "").lower().strip() in tree_types


# ───────────────────────────────────────────────────────────────────────────
# Main training entry point
# ───────────────────────────────────────────────────────────────────────────
def train(cfg: DotDict) -> dict[str, float]:
    """Run the full training loop for the given experiment config."""
    # ── 1. W&B ───────────────────────────────────────────────────────────────
    run = _init_wandb(cfg)

    # ── 2. Model ─────────────────────────────────────────────────────────────
    model_cfg = cfg.get("model") or DotDict({})
    log.info(
        "Building model — category: %s | type: %s",
        model_cfg.get("category", "?"),
        model_cfg.get("type", "?"),
    )
    model = get_model(cfg)

    # ── 3. Data ───────────────────────────────────────────────────────────────
    from src.data_loader import build_dataloaders

    train_loader, val_loader, _ = build_dataloaders(cfg)
    log.info(
        "Data ready — train: %d batches | val: %d batches",
        len(train_loader),
        len(val_loader),
    )

    # ── 4. Dispatch to correct training regime ───────────────────────────────
    if _is_tree_model(cfg):
        best_metrics = _train_tree(model, train_loader, val_loader, cfg, run)
    else:
        best_metrics = _train_neural(model, train_loader, val_loader, cfg, run)

    # ── 5. Final W&B summary ─────────────────────────────────────────────────
    if run is not None:
        run.summary.update(best_metrics)
        run.finish()
        log.info("W&B run finished.")

    return best_metrics


# ───────────────────────────────────────────────────────────────────────────
# Helper: Target scaling/unscaling (for regression stability)
# ───────────────────────────────────────────────────────────────────────────
def _transform_regression_values(
    values: torch.Tensor,
    cfg: DotDict,
    inverse: bool = False,
) -> torch.Tensor:
    """
    Transform/untransform regression values using config stats for stable training.

    Args:
        values: (B,) tensor of raw regression values
        cfg: experiment config with model.target_mean/std and dataset.log1p_target
        inverse: if True, reverse the transform (loss-space → original scale)
    Returns:
        transformed or untransformed values
    """
    ds_cfg = cfg.get("dataset") or DotDict({})
    model_cfg = cfg.get("model") or DotDict({})
    log1p_target = bool(ds_cfg.get("log1p_target", False))
    target_scale = bool(model_cfg.get("target_scale", False))
    mean = float(model_cfg.get("target_mean", 0.0))
    std = float(model_cfg.get("target_std", 1.0))

    if inverse:
        if target_scale:
            if std < 1e-6:
                log.warning("target_std=%.6f is too small; skipping unscaling.", std)
            else:
                values = values * std + mean
        if log1p_target:
            values = torch.expm1(values).clamp(min=0.0)
        return values

    if log1p_target:
        values = torch.log1p(values.clamp(min=0.0))

    if target_scale:
        if std < 1e-6:
            log.warning("target_std=%.6f is too small; skipping scaling.", std)
            return values
        values = (values - mean) / std

    return values


# ───────────────────────────────────────────────────────────────────────────
# Helper: Numpy → torch tensor (with None handling)
# ───────────────────────────────────────────────────────────────────────────
def _to_tensor(array: np.ndarray | None) -> torch.Tensor | None:
    """Convert numpy array to float32 torch tensor. Pass-through for None."""
    if array is None:
        return None
    try:
        return torch.from_numpy(array).float()
    except Exception as e:
        log.warning("Failed to convert array to tensor: %s", e)
        return None


# ───────────────────────────────────────────────────────────────────────────
# Neural network training (PyTorch gradient descent)
# ───────────────────────────────────────────────────────────────────────────
def _train_neural(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    cfg: DotDict,
    run: Any,
) -> dict[str, float]:
    """Train a PyTorch model with AdamW and Huber loss, with full modality support."""
    import torch.nn as nn

    training_cfg = cfg.get("training") or DotDict({})
    ds_cfg = cfg.get("dataset") or DotDict({})
    model_cfg_inner = cfg.get("model") or DotDict({})

    # ── Hyperparameters with safe type casting ─────────────────────────────
    num_epochs = int(training_cfg.get("epochs", 100))
    val_every = int(training_cfg.get("val_every", 5))

    # Learning rate: prefer training.lr, fallback to model.learning_rate, default 1e-4
    lr_raw = training_cfg.get("lr") or model_cfg_inner.get("learning_rate", 1e-4)
    lr = float(lr_raw) if lr_raw is not None else 1e-4

    weight_decay_raw = training_cfg.get("weight_decay", 1e-4)
    weight_decay = float(weight_decay_raw) if weight_decay_raw is not None else 1e-4

    grad_clip_raw = training_cfg.get("grad_clip", None)
    grad_clip = float(grad_clip_raw) if grad_clip_raw is not None else None

    log1p_target = bool(ds_cfg.get("log1p_target", False))

    save_dir = Path(training_cfg.get("save_dir", "logs/runs/checkpoints"))
    experiment_id = cfg.get("experiment_name") or "experiment"
    wandb_cfg = cfg.get("wandb") or DotDict({})
    run_name = wandb_cfg.get("run_name") or experiment_id
    threshold = training_cfg.get("risk_threshold", None)
    save_path = save_dir / f"{run_name}_weights.pth"

    # Logging
    if log1p_target:
        log.info("log1p_target=True — targets will be log1p-transformed.")
    if model_cfg_inner.get("target_scale", False):
        log.info(
            "target_scale=True — targets standardized (mean=%.2f, std=%.2f).",
            model_cfg_inner.get("target_mean", 0.0),
            model_cfg_inner.get("target_std", 1.0),
        )

    # ── Optimiser & loss ───────────────────────────────────────────────────
    # Differential LR: backbone gets lr × backbone_lr_factor (much smaller, to
    # preserve ImageNet weights), head gets the full lr (random init needs more signal).
    # Falls back to single LR for models that don't expose get_param_groups().
    backbone_lr_factor_raw = training_cfg.get("backbone_lr_factor", None)
    backbone_lr_factor = (
        float(backbone_lr_factor_raw) if backbone_lr_factor_raw is not None else None
    )

    if backbone_lr_factor is not None and hasattr(model, "get_param_groups"):
        param_groups = model.get_param_groups(
            head_lr=lr, backbone_lr_factor=backbone_lr_factor
        )
        log.info(
            "Differential LR — backbone: %.2e | head: %.2e",
            lr * backbone_lr_factor,
            lr,
        )
    else:
        param_groups = filter(lambda p: p.requires_grad, model._est.parameters())

    optimiser = torch.optim.AdamW(
        param_groups,
        lr=lr,
        weight_decay=weight_decay,
    )
    # delta=1.0 matches utils.py huber_loss(delta=1.0) so train_loss and val_huber
    # are directly comparable in the logs and W&B charts.
    criterion = nn.HuberLoss(delta=1.0)

    # ── LR Scheduler ──────────────────────────────────────────────────
    sched_name = training_cfg.get("scheduler", None)
    scheduler = None
    if sched_name == "ReduceLROnPlateau":
        sched_patience = int(training_cfg.get("scheduler_patience", 5))
        sched_factor = float(training_cfg.get("scheduler_factor", 0.5))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser,
            mode="min",
            patience=sched_patience,
            factor=sched_factor,
            min_lr=1e-6,
        )
        log.info(
            "LR scheduler: ReduceLROnPlateau (patience=%d, factor=%.2f)",
            sched_patience,
            sched_factor,
        )

    # ── AMP (Automatic Mixed Precision) ────────────────────────────────────
    # float16 activations halve the per-tile memory cost of InceptionV3 forward
    # passes; GradScaler compensates for reduced float16 dynamic range.
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        log.info("AMP enabled — using float16 activations to reduce GPU memory.")

    best_huber: float = float("inf")
    best_metrics: dict[str, float] = {}

    log.info(
        "Neural training — epochs: %d | val_every: %d | lr: %g | grad_clip: %s",
        num_epochs,
        val_every,
        lr,
        grad_clip,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    log.info("Using device: %s", device)

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_losses: list[float] = []

        for batch in train_loader:
            # ── Modality-safe tensor conversion ───────────────────────────
            image = (
                _to_tensor(batch["image"]).to(device, non_blocking=True)
                if batch["image"] is not None
                else None
            )
            tabular = (
                _to_tensor(batch["tabular"]).to(device, non_blocking=True)
                if batch["tabular"] is not None
                else None
            )
            target = _to_tensor(batch["target"])
            if target is None:
                continue
            target = target.to(device, non_blocking=True)

            # ── Optional log1p transform (stabilises heavy-tailed targets) ─
            if log1p_target:
                target = torch.log1p(target.clamp(min=0.0))

            # ── Skip batches with all-NaN targets ─────────────────────────
            valid = ~torch.isnan(target)
            if valid.sum() == 0:
                continue

            optimiser.zero_grad(set_to_none=True)
            # ── Forward pass (inside AMP autocast) ────────────────────────
            try:
                with torch.amp.autocast("cuda", enabled=use_amp):
                    preds = model(image, tabular).view(-1)
                if preds is None:
                    log.warning("Model returned None — skipping batch.")
                    continue
            except Exception as e:
                log.warning("Forward pass failed: %s — skipping batch.", e)
                continue

            # ── Numerical guards on predictions ───────────────────────────
            if not torch.isfinite(preds).all():
                log.warning("Non-finite predictions detected — skipping batch.")
                continue

            # ── Scale targets for stable loss computation ─────────────────
            transformed_target = _transform_regression_values(target, cfg)
            transformed_preds = _transform_regression_values(preds, cfg)

            # ── Loss on valid, scaled targets (float32) ─────────────────────
            loss = criterion(transformed_preds[valid], transformed_target[valid])

            # ── Guard against NaN/Inf loss ────────────────────────────────
            if not torch.isfinite(loss):
                log.warning("Epoch %d: non-finite loss — skipping batch.", epoch)
                continue

            # ── Backward + optimisation (AMP-aware) ───────────────────────
            scaler.scale(loss).backward()

            # Gradient clipping (unscale first so clip threshold is in true scale)
            if grad_clip is not None:
                scaler.unscale_(optimiser)
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model._est.parameters()),
                    grad_clip,
                )

            scaler.step(optimiser)
            scaler.update()
            epoch_losses.append(loss.item())

        # ── Epoch summary ─────────────────────────────────────────────────
        mean_loss = np.mean(epoch_losses) if epoch_losses else float("nan")

        # ── Validation at configured intervals ────────────────────────────
        if epoch % val_every == 0 or epoch == num_epochs:
            metrics = _neural_validate(model, val_loader, cfg, threshold=threshold)

            log.info(
                "Epoch %d/%d — train_loss: %.4f | val_huber: %.4f | R²: %.4f | lr: %.2e",
                epoch,
                num_epochs,
                mean_loss,
                metrics["huber"],
                metrics["r2"],
                optimiser.param_groups[0]["lr"],
            )

            # ── LR scheduler step (on val_huber) ──────────────────────────
            if scheduler is not None:
                scheduler.step(metrics["huber"])

            # ── W&B logging ───────────────────────────────────────────────
            if run is not None:
                import wandb

                wandb.log(
                    {
                        "epoch": epoch,
                        "train/loss": mean_loss,
                        **{f"val/{k}": v for k, v in metrics.items()},
                    },
                    step=epoch,
                )

            # ── Checkpointing ─────────────────────────────────────────────
            if metrics["huber"] < best_huber:
                best_huber = metrics["huber"]
                best_metrics = metrics
                save_checkpoint(model, save_path)
                log.info("  ↳ New best Huber=%.4f — checkpoint saved.", best_huber)

    # ── Optional model artifact upload ───────────────────────────────────
    if training_cfg.get("upload_pickled_model", False) and run is not None:
        import wandb

        artifact = wandb.Artifact(name="run_weights", type="model")
        artifact.add_file(str(save_path))
        run.log_artifact(artifact)

    return best_metrics


# ───────────────────────────────────────────────────────────────────────────
# Neural validation loop (with target unscaling for metrics)
# ───────────────────────────────────────────────────────────────────────────
def _neural_validate(
    model: Any,
    loader: Any,
    cfg: DotDict,
    threshold: float | None = None,
) -> dict[str, float]:
    """Validate a neural model and return all medical metrics."""
    import torch

    device = next(model.parameters()).device
    model.eval()

    all_preds: list[float] = []
    all_targets: list[float] = []

    with torch.no_grad():
        for batch in loader:
            # ── Modality-safe tensor conversion ───────────────────────────
            image = (
                _to_tensor(batch["image"]).to(device, non_blocking=True)
                if batch["image"] is not None
                else None
            )
            tabular = (
                _to_tensor(batch["tabular"]).to(device, non_blocking=True)
                if batch["tabular"] is not None
                else None
            )
            targets = batch["target"]  # Keep as numpy for metric computation

            # ── Forward pass ──────────────────────────────────────────────
            try:
                raw_preds = model(image, tabular).view(-1)

                # Unscale the log-predictions back to real-world TMB space!
                preds = _transform_regression_values(raw_preds, cfg, inverse=True)
                preds = preds.cpu().numpy()  # Perfectly safe here in validation!

            except Exception:
                continue  # Skip problematic batches

            # ── Collect finite predictions/targets ────────────────────────
            for p, t in zip(np.asarray(preds), np.asarray(targets)):
                if np.isfinite(p) and np.isfinite(t):
                    all_preds.append(float(p))
                    all_targets.append(float(t))

    # ── Compute metrics (handles empty lists gracefully) ─────────────────
    if not all_preds:
        log.warning("No valid predictions collected for validation.")
        return {"huber": float("inf"), "r2": float("nan"), "rmse": float("nan")}

    return compute_all_metrics(all_targets, all_preds, threshold=threshold)


# ───────────────────────────────────────────────────────────────────────────
# Tree-based model training (XGBoost / CatBoost / sklearn)
# ───────────────────────────────────────────────────────────────────────────
def _train_tree(
    model: Any,
    train_loader: Any,
    val_loader: Any,
    cfg: DotDict,
    run: Any,
) -> dict[str, float]:
    """Accumulate all tabular batches and call model.fit() in one shot."""
    log.info("Tree model training — collecting batches…")

    X_train, y_train = _collect_tabular(train_loader)
    X_val, y_val = _collect_tabular(val_loader)

    if X_train is None or y_train is None:
        raise RuntimeError(
            "No tabular data found for tree model. "
            "Verify cfg.modalities.tabular: true and that clinical.csv exists."
        )

    log.info("Fitting tree model on %d training patients…", len(y_train))
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    metrics = compute_all_metrics(y_val.tolist(), preds.tolist())

    log.info(
        "Tree model — RMSE: %.4f | R²: %.4f | C-Index: %.4f",
        metrics["rmse"],
        metrics["r2"],
        metrics["c_index"],
    )

    if hasattr(model, "print_feature_importances") and callable(
        model.print_feature_importances
    ):
        model_cfg = cfg.get("model") or DotDict({})
        model.print_feature_importances(
            feature_names=train_loader.feature_names,
            top_n=model_cfg.get("select_top_k"),
        )

    if run is not None:
        import wandb

        wandb.log({f"val/{k}": v for k, v in metrics.items()})

        # Save checkpoint
        training_cfg = cfg.get("training") or DotDict({})
        wandb_cfg = cfg.get("wandb") or DotDict({})
        save_dir = Path(training_cfg.get("save_dir", "logs/runs/checkpoints"))
        run_name = wandb_cfg.get("run_name") or "experiment"
        save_checkpoint(model, save_dir / f"{run_name}_weights.pkl")

        if training_cfg.get("upload_pickled_model", False):
            artifact = wandb.Artifact(name=f"{run_name}_weights", type="model")
            artifact.add_file(str(save_dir / f"{run_name}_weights.pkl"))
            run.log_artifact(artifact)

    return metrics


def _collect_tabular(
    loader: Any,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Drain a batch loader and stack all tabular arrays + targets."""
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    skipped_targets = 0

    for batch in loader:
        if batch["tabular"] is None:
            continue

        X_batch = np.asarray(batch["tabular"])
        y_batch = np.asarray(batch["target"], dtype=np.float32).reshape(-1)

        valid_mask = ~np.isnan(y_batch)
        skipped_targets += int((~valid_mask).sum())

        if not np.any(valid_mask):
            continue

        X_parts.append(X_batch[valid_mask])
        y_parts.append(y_batch[valid_mask])

    if skipped_targets:
        log.warning(
            "Skipped %d tabular rows with NaN targets while collecting data.",
            skipped_targets,
        )

    if not X_parts:
        return None, None

    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0).astype(
        np.float32
    )
