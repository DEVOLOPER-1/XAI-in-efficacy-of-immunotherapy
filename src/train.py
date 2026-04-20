"""
src/train.py — Multimodal Cancer Research training loop for WhiteBox.

Responsibilities:
  - Instantiate model via the registry (src.models.get_model).
  - Run training epochs over batched PatientSample dicts from data_loader.
  - Handle mixed inputs:  outputs = model(batch["image"], batch["tabular"])
  - Support tree-based models (XGBoost / CatBoost) that use fit() rather
    than gradient-descent epochs.
  - Log to W&B: config once at init; medical metrics at validation steps;
    final summary at end.
  - Save the best checkpoint based on validation RMSE.

Hardware efficiency metrics (latency, FLOPs, model size) are NOT computed
here — they are irrelevant to the cancer research task.

W&B logging strategy (bandwidth-conscious, per STANDARDS):
  - Hyperparameters → wandb.config   (once, at run init)
  - Validation metrics → wandb.log   (every cfg.training.val_every epochs)
  - Final summary → run.summary      (best values for leaderboard table)

Usage (called from main.py):
    from src.config import load_config
    from src.train import train
    cfg = load_config("configs/experiments/fusion_early.yaml")
    train(cfg)
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.config import DotDict
from src.models import get_model
from src.utils import compute_all_metrics, save_checkpoint

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# W&B initialisation (unchanged from original standards)
# ---------------------------------------------------------------------------

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
    run = wandb.init(
        project=wandb_cfg.get("project", "TMB-prediction"),
        entity=wandb_cfg.get("entity",   "mohamed-mourad-zewail-city"),
        name=wandb_cfg.get("run_name",
                           f"{datetime.now()}-{cfg.model.type}"),
        tags=wandb_cfg.get("tags",       []) + ["success",],
        config=cfg.to_dict(),   # full merged config as hyperparameters — reproducible
    )
    log.info("W&B run initialised: %s", run.url)
    return run


# ---------------------------------------------------------------------------
# Backend detection helper
# ---------------------------------------------------------------------------

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
    model_cfg  = cfg.get("model") or DotDict({})
    return (model_cfg.get("type") or "").lower() in tree_types


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train(cfg: DotDict) -> dict[str, float]:
    """Run the full training loop for the given experiment config.

    Args:
        cfg: Merged experiment config (DotDict from load_config).

    Returns:
        Dict of best validation metrics: rmse, mae, r2, pearson_r, c_index.
        Copy these values into logs/leaderboard.csv.
    """
    # ── 1. W&B ───────────────────────────────────────────────────────────────
    run = _init_wandb(cfg)

    # ── 2. Model ─────────────────────────────────────────────────────────────
    model_cfg = cfg.get("model") or DotDict({})
    log.info(
        "Building model — category: %s | type: %s",
        model_cfg.get("category", "?"),
        model_cfg.get("type",     "?"),
    )
    model = get_model(cfg)

    # ── 3. Data ───────────────────────────────────────────────────────────────
    from src.data_loader import build_dataloaders
    train_loader, val_loader = build_dataloaders(cfg)
    log.info(
        "Data ready — train: %d batches | val: %d batches",
        len(train_loader), len(val_loader),
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


# ---------------------------------------------------------------------------
# Neural network training (PyTorch gradient descent)
# ---------------------------------------------------------------------------

def _train_neural(
    model:        Any,
    train_loader: Any,
    val_loader:   Any,
    cfg:          DotDict,
    run:          Any,
) -> dict[str, float]:
    """Train a PyTorch model with AdamW and MSE loss."""
    import torch
    import torch.nn as nn

    training_cfg  = cfg.get("training") or DotDict({})
    num_epochs    = training_cfg.get("epochs",       100)
    val_every     = training_cfg.get("val_every",      5)
    lr            = training_cfg.get("lr",           1e-3)
    weight_decay  = training_cfg.get("weight_decay", 1e-4)
    save_dir      = Path(training_cfg.get("save_dir", "logs/runs/checkpoints"))
    experiment_id = cfg.get("experiment_name") or "experiment"
    threshold     = training_cfg.get("risk_threshold", None)

    optimiser = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay,
    )
    criterion = nn.MSELoss()

    best_rmse:     float             = float("inf")
    best_metrics:  dict[str, float]  = {}
    ckpt_path = save_dir / f"{experiment_id}_best.pth"

    log.info("Neural training — epochs: %d | val_every: %d | lr: %g", num_epochs, val_every, lr)

    for epoch in range(1, num_epochs + 1):

        # -- Train one epoch -------------------------------------------------
        model.train()
        epoch_losses: list[float] = []

        for batch in train_loader:
            image   = _to_tensor(batch["image"])
            tabular = _to_tensor(batch["tabular"])
            target  = _to_tensor(batch["target"])

            # Skip batches where every target is NaN (test/unlabelled patients)
            valid   = ~torch.isnan(target)
            if valid.sum() == 0:
                continue

            optimiser.zero_grad()
            # Mixed-input forward pass: model handles None modalities internally
            preds = model(image, tabular).squeeze(-1)          # (B,)
            loss  = criterion(preds[valid], target[valid])
            loss.backward()
            optimiser.step()
            epoch_losses.append(loss.item())

        mean_loss = np.mean(epoch_losses) if epoch_losses else float("nan")

        # -- Validate (and log) at configured intervals ----------------------
        if epoch % val_every == 0 or epoch == num_epochs:
            metrics = _neural_validate(model, val_loader, threshold=threshold)

            log.info(
                "Epoch %d/%d — train_loss: %.4f | RMSE: %.4f | R²: %.4f | C-Index: %.4f",
                epoch, num_epochs, mean_loss,
                metrics["rmse"], metrics["r2"], metrics["c_index"],
            )

            # Log to W&B only at validation steps (bandwidth-conscious)
            if run is not None:
                import wandb
                wandb.log(
                    {"epoch": epoch, "train/loss": mean_loss,
                     **{f"val/{k}": v for k, v in metrics.items()}},
                    step=epoch,
                )

            if metrics["rmse"] < best_rmse:
                best_rmse    = metrics["rmse"]
                best_metrics = metrics
                save_checkpoint(model, ckpt_path)
                log.info("  ↳ New best RMSE=%.4f — checkpoint saved.", best_rmse)

    return best_metrics


def _neural_validate(
    model:     Any,
    loader:    Any,
    threshold: float | None = None,
) -> dict[str, float]:
    """Validate a neural model and return all medical metrics."""
    import torch

    model.eval()
    all_preds:   list[float] = []
    all_targets: list[float] = []

    with torch.no_grad():
        for batch in loader:
            image   = _to_tensor(batch["image"])
            tabular = _to_tensor(batch["tabular"])
            targets = batch["target"]   # numpy (B,), may contain NaN

            preds = model(image, tabular).squeeze(-1).cpu().numpy()

            for p, t in zip(preds, targets):
                if not np.isnan(t):
                    all_preds.append(float(p))
                    all_targets.append(float(t))

    return compute_all_metrics(all_targets, all_preds, threshold=threshold)


def _to_tensor(array: "np.ndarray | None") -> "Any":
    """Convert numpy array to float32 torch tensor. Pass-through for None."""
    if array is None:
        return None
    try:
        import torch
        return torch.from_numpy(array).float()
    except ImportError:
        return array


# ---------------------------------------------------------------------------
# Tree-based model training (XGBoost / CatBoost)
# ---------------------------------------------------------------------------

def _train_tree(
    model:        Any,
    train_loader: Any,
    val_loader:   Any,
    cfg:          DotDict,
    run:          Any,
) -> dict[str, float]:
    """Accumulate all tabular batches and call model.fit() in one shot.

    Tree models do not iterate epochs — the loader is drained once into
    a single (N, F) matrix.
    """
    log.info("Tree model training — collecting batches…")

    X_train, y_train = _collect_tabular(train_loader)
    X_val,   y_val   = _collect_tabular(val_loader)

    if X_train is None or y_train is None:
        raise RuntimeError(
            "No tabular data found for tree model. "
            "Verify cfg.modalities.tabular: true and that clinical.csv exists."
        )

    log.info("Fitting tree model on %d training patients…", len(y_train))
    model.fit(X_train, y_train)

    preds   = model.predict(X_val)
    metrics = compute_all_metrics(y_val.tolist(), preds.tolist())

    log.info(
        "Tree model — RMSE: %.4f | R²: %.4f | C-Index: %.4f",
        metrics["rmse"], metrics["r2"], metrics["c_index"],
    )

    if hasattr(model, "print_feature_importances") and callable(model.print_feature_importances):
        model.print_feature_importances(
            feature_names=train_loader.feature_names,
            top_n=cfg.model.select_top_k,
        )
    else:
        log.debug("Model does not implement print_feature_importances; skipping.")

    if run is not None:
        import wandb
        wandb.log({f"val/{k}": v for k, v in metrics.items()})

        # Save checkpoint
        training_cfg  = cfg.get("training") or DotDict({})
        save_dir      = Path(training_cfg.get("save_dir", "logs/runs/checkpoints"))
        run_name = cfg.wandb.get("run_name") or "experiment"
        save_checkpoint(model, save_dir / f"{run_name}_weights.pkl")

        if (training_cfg.get('upload_pickeled_model', False)):
            artifact = wandb.Artifact(name=f"{run_name}_weights", type="model")
            artifact.add_file(str(save_dir / f"{run_name}_weights.pkl"))
            run.log_artifact(artifact)


    return metrics


def _collect_tabular(
    loader: Any,
) -> tuple["np.ndarray | None", "np.ndarray | None"]:
    """Drain a batch loader and stack all tabular arrays + targets.

    Rows with NaN targets are skipped so missing labels are not turned into
    real ``0.0`` targets for training or validation.
    """
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
            "Skipped %d tabular rows with NaN targets while collecting data for tree model.",
            skipped_targets,
        )

    if not X_parts:
        return None, None

    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0).astype(np.float32)