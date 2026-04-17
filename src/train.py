"""
src/train.py — Universal training loop for AIC-4 Zerone.

Responsibilities:
  - Accept a merged DotDict config (from src.config.load_config).
  - Instantiate the model via the registry (src.models.get_model).
  - Run training epochs; log to W&B **only** at validation steps and
    the final summary (not every epoch) to minimise bandwidth.
  - After training, measure efficiency metrics (CPU latency, FLOPs,
    params, model size) and compute the full scoring breakdown.
  - Save the best checkpoint and log its path to W&B.

W&B logging strategy (bandwidth-conscious):
  - Hyperparameters → wandb.config  (once, at run init)
  - Validation metrics → wandb.log  (every `cfg.training.val_every` epochs)
  - Efficiency metrics → wandb.log  (once, in the final summary)
  - run.summary        → final best values for the leaderboard

Usage (called from main.py, not directly):
    from src.config import load_config
    from src.train import train
    cfg = load_config("configs/experiments/siamfc_mobile.yaml")
    train(cfg)
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.config import DotDict
from src.models import get_model
from src.utils import (
    compute_all_scores,
    measure_cpu_latency,
    measure_flops,
    measure_model_size,
    measure_params,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# W&B initialisation helper
# ---------------------------------------------------------------------------

def _init_wandb(cfg: DotDict) -> Any:
    """Initialise a W&B run and return the run object.

    Logs the full merged config as hyperparameters so every run is
    fully reproducible from its W&B page.

    Returns None if W&B is not installed (graceful degradation).
    """
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
        project=wandb_cfg.get("project", "aic4-zerone"),
        entity=wandb_cfg.get("entity", None),    # None = personal account
        name=wandb_cfg.get("run_name", None),    # None = auto-generated
        tags=wandb_cfg.get("tags", []),
        config=cfg.to_dict(),                    # full merged config as hparams
        # Do NOT set `log_code=True` here — notebook outputs are stripped before
        # committing (STANDARDS §1), so code logging would capture nothing useful.
    )

    log.info("W&B run initialised: %s", run.url)
    return run


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(model: Any, path: Path) -> None:
    """Save model weights. Handles both torch and classical-CV models."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import torch  # type: ignore[import]
        if hasattr(model, "state_dict"):
            torch.save(model.state_dict(), path)
            log.info("Checkpoint saved → %s", path)
            return
    except ImportError:
        pass

    # For classical-CV models, use pickle as a fallback.
    import pickle
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    log.info("Checkpoint (pickle) saved → %s", path)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(cfg: DotDict) -> dict[str, float]:
    """Run the full training loop for the given experiment config.

    Args:
        cfg: Merged experiment config (DotDict from load_config).

    Returns:
        Dict of final scores (auc, norm_prec, s_acc, s_eff, final_score, …).
        These are the values you copy into logs/leaderboard.csv.
    """
    # ── 1. Initialise W&B ─────────────────────────────────────────────────
    run = _init_wandb(cfg)

    # ── 2. Instantiate model ───────────────────────────────────────────────
    log.info("Building model: type=%s", cfg.model.type)
    tracker = get_model(cfg)

    # ── 3. Load data ───────────────────────────────────────────────────────
    # Data loading is delegated to src.data_loader to keep this file focused.
    # We import lazily so missing optional deps don't break the import chain.
    from src.data_loader import build_dataloaders  # type: ignore[import]

    train_loader, val_loader = build_dataloaders(cfg)
    log.info(
        "Data loaded — train sequences: %d, val sequences: %d",
        len(train_loader),
        len(val_loader),
    )

    # ── 4. Training loop ───────────────────────────────────────────────────
    training_cfg  = cfg.get("training") or DotDict({})
    num_epochs    = training_cfg.get("epochs", 50)
    val_every     = training_cfg.get("val_every", 5)    # log to W&B every N epochs
    save_dir      = Path(training_cfg.get("save_dir", "logs/runs/checkpoints"))
    experiment_id = cfg.get("experiment_name") or cfg.model.type

    best_final_score: float = -1.0
    best_metrics:     dict[str, float] = {}

    log.info(
        "Starting training — epochs: %d, val_every: %d",
        num_epochs,
        val_every,
    )

    for epoch in range(1, num_epochs + 1):

        # -- 4a. Train one epoch --------------------------------------------
        _train_epoch(tracker, train_loader, cfg, epoch)

        # -- 4b. Validate (and log) at specified intervals -----------------
        if epoch % val_every == 0 or epoch == num_epochs:
            val_metrics = _validate(tracker, val_loader, cfg)

            log.info(
                "Epoch %d/%d — AUC: %.4f | NormPrec: %.4f | S_acc: %.4f",
                epoch, num_epochs,
                val_metrics["auc"],
                val_metrics["norm_prec"],
                val_metrics["s_acc"],
            )

            # Log accuracy metrics to W&B (but NOT efficiency — measured once later)
            if run is not None:
                import wandb  # type: ignore[import]
                wandb.log(
                    {
                        "epoch":     epoch,
                        "val/auc":        val_metrics["auc"],
                        "val/norm_prec":  val_metrics["norm_prec"],
                        "val/s_acc":      val_metrics["s_acc"],
                    },
                    step=epoch,
                )

            # Track best checkpoint
            if val_metrics["s_acc"] > best_final_score:
                best_final_score = val_metrics["s_acc"]
                best_metrics = val_metrics
                ckpt_path = save_dir / f"{experiment_id}_best.pth"
                _save_checkpoint(tracker, ckpt_path)

    # ── 5. Measure efficiency (post-training, CPU-only) ───────────────────
    log.info("Measuring efficiency metrics on CPU…")

    # Get a representative frame from the validation set for latency timing
    sample_frame = _get_sample_frame(val_loader)

    latency_ms = measure_cpu_latency(tracker, sample_frame)
    flops_g    = measure_flops(getattr(tracker, "model", tracker))
    params_m   = measure_params(getattr(tracker, "model", tracker))
    size_gb    = measure_model_size(save_dir / f"{experiment_id}_best.pth")

    log.info(
        "Efficiency — Latency: %.1f ms | FLOPs: %.2f G | Params: %.2f M | Size: %.3f GB",
        latency_ms, flops_g, params_m, size_gb,
    )

    # ── 6. Compute final scores ────────────────────────────────────────────
    final_scores = compute_all_scores(
        auc        = best_metrics.get("auc", 0.0),
        norm_prec  = best_metrics.get("norm_prec", 0.0),
        flops_g    = flops_g,
        params_m   = params_m,
        latency_ms = latency_ms,
        size_gb    = size_gb,
    )

    log.info(
        "Final Score: %.4f (S_acc=%.4f, S_eff=%.4f)",
        final_scores["final_score"],
        final_scores["s_acc"],
        final_scores["s_eff"],
    )

    # ── 7. Log final summary to W&B ────────────────────────────────────────
    if run is not None:
        import wandb  # type: ignore[import]

        # Log all efficiency + final metrics in one call
        wandb.log(
            {
                "efficiency/latency_ms": latency_ms,
                "efficiency/flops_g":    flops_g,
                "efficiency/params_m":   params_m,
                "efficiency/size_gb":    size_gb,
                "efficiency/s_eff":      final_scores["s_eff"],
                "final/s_acc":           final_scores["s_acc"],
                "final/final_score":     final_scores["final_score"],
                "checkpoint":            str(save_dir / f"{experiment_id}_best.pth"),
            }
        )

        # Also write to run.summary for the W&B leaderboard table
        run.summary.update(
            {k: v for k, v in final_scores.items()}
        )

        run.finish()
        log.info("W&B run finished.")

    return final_scores


# ---------------------------------------------------------------------------
# Epoch-level helpers (stubs — real logic lives in data_loader / model)
# ---------------------------------------------------------------------------

def _train_epoch(
    tracker: Any,
    train_loader: Any,
    cfg: DotDict,
    epoch: int,
) -> None:
    """Train for one epoch.

    The actual gradient update logic lives in the tracker / model class
    (e.g. SiamFCTracker.train_step). This function is the per-epoch driver.
    Classical trackers (CSRT, KCF) are online-only and have no training loop;
    for them this is a no-op.
    """
    if not hasattr(tracker, "train_step"):
        return  # Classical tracker — no training phase

    tracker.train()  # Switch to train mode if torch model
    for batch in train_loader:
        frames, bboxes = batch
        tracker.train_step(frames, bboxes, cfg)


def _validate(
    tracker: Any,
    val_loader: Any,
    cfg: DotDict,
) -> dict[str, float]:
    """Run one pass of validation and return accuracy metrics.

    Returns a dict with keys: auc, norm_prec, s_acc.
    """
    from src.utils import success_auc, norm_precision, compute_s_acc  # local import avoids circular

    all_preds: list[tuple[float, float, float, float]] = []
    all_gts:   list[tuple[float, float, float, float]] = []

    eval_mode = hasattr(tracker, "eval")
    if eval_mode:
        tracker.eval()

    for sequence in val_loader:
        frames, gts = sequence
        # Initialise on frame 0
        tracker.init(frames[0], gts[0])
        for frame, gt in zip(frames[1:], gts[1:]):
            pred = tracker.update(frame)
            all_preds.append(pred)
            all_gts.append(gt)

    auc       = success_auc(all_preds, all_gts)
    norm_prec = norm_precision(all_preds, all_gts)
    s_acc     = compute_s_acc(auc, norm_prec)

    return {"auc": auc, "norm_prec": norm_prec, "s_acc": s_acc}


def _get_sample_frame(val_loader: Any) -> "np.ndarray":
    """Return a single representative frame for latency benchmarking."""
    import numpy as np
    try:
        sequence = next(iter(val_loader))
        frames, _ = sequence
        return frames[0]
    except Exception:
        # Graceful fallback: black 480×640 frame
        return np.zeros((480, 640, 3), dtype=np.uint8)