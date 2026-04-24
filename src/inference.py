"""
src/inference.py — Online-only tracker evaluation for AIC-4 Zerone.

Competition rules (from README §1):
  - Tracking mode: ONLINE ONLY — no access to future frames.
  - No re-initialisation after frame 0.
  - Initialisation: ground-truth bbox given in frame 0 only.

This module handles two modes:
  1. eval   — runs on sequences with ground-truth annotations and
               returns accuracy + efficiency metrics.
  2. predict — runs on unannotated public-LB sequences and writes
               submission.csv in the Kaggle-required format.

Usage (called from main.py, not directly):
    from src.config import load_config
    from src.inference import evaluate, predict

    cfg = load_config("configs/experiments/random_forest.yaml")
    metrics = evaluate(cfg, split="val")
    predict(cfg, split="public_lb", output_path="submission.csv")
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.config import DotDict
from src.models import get_model
from src.utils import (
    compute_all_scores,
    compute_s_acc,
    measure_cpu_latency,
    measure_flops,
    measure_model_size,
    measure_params,
    norm_precision,
    success_auc,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core online-tracking engine
# ---------------------------------------------------------------------------


def _run_sequence_online(
    tracker: Any,
    frames: list["np.ndarray"],
    init_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int, int, int]]:
    """Process a single sequence in strict online fashion.

    Rules enforced here:
    - tracker.init() is called exactly once, on frame 0.
    - tracker.update() is called for frames 1..N in order.
    - The tracker NEVER receives frame k+1 before processing frame k.
    - No future frames are buffered or provided.

    Args:
        tracker:   An initialised tracker (TrackerProtocol).
        frames:    List of BGR numpy frames for the sequence.
        init_bbox: Ground-truth (x, y, w, h) for frame 0.

    Returns:
        List of predicted (x, y, w, h) bboxes for frames 1..N.
        (Frame 0 uses the GT bbox, so it is not included.)
    """
    if not frames:
        return []

    # Initialise ONCE on frame 0 with the ground-truth bbox
    tracker.init(frames[0], init_bbox)

    predictions: list[tuple[int, int, int, int]] = []

    # Process frames 1..N — strictly one at a time, no lookahead
    for frame_idx in range(1, len(frames)):
        pred_bbox = tracker.update(frames[frame_idx])
        predictions.append(pred_bbox)

    return predictions


# ---------------------------------------------------------------------------
# Evaluation (annotated sequences)
# ---------------------------------------------------------------------------


def evaluate(cfg: DotDict, split: str = "val") -> dict[str, float]:
    """Run online evaluation on annotated sequences and compute all scores.

    Args:
        cfg:   Merged experiment config.
        split: Which data split to evaluate on ("val" or "train").

    Returns:
        Dict with keys: auc, norm_prec, s_acc, flops_g, params_m,
        latency_ms, size_gb, s_eff, final_score.
        These values map directly to leaderboard.csv columns.
    """
    log.info("Starting evaluation — split: %s, model: %s", split, cfg.model.type)

    # -- Build tracker -------------------------------------------------------
    tracker = get_model(cfg)

    # Load weights if a checkpoint path is given
    ckpt_path = cfg.get("eval", None)
    if ckpt_path:
        ckpt_path = (
            ckpt_path.get("checkpoint", None) if hasattr(ckpt_path, "get") else None
        )
    if ckpt_path:
        _load_checkpoint(tracker, Path(ckpt_path))

    # -- Load sequences ------------------------------------------------------
    from src.data_loader import load_sequences  # type: ignore[import]

    sequences = load_sequences(cfg, split=split)
    log.info("Loaded %d sequences for evaluation.", len(sequences))

    all_preds: list[tuple[int, int, int, int]] = []
    all_gts: list[tuple[int, int, int, int]] = []
    sample_frame: "np.ndarray | None" = None

    # -- Run online tracking on each sequence --------------------------------
    for seq_id, (frames, gt_bboxes) in sequences.items():
        if not frames:
            log.warning("Sequence %s has no frames — skipping.", seq_id)
            continue

        log.debug("Evaluating sequence: %s (%d frames)", seq_id, len(frames))

        preds = _run_sequence_online(tracker, frames, gt_bboxes[0])

        # Collect predictions vs ground-truths (frames 1..N)
        all_preds.extend(preds)
        all_gts.extend(gt_bboxes[1:])

        # Keep one frame for latency benchmarking
        if sample_frame is None and len(frames) > 1:
            sample_frame = frames[1]

    # -- Accuracy metrics ----------------------------------------------------
    auc = success_auc(all_preds, all_gts)
    norm_prec = norm_precision(all_preds, all_gts)
    s_acc = compute_s_acc(auc, norm_prec)

    log.info(
        "Accuracy — AUC: %.4f | NormPrecision: %.4f | S_acc: %.4f",
        auc,
        norm_prec,
        s_acc,
    )

    # -- Efficiency metrics (CPU latency is mandatory) -----------------------
    if sample_frame is None:
        sample_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    log.info("Measuring CPU latency…")
    latency_ms = measure_cpu_latency(tracker, sample_frame)
    flops_g = measure_flops(getattr(tracker, "model", tracker))
    params_m = measure_params(getattr(tracker, "model", tracker))

    # Model size from checkpoint; fall back to 0.0 for classical trackers
    training_cfg = cfg.get("training") or DotDict({})
    experiment_id = cfg.get("experiment_name") or cfg.model.type
    save_dir = Path(training_cfg.get("save_dir", "logs/runs/checkpoints"))
    size_gb = measure_model_size(save_dir / f"{experiment_id}_best.pth")

    log.info(
        "Efficiency — Latency: %.1f ms | FLOPs: %.2f G | Params: %.2f M | Size: %.3f GB",
        latency_ms,
        flops_g,
        params_m,
        size_gb,
    )

    # -- Combine into final leaderboard row ----------------------------------
    scores = compute_all_scores(
        auc=auc,
        norm_prec=norm_prec,
        flops_g=flops_g,
        params_m=params_m,
        latency_ms=latency_ms,
        size_gb=size_gb,
    )

    log.info(
        "Final Score: %.4f  (S_acc=%.4f, S_eff=%.4f)",
        scores["final_score"],
        scores["s_acc"],
        scores["s_eff"],
    )

    return scores


# ---------------------------------------------------------------------------
# Prediction (unannotated public-LB sequences → submission CSV)
# ---------------------------------------------------------------------------


def predict(
    cfg: DotDict,
    split: str = "public_lb",
    output_path: str | Path = "submission.csv",
) -> Path:
    """Run online tracking on unannotated sequences and write submission CSV.

    Output CSV format (Kaggle requirement):
        id,x,y,w,h
        <seq_id>_<frame_idx>,x,y,w,h

    Args:
        cfg:         Merged experiment config.
        split:       Which data split to predict on ("public_lb").
        output_path: Path for the output CSV file.

    Returns:
        Path to the written CSV file.
    """
    output_path = Path(output_path)
    log.info("Generating submission: split=%s → %s", split, output_path)

    # -- Build tracker -------------------------------------------------------
    tracker = get_model(cfg)

    ckpt_path = cfg.get("eval", None)
    if ckpt_path:
        ckpt_path = (
            ckpt_path.get("checkpoint", None) if hasattr(ckpt_path, "get") else None
        )
    if ckpt_path:
        _load_checkpoint(tracker, Path(ckpt_path))

    # -- Load sequences (no annotations in public-LB split) ------------------
    from src.data_loader import load_sequences  # type: ignore[import]

    sequences = load_sequences(cfg, split=split)
    log.info("Loaded %d sequences for prediction.", len(sequences))

    # -- Write CSV -----------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []

    for seq_id, (frames, init_bboxes) in sequences.items():
        if not frames:
            log.warning("Sequence %s is empty — skipping.", seq_id)
            continue

        # Frame 0 uses the provided init bbox directly (no prediction needed)
        init_bbox = init_bboxes[0]
        rows.append(
            {
                "id": f"{seq_id}_0",
                "x": init_bbox[0],
                "y": init_bbox[1],
                "w": init_bbox[2],
                "h": init_bbox[3],
            }
        )

        # Online tracking for frames 1..N
        tracker.init(frames[0], init_bbox)
        for frame_idx in range(1, len(frames)):
            pred = tracker.update(frames[frame_idx])
            rows.append(
                {
                    "id": f"{seq_id}_{frame_idx}",
                    "x": pred[0],
                    "y": pred[1],
                    "w": pred[2],
                    "h": pred[3],
                }
            )

    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "x", "y", "w", "h"])
        writer.writeheader()
        writer.writerows(rows)

    log.info("Submission written: %d rows → %s", len(rows), output_path)
    return output_path


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------


def _load_checkpoint(tracker: Any, path: Path) -> None:
    """Load weights into a tracker from a checkpoint file.

    Handles torch state-dicts and pickle-serialised classical models.
    """
    if not path.exists():
        log.warning("Checkpoint not found: %s — running with untrained weights.", path)
        return

    try:
        import torch  # type: ignore[import]

        if hasattr(tracker, "load_state_dict"):
            state = torch.load(path, map_location="cpu", weights_only=True)
            tracker.load_state_dict(state)
            log.info("Loaded torch checkpoint from %s", path)
            return
    except ImportError:
        pass

    import pickle

    with open(path, "rb") as fh:
        loaded = pickle.load(fh)
    # For classical models the checkpoint IS the model; copy its state.
    if hasattr(loaded, "__dict__"):
        tracker.__dict__.update(loaded.__dict__)
    log.info("Loaded pickle checkpoint from %s", path)
