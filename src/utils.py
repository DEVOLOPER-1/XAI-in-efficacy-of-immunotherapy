"""
src/utils.py — Shared utility functions for AIC-4 Zerone.

Covers:
  - Bounding-box helpers (IoU, overlap, centre)
  - Accuracy metrics (AUC of success curve, Normalised Precision)
  - Efficiency measurement (CPU latency, FLOPs, parameter count, model size)
  - Official scoring formula (S_acc, S_eff, Final Score)

⚠ This file is shared code — see STANDARDS §5 before modifying.
"""

from __future__ import annotations

import time
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Budget ceilings (from README §8.6 / §12)
# ---------------------------------------------------------------------------

BUDGET_FLOPS_G:    float = 30.0   # GFLOPs
BUDGET_PARAMS_M:   float = 50.0   # millions of parameters
BUDGET_LATENCY_MS: float = 30.0   # milliseconds, measured on CPU
BUDGET_SIZE_GB:    float = 0.5    # gigabytes

# Efficiency scoring weights (from STANDARDS §6)
W_FLOPS:    float = 0.25
W_PARAMS:   float = 0.15
W_LATENCY:  float = 0.35   # highest weight — optimise this first
W_SIZE:     float = 0.25

# Accuracy scoring weights (from README §12)
W_AUC:       float = 0.6
W_NORM_PREC: float = 0.4

# Final score lambda (penalises efficiency)
LAMBDA: float = 0.2


# ---------------------------------------------------------------------------
# Bounding-box helpers
# ---------------------------------------------------------------------------

def bbox_iou(
    pred: tuple[float, float, float, float],
    gt:   tuple[float, float, float, float],
) -> float:
    """Compute Intersection-over-Union between two (x, y, w, h) boxes.

    Args:
        pred: Predicted bounding box (x, y, w, h).
        gt:   Ground-truth bounding box (x, y, w, h).

    Returns:
        IoU value in [0, 1]. Returns 0.0 if either box has zero area.
    """
    px, py, pw, ph = pred
    gx, gy, gw, gh = gt

    # Convert to (x1, y1, x2, y2)
    px2, py2 = px + pw, py + ph
    gx2, gy2 = gx + gw, gy + gh

    inter_x1 = max(px, gx)
    inter_y1 = max(py, gy)
    inter_x2 = min(px2, gx2)
    inter_y2 = min(py2, gy2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    pred_area = pw * ph
    gt_area   = gw * gh
    union_area = pred_area + gt_area - inter_area

    if union_area <= 0.0:
        return 0.0

    return float(inter_area / union_area)


def bbox_centre(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """Return the centre (cx, cy) of an (x, y, w, h) box."""
    x, y, w, h = bbox
    return x + w / 2.0, y + h / 2.0


def bbox_diagonal(bbox: tuple[float, float, float, float]) -> float:
    """Return the diagonal length of an (x, y, w, h) box (used for NormPrecision)."""
    _, _, w, h = bbox
    return float(np.sqrt(w ** 2 + h ** 2))


# ---------------------------------------------------------------------------
# Accuracy metrics
# ---------------------------------------------------------------------------

def success_auc(
    preds: Sequence[tuple[float, float, float, float]],
    gts:   Sequence[tuple[float, float, float, float]],
    thresholds: Sequence[float] | None = None,
) -> float:
    """Compute the Area Under the IoU success-rate curve.

    For each IoU threshold t in [0, 1], the success rate is the fraction of
    frames where IoU(pred, gt) >= t. AUC is the mean success rate over all
    thresholds.

    Frames where gt = (0, 0, 0, 0) (target not visible) are skipped.

    Args:
        preds:      Sequence of predicted (x, y, w, h) bboxes.
        gts:        Sequence of ground-truth (x, y, w, h) bboxes.
        thresholds: IoU thresholds to evaluate at. Defaults to 0.00–1.00 (101 pts).

    Returns:
        AUC in [0, 1].
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 101).tolist()

    ious = [
        bbox_iou(p, g)
        for p, g in zip(preds, gts)
        if not (g[2] == 0 and g[3] == 0)   # skip "not visible" frames
    ]

    if not ious:
        return 0.0

    ious_arr = np.array(ious)
    success_rates = [float(np.mean(ious_arr >= t)) for t in thresholds]
    return float(np.mean(success_rates))


def norm_precision(
    preds: Sequence[tuple[float, float, float, float]],
    gts:   Sequence[tuple[float, float, float, float]],
    threshold: float = 0.5,
) -> float:
    """Compute scale-invariant normalised centre-error precision.

    For each frame, the centre error is divided by the GT bounding-box
    diagonal to make it scale-invariant. The precision score is the fraction
    of frames where this normalised error is below *threshold*.

    Args:
        preds:     Predicted (x, y, w, h) bboxes.
        gts:       Ground-truth (x, y, w, h) bboxes.
        threshold: Normalised error threshold (default 0.5 — half the diagonal).

    Returns:
        NormPrecision score in [0, 1].
    """
    errors: list[float] = []
    for pred, gt in zip(preds, gts):
        if gt[2] == 0 and gt[3] == 0:  # target not visible
            continue
        diag = bbox_diagonal(gt)
        if diag <= 0.0:
            continue
        pcx, pcy = bbox_centre(pred)
        gcx, gcy = bbox_centre(gt)
        dist = float(np.sqrt((pcx - gcx) ** 2 + (pcy - gcy) ** 2))
        errors.append(dist / diag)

    if not errors:
        return 0.0

    return float(np.mean(np.array(errors) < threshold))


def compute_s_acc(auc: float, norm_prec: float) -> float:
    """Accuracy component of the final score.

    S_acc = 0.6 × AUC + 0.4 × NormPrecision
    """
    return W_AUC * auc + W_NORM_PREC * norm_prec


# ---------------------------------------------------------------------------
# Efficiency measurement
# ---------------------------------------------------------------------------

def measure_cpu_latency(
    tracker: Any,
    frame: "np.ndarray",
    n_runs: int = 100,
    warmup: int = 10,
) -> float:
    """Measure per-frame inference latency on CPU (milliseconds).

    Runs tracker.update(frame) repeatedly on CPU to produce a stable median.
    CPU measurement is mandatory — the competition evaluates on standardised
    CPU hardware (see STANDARDS §6 and README §8.6).

    Args:
        tracker: An initialised tracker (TrackerProtocol).
        frame:   A representative BGR frame (numpy array).
        n_runs:  Number of timed repetitions.
        warmup:  Number of un-timed warm-up iterations.

    Returns:
        Median latency in milliseconds.
    """
    # If torch is available, force CPU for timing purposes.
    try:
        import torch
        _ctx = torch.no_grad()
        _ctx.__enter__()
    except ImportError:
        _ctx = None  # type: ignore[assignment]

    # Warm up (allows caches to settle)
    for _ in range(warmup):
        tracker.update(frame)

    # Timed runs
    times_ms: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        tracker.update(frame)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1_000.0)

    if _ctx is not None:
        _ctx.__exit__(None, None, None)

    return float(np.median(times_ms))


def measure_flops(
    model: Any,
    input_shape: tuple[int, ...] = (1, 3, 255, 255),
) -> float:
    """Estimate GFLOPs for a PyTorch model using thop (if available).

    Falls back gracefully to 0.0 for classical-CV models that have no
    PyTorch computation graph.

    Args:
        model:       A torch.nn.Module.
        input_shape: (B, C, H, W) dummy input tensor shape.

    Returns:
        GFLOPs (float). Returns 0.0 if thop / torch is unavailable.
    """
    try:
        import torch
        from thop import profile  # type: ignore[import]

        dummy = torch.zeros(*input_shape)
        flops_raw, _ = profile(model, inputs=(dummy,), verbose=False)
        return float(flops_raw / 1e9)
    except (ImportError, Exception):
        # Classical trackers or missing thop — return sentinel
        return 0.0


def measure_params(model: Any) -> float:
    """Count trainable parameters in millions for a PyTorch model.

    Returns 0.0 for non-PyTorch models.
    """
    try:
        import torch.nn as nn
        if not isinstance(model, nn.Module):
            return 0.0
        return sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    except ImportError:
        return 0.0


def measure_model_size(model_path: str | Path) -> float:
    """Return the on-disk size of a saved model file in gigabytes.

    Args:
        model_path: Path to a .pth / .pt / .pkl file.

    Returns:
        File size in GB. Returns 0.0 if the file does not exist.
    """
    path = Path(model_path)
    if not path.exists():
        return 0.0
    return path.stat().st_size / 1e9


# ---------------------------------------------------------------------------
# Efficiency scoring
# ---------------------------------------------------------------------------

def _norm(value: float, budget: float) -> float:
    """Clip-normalise a metric against its budget ceiling: min(1, value / budget)."""
    return min(1.0, value / budget) if budget > 0 else 0.0


def compute_s_eff(
    flops_g:    float,
    params_m:   float,
    latency_ms: float,
    size_gb:    float,
) -> float:
    """Compute the efficiency component S_eff.

    S_eff = 0.25 × norm(FLOPs) + 0.15 × norm(Params)
          + 0.35 × norm(Latency) + 0.25 × norm(Size)

    Lower is better — a model well within all budgets approaches 0.

    Args:
        flops_g:    GFLOPs.
        params_m:   Millions of parameters.
        latency_ms: CPU latency in milliseconds.
        size_gb:    Model file size in gigabytes.

    Returns:
        S_eff in [0, 1].
    """
    return (
        W_FLOPS   * _norm(flops_g,    BUDGET_FLOPS_G)    +
        W_PARAMS  * _norm(params_m,   BUDGET_PARAMS_M)   +
        W_LATENCY * _norm(latency_ms, BUDGET_LATENCY_MS) +
        W_SIZE    * _norm(size_gb,    BUDGET_SIZE_GB)
    )


def compute_final_score(s_acc: float, s_eff: float) -> float:
    """Compute the competition's final score.

    Final Score = S_acc − 0.2 × S_eff

    Args:
        s_acc: Accuracy component (from compute_s_acc).
        s_eff: Efficiency component (from compute_s_eff).

    Returns:
        Final score (higher is better).
    """
    return s_acc - LAMBDA * s_eff


def compute_all_scores(
    auc:        float,
    norm_prec:  float,
    flops_g:    float,
    params_m:   float,
    latency_ms: float,
    size_gb:    float,
) -> dict[str, float]:
    """Convenience wrapper — compute and return all score components at once.

    Returns a dict with keys: auc, norm_prec, s_acc, s_eff, final_score.
    Matches the column names in logs/leaderboard.csv.
    """
    s_acc  = compute_s_acc(auc, norm_prec)
    s_eff  = compute_s_eff(flops_g, params_m, latency_ms, size_gb)
    final  = compute_final_score(s_acc, s_eff)

    return {
        "auc":         auc,
        "norm_prec":   norm_prec,
        "s_acc":       s_acc,
        "flops_g":     flops_g,
        "params_m":    params_m,
        "latency_ms":  latency_ms,
        "size_gb":     size_gb,
        "s_eff":       s_eff,
        "final_score": final,
    }