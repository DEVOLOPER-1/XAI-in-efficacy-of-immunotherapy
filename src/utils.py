"""
src/utils.py — Shared utilities for WhiteBox Cancer Research.

Covers:
  - Medical regression metrics:  RMSE, MAE, R², Pearson r, C-Index (survival)
  - Classification metrics:      AUROC, AUPRC (for binary tasks / thresholded survival)
  - Cross-validation helpers:    stratified k-fold split by target quantile
  - Prediction IO:               save / load result CSV in leaderboard format
  - Checkpoint helpers:          save / load torch or sklearn models

Hardware efficiency metrics (latency, FLOPs, model size) are intentionally
removed — the cancer research task has no efficiency budget constraint.

⚠ Shared code — see STANDARDS §5 before modifying.
"""

from __future__ import annotations

import logging
from pathlib import Path
import pickle
from typing import Any, Sequence
import joblib

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core regression metrics
# ---------------------------------------------------------------------------


def rmse(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Root Mean Squared Error — primary regression metric.

    Lower is better. Units match the target (e.g. months for survival).

    Args:
        y_true: Ground-truth values. NaN entries are ignored.
        y_pred: Predicted values.

    Returns:
        RMSE as a float.
    """
    yt, yp = _clean_pairs(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def mae(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Mean Absolute Error.

    More robust to outliers than RMSE; useful for clinical interpretability
    (e.g. "predictions are off by X months on average").
    """
    yt, yp = _clean_pairs(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")
    return float(np.mean(np.abs(yt - yp)))


def r2_score(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Coefficient of determination (R²).

    1.0 = perfect prediction. 0.0 = predicts the mean. Can be negative.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.

    Returns:
        R² as a float in (-∞, 1].
    """
    yt, yp = _clean_pairs(y_true, y_pred)
    if len(yt) < 2:
        return float("nan")
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    if ss_tot == 0.0:
        return float("nan")  # target has zero variance — metric undefined
    return float(1.0 - ss_res / ss_tot)


def pearson_r(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    """Pearson correlation coefficient between predictions and ground truth.

    Measures linear association regardless of scale. Range: [-1, 1].
    Particularly useful for survival regression where RMSE magnitude is
    harder to interpret across different datasets.
    """
    yt, yp = _clean_pairs(y_true, y_pred)
    if len(yt) < 2:
        return float("nan")
    corr = np.corrcoef(yt, yp)
    return float(corr[0, 1])


# ---------------------------------------------------------------------------
# Survival-specific metric
# ---------------------------------------------------------------------------


def concordance_index(
    durations: Sequence[float],
    predictions: Sequence[float],
    events: Sequence[int] | None = None,
) -> float:
    """Harrell's Concordance Index (C-Index) for survival analysis.

    Measures the probability that, for a random pair of patients where one
    has a shorter survival time and actually experienced the event, the model
    correctly predicts the shorter survival.

    Range: [0, 1].  0.5 = random.  1.0 = perfect ordering.

    Args:
        durations:   Observed survival times (or times-to-event).
        predictions: Model's predicted risk scores or survival times.
                     Higher score = higher risk = shorter predicted survival.
        events:      Binary event indicators (1 = event occurred, 0 = censored).
                     If None, all patients are assumed to have experienced the event.

    Returns:
        C-Index float. NaN if no comparable pairs exist.

    Note:
        This is a simplified O(n²) implementation suitable for typical
        cohort sizes (<10 000 patients). For large datasets, use lifelines:
            from lifelines.utils import concordance_index as lifelines_ci
    """
    d = np.array(durations, dtype=np.float64)
    p = np.array(predictions, dtype=np.float64)

    if events is None:
        e = np.ones(len(d), dtype=np.int8)
    else:
        e = np.array(events, dtype=np.int8)

    # Remove NaN rows
    mask = ~(np.isnan(d) | np.isnan(p))
    d, p, e = d[mask], p[mask], e[mask]

    concordant = 0
    discordant = 0
    tied_risk = 0

    for i in range(len(d)):
        if e[i] == 0:
            continue  # censored patient cannot be the "earlier" in a pair
        for j in range(len(d)):
            if i == j:
                continue
            if d[j] <= d[i]:
                continue  # j did not have a longer follow-up than i
            # Patient i had a shorter survival — check if model agrees
            if p[i] > p[j]:
                concordant += 1
            elif p[i] < p[j]:
                discordant += 1
            else:
                tied_risk += 1

    total = concordant + discordant + tied_risk
    if total == 0:
        return float("nan")

    return float((concordant + 0.5 * tied_risk) / total)


# ---------------------------------------------------------------------------
# Binary / threshold metrics (useful for high-risk / low-risk stratification)
# ---------------------------------------------------------------------------


def auroc(
    y_true: Sequence[int],
    y_score: Sequence[float],
) -> float:
    """Area Under the ROC Curve (AUROC) for binary outcomes.

    Args:
        y_true:  Binary labels (0 / 1).
        y_score: Model's continuous risk/probability scores.

    Returns:
        AUROC in [0, 1]. 0.5 = random. 1.0 = perfect.
    """
    yt = np.array(y_true, dtype=np.int8)
    ys = np.array(y_score, dtype=np.float64)

    mask = ~np.isnan(ys)
    yt, ys = yt[mask], ys[mask]

    if len(np.unique(yt)) < 2:
        log.warning("auroc: only one class present in y_true — returning NaN.")
        return float("nan")

    # Trapezoidal AUROC via sorting
    order = np.argsort(-ys)  # descending by score
    yt_sorted = yt[order]
    n_pos = yt_sorted.sum()
    n_neg = len(yt_sorted) - n_pos

    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp_cum = np.cumsum(yt_sorted)
    fp_cum = np.cumsum(1 - yt_sorted)
    tpr = tp_cum / n_pos
    fpr = fp_cum / n_neg

    return float(np.trapezoid(tpr, fpr))


def auprc(
    y_true: Sequence[int],
    y_score: Sequence[float],
) -> float:
    """Area Under the Precision-Recall Curve (AUPRC).

    Preferred over AUROC for imbalanced datasets (e.g. rare events).

    Args:
        y_true:  Binary labels (0 / 1).
        y_score: Continuous risk scores.

    Returns:
        AUPRC in [0, 1].
    """
    yt = np.array(y_true, dtype=np.int8)
    ys = np.array(y_score, dtype=np.float64)

    mask = ~np.isnan(ys)
    yt, ys = yt[mask], ys[mask]

    order = np.argsort(-ys)
    yt_sorted = yt[order]
    n_pos = yt_sorted.sum()

    if n_pos == 0:
        return float("nan")

    tp_cum = np.cumsum(yt_sorted)
    precision = tp_cum / np.arange(1, len(yt_sorted) + 1)
    recall = tp_cum / n_pos

    return float(np.trapezoid(precision, recall))

def huber_loss(y_true: Sequence[float], y_pred: Sequence[float], delta: float = 1.0) -> float:
    """Huber Loss — matches PyTorch nn.HuberLoss(delta=1.0).

    Robust regression metric that acts like MSE for small errors
    and MAE for large errors (outliers).
    """
    yt, yp = _clean_pairs(y_true, y_pred)
    if len(yt) == 0:
        return float("nan")

    abs_err = np.abs(yt - yp)
    quadratic = np.minimum(abs_err, delta)
    linear = abs_err - quadratic

    loss = 0.5 * (quadratic ** 2) + delta * linear
    return float(np.mean(loss))

# ---------------------------------------------------------------------------
# Omnibus metric dict — used by train.py and W&B logging
# ---------------------------------------------------------------------------


def compute_all_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    events: Sequence[int] | None = None,
    threshold: float | None = None,
) -> dict[str, float]:
    """Compute and return all relevant metrics in a single call.

    This dict maps directly to W&B log keys and leaderboard columns.

    Args:
        y_true:    Ground-truth regression targets (e.g. survival months).
        y_pred:    Model predictions.
        events:    Binary event indicators for C-Index. None = all events.
        threshold: If provided, binarise y_true at this value and compute
                   AUROC / AUPRC (useful for high-risk stratification).

    Returns:
        Dict with keys: rmse, mae, r2, pearson_r, c_index.
        If threshold is given, also: auroc, auprc.
    """
    metrics: dict[str, float] = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "huber": huber_loss(y_true, y_pred, delta=1.0),  # <-- NEW: Added Huber
        "r2": r2_score(y_true, y_pred),
        "pearson_r": pearson_r(y_true, y_pred),
        "c_index": concordance_index(y_true, y_pred, events),
    }

    if threshold is not None:
        yt, yp = _clean_pairs(y_true, y_pred)
        binary_true = [1 if t >= threshold else 0 for t in yt]
        metrics["auroc"] = auroc(binary_true, yp)
        metrics["auprc"] = auprc(binary_true, yp)

    return metrics


# ---------------------------------------------------------------------------
# Cross-validation split
# ---------------------------------------------------------------------------


def stratified_kfold_splits(
    patient_ids: list[str],
    y: Sequence[float],
    n_splits: int = 5,
    seed: int = 42,
) -> list[tuple[list[str], list[str]]]:
    """Stratified K-Fold splits by target quantile (for regression).

    Binning the continuous target into quartiles before stratification
    ensures each fold has a similar target distribution — important for
    survival analysis where high-risk patients may be rare.

    Args:
        patient_ids: All patient IDs.
        y:           Corresponding target values.
        n_splits:    Number of folds (default 5).
        seed:        Random seed for reproducibility.

    Returns:
        List of (train_ids, val_ids) tuples, one per fold.
    """
    y_arr = np.array(y, dtype=np.float64)

    # Bin into quartiles for stratification; NaN patients go into their own bin
    n_bins = min(4, len(np.unique(y_arr[~np.isnan(y_arr)])))
    bins = np.nanquantile(y_arr, np.linspace(0, 1, n_bins + 1))
    strata = np.digitize(y_arr, bins[1:-1])  # 0 … n_bins-1

    # Collect indices per stratum
    strata_map: dict[int, list[int]] = {}
    for i, s in enumerate(strata):
        strata_map.setdefault(int(s), []).append(i)

    rng = np.random.default_rng(seed)
    for indices in strata_map.values():
        rng.shuffle(indices)

    # Assign fold indices round-robin within each stratum
    fold_indices: list[list[int]] = [[] for _ in range(n_splits)]
    for stratum_indices in strata_map.values():
        for k, idx in enumerate(stratum_indices):
            fold_indices[k % n_splits].append(idx)

    splits: list[tuple[list[str], list[str]]] = []
    for fold in range(n_splits):
        val_idx = set(fold_indices[fold])
        train_idx = [i for i in range(len(patient_ids)) if i not in val_idx]
        val_list = [patient_ids[i] for i in fold_indices[fold]]
        train_list = [patient_ids[i] for i in train_idx]
        splits.append((train_list, val_list))

    return splits


# ---------------------------------------------------------------------------
# Result CSV helpers (leaderboard format)
# ---------------------------------------------------------------------------


def save_predictions(
    patient_ids: list[str],
    y_pred: Sequence[float],
    output_path: str | Path,
) -> Path:
    """Save predictions to a CSV in leaderboard/submission format.

    Output format:
        PATIENT_ID, prediction
        TCGA-AA-001, 24.3
        ...

    Args:
        patient_ids: Patient identifiers.
        y_pred:      Predicted values.
        output_path: Destination CSV path.

    Returns:
        Path to the written file.
    """
    import csv

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["PATIENT_ID", "prediction"])
        for pid, pred in zip(patient_ids, y_pred):
            writer.writerow([pid, float(pred)])

    log.info("Predictions saved → %s (%d rows)", path, len(patient_ids))
    return path


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def save_checkpoint(model: Any, path: str | Path) -> None:
    """Save a model checkpoint. Handles both torch and sklearn-style models.

    Torch models: saves state_dict (recommended — smaller, portable).
    sklearn / XGBoost / CatBoost: pickle serialisation.

    Args:
        model: The model to save.
        path:  Destination file path (.pth for torch, .pkl for others).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "_est") and not hasattr(model._est, "state_dict"):
        joblib.dump(model._est, path)
        log.info("Scikit-Learn (Joblib) checkpoint saved ➔ %s", path)
        return

    # 2. Standard PyTorch Save
    try:
        import torch
        if hasattr(model, "state_dict"):
            torch.save(model.state_dict(), path)
            log.info("Torch checkpoint saved ➔ %s", path)
            return
    except ImportError:
        pass

    # 3. Ultimate Fallback (Standard Pickle)
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    log.info("Pickle checkpoint saved ➔ %s", path)


def load_checkpoint(model: Any, path: str | Path) -> Any:
    """Load a checkpoint into *model* (in-place for torch, return value for sklearn).

    Args:
        model: The model object to load weights into.
        path:  Path to the checkpoint file.

    Returns:
        The model with loaded weights (always the same object for torch).
    """
    path = Path(path)

    # 1. Intercept Scikit-Learn wrappers
    if hasattr(model, "_est") and not hasattr(model._est, "state_dict"):
        model._est = joblib.load(path)
        log.info("Scikit-Learn (Joblib) checkpoint loaded ➔ %s", path)
        return

    # 2. Standard PyTorch Load
    try:
        import torch
        if hasattr(model, "load_state_dict"):
            state_dict = torch.load(path, weights_only=True)
            model.load_state_dict(state_dict)
            log.info("Torch checkpoint loaded ➔ %s", path)
            return
    except ImportError:
        pass

    # 3. Fallback
    with open(path, "rb") as fh:
        model = pickle.load(fh)
    log.info("Pickle checkpoint loaded ➔ %s", path)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clean_pairs(
    y_true: Sequence[float],
    y_pred: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert to float64 arrays and drop rows where either value is NaN."""
    yt = np.array(y_true, dtype=np.float64)
    yp = np.array(y_pred, dtype=np.float64)
    mask = ~(np.isnan(yt) | np.isnan(yp))
    return yt[mask], yp[mask]
