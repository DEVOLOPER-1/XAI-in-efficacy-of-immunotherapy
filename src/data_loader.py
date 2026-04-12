"""
src/data_loader.py — Dataset loading for AIC-4 Zerone.

Data layout on disk (from README §7):

    data/
    ├── contestant_manifest.json          ← seq_id → {video_path, annotation_path, …}
    ├── train/
    │   └── <dataset>/<seq_name>/
    │       ├── <seq_name>.mp4
    │       └── annotation.txt            ← one "x,y,w,h" per line
    └── public_lb/
        └── <dataset>/<seq_name>/
            └── <seq_name>.mp4            ← no annotation.txt

Annotation format:
    x,y,w,h   — top-left corner + size in pixels
    0,0,0,0   — target not visible in this frame

Public API
──────────
    load_sequences(cfg, split)        → dict[seq_id, (frames, bboxes)]
    build_dataloaders(cfg)            → (train_loader, val_loader)

Both functions read from cfg so no paths are hardcoded in .py files
(per STANDARDS §4).

⚠ Shared code — see STANDARDS §5 before modifying.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Iterator

import numpy as np

from src.config import DotDict

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

BBox   = tuple[int, int, int, int]           # (x, y, w, h)
Frames = list[np.ndarray]                    # list of BGR frames
SequenceData = tuple[Frames, list[BBox]]     # (frames, bboxes)


# ---------------------------------------------------------------------------
# Annotation parser
# ---------------------------------------------------------------------------

def _parse_annotation(path: Path) -> list[BBox]:
    """Parse an annotation.txt file into a list of (x, y, w, h) tuples.

    Args:
        path: Path to annotation.txt.

    Returns:
        List with one (x, y, w, h) per line. Lines where target is not visible
        are represented as (0, 0, 0, 0) — callers skip these when needed.

    Raises:
        ValueError: If a line cannot be parsed as four integers.
    """
    bboxes: list[BBox] = []
    with open(path) as fh:
        for line_num, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) != 4:
                raise ValueError(
                    f"{path}:{line_num} — expected 4 comma-separated values, "
                    f"got {len(parts)}: '{line}'"
                )
            try:
                x, y, w, h = (int(float(p)) for p in parts)
            except ValueError as exc:
                raise ValueError(
                    f"{path}:{line_num} — cannot parse as integers: '{line}'"
                ) from exc
            bboxes.append((x, y, w, h))
    return bboxes


# ---------------------------------------------------------------------------
# Video decoder
# ---------------------------------------------------------------------------

def _decode_video(path: Path, max_frames: int | None = None) -> Frames:
    """Decode an MP4 file into a list of BGR numpy arrays.

    Uses OpenCV's VideoCapture. Frames are stored in memory — for very long
    sequences consider using a lazy generator instead (see _iter_frames).

    Args:
        path:       Path to the .mp4 file.
        max_frames: Cap on the number of frames to load. None = load all.

    Returns:
        List of (H, W, 3) uint8 BGR arrays.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError:      If OpenCV cannot open the file.
    """
    try:
        import cv2  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required to decode video. "
            "Install with: uv pip install opencv-python"
        ) from exc

    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")

    frames: Frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        cap.release()

    return frames


def _iter_frames(path: Path) -> Iterator[np.ndarray]:
    """Lazy frame iterator — yields one BGR frame at a time without buffering.

    Use this inside inference.predict() for large sequences to avoid OOM.
    """
    try:
        import cv2  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "OpenCV is required to decode video. "
            "Install with: uv pip install opencv-python"
        ) from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {path}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------

def _load_manifest(data_root: Path) -> dict:
    """Load contestant_manifest.json from data_root.

    The manifest maps split names ("train", "public_lb") to dicts of
    seq_id → metadata (video path, annotation path, etc.).

    Returns:
        The parsed manifest dict.

    Raises:
        FileNotFoundError: If contestant_manifest.json is missing.
    """
    manifest_path = data_root / "contestant_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"contestant_manifest.json not found at '{manifest_path}'. "
            "Run 'make setup' and download the dataset via 'kaggle competitions download'."
        )
    with open(manifest_path) as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Single-sequence loader
# ---------------------------------------------------------------------------

def _load_sequence(
    seq_id: str,
    meta: dict,
    data_root: Path,
    max_frames: int | None = None,
) -> SequenceData:
    """Load frames and annotations for one sequence.

    For public-LB sequences (no annotation.txt), bboxes will contain only
    the init bbox from meta["init_bbox"] repeated once (frame 0 only).
    Callers that iterate frames[1:] will never access bboxes[1:] for public-LB
    sequences, so this is safe.

    Args:
        seq_id:     Sequence identifier string.
        meta:       Metadata dict from the manifest for this sequence.
        data_root:  Root directory of the dataset.
        max_frames: Optional cap on frames to load (useful for fast debug runs).

    Returns:
        (frames, bboxes) tuple.
    """
    # -- Video path ----------------------------------------------------------
    video_path = data_root / meta["video_path"]
    log.debug("Loading sequence %s from %s", seq_id, video_path)
    frames = _decode_video(video_path, max_frames=max_frames)

    if not frames:
        log.warning("Sequence %s decoded 0 frames — check the video file.", seq_id)
        return [], []

    # -- Annotations ---------------------------------------------------------
    if "annotation_path" in meta and meta["annotation_path"]:
        ann_path = data_root / meta["annotation_path"]
        bboxes   = _parse_annotation(ann_path)

        # Sanity-check: frame count should match annotation line count.
        if len(bboxes) != len(frames):
            log.warning(
                "Sequence %s: %d frames but %d annotation lines — "
                "truncating to the shorter of the two.",
                seq_id, len(frames), len(bboxes),
            )
            n = min(len(frames), len(bboxes))
            frames = frames[:n]
            bboxes = bboxes[:n]
    else:
        # Public-LB sequence: only the init bbox is known.
        init_bbox: BBox = tuple(meta.get("init_bbox", [0, 0, 0, 0]))  # type: ignore[assignment]
        bboxes = [init_bbox]

    return frames, bboxes


# ---------------------------------------------------------------------------
# Public: load_sequences
# ---------------------------------------------------------------------------

def load_sequences(
    cfg: DotDict,
    split: str = "train",
) -> dict[str, SequenceData]:
    """Load all sequences for a given data split.

    Args:
        cfg:   Merged experiment config. Reads cfg.dataset.data_root and
               cfg.dataset.max_frames_per_seq (optional debug cap).
        split: "train", "val", or "public_lb".
               "val" is a subset of "train" sequences held out per cfg.dataset.val_ratio.

    Returns:
        Dict of seq_id → (frames, bboxes).

    Notes:
        - The train/val split is deterministic (seeded by cfg.dataset.seed).
        - public_lb sequences return bboxes with only frame-0 GT (no future GTs).
    """
    dataset_cfg = cfg.get("dataset") or DotDict({})
    data_root   = Path(dataset_cfg.get("data_root", "data"))
    max_frames  = dataset_cfg.get("max_frames_per_seq", None)
    val_ratio   = dataset_cfg.get("val_ratio", 0.1)
    seed        = dataset_cfg.get("seed", 42)

    manifest = _load_manifest(data_root)

    # -- Resolve which manifest split to use ---------------------------------
    if split in ("train", "val"):
        raw_split = "train"
    elif split == "public_lb":
        raw_split = "public_lb"
    else:
        raise ValueError(
            f"Unknown split '{split}'. Choose from: 'train', 'val', 'public_lb'."
        )

    if raw_split not in manifest:
        raise KeyError(
            f"Manifest does not contain split '{raw_split}'. "
            f"Available keys: {list(manifest.keys())}"
        )

    all_ids: list[str] = list(manifest[raw_split].keys())

    # -- Train / val deterministic split ------------------------------------
    if split in ("train", "val"):
        rng = random.Random(seed)
        shuffled = all_ids[:]
        rng.shuffle(shuffled)
        n_val   = max(1, int(len(shuffled) * val_ratio))
        val_ids = set(shuffled[:n_val])

        if split == "val":
            seq_ids = [sid for sid in all_ids if sid in val_ids]
        else:  # "train"
            seq_ids = [sid for sid in all_ids if sid not in val_ids]
    else:
        seq_ids = all_ids

    log.info(
        "Loading %d '%s' sequences (data_root=%s, max_frames=%s)…",
        len(seq_ids), split, data_root, max_frames,
    )

    # -- Load each sequence -------------------------------------------------
    sequences: dict[str, SequenceData] = {}
    for seq_id in seq_ids:
        meta = manifest[raw_split][seq_id]
        try:
            frames, bboxes = _load_sequence(seq_id, meta, data_root, max_frames)
            sequences[seq_id] = (frames, bboxes)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            log.error("Failed to load sequence %s: %s — skipping.", seq_id, exc)

    log.info("Loaded %d/%d sequences successfully.", len(sequences), len(seq_ids))
    return sequences


# ---------------------------------------------------------------------------
# Iterable wrappers used by train.py
# ---------------------------------------------------------------------------

class _SequenceIterable:
    """Thin iterable that wraps a sequences dict for use in training loops.

    Yields (frames, bboxes) tuples one sequence at a time.
    This is intentionally simple — no batching, no shuffling across sequences,
    because the tracker is online and processes frames sequentially.
    """

    def __init__(self, sequences: dict[str, SequenceData], shuffle: bool = False, seed: int = 42) -> None:
        self._items = list(sequences.values())
        if shuffle:
            rng = random.Random(seed)
            rng.shuffle(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[SequenceData]:
        return iter(self._items)


# ---------------------------------------------------------------------------
# Public: build_dataloaders
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: DotDict) -> tuple[_SequenceIterable, _SequenceIterable]:
    """Build and return (train_loader, val_loader) iterables.

    Each iterable yields (frames: list[np.ndarray], bboxes: list[BBox]).
    The split is deterministic based on cfg.dataset.val_ratio and cfg.dataset.seed.

    Args:
        cfg: Merged experiment config.

    Returns:
        (train_loader, val_loader) — both are _SequenceIterable instances.
    """
    dataset_cfg = cfg.get("dataset") or DotDict({})
    seed        = dataset_cfg.get("seed", 42)

    train_seqs = load_sequences(cfg, split="train")
    val_seqs   = load_sequences(cfg, split="val")

    train_loader = _SequenceIterable(train_seqs, shuffle=True, seed=seed)
    val_loader   = _SequenceIterable(val_seqs,   shuffle=False)

    log.info(
        "Dataloaders ready — train: %d seqs, val: %d seqs",
        len(train_loader), len(val_loader),
    )
    return train_loader, val_loader