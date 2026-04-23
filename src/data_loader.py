"""
src/data_loader.py — Multimodal Cancer Research data loading for Zerone.

Expected data layout on disk (configured via cfg.dataset.*):

    data/
    ├── clinical.csv                 ← tabular data; must contain a PATIENT_ID column
    │                                   + genomics/clinical features + regression target
    ├── images/
    │   └── <PATIENT_ID>/
    │       └── slide.svs            ← whole-slide image (WSI, SVS or TIFF format)
    └── features/                    ← (optional) pre-extracted patch features
        └── <PATIENT_ID>.npy         ← (N_patches, feature_dim) float32 array

Modality availability per patient:
  - tabular-only : row in CSV, no SVS / .npy       → image is None in sample
  - image-only   : SVS present, not in CSV         → tabular is None in sample
  - both         : normal multimodal patient
  - neither      : logged as warning, patient excluded from all splits

Public API
──────────
    MultimodalDataset               ← iterable dataset, yields PatientSample dicts
    build_dataloaders(cfg)          → (train_loader, val_loader)

Config keys read (all under cfg.dataset.*):
    data_root          str   "data"
    tabular_file       str   "clinical.csv"
    images_dir         str   "images"
    features_dir       str   "features"
    target_col         str   "survival_months"
    patient_id_col     str   "PATIENT_ID"
    use_preextracted   bool  true     ← load .npy features; false = tile SVS on-the-fly
    patch_size         int   256      ← tile size when tiling SVS on-the-fly
    max_patches        int   16       ← patches sampled per slide per epoch
    val_ratio          float 0.15
    seed               int   42
    batch_size         int   32
    image_size         int   224      ← resize each patch/tile to this resolution

Modality flags (under cfg.modalities.*):
    image    bool  true
    tabular  bool  true

⚠ Shared code — see STANDARDS §5 before modifying.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from src.config import DotDict

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Patient sample — the unit that flows through the entire pipeline
# ---------------------------------------------------------------------------

@dataclass
class PatientSample:
    """One patient's multimodal data, ready for model input.

    Fields are None when that modality is unavailable for this patient.
    All downstream models must handle None inputs gracefully.

    Attributes:
        patient_id : Unique patient identifier (matches the CSV / folder name).
        tabular    : 1-D float32 array of shape (n_features,).
                     None if patient is absent from the tabular CSV.
        image      : float32 array of shape (N, C, H, W):
                       - Mode A (preextracted): shape is (N_patches, feature_dim, 1, 1)
                         or (N_patches, feature_dim) depending on the extractor.
                       - Mode B (on-the-fly):   shape is (N_tiles, 3, H, W).
                     None if no slide or feature file was found.
        target     : Float regression target. None for test patients.
        modalities : Set of active modality strings, e.g. {"tabular", "image"}.
    """
    patient_id: str
    tabular:    np.ndarray | None
    image:      np.ndarray | None
    target:     float | None
    modalities: set[str] = field(default_factory=set)


# ---------------------------------------------------------------------------
# Tabular store
# ---------------------------------------------------------------------------

class TabularStore:
    """Loads, imputes, and encodes the clinical/genomics CSV.

    Processing pipeline:
      1. Read CSV, set patient_id_col as the index.
      2. Separate the regression target column.
      3. Impute: numeric → median, categorical → most-frequent.
      4. Encode categorical columns with ordinal codes (model-agnostic;
         tree models handle ordinal fine; override in subclass for NN embeddings).
      5. Cast everything to float32 for downstream model consumption.
    """

    def __init__(
        self,
        file_path:       Path,
        patient_id_col: str,
        target_col:     str,
    ) -> None:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Tabular file not found: '{file_path}'. "
                "Check cfg.dataset.tabular_file and cfg.dataset.data_root."
            )

        if file_path.suffix == ".csv":
            raw = pd.read_csv(file_path)
        elif file_path.suffix == ".parquet":
            raw = pd.read_parquet(file_path)
        else:
            raise FileNotFoundError(f"Unsupported file type: {file_path.suffix}")

        if patient_id_col not in raw.columns:
            raise KeyError(
                f"Patient ID column '{patient_id_col}' not in {file_path.name}. "
                f"Available columns: {list(raw.columns)}"
            )

        raw = raw.set_index(patient_id_col)

        # Separate target from features
        if target_col in raw.columns:
            self._targets: pd.Series = raw.pop(target_col).astype(float)
        else:
            log.warning(
                "Target column '%s' not found in CSV — targets will be None for all patients.",
                target_col,
            )
            self._targets = pd.Series(dtype=float)

        # -- Data Preprocessing -----------------------------------------------
        #     -- Imputation -------------------------------------------------------
        # for col in raw.columns:
        #     if pd.api.types.is_numeric_dtype(raw[col]):
        #         raw[col] = raw[col].fillna(raw[col].median())
        #     else:
        #         mode_val = raw[col].mode()
        #         raw[col] = raw[col].fillna(
        #             mode_val.iloc[0] if not mode_val.empty else "UNKNOWN"
        #         )
        #
        # #     -- Categorical encoding ---------------------------------------------
        # for col in raw.select_dtypes(include=["object", "category"]).columns:
        #     raw[col] = raw[col].astype("category").cat.codes.astype(float)
        # for col in raw.select_dtypes(include=["bool"]).columns:
        #     raw[col] = raw[col].astype(float)

        self._features: pd.DataFrame = raw.astype(np.float32)

        log.info(
            "TabularStore ready: %d patients, %d features.",
            len(self._features), self.n_features,
        )

    @property
    def patient_ids(self) -> list[str]:
        return self._features.index.tolist()

    @property
    def n_features(self) -> int:
        return len(self._features.columns)

    @property
    def feature_names(self) -> list[str]:
        """Ordered list of feature column names after preprocessing.

        The order matches the axis-1 index of every array returned by
        get_features(), so feature_names[i] is the name of column i.
        Use this to label feature-importance plots or pass to
        model.print_feature_importances(feature_names=...).
        """
        return self._features.columns.tolist()

    def get_features(self, patient_id: str) -> np.ndarray | None:
        """Return float32 (n_features,) array, or None if patient absent."""
        if patient_id not in self._features.index:
            return None
        return self._features.loc[patient_id].to_numpy(dtype=np.float32)

    def get_target(self, patient_id: str) -> float | None:
        """Return regression target, or None if absent or NaN."""
        if patient_id not in self._targets.index:
            return None
        val = self._targets.loc[patient_id]
        return float(val) if pd.notna(val) else None


# ---------------------------------------------------------------------------
# Slide / WSI store
# ---------------------------------------------------------------------------

class SlideStore:
    """Loads pathology images (WSI) for each patient.

    Mode A — pre-extracted features (default, recommended):
        Loads a .npy array of shape (N_patches, feature_dim) saved by your
        feature extractor script (e.g. UNI, CONCH, PLIP).
        No OpenSlide dependency, very fast.

    Mode B — on-the-fly SVS tiling (requires openslide-python):
        Opens the SVS at full resolution, randomly samples cfg.dataset.max_patches
        non-overlapping tiles of size patch_size × patch_size, resizes each to
        image_size × image_size, and normalises to ImageNet mean/std.
        Use this path for end-to-end fine-tuning or when feature files are absent.

    Install:
        Mode A: no extra deps.
        Mode B: uv pip install openslide-python
    """

    # ImageNet statistics — standard starting point for pathology transfer learning
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        images_dir:       Path,
        features_dir:     Path,
        use_preextracted: bool = True,
        patch_size:       int  = 256,
        max_patches:      int  = 16,
        image_size:       int  = 224,
    ) -> None:
        self._images_dir       = images_dir
        self._features_dir     = features_dir
        self._use_preextracted = use_preextracted
        self._patch_size       = patch_size
        self._max_patches      = max_patches
        self._image_size       = image_size

    def patient_has_data(self, patient_id: str) -> bool:
        """Return True if any image/feature data exists for this patient."""
        if self._use_preextracted:
            return (self._features_dir / f"{patient_id}.npy").exists()
        slide_dir = self._images_dir / patient_id
        if not slide_dir.is_dir():
            return False
        return any(slide_dir.glob("*.svs")) or any(slide_dir.glob("*.tiff"))

    def get_image(self, patient_id: str) -> np.ndarray | None:
        """Load and return image data for *patient_id*, or None if absent."""
        if self._use_preextracted:
            return self._load_preextracted(patient_id)
        return self._tile_svs(patient_id)

    # ------------------------------------------------------------------
    # Mode A
    # ------------------------------------------------------------------

    def _load_preextracted(self, patient_id: str) -> np.ndarray | None:
        path = self._features_dir / f"{patient_id}.npy"
        if not path.exists():
            return None
        arr = np.load(path).astype(np.float32)
        log.debug("Pre-extracted features for %s: shape=%s", patient_id, arr.shape)
        return arr

    # ------------------------------------------------------------------
    # Mode B
    # ------------------------------------------------------------------

    def _tile_svs(self, patient_id: str) -> np.ndarray | None:
        """Sample random tiles from the patient's SVS file.

        Returns float32 (N, 3, H, W) tensor normalised to ImageNet stats.
        """
        slide_dir  = self._images_dir / patient_id
        if not slide_dir.is_dir():
            return None

        candidates = list(slide_dir.glob("*.svs")) + list(slide_dir.glob("*.tiff"))
        if not candidates:
            log.warning("No SVS/TIFF found for patient %s in %s", patient_id, slide_dir)
            return None

        try:
            import openslide  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "openslide-python is required for on-the-fly WSI tiling.\n"
                "Install: uv pip install openslide-python\n"
                "Or set cfg.dataset.use_preextracted: true to use .npy files instead."
            ) from exc

        svs_path = candidates[0]
        try:
            slide = openslide.OpenSlide(str(svs_path))
        except openslide.OpenSlideError as exc:
            log.error("Cannot open WSI %s: %s", svs_path, exc)
            return None

        width, height = slide.dimensions
        ps = self._patch_size
        grid_w = max(1, width  // ps)
        grid_h = max(1, height // ps)
        total_tiles = grid_w * grid_h

        # Random tile sampling — not seeded so each epoch sees different tiles.
        # This acts as data augmentation for the image modality.
        sampled = random.sample(range(total_tiles), min(self._max_patches, total_tiles))

        tiles: list[np.ndarray] = []
        for idx in sampled:
            # Row-major: scan left-to-right, top-to-bottom.
            # OpenSlide read_region(x, y) expects x=horizontal, y=vertical.
            # col = idx % grid_w  → horizontal position (x-axis)
            # row = idx // grid_w → vertical position   (y-axis)
            x_idx = idx % grid_w   # column index (horizontal)
            y_idx = idx // grid_w  # row index    (vertical)
            x_off = x_idx * ps
            y_off = y_idx * ps
            region = slide.read_region((x_off, y_off), 0, (ps, ps))
            tile   = np.array(region.convert("RGB"), dtype=np.float32) / 255.0
            tile   = _resize_hwc(tile, self._image_size)
            tile   = (tile - self._MEAN) / self._STD
            tile   = tile.transpose(2, 0, 1)   # HWC → CHW
            tiles.append(tile)

        slide.close()
        return np.stack(tiles, axis=0) if tiles else None   # (N, 3, H, W)


def _resize_hwc(img: np.ndarray, size: int) -> np.ndarray:
    """Resize (H, W, C) float32 array to (size, size, C)."""
    if img.shape[0] == size and img.shape[1] == size:
        return img
    try:
        import cv2  # type: ignore[import]
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        # Nearest-neighbour fallback — no extra deps
        h, w = img.shape[:2]
        rs = (np.arange(size) * h / size).astype(int)
        cs = (np.arange(size) * w / size).astype(int)
        return img[np.ix_(rs, cs)]


# ---------------------------------------------------------------------------
# MultimodalDataset
# ---------------------------------------------------------------------------

class MultimodalDataset:
    """Dataset for multimodal cancer research — one item per patient.

    Yields PatientSample objects. Missing modalities are None, NOT zero-padded.
    This allows fusion models to apply explicit masking (recommended) vs silent
    zero-fill (less reliable for missing-at-random genomics data).

    Args:
        tabular_store : TabularStore (None → tabular modality disabled).
        slide_store   : SlideStore   (None → image modality disabled).
        patient_ids   : Patient IDs belonging to this split.
        shuffle       : Shuffle order each epoch (set True for train only).
        seed          : RNG seed for shuffle.
    """

    def __init__(
        self,
        tabular_store: TabularStore | None,
        slide_store:   SlideStore   | None,
        patient_ids:   list[str],
        shuffle: bool = False,
        seed:    int  = 42,
    ) -> None:
        self._tabular = tabular_store
        self._slides  = slide_store
        self._ids     = patient_ids
        self._shuffle = shuffle
        self._seed    = seed

        log.info(
            "MultimodalDataset: %d patients | tabular=%s | image=%s",
            len(patient_ids),
            tabular_store is not None,
            slide_store is not None,
        )

    def __len__(self) -> int:
        return len(self._ids)

    def __getitem__(self, idx: int) -> PatientSample:
        pid        = self._ids[idx]
        modalities: set[str] = set()

        # -- Tabular ----------------------------------------------------------
        tabular = None
        target  = None
        if self._tabular is not None:
            tabular = self._tabular.get_features(pid)
            target  = self._tabular.get_target(pid)
            if tabular is not None:
                modalities.add("tabular")

        # -- Image ------------------------------------------------------------
        image = None
        if self._slides is not None:
            image = self._slides.get_image(pid)
            if image is not None:
                modalities.add("image")

        if not modalities:
            log.warning("Patient %s: no modality data found — sample will be empty.", pid)

        return PatientSample(
            patient_id=pid,
            tabular=tabular,
            image=image,
            target=target,
            modalities=modalities,
        )

    def __iter__(self) -> Iterator[PatientSample]:
        indices = list(range(len(self._ids)))
        if self._shuffle:
            rng = random.Random(self._seed)
            rng.shuffle(indices)
        for i in indices:
            yield self[i]


# ---------------------------------------------------------------------------
# Collation
# ---------------------------------------------------------------------------

def collate_patients(samples: list[PatientSample]) -> dict[str, Any]:
    """Collate a list of PatientSamples into a batched dict.

    Keys in the returned dict:
        "patient_ids" : list[str]               length B
        "tabular"     : float32 ndarray (B, F)  or None
        "image"       : float32 ndarray (B, N, ...) or None — zero-padded for missing
        "target"      : float32 ndarray (B,)    NaN where target is unknown
        "modalities"  : list[set[str]]           length B

    Padding rules:
        tabular: zero-vector (shape F) for patients missing tabular data.
        image:   zero-array  (shape of the largest array) for patients missing images.
                 Fusion models should use the "modalities" field to apply masking.
    """
    patient_ids = [s.patient_id for s in samples]
    modalities  = [s.modalities  for s in samples]

    # targets — NaN for unlabelled test patients
    targets = np.array(
        [s.target if s.target is not None else float("nan") for s in samples],
        dtype=np.float32,
    )

    # tabular — zero-pad missing patients
    tab_present = [s.tabular for s in samples if s.tabular is not None]
    if tab_present:
        n_feat  = tab_present[0].shape[0]
        tabular = np.stack(
            [s.tabular if s.tabular is not None else np.zeros(n_feat, dtype=np.float32)
             for s in samples],
        )
    else:
        tabular = None

    # image — zero-pad missing patients and variable N_patches
    img_present = [s.image for s in samples if s.image is not None]
    if img_present:
        max_n     = max(a.shape[0] for a in img_present)
        rest_dims = img_present[0].shape[1:]           # (C, H, W) or (feature_dim,)
        image = np.zeros((len(samples), max_n, *rest_dims), dtype=np.float32)
        for i, s in enumerate(samples):
            if s.image is not None:
                n = s.image.shape[0]
                image[i, :n] = s.image
    else:
        image = None

    return {
        "patient_ids": patient_ids,
        "tabular":     tabular,
        "image":       image,
        "target":      targets,
        "modalities":  modalities,
    }


# ---------------------------------------------------------------------------
# Batch loader
# ---------------------------------------------------------------------------

class _BatchLoader:
    """Iterates a MultimodalDataset in mini-batches, yielding collated dicts.

    Intentionally avoids a hard torch.DataLoader dependency so tree-model
    members (XGBoost / CatBoost) can use it without installing PyTorch.
    train.py is responsible for converting numpy arrays to tensors.

    Attributes:
        feature_names: Ordered list of tabular column names (strings), matching
                       axis-1 of every batch["tabular"] array. None if the
                       tabular modality is disabled. Pass this directly to
                       model.print_feature_importances(feature_names=loader.feature_names).
    """

    def __init__(
        self,
        dataset:       MultimodalDataset,
        batch_size:    int         = 32,
        shuffle:       bool        = False,
        seed:          int         = 42,
        feature_names: list[str] | None = None,
    ) -> None:
        self._dataset    = dataset
        self._batch_size = batch_size
        self._shuffle    = shuffle
        self._seed       = seed
        self.feature_names: list[str] | None = feature_names

    def __len__(self) -> int:
        return math.ceil(len(self._dataset) / self._batch_size)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        indices = list(range(len(self._dataset)))
        if self._shuffle:
            rng = random.Random(self._seed)
            rng.shuffle(indices)

        batch: list[PatientSample] = []
        for idx in indices:
            batch.append(self._dataset[idx])
            if len(batch) == self._batch_size:
                yield collate_patients(batch)
                batch = []
        if batch:
            yield collate_patients(batch)


# ---------------------------------------------------------------------------
# Public: build_dataloaders
# ---------------------------------------------------------------------------

def build_dataloaders(cfg: DotDict) -> tuple[_BatchLoader, _BatchLoader]:
    """Construct train and validation dataloaders from the merged config.

    Args:
        cfg: Merged experiment config (from src.config.load_config).

    Returns:
        (train_loader, val_loader) — _BatchLoader instances.
    """
    ds_cfg  = cfg.get("dataset")   or DotDict({})
    mod_cfg = cfg.get("modalities") or DotDict({})

    data_root    = Path(ds_cfg.get("data_root",      "data"))
    tabular_file = data_root / ds_cfg.get("tabular_file",  "clinical.csv")
    images_dir   = data_root / ds_cfg.get("images_dir",    "images")
    features_dir = data_root / ds_cfg.get("features_dir",  "features")
    target_col   = ds_cfg.get("target_col",      "survival_months")
    id_col       = ds_cfg.get("patient_id_col",  "PATIENT_ID")
    val_ratio    = ds_cfg.get("val_ratio",        0.15)
    seed         = ds_cfg.get("seed",             42)
    batch_size   = ds_cfg.get("batch_size",       32)
    preextracted = ds_cfg.get("use_preextracted", True)
    patch_size   = ds_cfg.get("patch_size",       256)
    max_patches  = ds_cfg.get("max_patches",      16)
    image_size   = ds_cfg.get("image_size",       224)
    use_tabular  = mod_cfg.get("tabular",  True)
    use_image    = mod_cfg.get("image",    True)

    # -- Instantiate stores -------------------------------------------------
    tabular_store: TabularStore | None = None
    slide_store:   SlideStore   | None = None

    if use_tabular:
        tabular_store = TabularStore(tabular_file, id_col, target_col)

    if use_image:
        slide_store = SlideStore(
            images_dir=images_dir,
            features_dir=features_dir,
            use_preextracted=preextracted,
            patch_size=patch_size,
            max_patches=max_patches,
            image_size=image_size,
        )

    # -- Union of all known patient IDs ------------------------------------
    all_ids: set[str] = set()
    if tabular_store:
        all_ids.update(tabular_store.patient_ids)
    if slide_store:
        scan_dir = features_dir if preextracted else images_dir
        if scan_dir.exists():
            img_ids = (
                {p.stem for p in scan_dir.glob("*.npy")}
                if preextracted
                else {p.name for p in scan_dir.iterdir() if p.is_dir()}
            )
            all_ids.update(img_ids)

    if not all_ids:
        raise RuntimeError(
            "No patients found. Verify cfg.dataset.data_root, tabular_file, "
            "images_dir, and features_dir."
        )

    # -- Deterministic train / val split -----------------------------------
    sorted_ids = sorted(all_ids)
    rng        = random.Random(seed)
    rng.shuffle(sorted_ids)
    n_val      = max(1, int(len(sorted_ids) * val_ratio))
    val_ids    = sorted_ids[:n_val]
    train_ids  = sorted_ids[n_val:]

    log.info(
        "Patient split: train=%d, val=%d (seed=%d, val_ratio=%.2f)",
        len(train_ids), len(val_ids), seed, val_ratio,
    )

    train_ds = MultimodalDataset(tabular_store, slide_store, train_ids, shuffle=True,  seed=seed)
    val_ds   = MultimodalDataset(tabular_store, slide_store, val_ids,   shuffle=False)

    train_loader = _BatchLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        feature_names=tabular_store.feature_names if tabular_store else None,
    )
    val_loader = _BatchLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        feature_names=tabular_store.feature_names if tabular_store else None,
    )

    log.info(
        "Dataloaders: train=%d batches, val=%d batches (batch_size=%d)",
        len(train_loader), len(val_loader), batch_size,
    )
    return train_loader, val_loader