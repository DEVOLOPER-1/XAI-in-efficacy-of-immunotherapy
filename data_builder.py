# src/data_builder.py
import os
import shutil
import cv2
import torch
import random
import openslide
import numpy as np
import polars as pl
import timm
from pathlib import Path
import subprocess
from tqdm import tqdm
from typing import Dict, Any, Optional
import yaml
import torch.nn as nn


def load_config(base_path: str, override_path: Optional[str] = None) -> Dict[str, Any]:
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    if override_path and Path(override_path).exists():
        with open(override_path) as f:
            override = yaml.safe_load(f)
        for k, v in override.items():
            if isinstance(v, dict) and k in cfg:
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg


def setup_dirs(cfg: Dict[str, Any]) -> None:
    for dir_key in ["root_data_dir", "features_dir", "temp_svs_dir"]:
        Path(cfg["paths"][dir_key]).mkdir(parents=True, exist_ok=True)


def prepare_data(cfg: Dict[str, Any]) -> pl.DataFrame:
    manifest = pl.read_csv(cfg["paths"]["manifest_path"], separator="\t")
    manifest = manifest.with_columns(
        pl.col("filename")
        .map_elements(lambda s: "-".join(s.split("-")[:3]), return_dtype=pl.String)
        .alias("Case ID")
    )

    splits = pl.read_csv(cfg["paths"]["splits_path"])
    targets_path = Path(cfg["paths"]["targets_path"])
    targets = (
        pl.read_parquet(targets_path)
        if targets_path.suffix == ".parquet"
        else pl.read_csv(targets_path)
    )

    valid_pids = set(splits["PATIENT_ID"].to_list()) & set(
        targets["PATIENT_ID"].to_list()
    )
    return manifest.filter(pl.col("Case ID").is_in(valid_pids)).unique(
        subset=["Case ID"]
    )


def load_feature_extractor(cfg: Dict[str, Any]) -> tuple:
    """
    Load model with support for both timm and transformers libraries.
    Returns: (model, processor_or_none, device, feature_dim, normalize_cfg)
    """
    train_cfg = cfg["training"]

    # Device selection
    if train_cfg.get("device", "auto") == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(train_cfg["device"])

    model_name = train_cfg["model_name"]
    library = train_cfg.get("model_library", "timm")

    processor = None
    feature_dim = None

    if library == "timm":
        print(f"🔍 Loading via timm: {model_name}")
        model = timm.create_model(
            model_name, pretrained=train_cfg.get("pretrained", True), num_classes=0
        )
        # Get feature dim by running a dummy forward pass
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(device)
            feature_dim = model(dummy).shape[-1]

    elif library == "transformers":
        print(f"🔍 Loading via transformers: {model_name}")
        from transformers import AutoModel, AutoImageProcessor

        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

        # Strip classification head if present
        if hasattr(model, "classifier"):
            model.classifier = nn.Identity()
        elif hasattr(model, "head"):
            model.head = nn.Identity()

        # Get feature dim from config or dummy pass
        if hasattr(model.config, "hidden_size"):
            feature_dim = model.config.hidden_size
        else:
            with torch.inference_mode():
                # Create dummy input using processor defaults
                dummy_img = torch.randn(3, 224, 224)
                inputs = processor(images=[dummy_img], return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                # Handle different output formats
                if hasattr(outputs, "last_hidden_state"):
                    feature_dim = outputs.last_hidden_state.shape[-1]
                else:
                    feature_dim = outputs[0].shape[-1]
    else:
        raise ValueError(
            f"Unknown model_library: {library}. Use 'timm' or 'transformers'"
        )

    # Optional finetuning setup
    if train_cfg.get("do_training", False):
        print("🔧 Finetuning mode enabled")
        if train_cfg.get("freeze_backbone", True):
            for name, param in model.named_parameters():
                param.requires_grad = False

    model.eval().to(device)

    # Normalization config
    normalize_cfg = {
        "enabled": train_cfg.get("normalize", True),
        "mean": train_cfg.get("norm_mean", [0.485, 0.456, 0.406]),
        "std": train_cfg.get("norm_std", [0.229, 0.224, 0.225]),
    }

    print(
        f"✅ Model loaded | Library: {library} | Device: {device} | Feature dim: {feature_dim}"
    )
    return model, processor, device, feature_dim, normalize_cfg


def extract_patient_features(
    svs_path: Path, model: torch.nn.Module, device: torch.device, cfg: Dict[str, Any]
) -> np.ndarray:
    """Generalized patch feature extraction. Optional normalization via config."""
    slide = openslide.OpenSlide(str(svs_path))
    w, h = slide.dimensions
    pw, ph = cfg["processing"]["patch_size"], cfg["processing"]["patch_size"]
    gw, gh = max(1, w // pw), max(1, h // ph)
    indices = list(range(gw * gh))
    random.shuffle(indices)

    features = []
    resize_to = cfg["processing"]["resize_to"]
    max_patches = cfg["processing"]["max_patches"]
    white_thresh = cfg["processing"]["white_thresh"]
    white_tol = cfg["processing"]["white_tolerance"]
    do_norm = cfg["training"].get("normalize", True)

    # Precompute normalization tensors if enabled
    if do_norm:
        mean = torch.tensor(
            cfg["training"].get("norm_mean", [0.485, 0.456, 0.406]), device=device
        ).view(3, 1, 1)
        std = torch.tensor(
            cfg["training"].get("norm_std", [0.229, 0.224, 0.225]), device=device
        ).view(3, 1, 1)

    with torch.no_grad():
        for i in indices:
            if len(features) >= max_patches:
                break
            x, y = (i % gw) * pw, (i // gw) * ph
            try:
                tile = np.array(slide.read_region((x, y), 0, (pw, ph)).convert("RGB"))
            except Exception:
                continue

            # White glass filter
            gray = cv2.cvtColor(tile, cv2.COLOR_RGB2GRAY)
            if np.mean(gray > white_thresh) > white_tol:
                continue

            # Resize & convert to tensor [3, H, W] in [0, 1]
            tile_resized = cv2.resize(tile, (resize_to, resize_to))
            img_t = (
                torch.from_numpy(tile_resized)
                .permute(2, 0, 1)
                .float()
                .div(255.0)
                .to(device)
            )

            # Apply optional normalization
            if do_norm:
                img_t = (img_t - mean) / std

            # Forward pass
            feat = model(img_t.unsqueeze(0))

            # Handle models that return tuples, dicts, or extra dimensions
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            if isinstance(feat, dict):
                feat = feat[list(feat.keys())[0]]

            feat = feat.squeeze().cpu().numpy()
            features.append(feat)

    slide.close()

    # Handle empty slides gracefully
    if not features:
        with torch.no_grad():
            dummy = torch.zeros(1, 3, resize_to, resize_to, device=device)
            dummy_feat = model(dummy)
            if isinstance(dummy_feat, (tuple, list)):
                dummy_feat = dummy_feat[0]
            if isinstance(dummy_feat, dict):
                dummy_feat = dummy_feat[list(dummy_feat.keys())[0]]
            feat_dim = dummy_feat.squeeze().shape[0]
            features = [np.zeros(feat_dim, dtype=np.float32)]

    # Pad to max_patches
    while len(features) < max_patches:
        features.append(features[-1])

    return np.stack(features)  # Shape: (16, feature_dim)


def download_chunk(chunk_df: pl.DataFrame, cfg: Dict[str, Any], chunk_idx: int) -> None:
    chunk_manifest = Path(f"temp_chunk_{chunk_idx}.txt")
    chunk_df.write_csv(chunk_manifest, separator="\t")

    cmd = [
        "gdc-client",
        "download",
        "-m",
        str(chunk_manifest),
        "-d",
        cfg["paths"]["temp_svs_dir"],
        "-n",
        "10",
    ]
    subprocess.run(cmd, check=True)
    chunk_manifest.unlink()


def main(base_config: str, override_config: Optional[str] = None) -> None:
    cfg = load_config(base_config, override_config)
    setup_dirs(cfg)

    torch.manual_seed(cfg["training"]["seed"])
    np.random.seed(cfg["training"]["seed"])
    random.seed(cfg["training"]["seed"])

    manifest_valid = prepare_data(cfg)
    model, device = load_feature_extractor(cfg)

    chunk_size = cfg["processing"]["chunk_size"]
    for start_idx in tqdm(range(0, manifest_valid.shape[0], chunk_size), desc="Chunks"):
        chunk_df = manifest_valid.slice(start_idx, chunk_size)
        download_chunk(chunk_df, cfg, start_idx)

        for row in chunk_df.iter_rows(named=True):
            pid = row["Case ID"]
            svs_dir = Path(cfg["paths"]["temp_svs_dir"]) / row["id"]
            svs_path = svs_dir / row["filename"]
            feat_path = Path(cfg["paths"]["features_dir"]) / f"{pid}.npy"

            if feat_path.exists():
                continue
            if not svs_path.exists():
                print(f"⚠️ Missing WSI for {pid}")
                continue

            try:
                features = extract_patient_features(svs_path, model, device, cfg)
                np.save(feat_path, features)
            except Exception as e:
                print(f"❌ Failed {pid}: {e}")

        if Path(cfg["paths"]["temp_svs_dir"]).exists():
            shutil.rmtree(cfg["paths"]["temp_svs_dir"])
        Path(cfg["paths"]["temp_svs_dir"]).mkdir(parents=True, exist_ok=True)
        print("🧹 Chunk complete. SVS wiped.")


if __name__ == "__main__":
    import sys

    override = sys.argv[1] if len(sys.argv) > 1 else None
    main("configs/data_builder_config.yaml", override)
