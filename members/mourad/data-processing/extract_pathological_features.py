import os
import random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import cv2
from tqdm import tqdm

try:
    import openslide
    import albumentations as A
    import torchstain
except ImportError:
    raise ImportError(
        "Missing libraries. Run: pip install pandas openslide-python albumentations opencv-python tqdm torchstain")

# =====================================================================
# 1. CONFIGURATION & ABSOLUTE PATHS
# =====================================================================
IMAGES_DIR = Path("/media/maro/Mom0-0/Datasets/TCGA/pathological/raw")
FEATURES_DIR = Path("/media/maro/Mom0-0/Datasets/TCGA/pathological/features")
MANIFEST_TSV = Path("/media/maro/Mom0-0/Datasets/TCGA/pathological/raw/gdc_sample_sheet.2026-05-10.tsv")
WEIGHTS_PATH = Path("/home/maro/final-projects/DSAI_305_XAI/freezed-models/runs/checkpoints/best_wsi_model.pth")

PATCH_SIZE = 256
IMAGE_SIZE = 224
MAX_PATCHES = 16

# ImageNet Stats
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =====================================================================
# 2. MODEL SETUP (Matching Colab Architecture)
# =====================================================================
class GoogLeNetWSI(nn.Module):
    """Must match the Colab training architecture to load the StateDict properly."""
    def __init__(self):
        super().__init__()
        # FIX: Removed aux_logits=False so it perfectly matches your Colab run
        self.cnn = models.googlenet(weights=models.GoogLeNet_Weights.DEFAULT)
        self.cnn.fc = nn.Identity()
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(1024, 1))

def get_feature_extractor(device):
    print("Loading Fine-Tuned GoogLeNet...")

    # 1. Instantiate the wrapper
    wrapper = GoogLeNetWSI()

    # 2. Load the weights safely across device architectures
    state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    wrapper.load_state_dict(state_dict)

    # 3. Extract just the CNN (already outputs 1024D because fc is Identity)
    extractor = wrapper.cnn
    extractor = extractor.to(device)
    extractor.eval()

    return extractor


# =====================================================================
# 3. HELPER FUNCTIONS
# =====================================================================
def resize_hwc(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def process_patient_slide(svs_path, max_patches, patch_size, image_size, augmentor, normalizer):
    """Extracts valid, Macenko-normalized tissue patches from an SVS file."""
    try:
        slide = openslide.OpenSlide(str(svs_path))
    except Exception as e:
        print(f"\nError opening {svs_path}: {e}")
        return None

    width, height = slide.dimensions
    grid_w, grid_h = max(1, width // patch_size), max(1, height // patch_size)

    all_indices = list(range(grid_w * grid_h))
    random.shuffle(all_indices)

    valid_patches = []

    for idx in all_indices:
        if len(valid_patches) >= max_patches:
            break

        x_off = (idx % grid_w) * patch_size
        y_off = (idx // grid_w) * patch_size

        try:
            region = slide.read_region((x_off, y_off), 0, (patch_size, patch_size)).convert("RGB")
        except Exception:
            continue

        tile_np = np.array(region)

        # 1. Tissue Detection
        gray = cv2.cvtColor(tile_np, cv2.COLOR_RGB2GRAY)
        white_ratio = np.sum(gray > 220) / (patch_size * patch_size)
        if white_ratio > 0.5:
            continue

        # 2. Macenko Color Normalization
        try:
            tile_np, _, _ = normalizer.normalize(I=tile_np, stains=False)
        except Exception:
            continue

        # 3. Augmentation (Usually disabled for feature extraction)
        if augmentor:
            tile_np = augmentor(image=tile_np)["image"]

        # 4. Resize and Standardize
        tile_np = tile_np.astype(np.float32) / 255.0
        tile_np = resize_hwc(tile_np, image_size)
        tile_np = (tile_np - MEAN) / STD
        tile_np = tile_np.transpose(2, 0, 1)  # HWC to CHW

        valid_patches.append(tile_np)

    slide.close()

    # 5. Oversampling Guarantee (Ensures exactly [16, 1024] embeddings)
    if not valid_patches:
        return None

    while len(valid_patches) < max_patches:
        valid_patches.append(random.choice(valid_patches))

    return np.stack(valid_patches, axis=0)


# =====================================================================
# 4. MAIN EXECUTION
# =====================================================================
def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    extractor = get_feature_extractor(device)
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='numpy')

    # Intentionally set to None for deterministic Phase 2 extraction
    augmentor = None

    print("Loading TSV manifest...")
    manifest = pd.read_csv(MANIFEST_TSV, sep='\t')
    manifest = manifest[manifest['Data Type'] == 'Slide Image']

    processed_patients = set()

    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Extracting WSI"):
        patient_id = row['Case ID']
        file_id = row['File ID']
        file_name = row['File Name']

        if patient_id in processed_patients:
            continue

        output_file = FEATURES_DIR / f"{patient_id}.npy"
        if output_file.exists():
            processed_patients.add(patient_id)
            continue

        svs_path = IMAGES_DIR / file_id / file_name
        if not svs_path.exists():
            print(f"\nWarning: File not found on disk: {svs_path}")
            continue

        patches_np = process_patient_slide(
            svs_path, MAX_PATCHES, PATCH_SIZE, IMAGE_SIZE, augmentor, normalizer
        )

        if patches_np is None:
            print(f"\nWarning: No valid tissue found for {patient_id}")
            continue

        # batch_tensor shape: [16, 3, 224, 224]
        batch_tensor = torch.from_numpy(patches_np).to(device)
        with torch.no_grad():
            # embeddings shape: [16, 1024]
            embeddings = extractor(batch_tensor)

        np.save(output_file, embeddings.cpu().numpy())
        processed_patients.add(patient_id)

    print("\nExtraction complete! Ready for multimodal fusion.")


if __name__ == "__main__":
    main()
