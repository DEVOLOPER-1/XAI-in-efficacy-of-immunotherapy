import sys
import random
from pathlib import Path
import numpy as np
import torch
import cv2
from tqdm import tqdm

# =====================================================================
# 1. FIX PYTHON PATH (Allows importing from src)
# =====================================================================
# Dynamically add the project root (4 levels up) to the Python path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config import DotDict
from src.models.image import GoogLeNetWSI

# =====================================================================
# 2. CONFIGURATION & ABSOLUTE PATHS
# =====================================================================
IMAGES_DIR = Path("/media/maro/Mom0-0/Datasets/TCGA/pathological/images")
FEATURES_DIR = Path("/media/maro/Mom0-0/Datasets/TCGA/pathological/features")
WEIGHTS_PATH = Path(
    "/home/maro/final-projects/DSAI_305_XAI/logs/runs/checkpoints/Gnet_unfrozen_2layers+AttentionMIL-btch_16-lr_1e-5-log1p_true_fixed_final_weights.pth"
)

IMAGE_SIZE = 224
MAX_PATCHES = 16

# ImageNet Stats for Normalization
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =====================================================================
# 3. MODEL SETUP
# =====================================================================
def get_feature_extractor(device):
    print("Loading Fine-Tuned GoogLeNet from project pipeline...")

    cfg = DotDict({"model": {"dropout": 0.3}})
    wrapper = GoogLeNetWSI(cfg)

    state_dict = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
    wrapper.load_state_dict(state_dict)

    # Extract just the CNN backbone to get the [16, 1024] embeddings
    extractor = wrapper._est.cnn
    extractor = extractor.to(device)
    extractor.eval()

    return extractor


# =====================================================================
# 4. TILE PROCESSING
# =====================================================================
def process_patient_folder(patient_dir, max_patches, image_size):
    """Reads PNG tiles directly from the patient's folder."""

    # Grab all tile_XXX.png files in the folder
    tile_paths = sorted(list(patient_dir.glob("tile_*.png")))

    if not tile_paths:
        return None

    # Optional: shuffle so we don't always grab just the top-left corner of the slide
    random.shuffle(tile_paths)

    valid_patches = []

    for path in tile_paths:
        if len(valid_patches) >= max_patches:
            break

        img = cv2.imread(str(path))
        if img is None:
            continue

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize and Standardize
        img = img.astype(np.float32) / 255.0
        img = cv2.resize(img, (image_size, image_size))
        img = (img - MEAN) / STD

        # HWC to CHW for PyTorch
        img = img.transpose(2, 0, 1)
        valid_patches.append(img)

    if not valid_patches:
        return None

    # --- THE FIX: Oversampling Guarantee ---
    # If the patient has less than 16 valid patches, duplicate existing ones
    while len(valid_patches) < max_patches:
        valid_patches.append(random.choice(valid_patches))

    # Stack into exactly [16, 3, 224, 224]
    return np.stack(valid_patches, axis=0)


# =====================================================================
# 5. MAIN EXECUTION
# =====================================================================
def main():
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    extractor = get_feature_extractor(device)

    # Automatically find all patient folders in the raw directory
    patient_folders = [p for p in IMAGES_DIR.iterdir() if p.is_dir()]
    print(f"Found {len(patient_folders)} patient folders to process.")

    for patient_dir in tqdm(patient_folders, desc="Extracting 1024D Features"):
        patient_id = patient_dir.name

        output_file = FEATURES_DIR / f"{patient_id}.npy"

        # Skip if already extracted
        if output_file.exists():
            continue

        # Load, resize, normalize, and pad to exactly 16 patches
        patches_np = process_patient_folder(patient_dir, MAX_PATCHES, IMAGE_SIZE)

        if patches_np is None:
            tqdm.write(f"Warning: No valid tiles found for {patient_id}. Skipping.")
            continue

        # batch_tensor shape: [16, 3, 224, 224]
        batch_tensor = torch.from_numpy(patches_np).to(device)

        with torch.no_grad():
            # embeddings shape: [16, 1024]
            embeddings = extractor(batch_tensor)

        # Save exactly [16, 1024] feature block
        np.save(output_file, embeddings.cpu().numpy())

    print("\nFeature Extraction Complete! Your ABMIL fusion layer is ready to go.")


if __name__ == "__main__":
    main()
