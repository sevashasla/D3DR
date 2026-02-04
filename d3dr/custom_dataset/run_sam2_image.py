import cv2
import numpy as np
import torch
from pathlib import Path

import hydra
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# --- Paths ---
CONFIG_DIR = "/home/skorokho/coding/voi_gs/render_dataset/"
CONFIG_NAME = "sam2.1_hiera_s.yaml"
# CONFIG_PATH = "/home/skorokho/coding/voi_gs/render_dataset/sam2.1_hiera_s.yaml"
CHECKPOINT_PATH = "/scratch/izar/skorokho/sam/sam2.1_hiera_small.pt"
IMAGE_PATH = "/scratch/izar/skorokho/data/toster/processed_dark_toster/images/frame_00001.jpg"
OUT_MASK_PATH = "/home/skorokho/coding/voi_gs/tmp/center_mask.png"

# --- Load model ---
hydra.core.global_hydra.GlobalHydra.instance().clear()
hydra.initialize_config_dir(version_base="2.1", config_dir="/home/skorokho/coding/voi_gs/render_dataset/")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading SAM2 on {device}...")
sam2_model = build_sam2(CONFIG_NAME, CHECKPOINT_PATH, device=device)
predictor = SAM2ImagePredictor(sam2_model)

# --- Load image ---
image_bgr = cv2.imread(IMAGE_PATH)
if image_bgr is None:
    raise FileNotFoundError(f"Could not read image: {IMAGE_PATH}")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
H, W = image_rgb.shape[:2]

# --- Pick center point ---
center_point = np.array([[W / 2, H / 2]], dtype=np.float32)
center_label = np.array([1], dtype=np.int32)  # 1 = foreground

# --- Predict mask ---
predictor.set_image(image_rgb)
masks, scores, _ = predictor.predict(
    point_coords=center_point,
    point_labels=center_label,
    multimask_output=False  # only one mask
)

mask = masks[0].astype(np.uint8) * 255

# --- Save mask ---
cv2.imwrite(OUT_MASK_PATH, mask)
print(f"Mask saved to {OUT_MASK_PATH}")
