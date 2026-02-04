#!/usr/bin/env python3
import argparse
import shutil

# SAM 2: build the video predictor
# Repo docs show build_sam2_video_predictor and API usage:
#   - init_state(video_dir)
#   - add_new_points_or_box(...)
#   - propagate_in_video(...)
# See: https://github.com/facebookresearch/sam2  (README video section)
import sys
from pathlib import Path

import cv2
import hydra
import numpy as np
import torch
from PIL import Image

sys.path.append("./sam2")  # for local import of sam2.build_sam
from sam2.build_sam import build_sam2_video_predictor  # type: ignore


def center_crop_to_square(cv2_image):
    h, w = cv2_image.shape[:2]
    if h > w:
        start = (h - w) // 2
        return cv2_image[start : h - start, ...]
    elif w > h:
        start = (w - h) // 2
        return cv2_image[:, start : w - start, ...]
    else:
        return cv2_image


def save_mask(
    mask_logits_tensor, obj_ids, output_folder, frame_idx, save_masks_oos=True
):
    """
    Save 1-bit mask PNG(s) for this frame.
    mask_logits_tensor: Tensor [N_obj, 1, H, W] (logits)
    """
    output_folder.mkdir(parents=True, exist_ok=True)

    # Threshold logits at 0 to get binary mask, per examples
    # (logits > 0) -> foreground
    # Save one PNG per object ID
    for i, oid in enumerate(obj_ids):
        logits = mask_logits_tensor[i]  # [1, H, W]
        binary = (
            (logits > 0.0).cpu().numpy().astype(np.uint8)[0]
        )  # [H, W] in {0,1}
        if save_masks_oos:
            mask = binary * 255
            mask = center_crop_to_square(mask)
            mask = cv2.resize(mask, (800, 800))
            cv2.imwrite(str(output_folder / f"mask_{frame_idx:05d}.png"), mask)
        else:
            img = Image.fromarray(binary * 255, mode="L")
            img.save(output_folder / f"frame_{frame_idx:06d}_obj_{oid}.png")


def main():
    p = argparse.ArgumentParser(description="SAM2 video masks from one click")
    p.add_argument(
        "--checkpoint",
        default="./checkpoints/sam2.1_hiera_small.pt",
        help="Path to sam2.*.pt checkpoint",
    )
    p.add_argument(
        "--config",
        default="./d3dr/custom_dataset/sam2.1_hiera_s.yaml",
        help="Path to sam2.*.yaml config",
    )
    p.add_argument(
        "--frames_dir",
        required=True,
        help="Directory containing JPG frames (SAM2 loads all .jpgs inside)",
    )
    p.add_argument(
        "--output_folder",
        required=True,
        help="Where to save mask PNGs",
    )

    p.add_argument(
        "--click_x",
        type=float,
        required=True,
        nargs="+",
        help="X coordinate (pixels) of the positive click on the chosen frame",
    )
    p.add_argument(
        "--click_y",
        type=float,
        required=True,
        nargs="+",
        help="Y coordinate (pixels) of the positive click on the chosen frame",
    )
    p.add_argument(
        "--frame_idx",
        type=int,
        default=0,
        help="Index of the frame to click on (0-based, relative to SAM2's load order)",
    )
    p.add_argument(
        "--obj_id",
        type=int,
        default=1,
        help="Numeric object id to track (use different ids for multiple objects)",
    )
    p.add_argument(
        "--vos_optimized",
        action="store_true",
        help="Enable SAM2's VOS optimized predictor (torch.compile) for speed",
    )
    p.add_argument(
        "--frame_pattern",
        type=str,
        default="*",
    )
    p.add_argument(
        "--save_masks_oos",
        action="store_true",
    )
    args = p.parse_args()
    print(args)
    assert len(args.click_x) == len(args.click_y), (
        "len(x) should be equal to len(y)"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build the predictor (supports vos_optimized=True per release notes)
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize_config_dir(
        version_base="2.1", config_dir=str(Path(args.config).absolute().parent)
    )

    predictor = build_sam2_video_predictor(
        Path(args.config).name,
        args.checkpoint,
        device=device,
        vos_optimized=args.vos_optimized,
    )

    # Initialize SAM2 state over a directory of JPEG frames.
    # According to the official examples, init_state(video_dir) loads all *.jpg frames.
    video_dir = Path(args.frames_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {video_dir}")

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    video_dir_real = output_folder / "input_dir"
    (video_dir_real).mkdir(exist_ok=True)
    _i = 0
    for frame_path in sorted(list(video_dir.rglob(args.frame_pattern))):
        img = Image.open(frame_path).convert("RGB")
        img.save(str(video_dir_real / f"{_i:05}.jpg"))
        _i += 1

    # Mixed precision + inference mode recommended in README
    with torch.inference_mode(), torch.autocast(
        device_type=device,
        dtype=torch.float16 if device == "cuda" else torch.float32,
    ):
        inference_state = predictor.init_state(str(video_dir_real))

        # Positive click (label=1) at (x, y) on the chosen frame
        # API: add_new_points_or_box(inference_state, frame_idx, obj_id, points=..., labels=...)
        points = np.array(
            [[x, y] for (x, y) in zip(args.click_x, args.click_y)],
            dtype=np.float32,
        )
        labels = np.array(
            [1] * len(args.click_x), dtype=np.int32
        )  # 1 = positive point

        predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=args.frame_idx,
            obj_id=args.obj_id,
            points=points,
            labels=labels,
            # normalize_coords=True is the default in wrappers; the official notebook
            # examples pass pixel coords directly. If results look off, try normalize_coords=False.
        )

        # Propagate across video and write masks for each frame
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in predictor.propagate_in_video(inference_state):
            save_mask(
                out_mask_logits,
                out_obj_ids,
                output_folder,
                out_frame_idx,
                save_masks_oos=args.save_masks_oos,
            )

    shutil.rmtree(str(video_dir_real))
    print(f"Done. Masks saved under: {output_folder.resolve()}")


if __name__ == "__main__":
    main()
