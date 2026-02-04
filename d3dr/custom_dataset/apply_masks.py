import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image


def apply_masks(images_dir, masks_dir, output_folder):
    """Apply masks to images and save results."""
    os.makedirs(output_folder, exist_ok=True)

    image_files = sorted(Path(images_dir).iterdir())
    masks_files = sorted(Path(masks_dir).iterdir())

    for image_path, mask_path in zip(image_files, masks_files):
        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Ensure same size
        if image.size != mask.size:
            mask = mask.resize(image.size, Image.BILINEAR)

        # Apply mask: multiply image by normalized mask
        image_array = np.array(image, dtype=np.float32)
        mask_array = np.array(mask, dtype=np.float32) / 255.0

        masked_image = image_array * mask_array[:, :, np.newaxis]
        masked_image = np.uint8(masked_image)

        # Save result
        output_path = os.path.join(output_folder, image_path.name)
        Image.fromarray(masked_image).save(output_path)
        print(f"Saved masked image to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Apply masks to images")
    parser.add_argument(
        "--images_dir", type=str, help="Directory containing input images"
    )
    parser.add_argument(
        "--masks_dir", type=str, help="Directory containing mask images"
    )
    parser.add_argument(
        "--output_folder", type=str, help="Output directory for masked images"
    )

    args = parser.parse_args()

    apply_masks(args.images_dir, args.masks_dir, args.output_folder)


if __name__ == "__main__":
    main()
