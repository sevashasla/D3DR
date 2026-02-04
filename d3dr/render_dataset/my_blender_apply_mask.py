import argparse
import os

import cv2
import numpy as np


def apply_mask(image_path, mask_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        print(f"Error reading image or mask: {image_path}, {mask_path}")
        return

    mask = (mask > 127).astype(np.uint8)
    masked_image = (image * mask[:, :, np.newaxis]).astype(np.uint8)

    cv2.imwrite(output_path, masked_image)


def process_images(images_folder, masks_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_name in os.listdir(images_folder):
        image_num = image_name.split("_")[1].split(".")[0]
        mask_name = "mask_" + image_num + ".png"
        image_path = os.path.join(images_folder, image_name)
        mask_path = os.path.join(masks_folder, mask_name)
        output_path = os.path.join(output_folder, image_name)

        if os.path.exists(mask_path):
            apply_mask(image_path, mask_path, output_path)
        else:
            print(f"Mask not found for image: {image_name}")


def main():
    parser = argparse.ArgumentParser(description="Apply masks to images.")
    parser.add_argument(
        "--images_folder",
        type=str,
        help="Path to the folder containing images.",
    )
    parser.add_argument(
        "--masks_folder", type=str, help="Path to the folder containing masks."
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        help="Path to the folder to save masked images.",
    )

    args = parser.parse_args()

    process_images(args.images_folder, args.masks_folder, args.output_folder)


if __name__ == "__main__":
    main()
