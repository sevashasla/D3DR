"""
Dataset containing images and masks for cropping and generation.
"""

import json

import cv2
import numpy as np
import torch
from dataset import ImageDataset
from PIL import Image as PILImage


class ImageDatasetWImg(ImageDataset):
    """
    A dataset for Texture-Preserving Diffusion Personalization
    """
    def __init__(
        self, make_dark_background, crop_type, take_only, *args, **kwargs
    ):
        self.make_dark_background = make_dark_background
        self.crop_type = crop_type
        self.take_only = take_only
        super().__init__(*args, **kwargs)

        with open(self.save_iclight_dir / "prompts.json") as f:
            self.data_prompts = json.load(f)

        self.cached = dict()

    def make_random_crop_center(self, images, mask):
        """
        make random crop at the center of the image
        """
        was_list = True
        if not isinstance(images, list):
            was_list = False
            images = [images]

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        max_contour = max(contours, key=cv2.contourArea)

        # rectangle coordinates in mask shape
        x_r, y_r, wh_r = self._xywh2square(*cv2.boundingRect(max_contour))

        for i in range(len(images)):
            wh_new = int(wh_r * self.crop_ratio)
            images[i] = images[i][y_r : y_r + wh_new, x_r : x_r + wh_new, ...]
            images[i] = cv2.resize(images[i], (self.width, self.height))

        if was_list:
            return images
        else:
            return images[0]

    @staticmethod
    def _create_kernel(ksize):
        x, y = np.meshgrid(np.arange(ksize), np.arange(ksize))
        x_mean = ksize / 2.0 - 0.5
        y_mean = ksize / 2.0 - 0.5
        dist = np.absolute(x - x_mean) ** 2 + np.absolute(y - y_mean) ** 2
        kernel = 1.0 - dist / dist.max()
        kernel /= kernel.sum()
        return kernel

    @staticmethod
    def _xywh2square(x, y, w, h):
        x_c = x + w / 2  # x center
        y_c = y + h / 2  # y center
        wh = max(w, h)  # now the image is square
        x = int(x_c - wh / 2)
        y = int(y_c - wh / 2)
        x = max(x, 0)
        y = max(y, 0)
        wh = int(wh)
        return x, y, wh

    def make_random_crop_mask(self, images, mask):
        """
        make random crop such that the cropped part has a big
        overlap with the object
        """
        was_list = True
        if not isinstance(images, list):
            was_list = False
            images = [images]

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        max_contour = max(contours, key=cv2.contourArea)

        # rectangle coordinates in mask shape
        x_r, y_r, wh_r = self._xywh2square(*cv2.boundingRect(max_contour))

        # crop the mask, because we want to take
        # object only from the close views
        mask_crop = mask[y_r : y_r + wh_r, x_r : x_r + wh_r, ...]
        wh_new = int(wh_r * self.crop_ratio)

        # find pixels which have a lot of neighbors
        # by averaging the intensity
        kernel = self._create_kernel(wh_new)
        mask_crop_smooth = cv2.filter2D(mask_crop, -1, kernel)
        quant = np.quantile(mask_crop_smooth[mask_crop_smooth > 0.0], 0.9) - 0.5
        xy_good = np.where(mask_crop_smooth >= quant)  # here are the pixels

        # if pixels are close to the borders then remove them
        x_good_mask = (xy_good[1] >= wh_new // 2 + 1) & (
            xy_good[1] <= wh_r - wh_new // 2 - 1
        )
        y_good_mask = (xy_good[0] >= wh_new // 2 + 1) & (
            xy_good[0] <= wh_r - wh_new // 2 - 1
        )
        # no such good pixels -> just return the center
        if not np.any(x_good_mask & y_good_mask):
            c_y, c_x = wh_r // 2, wh_r // 2
            print("[INFO] No good pixels found :(")
        else:
            xy_good = np.stack(
                [
                    xy_good[0][x_good_mask & y_good_mask],
                    xy_good[1][x_good_mask & y_good_mask],
                ],
                axis=1,
            )
            random_center_idx = np.random.choice(len(xy_good))
            c_y, c_x = xy_good[random_center_idx]

        # just to make one crop, these are the global coordinates
        y1 = y_r + c_y - wh_new // 2
        y2 = y1 + wh_new
        x1 = x_r + c_x - wh_new // 2
        x2 = x1 + wh_new

        # crop every image (assume they have the same w,h)
        for i in range(len(images)):
            images[i] = images[i][y1:y2, x1:x2, ...]
            images[i] = cv2.resize(images[i], (self.width, self.height))

        if was_list:
            return images
        else:
            return images[0]

    def store_crop(self, crop_ratio):
        self.crop_ratio = crop_ratio

    def postprocess(self, item):
        (image_0, image_1, emb_idx, mask) = item
        image_0 = np.asarray(image_0)
        image_1 = np.asarray(image_1)
        mask = np.asarray(mask)

        try:
            if self.crop_type == "center":
                image_0_cropped, image_1_cropped, mask_cropped = (
                    self.make_random_crop_center(
                        [image_0, image_1, mask], mask * 255
                    )
                )
            elif self.crop_type == "mask":
                image_0_cropped, image_1_cropped, mask_cropped = (
                    self.make_random_crop_mask(
                        [image_0, image_1, mask], mask * 255
                    )
                )
        except Exception as e:
            print(e)
            torch.save(
                [image_0, image_1, mask, self.crop_ratio], "./exception.pth"
            )
            raise e

        image_0_cropped = self.transform(image_0_cropped)
        image_1_cropped = self.transform(image_1_cropped)
        mask_cropped = self.transform(mask_cropped * 255)

        return image_0_cropped, image_1_cropped, emb_idx, mask_cropped

    def __len__(self):
        if self.take_only < 0:
            return len(self.images_iclight)
        else:
            return self.take_only

    def __getitem__(self, idx):
        if idx in self.cached:
            return self.postprocess(self.cached[idx])

        image_data = self.data_prompts[idx]
        image_path_0 = self.image_dir / image_data["fg_name"]
        image_path_1 = self.save_iclight_dir / image_data["result_image"]

        emb_idx = torch.tensor([1])

        image_0 = PILImage.open(image_path_0)
        image_1 = PILImage.open(image_path_1)

        mask_path_0 = (
            self.image_dir.parent
            / "mask"
            / image_data["fg_name"].replace(".png", ".jpg")
        )
        mask = cv2.imread(str(mask_path_0), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 64).astype(np.uint8)

        # remove background from image_1
        if self.make_dark_background:
            image_1 = np.asarray(image_1)
            image_1 = image_1 * (mask)
            image_1 = image_1.clip(0, 255).astype(np.uint8)
            image_1 = PILImage.fromarray(image_1)

        self.cached[idx] = (image_0, image_1, emb_idx, mask)

        return self.postprocess(self.cached[idx])
