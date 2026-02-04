# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# A fixed script for my validation pipeline

#!/usr/bin/env python
"""
eval.py
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import tyro

import os
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.cameras.cameras import Cameras

from d3dr.utils.read_and_rotate import get_Rt_from_json

import numpy as np
from PIL import Image as PILImage
import cv2

class DatasetSimple:
    def __init__(self, obj_scene_datapath):
        self.obj_scene_datapath = obj_scene_datapath
        self.device = "cuda"
        with open(os.path.join(self.obj_scene_datapath, "transforms.json"), "r") as f:
            data = json.load(f)

        self.eval_fl = data["fl_x"]
        self.resolution = data["w"]
        self.eval_poses = torch.tensor([f["transform_matrix"] for f in data["frames"]])
        self.image_paths = [
            os.path.join(
                self.obj_scene_datapath, 
                f["file_path"]
            ) for f in data["frames"]
        ]
        self.images = [np.asarray(PILImage.open(p)) for p in self.image_paths]
        self.images = [torch.tensor(im).float() for im in self.images]
        # self.masks_paths = [
        #     os.path.join(
        #         self.obj_scene_datapath, 
        #         f["file_path"]\
        #         .replace("images", "masks")\
        #         .replace("color", "mask").replace("frame", "mask")
        #     ) for f in data["frames"]
        # ]
        self.masks_paths = sorted(list((Path(self.obj_scene_datapath) / "masks").glob("*")))
        self.masks = [np.asarray(PILImage.open(p)) for p in self.masks_paths]
        self.masks = [torch.tensor(im).float() for im in self.masks]
        # update with good eval ids (where mask is not empty)
        good_eval_ids = [i for i, mask in enumerate(self.masks) if torch.sum(mask > 0.1) > 150]

        self.eval_poses = self.eval_poses[good_eval_ids]
        self.images = [self.images[i] for i in good_eval_ids]
        self.masks = [self.masks[i] for i in good_eval_ids]

        self.eval_cameras = Cameras(
            camera_to_worlds=self.eval_poses,
            fx=self.eval_fl,
            fy=self.eval_fl,
            cx=self.resolution / 2,
            cy=self.resolution / 2,
        )
        self.eval_count = 0

    def _get_obj_scene_el(self, idx_global):
        idx_local = idx_global % len(self.eval_poses)
        images = self.images[idx_local].to(self.device) / 255.0
        masks = self.masks[idx_local].to(self.device)
        cameras = self.eval_cameras[idx_local:idx_local + 1].to(self.device)

        return {
            "cameras": cameras,
            "images": images,
            "masks": masks,
        }

    def next_eval_image(self, step: int) -> Tuple[Cameras, Dict]:
        torch.manual_seed(42)
        self.eval_count += 1

        curr_el = self._get_obj_scene_el(step)

        cameras = curr_el["cameras"].to(self.device)
        data = {
            "image": curr_el["images"].to(self.device),
            "mask": curr_el["masks"].to(self.device),
        }

        return cameras, data

import torch

class Metrics:
    def __init__(self):
        self.device = "cuda"
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=11).to(self.device)
        self.lpips = LearnedPerceptualImagePatchSimilarity().to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

    @torch.no_grad()
    def _psnr_masked(self, x, y, mask):
        if torch.any(mask):
            x = x.permute(2, 3, 1, 0) # [1, C, H, W] -> [H, W, C, 1]
            y = y.permute(2, 3, 1, 0)
            mse = ((x - y)[mask] ** 2).sum() / mask.float().sum()
            return -10.0 * torch.log10(mse)
        else:
            raise RuntimeError("Mask is empty! in _masked_mse")

    def get_image_metrics_and_images(
            self,
            outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
        ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
            """Writes the test image outputs.

            Args:
                image_idx: Index of the image.
                step: Current step.
                batch: Batch of data.
                outputs: Outputs of the model.

            Returns:
                A dictionary of metrics.
            """
            gt_rgb = batch["image"]
            gt_mask = batch["mask"]
            gt_mask_01 = (gt_mask > 0.1)
            x, y, w, h = cv2.boundingRect(gt_mask_01.squeeze().cpu().numpy().astype(np.uint8))

            predicted_rgb = outputs["rgb"]
            combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

            # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
            gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
            cropped_gt_rgb = gt_rgb[:, :, y:y+h, x:x+w]
            predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]
            cropped_predicted_rgb = predicted_rgb[:, :, y:y+h, x:x+w]
            
            psnr = self.psnr(gt_rgb, predicted_rgb)
            ssim = self.ssim(gt_rgb, predicted_rgb)
            psnr_cropped = self.psnr(cropped_gt_rgb, cropped_predicted_rgb)
            lpips = self.lpips(gt_rgb, predicted_rgb)

            psnr_part = self._psnr_masked(gt_rgb, predicted_rgb, gt_mask_01)
            ssim_part = self.ssim(cropped_gt_rgb, cropped_predicted_rgb)

            # all of these metrics will be logged as scalars
            metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
            metrics_dict["lpips"] = float(lpips)
            metrics_dict["psnr_part"] = float(psnr_part.item())
            metrics_dict["psnr_cropped"] = float(psnr_cropped.item())
            metrics_dict["ssim_part"] = float(ssim_part)

            images_dict = {"img": combined_rgb}

            # torch.save(
            #     {
            #         "combined_rgb": combined_rgb,
            #         "cropped_gt_rgb": cropped_gt_rgb,
            #         "cropped_predicted_rgb": cropped_predicted_rgb,
            #         "metrics": metrics_dict,
            #         "gt_mask_01": gt_mask_01,
            #         "gt_mask": gt_mask, 
            #     }, 
            #     "/home/skorokho/coding/voi_gs/tmp/metrics_debug.pth"
            # )
            
            return metrics_dict, images_dict

@dataclass
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.json")
    # Optional path to save rendered outputs to.
    render_output_path: Optional[Path] = None
    obj_scene_datapath: Optional[Path] = None
    """Path to the true dataset"""

    get_std: Optional[bool] = False

    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)

        metrics_cls = Metrics()

        if self.obj_scene_datapath is None:
            _, _, obj_scene_datapath = get_Rt_from_json(None, pipeline.model.config.init_obj_path)
            obj_scene_datapath = obj_scene_datapath.replace("obj_scene", "obj_scene_eval")
        else:
            obj_scene_datapath = self.obj_scene_datapath
        print(f"obj_scene_datapath: {obj_scene_datapath}")
        dataset = DatasetSimple(obj_scene_datapath)

        assert self.output_path.suffix == ".json"
        if self.render_output_path is not None:
            self.render_output_path.mkdir(parents=True, exist_ok=True)

        num_images = len(dataset.images)
        # for i in range(dataset.images)

        metrics_dict_list = []
        pipeline.eval()
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task("[green]Evaluating all images...", total=num_images)
            
            for idx in range(num_images):
                camera, batch = dataset.next_eval_image(idx)
                outputs = pipeline.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, image_dict = metrics_cls.get_image_metrics_and_images(outputs, batch)
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)

        metrics_dict = {}
        for key in metrics_dict_list[0].keys():
            if self.get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list])
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(torch.tensor([metrics_dict[key] for metrics_dict in metrics_dict_list]))
                )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        # Save output to output file
        self.output_path.write_text(json.dumps(benchmark_info, indent=2), "utf8")
        CONSOLE.print(f"Saved results to: {self.output_path}")

def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputePSNR).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputePSNR)  # noqa
