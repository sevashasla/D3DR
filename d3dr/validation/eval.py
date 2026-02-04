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
from typing import Optional

import torch
import tyro
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import CONSOLE
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)


@dataclass
class ComputePSNR:
    """Load a checkpoint, compute some PSNR metrics, and save it to a JSON file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the output file.
    output_path: Path = Path("output.json")

    turn_off_shadows: bool = False
    """Whether to turn off scene shadows during metrics computation"""

    get_std: Optional[bool] = False

    def main(self) -> None:
        """Main function."""
        config, pipeline, checkpoint_path, _ = eval_setup(self.load_config)
        if self.turn_off_shadows:
            pipeline.model.config.scene_shadows = "none"

        assert self.output_path.suffix == ".json"

        num_images = len(pipeline.datamanager.images)

        metrics_dict_list = []
        pipeline.eval()
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeElapsedColumn(),
            MofNCompleteColumn(),
            transient=True,
        ) as progress:
            task = progress.add_task(
                "[green]Evaluating all images...", total=num_images
            )

            for idx in range(num_images):
                camera, batch = pipeline.datamanager.next_eval(idx)
                outputs = pipeline.model.get_outputs_for_camera(camera=camera)
                height, width = camera.height, camera.width
                num_rays = height * width
                metrics_dict, image_dict = (
                    pipeline.model.get_image_metrics_and_images(outputs, batch)
                )
                metrics_dict_list.append(metrics_dict)
                progress.advance(task)

        metrics_dict = {}

        for key in metrics_dict_list[0].keys():
            if self.get_std:
                key_std, key_mean = torch.std_mean(
                    torch.tensor(
                        [
                            metrics_dict[key]
                            for metrics_dict in metrics_dict_list
                            if len(metrics_dict) > 0
                        ]
                    )
                )
                metrics_dict[key] = float(key_mean)
                metrics_dict[f"{key}_std"] = float(key_std)
            else:
                metrics_dict[key] = float(
                    torch.mean(
                        torch.tensor(
                            [
                                metrics_dict[key]
                                for metrics_dict in metrics_dict_list
                                if len(metrics_dict) > 0
                            ]
                        )
                    )
                )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        # Get the output and define the names to save to
        benchmark_info = {
            "experiment_name": config.experiment_name,
            "method_name": config.method_name,
            "checkpoint": str(checkpoint_path),
            "results": metrics_dict,
        }
        print(benchmark_info)
        # Save output to output file
        self.output_path.write_text(
            json.dumps(benchmark_info, indent=2), "utf8"
        )
        CONSOLE.print(f"Saved results to: {self.output_path}")


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(ComputePSNR).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ComputePSNR)  # noqa
