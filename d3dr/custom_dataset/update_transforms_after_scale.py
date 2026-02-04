import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
import tyro


@dataclass
class Arguments:
    data_folder: Path
    downscale_factor: float = 1.0
    poses_scale: float = 1.0
    scale_point_cloud: bool = False


def main(params: Arguments):
    transforms_file_path = params.data_folder / "transforms.json"
    with open(transforms_file_path) as f:
        transforms = json.load(f)

    transforms["fl_x"] /= params.downscale_factor
    transforms["fl_y"] /= params.downscale_factor
    transforms["cx"] /= params.downscale_factor
    transforms["cy"] /= params.downscale_factor
    transforms["w"] = int(transforms["w"] // params.downscale_factor)
    transforms["h"] = int(transforms["h"] // params.downscale_factor)
    for i in range(len(transforms["frames"])):
        tf_matrix = np.array(transforms["frames"][i]["transform_matrix"])
        tf_matrix[:3, 3] *= params.poses_scale
        transforms["frames"][i]["transform_matrix"] = tf_matrix.tolist()

    with open(params.transforms_file_path, "w") as f:
        json.dump(transforms, f, indent=4)

    if params.scale_point_cloud:
        pcd = o3d.io.read_point_cloud(
            str(params.data_folder / "point_cloud.ply")
        )
        pcd.scale(params.poses_scale, center=pcd.get_center())
        o3d.io.write_point_cloud(
            str(params.data_folder / "point_cloud.ply"), pcd
        )


if __name__ == "__main__":
    params = tyro.cli(Arguments)
    main(params)
