import json
import os
from typing import Dict, Optional

import torch
from scipy.spatial.transform import Rotation as R
from torch import Tensor

from d3dr.utils.rotate_splats import affine_transform_dn_splats


def find_ckpt_path(input_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".ckpt"):
                return os.path.join(root, file)
    raise RuntimeError(f"No .ckpt file found in {input_dir}")


def get_Rt_from_json(transforms_obj: str, directory_path: str):
    # find the path to the object transforms
    if transforms_obj is None:
        with open(os.path.join(directory_path, "config.yml")) as f:
            lines = f.read().splitlines()
        datapath = []
        for i in range(len(lines)):
            line = lines[i].strip()
            if not (line.startswith("data") and "PosixPath" in line):
                continue
            datapath = []
            for k in range(i + 1, len(lines)):
                line = lines[k].strip()
                if not line.startswith("-"):
                    break
                datapath.append(line.split()[1])
            datapath = "/".join(datapath)[1:]
            datapath = os.path.join(os.path.dirname(datapath), "obj_scene_eval")
            if os.path.exists(datapath):
                break
    else:
        datapath = transforms_obj

    # load the transforms
    with open(os.path.join(datapath, "transforms.json")) as f:
        transforms = json.load(f)
    Rot = R.from_euler("xyz", transforms["euler_rotation"]).as_matrix()
    Rot = torch.tensor(Rot, dtype=torch.float32)
    trans = torch.tensor(transforms["object_center"], dtype=torch.float32)
    return Rot, trans, datapath


def load_gauss_params(
    ckpt_path: str, device: str = "cuda"
) -> Dict[str, torch.Tensor]:
    data = torch.load(ckpt_path, map_location=device)
    gauss_params = {}
    pipeline = data["pipeline"]
    for key, value in pipeline.items():
        if "gauss_params" in key:
            real_key = key.split(".")[-1]
            gauss_params[real_key] = value
    return gauss_params


def maybe_cat(tensors, dim: int = 0) -> Optional[Tensor]:
    tensors = [t for t in tensors if t is not None]
    if len(tensors) == 0:
        return None
    return torch.cat(tensors, dim=dim)


def read_from_file_and_rotate(
    init_obj_path,
    init_scene_path,
    transforms_obj,
    comp_device="cuda",
) -> Dict[str, Tensor]:
    # load gauss parameters from checkpoints
    init_gauss_params = {}
    # we need to load transforms
    init_scene_ckpt_path = find_ckpt_path(init_scene_path)
    init_obj_ckpt_path = find_ckpt_path(init_obj_path)

    transform_scene = torch.eye(4).float().to(comp_device)
    transform_obj = torch.eye(4).float().to(comp_device)
    # if False:
    if init_scene_path is not None:
        # BUG: In some cases I need to rotate the scene coefficients too
        # In case when obj_scene transforms is different from scene transforms
        with open(
            os.path.join(init_scene_path, "dataparser_transforms.json")
        ) as f:
            transform_scene[:3, :4] = (
                torch.tensor(json.load(f)["transform"]).float().to(comp_device)
            )
        scene_gauss_params = load_gauss_params(
            init_scene_ckpt_path, device=comp_device
        )
        init_gauss_params.update(scene_gauss_params)
    if init_obj_path is not None:
        # If for some reason object_transforms is different from scene_transforms
        # then we need to rotate the object coefficients
        with open(
            os.path.join(init_obj_path, "dataparser_transforms.json")
        ) as f:
            transform_obj[:3, :4] = (
                torch.tensor(json.load(f)["transform"]).float().to(comp_device)
            )

        # load config file
        rotation_matrix, movement_vector, _ = get_Rt_from_json(
            transforms_obj, init_obj_path
        )

        # obj space to scene space
        transform_obj_to_scene = transform_scene @ torch.linalg.inv(
            transform_obj
        )

        # initial transfromation in object space
        transform_obj_init = torch.eye(4).float().to(comp_device)
        transform_obj_init[:3, :3] = rotation_matrix
        transform_obj_init[:3, 3] = movement_vector

        # initial transformation in scene space
        transform_obj_real = transform_obj_to_scene @ transform_obj_init
        rotation_matrix_real = transform_obj_real[:3, :3]
        movement_vector_real = transform_obj_real[:3, 3]

        obj_gauss_params = load_gauss_params(
            init_obj_ckpt_path, device=comp_device
        )
        obj_gauss_params = affine_transform_dn_splats(
            obj_gauss_params,
            rotation_matrix_real,
            movement_vector_real,
            comp_device=comp_device,
        )
        for key in obj_gauss_params:
            init_gauss_params[key] = maybe_cat(
                [init_gauss_params.get(key, None), obj_gauss_params[key]],
                dim=0,
            )
        init_gauss_params["obj_ids"] = torch.zeros(
            init_gauss_params["means"].shape[0],
            dtype=torch.bool,
            device=comp_device,
        )
        init_gauss_params["obj_ids"][-obj_gauss_params["means"].shape[0] :] = (
            True
        )
        init_gauss_params["obj_ids"].requires_grad = False
    return init_gauss_params
