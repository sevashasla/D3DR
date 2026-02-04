from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional, Type

import numpy as np
import open3d as o3d
import torch
from d3dr.scripts.align_depth import MaybeColmapToAlignedMonoDepths
from d3dr.scripts.normals_from_pretrain import (
    NormalsFromPretrained,
    normals_from_depths,
)
from natsort import natsorted
from rich.console import Console

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import CAMERA_MODEL_TO_TYPE, Cameras, CameraType
from nerfstudio.data.dataparsers.nerfstudio_dataparser import (
    Nerfstudio,
    NerfstudioDataParserConfig,
    DataparserOutputs,
)

from nerfstudio.data.utils.dataparsers_utils import (
    get_train_eval_split_all,
    get_train_eval_split_filename,
    get_train_eval_split_fraction,
    get_train_eval_split_interval,
)

from nerfstudio.data.scene_box import SceneBox
from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from nerfstudio.process_data.colmap_utils import colmap_to_json
from nerfstudio.utils.io import load_from_json
from nerfstudio.utils.rich_utils import CONSOLE

MAX_AUTO_RESOLUTION = 1600
CONSOLE = Console()


@dataclass
class NSDNDataParserConfig(NerfstudioDataParserConfig):
    _target: Type = field(default_factory=lambda: NSDNDataParser)

    depth_mode: Literal["mono", "none"] = "mono"
    """Which depth data to load"""
    is_euclidean_depth: bool = False
    """Whether input depth maps are Euclidean distances (or z-distances)."""
    load_depths: bool = True
    """Whether to load depth maps"""
    mono_pretrain: Literal["zoe"] = "zoe"
    """Which mono depth pretrain model to use."""
    load_normals: bool = True
    """Set to true to use ground truth normal maps"""
    normal_format: Literal["opencv", "opengl"] = "opengl"
    """Which format the normal maps in camera frame are saved in."""
    normals_from: Literal["depth", "pretrained"] = "pretrained"
    """If no ground truth normals, generate normals either from sensor depths or from pretrained model."""
    load_pcd_normals: bool = False
    """Whether to load pcd normals for normal initialisation"""
    load_3D_points: bool = True
    """Whether to load the 3D points from the sparse_pc.ply."""
    eval_mode: Literal["fraction", "filename", "interval", "all"] = "interval"
    """
    Interval uses every nth frame for eval (used by most academic papers, e.g. MipNerf360, GSplat).
    """
    load_every: int = 1  # 30 for eval train split
    """load every n'th frame from the dense trajectory from the train split"""
    eval_interval: int = -1
    """eval interval"""
    depth_unit_scale_factor: float = 1
    """Scales the depth values to meters. Default value is 0.001 for a millimeter to meter conversion."""
    images_path: Path = Path("images")
    """Path to images directory relative to the data path."""
    depths_path: Optional[Path] = None
    """Path to depth maps directory. If not set, depths are not loaded."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "none"
    """The method to use for orientation."""
    center_method: Literal["poses", "focus", "none"] = "none"
    """The method to use to center the poses."""
    auto_scale_poses: bool = False
    """Whether to automatically scale the poses to fit in +/- 1 bounding box."""
    downscale_factor: int = 1


class NSDNDataParser(Nerfstudio):
    config: NSDNDataParserConfig

    def __init__(self, config: NSDNDataParserConfig):
        super().__init__(config)
        self.config = config
        self._downscale_factor = None

    def get_depth_filepaths(self):
        # TODO this only returns aligned monodepths right now
        depth_paths = natsorted(
            glob.glob(f"{self.config.data}/mono_depth/*_aligned.npy")
        )
        if not depth_paths:
            CONSOLE.log("Could not find _aligned.npy depths, trying *.npy")
            depth_paths = natsorted(glob.glob(f"{self.config.data}/mono_depth/*.npy"))
        if depth_paths:
            CONSOLE.log("Found depths ending in *.npy")
        else:
            CONSOLE.log("Could not find depths :(")
            # quit()
        return depth_paths

    def get_normal_filepaths(self):
        return natsorted(glob.glob(f"{self.normal_save_dir}/*.png"))

    def _generate_dataparser_outputs(self, split: str = "train", **kwargs):
        assert (
            self.config.data.exists()
        ), f"Data directory {self.config.data} does not exist."
        
        # choose normal path
        if self.config.normals_from == "depth":
            self.normal_save_dir = self.config.data / Path("normals_from_depth")
        else:
            self.normal_save_dir = self.config.data / Path("normals_from_pretrain")
        
        meta = load_from_json(self.config.data / "transforms.json")
        data_dir = self.config.data

        image_filenames = []
        mask_filenames = []
        depth_filenames = []
        poses = []

        fx_fixed = "fl_x" in meta
        fy_fixed = "fl_y" in meta
        cx_fixed = "cx" in meta
        cy_fixed = "cy" in meta
        height_fixed = "h" in meta
        width_fixed = "w" in meta
        distort_fixed = False
        for distort_key in ["k1", "k2", "k3", "p1", "p2", "distortion_params"]:
            if distort_key in meta:
                distort_fixed = True
                break
        fisheye_crop_radius = meta.get("fisheye_crop_radius", None)
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        # sort the frames by fname
        fnames = []
        for frame in meta["frames"]:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)
            fnames.append(fname)
        inds = np.argsort(fnames)
        frames = [meta["frames"][ind] for ind in inds]

        for frame in frames:
            filepath = Path(frame["file_path"])
            fname = self._get_fname(filepath, data_dir)

            if not fx_fixed:
                assert "fl_x" in frame, "fx not specified in frame"
                fx.append(float(frame["fl_x"]))
            if not fy_fixed:
                assert "fl_y" in frame, "fy not specified in frame"
                fy.append(float(frame["fl_y"]))
            if not cx_fixed:
                assert "cx" in frame, "cx not specified in frame"
                cx.append(float(frame["cx"]))
            if not cy_fixed:
                assert "cy" in frame, "cy not specified in frame"
                cy.append(float(frame["cy"]))
            if not height_fixed:
                assert "h" in frame, "height not specified in frame"
                height.append(int(frame["h"]))
            if not width_fixed:
                assert "w" in frame, "width not specified in frame"
                width.append(int(frame["w"]))
            if not distort_fixed:
                distort.append(
                    torch.tensor(frame["distortion_params"], dtype=torch.float32)
                    if "distortion_params" in frame
                    else camera_utils.get_distortion_params(
                        k1=float(frame["k1"]) if "k1" in frame else 0.0,
                        k2=float(frame["k2"]) if "k2" in frame else 0.0,
                        k3=float(frame["k3"]) if "k3" in frame else 0.0,
                        k4=float(frame["k4"]) if "k4" in frame else 0.0,
                        p1=float(frame["p1"]) if "p1" in frame else 0.0,
                        p2=float(frame["p2"]) if "p2" in frame else 0.0,
                    )
                )

            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            if "mask_path" in frame:
                mask_filepath = Path(frame["mask_path"])
                mask_fname = self._get_fname(
                    mask_filepath,
                    data_dir,
                    downsample_folder_prefix="masks_",
                )
                mask_filenames.append(mask_fname)
        
        assert len(mask_filenames) == 0 or (
            len(mask_filenames) == len(image_filenames)
        ), """
        Different number of image and mask filenames.
        """
        depth_filenames = self.get_depth_filepaths()
        assert len(depth_filenames) == 0 or (
            len(depth_filenames) == len(image_filenames)
        ), """
        Different number of image and depth filenames.
        """
        normal_filenames = self.get_normal_filepaths()
        assert len(normal_filenames) == 0 or (
            len(normal_filenames) == len(image_filenames)
        ), """
        Different number of image and normal filenames.
        """

        poses = [
            pose for img, pose in natsorted(zip(image_filenames, poses), lambda x: x[0])
        ]
        image_filenames = natsorted(image_filenames)
        mask_filenames = natsorted(mask_filenames)

        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, transform_matrix = camera_utils.auto_orient_and_center_poses(
            poses,
            method=self.config.orientation_method,
            center_method=self.config.center_method,
        )

        # Scale poses
        scale_factor = 1.0
        if self.config.auto_scale_poses:
            scale_factor /= float(torch.max(torch.abs(poses[:, :3, 3])))
        scale_factor *= self.config.scale_factor
        poses[:, :3, 3] *= scale_factor

        # Choose image_filenames and poses based on split, but after auto orient and scaling the poses.
        has_split_files_spec = any(
            f"{split}_filenames" in meta for split in ("train", "val", "test")
        )
        if f"{split}_filenames" in meta:
            # Validate split first
            split_filenames = set(
                self._get_fname(Path(x), data_dir) for x in meta[f"{split}_filenames"]
            )
            unmatched_filenames = split_filenames.difference(image_filenames)
            if unmatched_filenames:
                raise RuntimeError(
                    f"Some filenames for split {split} were not found: {unmatched_filenames}."
                )

            indices = [
                i for i, path in enumerate(image_filenames) if path in split_filenames
            ]
            CONSOLE.log(f"[yellow] Dataset is overriding {split}_indices to {indices}")
            indices = np.array(indices, dtype=np.int32)
        elif has_split_files_spec:
            raise RuntimeError(
                f"The dataset's list of filenames for split {split} is missing."
            )
        else:
            # find train and eval indices based on the eval_mode specified
            if self.config.eval_mode == "fraction":
                i_train, i_eval = get_train_eval_split_fraction(
                    image_filenames, self.config.train_split_fraction
                )
            elif self.config.eval_mode == "filename":
                i_train, i_eval = get_train_eval_split_filename(image_filenames)
            elif self.config.eval_mode == "interval":
                i_train, i_eval = get_train_eval_split_interval(
                    image_filenames, 
                    self.config.eval_interval if self.config.eval_interval > 0 else len(image_filenames) + 1
                )
                print(i_train, i_eval)
            elif self.config.eval_mode == "all":
                CONSOLE.log(
                    "[yellow] Be careful with '--eval-mode=all'. If using camera optimization, the cameras may diverge in the current implementation, giving unpredictable results."
                )
                i_train, i_eval = get_train_eval_split_all(image_filenames)
            else:
                raise ValueError(f"Unknown eval mode {self.config.eval_mode}")

            if split == "train":
                indices = i_train
            elif split in ["val", "test"]:
                indices = i_eval
            else:
                raise ValueError(f"Unknown dataparser split {split}")

        # get only images from indices
        image_filenames = [image_filenames[i] for i in indices]
        depth_filenames = (
            [depth_filenames[i] for i in indices] if len(depth_filenames) > 0 else []
        )
        mask_filenames = (
            [mask_filenames[i] for i in indices] if len(mask_filenames) > 0 else []
        )

        normal_filenames = (
            [normal_filenames[i] for i in indices] if len(normal_filenames) > 0 else []
        )

        idx_tensor = torch.tensor(indices, dtype=torch.long)
        poses = poses[idx_tensor]

        metadata = {}

        # load depths
        if self.config.depth_mode != "none" and self.config.load_depths:
            # assert (self.config.data / "mono_depth").exists(), "depth folder not found"
            metadata["mono_depth_filenames"] = [Path(f) for f in depth_filenames]

        # in x,y,z order
        # assumes that the scene is centered at the origin
        aabb_scale = self.config.scene_scale
        scene_box = SceneBox(
            aabb=torch.tensor(
                [
                    [-aabb_scale, -aabb_scale, -aabb_scale],
                    [aabb_scale, aabb_scale, aabb_scale],
                ],
                dtype=torch.float32,
            )
        )

        if "camera_model" in meta:
            camera_type = CAMERA_MODEL_TO_TYPE[meta["camera_model"]]
        else:
            camera_type = CameraType.PERSPECTIVE

        fx = (
            float(meta["fl_x"])
            if fx_fixed
            else torch.tensor(fx, dtype=torch.float32)[idx_tensor]
        )
        fy = (
            float(meta["fl_y"])
            if fy_fixed
            else torch.tensor(fy, dtype=torch.float32)[idx_tensor]
        )
        cx = (
            float(meta["cx"])
            if cx_fixed
            else torch.tensor(cx, dtype=torch.float32)[idx_tensor]
        )
        cy = (
            float(meta["cy"])
            if cy_fixed
            else torch.tensor(cy, dtype=torch.float32)[idx_tensor]
        )
        height = (
            int(meta["h"])
            if height_fixed
            else torch.tensor(height, dtype=torch.int32)[idx_tensor]
        )
        width = (
            int(meta["w"])
            if width_fixed
            else torch.tensor(width, dtype=torch.int32)[idx_tensor]
        )
        if distort_fixed:
            distortion_params = (
                torch.tensor(meta["distortion_params"], dtype=torch.float32)
                if "distortion_params" in meta
                else camera_utils.get_distortion_params(
                    k1=float(meta["k1"]) if "k1" in meta else 0.0,
                    k2=float(meta["k2"]) if "k2" in meta else 0.0,
                    k3=float(meta["k3"]) if "k3" in meta else 0.0,
                    k4=float(meta["k4"]) if "k4" in meta else 0.0,
                    p1=float(meta["p1"]) if "p1" in meta else 0.0,
                    p2=float(meta["p2"]) if "p2" in meta else 0.0,
                )
            )
        else:
            distortion_params = torch.stack(distort, dim=0)[idx_tensor]


        if (camera_type in [CameraType.FISHEYE, CameraType.FISHEYE624]) and (
            fisheye_crop_radius is not None
        ):
            metadata["fisheye_crop_radius"] = fisheye_crop_radius

        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            distortion_params=distortion_params,
            height=height,
            width=width,
            camera_to_worlds=poses[:, :3, :4],
            camera_type=camera_type,
        )

        # cameras.rescale_output_resolution(scaling_factor=1.0 / downscale_factor)

        if "applied_transform" in meta:
            applied_transform = torch.tensor(
                meta["applied_transform"], dtype=transform_matrix.dtype
            )
            transform_matrix = transform_matrix @ torch.cat(
                [
                    applied_transform,
                    torch.tensor([[0, 0, 0, 1]], dtype=transform_matrix.dtype),
                ],
                0,
            )
        if "applied_scale" in meta:
            applied_scale = float(meta["applied_scale"])
            scale_factor *= applied_scale

        if self.config.load_3D_points:
            # Load 3D points
            ply_file_path = data_dir / "sparse_pc.ply"
            assert ply_file_path.exists(), "sparse_pc.ply not found"

            sparse_points = self._load_3D_points(
                ply_file_path, transform_matrix, scale_factor
            )
            metadata.update(sparse_points)


        metadata.update({"depth_mode": self.config.depth_mode})
        metadata.update({"load_depths": self.config.load_depths})
        metadata.update({"is_euclidean_depth": self.config.is_euclidean_depth})

        if self.config.load_normals and (
            not (self.normal_save_dir).exists()
            or len(os.listdir(self.normal_save_dir)) == 0
        ):
            CONSOLE.print(
                f"[bold yellow]Could not find normals, generating them into {str(self.normal_save_dir)}"
            )
            self.normal_save_dir.mkdir(exist_ok=True, parents=True)
            if self.config.normals_from == "depth":
                normals_from_depths(
                    path_to_transforms=Path(image_filenames[0]).parent.parent
                    / "transforms.json",
                    normal_format=self.config.normal_format,
                )
            elif self.config.normals_from == "pretrained":
                NormalsFromPretrained(data_dir=self.config.data).main()
            else:
                raise NotImplementedError

        if self.config.load_normals:
            normal_filenames = self.get_normal_filepaths()
            metadata.update(
                {"normal_filenames": [Path(normal_filenames[idx]) for idx in indices]}
            )
            metadata.update({"normal_format": self.config.normal_format})

        metadata.update({"load_normals": self.config.load_normals})
        if self.config.load_pcd_normals:
            metadata.update(
                self._load_points3D_normals(points_3d=metadata["points3D_xyz"])
            )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            mask_filenames=mask_filenames if len(mask_filenames) > 0 else None,
            dataparser_scale=scale_factor,
            dataparser_transform=transform_matrix,
            metadata={
                "depth_unit_scale_factor": self.config.depth_unit_scale_factor,
                **metadata,
            },
        )

        return dataparser_outputs

    def _load_points3D_normals(self, points_3d):
        transform_matrix = torch.eye(4, dtype=torch.float, device="cpu")[:3, :4]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d.cpu().numpy())
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        pcd.normalize_normals()
        points3D_normals = torch.from_numpy(np.asarray(pcd.normals, dtype=np.float32))
        points3D_normals = (
            torch.cat(
                (points3D_normals, torch.ones_like(points3D_normals[..., :1])), -1
            )
            @ transform_matrix.T
        )
        return {"points3D_normals": points3D_normals}


NSDNDataParserSpecification = DataParserSpecification(
    config=NSDNDataParserConfig(),
    description="NSND: modified version of Nerfstudio dataparser for DN-Splatter Dataset",
)
