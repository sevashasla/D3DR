import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import numpy as np
import open3d as o3d
import torch
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManager,
    DataManagerConfig,
)
from nerfstudio.data.datamanagers.random_cameras_datamanager import (
    TrivialDataset,
)
from nerfstudio.data.utils.dataloaders import RandIndicesEvalDataloader
from PIL import Image as PILImage
from rich.progress import Console
from torch import Tensor
from torch.nn.parameter import Parameter
from typing_extensions import Literal

CONSOLE = Console(width=120)

from d3dr.utils.dummy_class import DummyClass
from d3dr.utils.read_and_rotate import (
    find_ckpt_path,
    get_Rt_from_json,
    load_gauss_params,
)


@dataclass
class RandomCamerasDataManagerObjSceneConfig(DataManagerConfig):
    """Configuration for data manager that does not load from a dataset. Instead, it generates random poses."""

    _target: Type = field(
        default_factory=lambda: RandomCamerasDataManagerObjScene
    )

    eval_resolution: Optional[int] = None
    """It be taken from init_obj_path"""
    train_resolution: Optional[int] = None
    """It be taken from init_obj_path"""
    camera_angle_x: Optional[float] = None
    """It be taken from init_obj_path"""
    focal: Optional[float] = None
    """It be taken from init_obj_path"""
    center: Optional[Tuple[float, float, float]] = None
    """It be taken from init_obj_path"""

    num_eval_angles: int = 256
    """Number of evaluation angles"""
    train_images_per_batch: int = 1
    """Number of images per batch for training"""
    eval_images_per_batch: int = 1
    """Number of images per batch for evaluation"""
    betas: Tuple[float, float] = (0.1, 0.6)
    """Std of radius of camera orbit"""
    betas_refine: Tuple[float, float] = (0.5, 0.9)
    """Std of radius of camera orbit"""
    vertical_rotation_range: Tuple[float, float] = (-90, 0)
    """Range of vertical rotation"""
    position_jitter: float = 0.05
    """Std of camera direction jitter, so we don't just point the cameras towards the center every time"""
    transforms_obj: Optional[str] = None
    """Path to the json file with the transforms for the object"""
    init_obj_path: Optional[str] = None
    """Path to the initial object"""
    init_scene_path: Optional[str] = None
    """Path to the initial scene"""
    voxel_size: float = 0.1
    """Voxel size for the voxelization of the scene"""
    shift_from_hit: float = 0.1
    """Shift from from the hit point to the object (to avoid huge gaussians)"""
    num_fixed_train_angles: int = 100
    """Take fixed train angles (directions). If <= 0 then the dataset has only random"""
    use_min_for_generation: int = 0
    """Whether to use min or mean for image generation"""
    seed: int = 42
    """Random seed"""


class RandomCamerasDataManagerObjScene(DataManager):  # pylint: disable=abstract-method
    """Basic stored data manager implementation.

    This is pretty much a port over from our old dataloading utilities, and is a little jank
    under the hood. We may clean this up a little bit under the hood with more standard dataloading
    components that can be strung together, but it can be just used as a black box for now since
    only the constructor is likely to change in the future, or maybe passing in step number to the
    next_train and next_eval functions.

    Args:
        config: the DataManagerConfig used to instantiate class
    """

    config: RandomCamerasDataManagerObjSceneConfig

    def generate_points_on_sphere_dist(
        self,
        a=0.1,
        vertical_rotation_range=(-90, 90),
        r=1.0,
    ):
        """
        vetrical_angles_range: (min, max)
        down is 90 and up is -90
        (nerfstudio did this not me...)
        """
        result = []
        curr_phi = -np.deg2rad(vertical_rotation_range[1])
        phi_add = 2 * math.atan(a / (2 * r))
        while curr_phi < -np.deg2rad(vertical_rotation_range[0]):
            curr_radius = r * math.cos(curr_phi)
            num_generate_curr = np.ceil(2 * math.pi * curr_radius / a)
            for i in range(int(num_generate_curr)):
                curr_theta = 2 * math.pi * i / num_generate_curr
                x = r * math.cos(curr_phi) * math.cos(curr_theta)
                y = r * math.cos(curr_phi) * math.sin(curr_theta)
                z = r * math.sin(curr_phi)
                result.append([x, y, z])
            curr_phi += phi_add
        return torch.tensor(result)

    def generate_points_on_sphere(
        self,
        n=100,
        vertical_rotation_range=(90, -90),
    ):

        left = 1.0 / n
        right = 1.0
        while (right - left) > 1e-3:
            mid = (left + right) / 2
            points = self.generate_points_on_sphere_dist(
                a=mid, vertical_rotation_range=vertical_rotation_range
            )
            if len(points) > n:
                left = mid
            else:
                right = mid
        return points

    def _generate_voxel_scene(
        self,
        scene_pts: Tensor,
        scene_pts_width: Tensor,
        vs: float,
        device: str = "cuda",
    ):
        bb_scene = (scene_pts.min(dim=0).values, scene_pts.max(dim=0).values)
        print("bb_scene:", bb_scene)

        # create a mesh from scene_pts
        N_voxels = torch.ceil((bb_scene[1] - bb_scene[0]) / vs).to(torch.long)

        voxels_ids = torch.floor((scene_pts - bb_scene[0]) / vs)
        voxels_ids = torch.stack(
            [
                voxels_ids[:, i].clip(0, N_voxels[i] - 1).to(torch.long)
                for i in range(3)
            ],
            axis=1,
        )
        voxels_ids_uniq, inverse_indices = torch.unique(
            voxels_ids, return_inverse=True, dim=0
        )

        ids_x = voxels_ids_uniq[:, 0]
        ids_y = voxels_ids_uniq[:, 1]
        ids_z = voxels_ids_uniq[:, 2]

        # find voxels representations
        voxels_x = (
            bb_scene[0][0] + torch.arange(0, N_voxels[0], device=device) * vs
        )
        voxels_y = (
            bb_scene[0][1] + torch.arange(0, N_voxels[1], device=device) * vs
        )
        voxels_z = (
            bb_scene[0][2] + torch.arange(0, N_voxels[2], device=device) * vs
        )

        voxels_x = voxels_x[ids_x]
        voxels_y = voxels_y[ids_y]
        voxels_z = voxels_z[ids_z]

        # generate cubes
        vectors = torch.cat(
            [
                torch.stack([voxels_x + vs, voxels_y, voxels_z], dim=1),
                torch.stack([voxels_x + vs, voxels_y + vs, voxels_z], dim=1),
                torch.stack([voxels_x, voxels_y + vs, voxels_z], dim=1),
                torch.stack([voxels_x, voxels_y, voxels_z], dim=1),
                torch.stack([voxels_x + vs, voxels_y, voxels_z + vs], dim=1),
                torch.stack(
                    [voxels_x + vs, voxels_y + vs, voxels_z + vs], dim=1
                ),
                torch.stack([voxels_x, voxels_y + vs, voxels_z + vs], dim=1),
                torch.stack([voxels_x, voxels_y, voxels_z + vs], dim=1),
            ],
            dim=0,
        )

        # update the cubes sizes because of their widths
        # find centers of the cubes
        cubes_centers = torch.stack(
            [ch for ch in vectors.chunk(8, dim=0)], dim=0
        ).mean(dim=0)
        cubes_centers = cubes_centers.repeat(8, 1)
        # find final widths in every cube as mean of its gaussians
        scene_pts_width_ids = torch.zeros_like(voxels_x)
        scene_pts_width_ids.scatter_reduce_(
            dim=0,
            index=inverse_indices,
            src=scene_pts_width,
            reduce="mean" if not self.config.use_min_for_generation else "min",
        )

        scene_pts_width_ids = scene_pts_width_ids.repeat(8, 1)
        # update the width
        vectors = cubes_centers + (vectors - cubes_centers) / (vs * 0.5) * (
            vs * 0.5 + scene_pts_width_ids.reshape(-1, 1)
        )

        # generate faces
        N = ids_x.shape[0]
        arn = torch.arange(N, device=device)
        faces = torch.cat(
            [
                torch.stack(
                    [arn + (1 - 1) * N, arn + (2 - 1) * N, arn + (5 - 1) * N],
                    dim=1,
                ),
                torch.stack(
                    [arn + (2 - 1) * N, arn + (6 - 1) * N, arn + (5 - 1) * N],
                    dim=1,
                ),
                torch.stack(
                    [arn + (2 - 1) * N, arn + (3 - 1) * N, arn + (6 - 1) * N],
                    dim=1,
                ),
                torch.stack(
                    [arn + (3 - 1) * N, arn + (7 - 1) * N, arn + (6 - 1) * N],
                    dim=1,
                ),
                torch.stack(
                    [arn + (3 - 1) * N, arn + (4 - 1) * N, arn + (7 - 1) * N],
                    dim=1,
                ),
                torch.stack(
                    [arn + (4 - 1) * N, arn + (8 - 1) * N, arn + (7 - 1) * N],
                    dim=1,
                ),
                torch.stack(
                    [arn + (4 - 1) * N, arn + (1 - 1) * N, arn + (8 - 1) * N],
                    dim=1,
                ),
                torch.stack(
                    [arn + (1 - 1) * N, arn + (5 - 1) * N, arn + (8 - 1) * N],
                    dim=1,
                ),
                torch.stack(
                    [arn + (1 - 1) * N, arn + (4 - 1) * N, arn + (2 - 1) * N],
                    dim=1,
                ),
                torch.stack(
                    [arn + (4 - 1) * N, arn + (3 - 1) * N, arn + (2 - 1) * N],
                    dim=1,
                ),
                torch.stack(
                    [arn + (5 - 1) * N, arn + (6 - 1) * N, arn + (8 - 1) * N],
                    dim=1,
                ),
                torch.stack(
                    [arn + (6 - 1) * N, arn + (7 - 1) * N, arn + (8 - 1) * N],
                    dim=1,
                ),
            ],
            dim=0,
        )

        if self.config.use_min_for_generation != -1:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vectors.cpu().numpy())
            mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
            mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
            scene = o3d.t.geometry.RaycastingScene()
            scene.add_triangles(mesh_legacy)
        else:
            scene = o3d.t.geometry.RaycastingScene()
        return scene

    def _calculate_dist(
        self,
        camera_angle_x,
        bb_obj,
        beta,
    ):
        # generate poses
        obj_width = (bb_obj[1] - bb_obj[0]).max().item()

        # (w / 2)/ (d * tan(fov / 2)) = sqrt(beta) =>
        # d = (w / 2) / (tan(fov / 2) * sqrt(beta))
        min_dist = (
            0.5 * obj_width / (math.tan(camera_angle_x / 2) * math.sqrt(beta))
        )
        return min_dist

    def generate_random_cameras(
        self,
        num_poses: int = 1,
        chunk_size: int = 64,
        resolution: int = 800,
        directions: Optional[Tensor] = None,
    ):
        # generate poses
        poses = []
        bb_obj_np = (self.bb_obj[0].cpu().numpy(), self.bb_obj[1].cpu().numpy())
        while len(poses) < num_poses:
            # random positions
            random_center = (
                self.bb_obj[0] + self.bb_obj[1]
            ) / 2 + self.config.position_jitter * torch.rand(
                (chunk_size, 3), device=self.device
            ) * (self.bb_obj[1] - self.bb_obj[0]) / 2

            # generate random directions
            if directions is None:
                random_theta = (
                    torch.rand(chunk_size, device=self.device) * 2 * torch.pi
                )
                random_phi = np.deg2rad(
                    self.config.vertical_rotation_range[0]
                ) + torch.rand(chunk_size, device=self.device) * (
                    np.deg2rad(self.config.vertical_rotation_range[1])
                    - np.deg2rad(self.config.vertical_rotation_range[0])
                )
                random_phi = -random_phi
                random_directions = torch.stack(
                    [
                        torch.cos(random_phi) * torch.cos(random_theta),
                        torch.cos(random_phi) * torch.sin(random_theta),
                        torch.sin(random_phi),
                    ],
                    dim=1,
                )
            else:
                random_ids = torch.randint(
                    0, directions.shape[0], (chunk_size,)
                )
                random_directions = directions[random_ids]

            rays = (
                torch.cat([random_center, random_directions], axis=1)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            # find intersections
            ans = self.voxel_scene.cast_rays(rays)
            t_hit = ans["t_hit"].numpy()
            t_hit = np.clip(t_hit, 0.0, 1e6)
            x_hit = rays[:, :3] + rays[:, 3:] * t_hit.reshape(-1, 1)

            clamped_point = np.maximum(
                bb_obj_np[0].reshape(1, -1),
                np.minimum(x_hit, bb_obj_np[1].reshape(1, -1)),
            )
            dist_x_hit_obj = np.linalg.norm(x_hit - clamped_point, axis=1)
            good_mask = dist_x_hit_obj >= self.dist_range[0]
            # IMPORTANT: Update t_hit
            t_hit = dist_x_hit_obj / np.linalg.norm(rays[:, 3:], axis=1)

            # take only poses which are kinda far from the object
            rays = rays[good_mask]
            t_hit = t_hit[good_mask]
            t_hit = np.clip(t_hit, 0, self.dist_range[1])
            t_hit = t_hit - self.config.shift_from_hit
            t_hit = self.dist_range[0] + np.random.rand(t_hit.shape[0]) * (
                t_hit - self.dist_range[0]
            )

            # get look_at (object), look_from (somewhere in the scene)
            look_at = rays[:, :3]
            look_from = look_at + rays[:, 3:] * t_hit.reshape(-1, 1)

            poses_chunk = []
            for i in range(look_at.shape[0]):
                z_dir = look_from[i] - look_at[i]
                z_dir /= np.linalg.norm(z_dir)
                x_dir = np.array([-z_dir[1], z_dir[0], 0.0])
                x_dir /= np.linalg.norm(x_dir)
                y_dir = np.cross(z_dir, x_dir)
                y_dir /= np.linalg.norm(y_dir)

                curr_pose = np.eye(4)
                curr_pose[:3, 0] = x_dir
                curr_pose[:3, 1] = y_dir
                curr_pose[:3, 2] = z_dir
                curr_pose[:3, 3] = look_from[i]
                poses_chunk.append(curr_pose[:3, :].tolist())

            poses.extend(poses_chunk)

        poses = poses[:num_poses]
        camera_to_worlds = torch.tensor(poses)

        cameras = Cameras(
            camera_to_worlds=camera_to_worlds,
            fx=self.config.focal * resolution,
            fy=self.config.focal * resolution,
            cx=resolution / 2,
            cy=resolution / 2,
        ).to(self.device)

        return cameras

    def _load_obj_scene(self):
        curr_obj_scene_datapath = Path(self.obj_scene_datapath)
        if curr_obj_scene_datapath.name == "obj_scene":
            curr_obj_scene_datapath = (
                self.obj_scene_datapath.parent / "obj_scene_eval"
            )
        with open(curr_obj_scene_datapath / "transforms.json", "r") as f:
            data = json.load(f)

        self.eval_fl = data["fl_x"]
        self.eval_poses = torch.tensor(
            [f["transform_matrix"] for f in data["frames"]]
        )
        self.image_paths = [
            curr_obj_scene_datapath / f["file_path"] for f in data["frames"]
        ]
        self.images = [np.asarray(PILImage.open(p)) for p in self.image_paths]
        self.images = [torch.tensor(im).float() for im in self.images]
        self.masks_paths = [
            curr_obj_scene_datapath
            / f["file_path"].replace("images", "masks").replace("color", "mask")
            for f in data["frames"]
        ]

        self.masks_paths = [
            p if os.path.exists(p) else im_p
            for (p, im_p) in zip(self.masks_paths, self.image_paths)
        ]
        self.masks = [
            np.asarray(PILImage.open(p).convert("L")) for p in self.masks_paths
        ]
        self.masks = [torch.tensor(im).float() for im in self.masks]
        # update with good eval ids (where mask is not empty)
        good_eval_ids = [
            i
            for i, mask in enumerate(self.masks)
            if torch.sum(mask > 0.1) > 150
        ]

        self.eval_poses = self.eval_poses[good_eval_ids]
        self.images = [self.images[i] for i in good_eval_ids]
        self.masks = [self.masks[i] for i in good_eval_ids]

        self.eval_cameras = Cameras(
            camera_to_worlds=self.eval_poses,
            fx=self.eval_fl,
            fy=self.eval_fl,
            cx=self.config.eval_resolution / 2,
            cy=self.config.eval_resolution / 2,
        )

    def _get_obj_scene_el(self, idx_global):
        idx_local = idx_global % len(self.eval_poses)
        images = self.images[idx_local].to(self.device) / 255.0
        masks = self.masks[idx_local].to(self.device)
        cameras = self.eval_cameras[idx_local : idx_local + 1].to(self.device)

        return {
            "cameras": cameras,
            "images": images,
            "masks": masks,
        }

    def lighting_phase(self):
        if self.is_lighting_phase:
            return
        if self.is_refining_phase:
            raise RuntimeError("lighting phase should be before!")
        self.is_lighting_phase = True
        self.is_refining_phase = False
        self.dist_range = (
            self._calculate_dist(
                self.config.camera_angle_x, self.bb_obj, self.config.betas[1]
            ),
            self._calculate_dist(
                self.config.camera_angle_x, self.bb_obj, self.config.betas[0]
            ),
        )

    def refining_phase(self):
        if self.is_refining_phase:
            return
        self.is_lighting_phase = False
        self.is_refining_phase = True
        self.dist_range = (
            self._calculate_dist(
                self.config.camera_angle_x,
                self.bb_obj,
                self.config.betas_refine[1],
            ),
            self._calculate_dist(
                self.config.camera_angle_x,
                self.bb_obj,
                self.config.betas_refine[0],
            ),
        )

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        config: RandomCamerasDataManagerObjSceneConfig,
        device: Union[torch.device, str] = "cpu",
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        **kwargs,  # pylint: disable=unused-argument
    ):
        self.config = config
        self.device = device
        self.world_size = world_size
        self.local_rank = local_rank
        self.sampler = None
        self.test_mode = test_mode
        self.test_split = (
            "test" if test_mode in ["test", "inference"] else "val"
        )

        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)

        rotation_matrix, movement_vector, datapath = get_Rt_from_json(
            self.config.transforms_obj, self.config.init_obj_path
        )

        self.obj_scene_datapath = datapath

        rotation_matrix = rotation_matrix.to(device)
        movement_vector = movement_vector.to(device)

        self.datapath = datapath
        with open(os.path.join(datapath, "transforms.json"), "r") as f:
            data = json.load(f)
            self.config.camera_angle_x = data["camera_angle_x"]
            self.config.train_resolution = data["w"]
            self.config.eval_resolution = data["w"]
            # f / width = 0.5 / tan(camera_angle_x / 2)
            self.config.focal = 0.5 / math.tan(self.config.camera_angle_x / 2)

        CONSOLE.print(f"focal: {self.config.focal}")
        CONSOLE.print(f"train_resolution: {self.config.train_resolution}")
        CONSOLE.print(f"eval_resolution: {self.config.eval_resolution}")

        init_obj_ckpt_path = find_ckpt_path(self.config.init_obj_path)
        init_scene_ckpt_path = find_ckpt_path(self.config.init_scene_path)
        obj_gauss_params = load_gauss_params(init_obj_ckpt_path, device=device)
        scene_gauss_params = load_gauss_params(
            init_scene_ckpt_path, device=device
        )

        obj_pts = obj_gauss_params["means"]
        obj_pts = obj_pts @ rotation_matrix.t() + movement_vector
        self.bb_obj = (obj_pts.min(dim=0).values, obj_pts.max(dim=0).values)
        self.config.center = (
            (0.5 * (self.bb_obj[0] + self.bb_obj[1])).detach().cpu().tolist()
        )
        CONSOLE.print(f"center: {self.config.center}")

        scene_pts = scene_gauss_params["means"]
        scene_pts_width = torch.exp(
            scene_gauss_params["scales"].max(dim=1).values
        )
        self.voxel_scene = self._generate_voxel_scene(
            scene_pts, scene_pts_width, self.config.voxel_size, device=device
        )

        self.is_lighting_phase = False
        self.is_refining_phase = False

        # turn on lighting phase
        self.lighting_phase()
        assert self.dist_range[0] <= self.dist_range[1], (
            "dist_range[0] should be less than dist_range[1]"
        )

        # Unfortunately I have the old nerfstudio version
        # It might be better to use the new version but it may break
        # the already existing code
        # this does nothing and it is useless
        self.train_dataparser_outputs = DummyClass(
            fields=["metadata"],
            fields_values={"metadata": {}},
            methods=["save_dataparser_transform"],
        )

        self.fixed_directions_eval = torch.tensor(
            self.generate_points_on_sphere(
                n=self.config.num_eval_angles,
                vertical_rotation_range=self.config.vertical_rotation_range,
            ),
            device=self.device,
        )

        if self.config.num_fixed_train_angles > 0:
            self.fixed_directions_train = torch.tensor(
                self.generate_points_on_sphere(
                    n=self.config.num_fixed_train_angles,
                    vertical_rotation_range=self.config.vertical_rotation_range,
                ),
                device=self.device,
            )
        else:
            self.fixed_directions_train = None

        # change devices to cpu
        cameras = self.generate_random_cameras(
            num_poses=self.config.num_eval_angles,
            chunk_size=self.config.num_eval_angles,
            resolution=self.config.eval_resolution,
            directions=self.fixed_directions_eval,
        )

        # eval dataset it here!
        self._load_obj_scene()

        self.train_dataset = TrivialDataset(cameras)
        self.eval_dataset = TrivialDataset(self.eval_cameras)

        self.eval_dataloader = RandIndicesEvalDataloader(
            input_dataset=self.eval_dataset,
            device=self.device,
            num_workers=self.world_size * 4,
        )

        self.train_dataset.cameras = self.train_dataset.cameras.to("cpu")
        self.eval_dataset.cameras = self.eval_dataset.cameras.to("cpu")

        self.current_train = None
        self.current_train_num = 0

        DataManager.__init__(self)

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next train batch

        Returns a Camera instead of raybundle"""
        return self.next_train_image(step=step)

    def next_eval(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next evaluation batch

        Returns a Camera instead of raybundle"""
        return self.next_eval_image(step=step)

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

    def next_train_image(self, step: int) -> Tuple[Cameras, Dict]:
        # torch.manual_seed(42)
        self.train_count += 1

        cameras = self.generate_random_cameras(
            num_poses=self.config.train_images_per_batch,
            chunk_size=2,
            resolution=self.config.train_resolution,
            directions=self.fixed_directions_train,
        )

        return cameras, {}

    def get_datapath(self) -> int:
        from pathlib import Path

        return Path(".")

    def get_train_rays_per_batch(self) -> int:
        return self.config.train_resolution**2

    def get_eval_rays_per_batch(self) -> int:
        return self.config.eval_resolution**2

    def get_param_groups(
        self,
    ) -> Dict[str, List[Parameter]]:  # pylint: disable=no-self-use
        """Get the param groups for the data manager.
        Returns:
            A list of dictionaries containing the data manager's param groups.
        """
        param_groups = {}

        return param_groups
