"""
Generates poses such that the object is visible and not fully blocked

1 - i + 1,  j,      k
2 - i + 1,  j + 1,  k
3 - i,      j + 1,  k
4 - i       j       k
5 - i + 1,  j,      k + 1
6 - i + 1,  j + 1,  k + 1
7 - i,      j + 1,  k + 1
8 - i       j       k + 1

        ^ z
        |
        8-------------7
      / '            / |
     /  |           /  |
    5 -------------6   |
    |   |          |   |
    |   |          |   |
    |   4----------|---3---> y
    |  /           |  /
    | /            | /
    |/             |/
    1--------------2
   /
  /
 v x

The triangle faces are
1 2 5, 2 6 5
2 3 6, 3 7 6
3 4 7. 4 8 7
4 1 8, 1 5 8
1 4 2, 4 3 2
5 6 8, 6 7 8
"""

import argparse
import json

import numpy as np
import open3d as o3d
import torch
from generate_camera_on_sphere import _calculate_dist
from torch import Tensor

from d3dr.utils.read_and_rotate import (
    find_ckpt_path,
    get_Rt_from_json,
    load_gauss_params,
)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--beta",
        type=float,
        default=0.25,
        help="What ~ratio of area it takes on the image",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.1,
        help="Voxel size when voxelizing the scene",
    )
    parser.add_argument("--fov", type=float, default=60, help="fov in degrees")
    parser.add_argument(
        "--poses_num", type=int, default=100, help="how many poses to generate"
    )
    parser.add_argument(
        "--resolution", type=int, default=800, help="width and height"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./transforms_lao.json",
        help="where to store the file",
    )
    parser.add_argument(
        "--init_obj_path",
        type=str,
        default=None,
        help="path to the directory with obj params",
    )
    parser.add_argument(
        "--init_scene_path",
        type=str,
        default=None,
        help="path to the directory with scene params",
    )
    parser.add_argument(
        "--vertical_rotation_range_str",
        type=str,
        default="-45 0",
        help="path to the directory with scene params",
    )
    parser.add_argument(
        "--position_jitter",
        type=float,
        default=0.0,
        help="how much to jitter the center position (of the object, i.e. the look_at)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--for_video",
        type=int,
        default=0,
        help="whether to use those for video",
    )
    parser.add_argument("--transforms_obj", type=str, default=None)
    parser.add_argument("--ignore_scene", type=int, default=0)

    args = parser.parse_args()
    args.vertical_rotation_range = [
        int(x) for x in args.vertical_rotation_range_str.split()
    ]
    assert len(args.vertical_rotation_range) == 2, (
        "vertical_rotation_range should have 2 elements"
    )
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args


def _generate_voxel_scene(
    # self,
    scene_pts: Tensor,
    scene_pts_width: Tensor,
    vs: float,
    device: str = "cuda",
    ignore_scene: int = 0,
):

    # simply return an empty scene!)
    if ignore_scene != 0:
        return o3d.t.geometry.RaycastingScene()
    bb_scene = (scene_pts.min(dim=0).values, scene_pts.max(dim=0).values)

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
    voxels_x = bb_scene[0][0] + torch.arange(0, N_voxels[0], device=device) * vs
    voxels_y = bb_scene[0][1] + torch.arange(0, N_voxels[1], device=device) * vs
    voxels_z = bb_scene[0][2] + torch.arange(0, N_voxels[2], device=device) * vs

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
            torch.stack([voxels_x + vs, voxels_y + vs, voxels_z + vs], dim=1),
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
        dim=0, index=inverse_indices, src=scene_pts_width, reduce="mean"
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
                [arn + (1 - 1) * N, arn + (2 - 1) * N, arn + (5 - 1) * N], dim=1
            ),
            torch.stack(
                [arn + (2 - 1) * N, arn + (6 - 1) * N, arn + (5 - 1) * N], dim=1
            ),
            torch.stack(
                [arn + (2 - 1) * N, arn + (3 - 1) * N, arn + (6 - 1) * N], dim=1
            ),
            torch.stack(
                [arn + (3 - 1) * N, arn + (7 - 1) * N, arn + (6 - 1) * N], dim=1
            ),
            torch.stack(
                [arn + (3 - 1) * N, arn + (4 - 1) * N, arn + (7 - 1) * N], dim=1
            ),
            torch.stack(
                [arn + (4 - 1) * N, arn + (8 - 1) * N, arn + (7 - 1) * N], dim=1
            ),
            torch.stack(
                [arn + (4 - 1) * N, arn + (1 - 1) * N, arn + (8 - 1) * N], dim=1
            ),
            torch.stack(
                [arn + (1 - 1) * N, arn + (5 - 1) * N, arn + (8 - 1) * N], dim=1
            ),
            torch.stack(
                [arn + (1 - 1) * N, arn + (4 - 1) * N, arn + (2 - 1) * N], dim=1
            ),
            torch.stack(
                [arn + (4 - 1) * N, arn + (3 - 1) * N, arn + (2 - 1) * N], dim=1
            ),
            torch.stack(
                [arn + (5 - 1) * N, arn + (6 - 1) * N, arn + (8 - 1) * N], dim=1
            ),
            torch.stack(
                [arn + (6 - 1) * N, arn + (7 - 1) * N, arn + (8 - 1) * N], dim=1
            ),
        ],
        dim=0,
    )

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vectors.cpu().numpy())
    mesh.triangles = o3d.utility.Vector3iVector(faces.cpu().numpy())
    mesh_legacy = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    o3d.io.write_triangle_mesh(
        "/home/skorokho/coding/voi_gs/tmp/test_scene.ply", mesh
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_legacy)
    return scene


def filter_poses(
    scene,
    random_center,
    random_directions,
    min_dist,
    bb_obj,
    for_video,
):
    rays = (
        torch.cat([random_center, random_directions], axis=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    # find intersections
    ans = scene.cast_rays(rays)
    t_hit = ans["t_hit"].numpy()
    x_hit = rays[:, :3] + rays[:, 3:] * t_hit.reshape(-1, 1)

    bb_obj_np = (bb_obj[0].cpu().numpy(), bb_obj[1].cpu().numpy())
    clamped_point = np.maximum(
        bb_obj_np[0].reshape(1, -1),
        np.minimum(x_hit, bb_obj_np[1].reshape(1, -1)),
    )
    dist_x_hit_obj = np.linalg.norm(x_hit - clamped_point, axis=1)

    good_mask = ~(dist_x_hit_obj < min_dist)

    # find poses
    if not for_video:
        rays = rays[good_mask]
        t_hit = t_hit[good_mask]
    t_hit = np.clip(t_hit, 0, min_dist)
    look_at = rays[:, :3]
    look_from = look_at + rays[:, 3:] * t_hit.reshape(-1, 1)
    return look_at, look_from


def main():
    args = get_arguments()

    rotation_matrix, movement_vector, datapath = get_Rt_from_json(
        args.transforms_obj, args.init_obj_path
    )
    rotation_matrix = rotation_matrix.to(args.device)
    movement_vector = movement_vector.to(args.device)

    vs = args.voxel_size

    init_obj_ckpt_path = find_ckpt_path(args.init_obj_path)
    init_scene_ckpt_path = find_ckpt_path(args.init_scene_path)
    obj_gauss_params = load_gauss_params(init_obj_ckpt_path, device=args.device)
    scene_gauss_params = load_gauss_params(
        init_scene_ckpt_path, device=args.device
    )

    obj_pts = obj_gauss_params["means"]
    obj_pts = obj_pts @ rotation_matrix.t() + movement_vector
    bb_obj = (obj_pts.min(dim=0).values, obj_pts.max(dim=0).values)

    scene_pts = scene_gauss_params["means"]
    # scene_pts_width = torch.exp(scene_gauss_params["scales"].max(dim=1).values) * sps.norm.ppf(0.995)
    scene_pts_width = torch.exp(scene_gauss_params["scales"].max(dim=1).values)
    print("median scale:", torch.median(scene_pts_width))

    bb_scene = (scene_pts.min(dim=0).values, scene_pts.max(dim=0).values)

    scene = _generate_voxel_scene(
        scene_pts,
        scene_pts_width,
        vs,
        device=args.device,
        ignore_scene=args.ignore_scene,
    )

    # generate poses
    focal = args.resolution / (2 * np.tan(np.deg2rad(args.fov) / 2))
    min_dist = _calculate_dist(np.deg2rad(args.fov), bb_obj, args.beta)
    print("min_dist:", min_dist)

    result_look_at = np.empty((0, 3))
    result_look_from = np.empty((0, 3))

    batch_size = 64
    random_phi_single = None
    while len(result_look_at) < args.poses_num:
        # generate random directions
        if not args.for_video:
            random_theta = (
                torch.rand(batch_size, device=args.device) * 2 * torch.pi
            )
        else:
            args.poses_num - len(result_look_at)
            args.poses_num
            random_theta = (
                torch.linspace(
                    len(result_look_at) / args.poses_num,
                    1.0,
                    steps=args.poses_num - len(result_look_at) + 1,
                    device=args.device,
                )[:batch_size]
                * 2
                * torch.pi
            )

        # generate centers
        random_center = (bb_obj[0] + bb_obj[1]) / 2
        random_center = random_center.reshape(1, 3).expand(len(random_theta), 3)
        if not args.for_video:
            random_center += (
                args.position_jitter
                * torch.rand((len(random_theta), 3), device=args.device)
                * (bb_obj[1] - bb_obj[0])
                / 2
            )

        # this doesn't work for some reason...
        # sampled_uniform = (
        #     torch.rand(batch_size, device=args.device) * (args.vertical_rotation_range[1] - args.vertical_rotation_range[0]) + args.vertical_rotation_range[0] + 90
        # ) / 180
        # random_phi = torch.arccos(1 - 2 * sampled_uniform)

        if not args.for_video:
            random_phi = np.deg2rad(
                args.vertical_rotation_range[0]
            ) + torch.rand(batch_size, device=args.device) * (
                np.deg2rad(args.vertical_rotation_range[1])
                - np.deg2rad(args.vertical_rotation_range[0])
            )
        else:
            if random_phi_single is None:
                random_phi_single = np.deg2rad(
                    args.vertical_rotation_range[0]
                ) + torch.rand(1, device=args.device) * (
                    np.deg2rad(args.vertical_rotation_range[1])
                    - np.deg2rad(args.vertical_rotation_range[0])
                )
            random_phi = random_phi_single.reshape(-1).expand(len(random_theta))

        random_phi = -random_phi
        random_directions = torch.stack(
            [
                torch.cos(random_phi) * torch.cos(random_theta),
                torch.cos(random_phi) * torch.sin(random_theta),
                torch.sin(random_phi),
            ],
            dim=1,
        )

        curr_look_at, curr_look_from = filter_poses(
            scene,
            random_center,
            random_directions,
            min_dist,
            bb_obj,
            args.for_video,
        )

        assert len(curr_look_at) == len(curr_look_from), (
            "len(curr_look_at) != len(curr_look_from)"
        )

        result_look_at = np.concatenate([result_look_at, curr_look_at], axis=0)
        result_look_from = np.concatenate(
            [result_look_from, curr_look_from], axis=0
        )

    result_look_at = result_look_at[: args.poses_num]
    result_look_from = result_look_from[: args.poses_num]

    poses = []
    for i in range(result_look_at.shape[0]):
        z_dir = result_look_from[i] - result_look_at[i]
        z_dir /= np.linalg.norm(z_dir)

        x_dir = np.array([-z_dir[1], z_dir[0], 0.0])
        x_dir /= np.linalg.norm(x_dir)

        y_dir = np.cross(z_dir, x_dir)
        y_dir /= np.linalg.norm(y_dir)

        curr_pose = np.eye(4)
        curr_pose[:3, 0] = x_dir
        curr_pose[:3, 1] = y_dir
        curr_pose[:3, 2] = z_dir
        curr_pose[:3, 3] = result_look_from[i] - args.voxel_size / 1.5
        poses.append(curr_pose)

    transforms = {}
    transforms["camera_angle_x"] = np.deg2rad(args.fov)
    transforms["w"] = args.resolution
    transforms["h"] = args.resolution
    transforms["fl_x"] = focal
    transforms["fl_y"] = focal
    transforms["cx"] = args.resolution / 2
    transforms["cy"] = args.resolution / 2
    transforms["k1"] = 0.0
    transforms["k2"] = 0.0
    transforms["p1"] = 0.0
    transforms["p2"] = 0.0
    transforms["camera_model"] = "OPENCV"

    transforms["frames"] = []
    for pose in poses:
        transforms["frames"].append({"transform_matrix": pose.tolist()})

    with open(args.save_path, "w") as f:
        json.dump(transforms, f, indent=4)


if __name__ == "__main__":
    main()
