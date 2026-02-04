"""
Generate a camera circle with a fixed radius and a fixed number of cameras.
"""

import argparse
import json
import math
import os
from copy import deepcopy

import numpy as np
import open3d as o3d


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--betas_str", type=str, default="0.1 0.3")
    parser.add_argument("--angles_z_str", type=str, default="0 45")
    parser.add_argument("--num", type=int, default=64)
    parser.add_argument("--fov_degrees", type=float, default=60.0)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--store_path", type=str, required=True)

    args = parser.parse_args()
    args.betas = [float(el) for el in args.betas_str.split(" ")]
    args.angles_z = [int(el) for el in args.angles_z_str.split(" ")]
    args.fov = np.deg2rad(float(args.fov_degrees))

    return args


def generate_points_on_sphere_dist(
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
    return np.array(result)


def generate_points_on_sphere(
    n=100,
    vertical_rotation_range=(90, -90),
):

    left = 1.0 / n
    right = 1.0
    # binsearch
    while (right - left) > 1e-3:
        mid = (left + right) / 2
        points = generate_points_on_sphere_dist(
            a=mid, vertical_rotation_range=vertical_rotation_range
        )
        if len(points) > n:
            left = mid
        else:
            right = mid
    return points


def _calculate_dist(
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


def main():
    args = get_args()
    obj_pc = o3d.io.read_point_cloud(
        os.path.join(args.data_root, "obj/sparse_pc.ply")
    )
    bounding_box = obj_pc.get_axis_aligned_bounding_box()
    center = bounding_box.get_center().reshape(-1)  # lookat
    bb_points = np.array(bounding_box.get_box_points())
    bb_min, bb_max = bb_points.min(axis=0), bb_points.max(axis=0)
    radius_min = _calculate_dist(args.fov, (bb_min, bb_max), args.betas[1])
    radius_max = _calculate_dist(args.fov, (bb_min, bb_max), args.betas[0])
    print("radius_min:", radius_min)
    print("radius_max:", radius_max)

    points_on_sphere = generate_points_on_sphere(
        n=args.num,
        vertical_rotation_range=(-args.angles_z[1], -args.angles_z[0]),
    )

    generated_poses = []
    for i in range(len(points_on_sphere)):
        points_on_sphere[i]

        curr_radius = np.random.uniform(radius_min, radius_max)
        xyz = center + points_on_sphere[i] * curr_radius

        z_dir = xyz - center
        assert np.linalg.norm(z_dir) > 0
        z_dir /= np.linalg.norm(z_dir)

        x_dir = np.array([-z_dir[1], z_dir[0], 0.0])
        x_dir /= np.linalg.norm(x_dir)

        y_dir = np.cross(z_dir, x_dir)
        assert np.linalg.norm(y_dir) > 0
        y_dir /= np.linalg.norm(y_dir)

        curr_pose = np.eye(4)
        curr_pose[:3, 0] = x_dir
        curr_pose[:3, 1] = y_dir
        curr_pose[:3, 2] = z_dir
        curr_pose[:3, 3] = xyz

        generated_poses.append(curr_pose)
    generated_poses = np.array(generated_poses)

    with open(os.path.join(args.data_root, "obj/transforms.json")) as f:
        data = json.load(f)
    new_data = deepcopy(data)
    new_data["camera_angle_x"] = args.fov
    new_data["camera_angle_y"] = args.fov
    # f * tan(fov / 2) = w / 2 => f = w / (2 * tan(fov / 2))
    new_data["fl_x"] = new_data["w"] / (2 * math.tan(args.fov / 2))
    new_data["fl_y"] = new_data["h"] / (2 * math.tan(args.fov / 2))

    new_data["frames"] = []
    for gp in generated_poses:
        new_data["frames"].append({"transform_matrix": gp.tolist()})

    with open(args.store_path, "w") as f:
        json.dump(new_data, f)

    print("Done!")


if __name__ == "__main__":
    main()
