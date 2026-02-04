import argparse
import json
import os
from copy import deepcopy

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--generate_json", type=str, default=None)
parser.add_argument("--tr_path", type=str, default=None)
parser.add_argument("--scene_name", type=str, default=None)
parser.add_argument("--scenes_root", type=str, required=True)
parser.add_argument("--N", type=int, default=25)
parser.add_argument("--lookat_str", type=str, default="[0.0, 0.0, 0.0]")
parser.add_argument("--height_range_str", type=str, default="[-0.5, 0.5]")
parser.add_argument("--radius", type=float, default=1.0)

args = parser.parse_args()

if args.generate_json:
    curr_dict = vars(args)
    with open(args.generate_json) as f:
        data_to_run = json.load(f)
    curr_dict.update(data_to_run)

args.lookat = np.array(eval(args.lookat_str))
args.height_range = np.array(eval(args.height_range_str))

generated_poses = []
for i in range(args.N):
    angle = 2 * np.pi * i / args.N

    curr_height = np.random.uniform(args.height_range[0], args.height_range[1])

    x = args.lookat[0] + args.radius * np.cos(angle)
    y = args.lookat[1] + args.radius * np.sin(angle)
    z = args.lookat[2] + curr_height

    xyz = np.array([x, y, z])

    z_dir = xyz - args.lookat
    assert np.linalg.norm(z_dir) > 0
    z_dir /= np.linalg.norm(z_dir)

    x_dir = np.array([-z_dir[1], z_dir[0], 0.0])
    # helper = np.array([lookat[0], lookat[1], 0.0]) - np.array([xyz[0], xyz[1], 0.0])
    # x_dir = np.cross(-z_dir, helper)
    # assert np.linalg.norm(x_dir) > 0
    x_dir /= np.linalg.norm(x_dir)

    y_dir = np.cross(z_dir, x_dir)
    assert np.linalg.norm(y_dir) > 0
    y_dir /= np.linalg.norm(y_dir)

    curr_pose = np.eye(4)
    curr_pose[:3, 0] = x_dir
    curr_pose[:3, 1] = y_dir
    curr_pose[:3, 2] = z_dir
    curr_pose[:3, 3] = np.array([x, y, z])

    generated_poses.append(curr_pose)
generated_poses = np.array(generated_poses)

with open(
    os.path.join(args.scenes_root, args.scene_name, "obj/transforms.json")
) as f:
    data = json.load(f)
new_data = deepcopy(data)

new_data["frames"] = []
for gp in generated_poses:
    new_data["frames"].append({"transform_matrix": gp.tolist()})

with open(args.tr_path, "w") as f:
    json.dump(new_data, f)

print("Done!")
