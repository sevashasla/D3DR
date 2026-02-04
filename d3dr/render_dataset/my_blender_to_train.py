import argparse
import datetime
import json
import os
import time
from copy import deepcopy

import numpy as np
from PIL import Image
from tqdm import tqdm

time_start = time.time()
print("Start at:", datetime.datetime.now().strftime("%H:%M:%S"))

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, help="input folder")
parser.add_argument("--output_dir", type=str, help="output folder")
parser.add_argument(
    "--help_eval_dir",
    type=str,
    default=None,
    help="where to take transforms and other stuff for eval folder",
)
parser.add_argument(
    "--dn_splatter_path",
    type=str,
    help="dir with dn-splatter",
    default="/home/skorokho/coding/voi_gs/dn-splatter",
)
parser.add_argument("--run_depth", type=int, default=1, help="do depth?")
parser.add_argument("--run_normals", type=int, default=1, help="do normals?")
parser.add_argument(
    "--normal", type=str, help="", choices=["omni", "dsine"], default="dsine"
)
parser.add_argument("--run_colmap", action="store_true")
parser.add_argument(
    "--type",
    type=str,
    nargs="+",
    help="what to preprocess",
    choices=[
        "all",
        "scene",
        "obj",
        "obj_scene",
        "obj_scene_eval",
        "scene_eval",
    ],
    default=["all"],
)
parser.add_argument(
    "--ply_name",
    type=str,
    help="filename of ply_name (I made an error and wrote sparce instead of sparse...)",
    default="sparce_pc.ply",
)

args = parser.parse_args()
dn_splatter_path = os.path.normpath(args.dn_splatter_path)

if "all" in args.type:
    DIRS = ["obj", "scene", "obj_scene", "obj_scene_eval", "scene_eval"]
else:
    DIRS = args.type

os.makedirs(args.output_dir, exist_ok=True)

for d in DIRS:
    # rename images
    curr_path_out = os.path.join(args.output_dir, d)
    curr_path_in = os.path.join(args.input_dir, d)
    os.makedirs(curr_path_out, exist_ok=True)
    os.makedirs(os.path.join(curr_path_out, "images"), exist_ok=True)
    all_files = os.listdir(curr_path_in)
    all_images = [f for f in all_files if f.startswith("color_")]

    if len(os.listdir(os.path.join(curr_path_out, "images"))) != len(
        all_images
    ):
        for name in tqdm(all_images):
            image_number = int(name[6:-4])
            im = Image.open(os.path.join(curr_path_in, name))
            im = im.convert("RGB")
            im.save(
                os.path.join(
                    curr_path_out, "images", f"color_{image_number:05}.png"
                )
            )
    else:
        print(f"Images in {curr_path_out} already exist")

    # need to replace masks
    if d in ["obj", "obj_scene", "obj_scene_eval"]:
        all_masks = [f for f in all_files if f.startswith("mask_")]
        os.makedirs(os.path.join(curr_path_out, "masks"), exist_ok=True)
        if len(os.listdir(os.path.join(curr_path_out, "masks"))) != len(
            all_masks
        ):
            for name in tqdm(all_masks):
                mask_number = int(name[5:-8])
                im = Image.open(os.path.join(curr_path_in, name))
                im = im.convert("L")
                im.save(
                    os.path.join(
                        curr_path_out, "masks", f"mask_{mask_number:05}.png"
                    )
                )
        else:
            print(f"Masks in {curr_path_out} already exist")

    # process json
    with open(os.path.join(curr_path_in, "transforms.json")) as f:
        data = json.load(f)
    new_data = deepcopy(data)
    for i, frame_data in enumerate(new_data["frames"]):
        file_number = int(os.path.basename(frame_data["file_path"])[6:])
        frame_data["file_path"] = f"images/color_{file_number:05}.png"
        # if d == "obj":
        #     frame_data["mask_path"] = f"masks/mask_{file_number:05}.png"

    # add new information
    new_data["w"] = 800
    new_data["h"] = 800
    # f tg(0.5 * alpha) = 0.5 * w => f = 0.5 * w / tg(0.5 * alpha)
    new_data["fl_x"] = (
        0.5 * new_data["w"] / np.tan(0.5 * new_data["camera_angle_x"])
    )
    new_data["fl_y"] = (
        0.5 * new_data["h"] / np.tan(0.5 * new_data["camera_angle_x"])
    )
    new_data["cx"] = new_data["w"] / 2
    new_data["cy"] = new_data["h"] / 2
    new_data["k1"] = 0.0
    new_data["k2"] = 0.0
    new_data["p1"] = 0.0
    new_data["p2"] = 0.0
    new_data["camera_model"] = "OPENCV"

    if d == "obj_scene_eval":
        help_eval_dir = args.help_eval_dir
        if help_eval_dir is None:
            help_eval_dir = args.input_dir

        with open(
            os.path.join(help_eval_dir, "obj_scene/transforms.json")
        ) as f:
            data_obj_scene = json.load(f)
        new_data["euler_rotation"] = data_obj_scene["euler_rotation"]
        new_data["object_center"] = data_obj_scene["object_center"]

    if d in ["obj_scene_eval", "scene_eval"]:
        curr_ply_path = os.path.join(help_eval_dir, d[:-5], args.ply_name)
    else:
        curr_ply_path = os.path.join(curr_path_in, args.ply_name)

    if os.path.exists(curr_ply_path):
        print("COPY PLY")
        os.system(f"cp {curr_ply_path} {curr_path_out}/sparse_pc.ply")
        new_data["ply_file_path"] = os.path.join(curr_path_out, "sparse_pc.ply")
    else:
        print("ply file not found")

    with open(os.path.join(curr_path_out, "transforms.json"), "w") as f:
        json.dump(new_data, f, indent=4)

    # run colmap
    if args.run_colmap:
        print("RUN COLMAP")
        if not os.path.exists(os.path.join(curr_path_out, "colmap")):
            os.system(
                f"python3 d3dr/scripts/poses_to_colmap_sfm.py "
                f"--transforms-path {os.path.join(curr_path_out, 'transforms.json')} "
            )
            os.system(f"mkdir {curr_path_out}/colmap")
            os.system(f"mkdir {curr_path_out}/useless")
            os.system(
                f"mv {curr_path_out}/sparse/0/*.txt {curr_path_out}/useless"
            )
            os.system(f"mv {curr_path_out}/sparse {curr_path_out}/colmap")

    if args.run_normals:
        if not os.path.exists(
            os.path.join(curr_path_out, "normals_from_pretrain")
        ) or len(
            os.listdir(os.path.join(curr_path_out, "normals_from_pretrain"))
        ) != len(all_images):
            print("RUN NORMALS")
            if args.normal == "omni":
                os.system(
                    f"python3 d3dr/scripts/normals_from_pretrain.py "
                    f"--data-dir {curr_path_out} --resolution=low "
                )
            else:
                os.system(
                    f"python3 d3dr/scripts/normals_from_pretrain.py "
                    f"--data-dir {curr_path_out} --model-type dsine"
                )

    if args.run_depth:
        if not os.path.exists(os.path.join(curr_path_out, "mono_depth")) or len(
            os.listdir(os.path.join(curr_path_out, "mono_depth"))
        ) != 2 * len(all_images):
            print("RUN MONO DEPTH")
            os.system(
                f"python3 dn_splatter/scripts/align_depth.py "
                f"--data {curr_path_out} "
                "--no-skip-colmap-to-depths --no-skip-mono-depth-creation "
                + ("--no-colmap-format" if not args.run_colmap else "")
            )

    # need to process normals and depths based on mask
    if d == "obj" and args.run_normals and args.run_depth:
        all_masks = [f for f in all_files if f.startswith("mask_")]
        for name in tqdm(all_masks, desc="Apply masks to normals and depths"):
            mask_number = int(name[5:-8])
            mask_numpy = np.asarray(
                Image.open(os.path.join(curr_path_in, name))
            )[..., 0]
            mask_numpy_bool = mask_numpy > 0

            # process normals
            curr_normal_name = os.path.join(
                curr_path_out,
                "normals_from_pretrain",
                f"color_{mask_number:05}.png",
            )
            normal_im = Image.open(curr_normal_name)
            normal_numpy = np.asarray(normal_im)
            normal_numpy = (
                normal_numpy * mask_numpy_bool[..., None]
                + 128 * (~mask_numpy_bool)[..., None]
            )
            normal_im = Image.fromarray(normal_numpy.astype(np.uint8))
            normal_im.save(os.path.join(curr_normal_name))

            # process depths
            depth_name = os.path.join(
                curr_path_out, "mono_depth", f"color_{mask_number:05}.npy"
            )
            depth_name_aligned = os.path.join(
                curr_path_out,
                "mono_depth",
                f"color_{mask_number:05}_aligned.npy",
            )
            depth_numpy = np.load(depth_name) * mask_numpy_bool[..., None]
            depth_numpy_aligned = (
                np.load(depth_name_aligned) * mask_numpy_bool[..., None]
            )
            np.save(depth_name, depth_numpy)
            np.save(depth_name_aligned, depth_numpy_aligned)

print("End at:", datetime.datetime.now().strftime("%H:%M:%S"))
print(f"took {(time.time() - time_start) / 60:.2f} minutes")
