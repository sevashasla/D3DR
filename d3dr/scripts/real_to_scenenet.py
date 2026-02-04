import argparse
import os
from pathlib import Path
from tqdm import tqdm
import cv2
import json
import math
import shutil
from scipy.spatial.transform import Rotation as R
import numpy as np
from copy import deepcopy
import open3d as o3d

def get_args():
    parser = argparse.ArgumentParser(description="Process directory and image dimensions.")
    parser.add_argument("--input_dir", type=str, help="Path to the directory")
    parser.add_argument("--save_dir", type=str, help="Path to the directory")
    parser.add_argument("--wh", type=int, default=800, help="Width (integer)")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a valid directory.")
        return
    return args
    
def center_crop_to_square(cv2_image):
    h, w = cv2_image.shape[:2]
    if h > w:
        start = (h - w) // 2
        return cv2_image[start: h - start, ...]
    elif w > h:
        start = (w - h) // 2
        return cv2_image[:, start: w - start, ...]
    else:
        return cv2_image

def main():
    args = get_args()
    save_dir = Path(args.save_dir)
    input_dir = Path(args.input_dir)
    if (save_dir / "images").exists():
        shutil.rmtree(str(save_dir / "images"))
    (save_dir / "images").mkdir(parents=True, exist_ok=True)

    # reshape images to width/height:
    all_images_paths = (input_dir / "images").rglob("*.jpg")
    all_images_paths = sorted(list(all_images_paths)) # just to be sure
    for image_path in tqdm(all_images_paths):
        image = cv2.imread(str(image_path))
        # 1. crop the image
        image = center_crop_to_square(image)

        # 2. resize it
        image = cv2.resize(image, (args.wh, args.wh))
        cv2.imwrite(str(save_dir / "images" / image_path.name), image)
    print("Done reshaping images!")

    # read to_obj_scene
    with open(str(input_dir / "to_obj_scene.json")) as f:
        to_obj_scene = json.load(f)
    transform_to_obj_scene = np.eye(4)
    transform_to_obj_scene[:3, :3] = R.from_euler("xyz", to_obj_scene["rotation_euler"]).as_matrix()
    transform_to_obj_scene[:3, 3] = to_obj_scene["location"]

    result_data = {}
    with open(str(input_dir / "transforms.json")) as f:
        data = json.load(f)

    # find how to rescale data
    scale = min(data["w"], data["h"]) / args.wh

    result_data["w"] = args.wh
    result_data["h"] = args.wh
    result_data["fl_x"] = data["fl_x"] / scale
    result_data["fl_y"] = data["fl_y"] / scale
    result_data["cx"] = args.wh / 2
    result_data["cy"] = args.wh / 2

    result_data["k1"] = data["k1"]
    result_data["k2"] = data["k2"]
    result_data["p1"] = data["p1"]
    result_data["p2"] = data["p2"]
    result_data["camera_model"] = "OPENCV"
    # camera_angle_x = 2 * arctan(w / (2 * fl_x))
    # fx * tan(alpha / 2) = x/2 => alpha = atan(x / (2 fx)) * 2
    result_data["camera_angle_x"] = 2 * math.atan(result_data["w"] / (2 * result_data["fl_x"]))
    result_data["ply_file_path"] = str(save_dir / "sparse_pc.ply")

    result_data["frames"] = deepcopy(data["frames"])
    print("transform_to_obj_scene:")
    print(transform_to_obj_scene)
    for i in range(len(result_data["frames"])):
        transformed = \
            transform_to_obj_scene @ \
            np.array(result_data["frames"][i]["transform_matrix"])
        result_data["frames"][i]["transform_matrix"] = transformed.tolist()

    with open(str(save_dir / "transforms.json"), "w") as f:
        json.dump(result_data, f, indent=4)
    print("Done transforms.json")

    # read and transform the ply file
    pcd = o3d.io.read_point_cloud(str(input_dir / "sparse_pc.ply"))
    pcd.transform(transform_to_obj_scene)
    o3d.io.write_point_cloud(str(save_dir / "sparse_pc.ply"), pcd)
    print("Done changing ply file!")

if __name__ == "__main__":
    main()
