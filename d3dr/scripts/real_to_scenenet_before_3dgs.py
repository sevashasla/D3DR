import argparse
import os
from pathlib import Path
import subprocess
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
    parser.add_argument("--frame_pattern", type=str, default="frame_*.jpg")
    parser.add_argument("--as_scene", action="store_true")
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
    if (save_dir / "masks").exists() and not (save_dir / "masks_sam").exists():
        shutil.move(str(save_dir / "masks"), str(save_dir / "masks_sam"))
    (save_dir / "images").mkdir(parents=True, exist_ok=True)

    # read to_obj_scene
    with open(str(input_dir / "to_obj_scene.json")) as f:
        to_obj_scene = json.load(f)
    
    # process all images using sam
    has_masks = False
    if "points" in to_obj_scene:
        print("Find masks!")
        points_to_sam_x = [str(el[0]) for el in to_obj_scene["points"]]
        points_to_sam_y = [str(el[1]) for el in to_obj_scene["points"]]
        cmd = "conda run -n sam2 python3 render_dataset/run_sam2_video.py "    + \
            f"--click_x {' '.join(points_to_sam_x)} "        + \
            f"--click_y {' '.join(points_to_sam_y)} "        + \
            f"--frame_pattern {args.frame_pattern} "         + \
            f"--frames_dir {str(input_dir / 'images')} "     + \
            f"--out_dir {str(save_dir / 'masks_sam')}"
        # run masks processing
        os.system(cmd)
        has_masks = True
        masks_paths = sorted((save_dir / 'masks_sam').glob("*.png"))
        print("Done masks!")
        (save_dir / "masks").mkdir(parents=True, exist_ok=True)

    # reshape images to width/height:
    all_images_paths = (input_dir / "images").rglob("*.jpg")
    all_images_paths = sorted(list(all_images_paths)) # just to be sure
    for i, image_path in tqdm(enumerate(all_images_paths)):
    # for i, image_path in enumerate(all_images_paths):
        image = cv2.imread(str(image_path))
        # 1. crop the image
        image = center_crop_to_square(image)

        # 1.1 read a mask and apply it to the corresponding image
        if has_masks:
            mask = cv2.imread(str(masks_paths[i]), cv2.IMREAD_GRAYSCALE)
            mask = center_crop_to_square(mask)[..., None]
            mask = (mask > 64).astype(np.uint8)
            if not args.as_scene:
                image = image * mask
            resized_mask = cv2.resize(mask, (args.wh, args.wh))
            cv2.imwrite(str(save_dir / "masks" / f"mask_{i:05}.png"), resized_mask * 255)

        # 2. resize the image
        image = cv2.resize(image, (args.wh, args.wh))
        cv2.imwrite(str(save_dir / "images" / f"color_{i:05}.png"), image)
    print("Done reshaping images!")

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
        result_data["frames"][i]["file_path"] = f"./images/color_{i:05}.png"

    with open(str(save_dir / "transforms.json"), "w") as f:
        json.dump(result_data, f, indent=4)
    print("Done transforms.json")

    # read and transform the ply file
    pcd = o3d.io.read_point_cloud(str(input_dir / "sparse_pc.ply"))
    pcd.transform(transform_to_obj_scene)
    if "bbox" in to_obj_scene and not args.as_scene:
        bbox = to_obj_scene["bbox"]
        min_bound = np.array(bbox[0]).reshape(1, 3)
        max_bound = np.array(bbox[1]).reshape(1, 3)

        points = np.asarray(pcd.points)
        mask = np.all((min_bound <= points) & (points <= max_bound), axis=1)
        
        colors = None
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
        normals = None
        if pcd.has_normals():
            normals = np.asarray(pcd.normals)
        pcd.points = o3d.utility.Vector3dVector(points[mask])
        
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(colors[mask])
        if normals is not None:
            pcd.normals = o3d.utility.Vector3dVector(normals[mask])

    o3d.io.write_point_cloud(str(save_dir / "sparse_pc.ply"), pcd)
    print("Done changing ply file!")

if __name__ == "__main__":
    main()
