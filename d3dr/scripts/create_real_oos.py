import torch
import argparse
import os
import shutil
from pathlib import Path
import json
import open3d as o3d
import cv2
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Copy directories into save_dir with new names.")
    parser.add_argument('--obj_scene_dir', required=True, help='Path to obj_scene_dir')
    parser.add_argument('--scene_dir', required=True, help='Path to scene_dir')
    parser.add_argument('--obj_dir', required=True, help='Path to obj_dir')
    parser.add_argument('--obj_splat_ckpt', required=False, help='Path to obj_path', default=None)
    parser.add_argument('--save_dir', required=True, help='Path to save_dir')
    args = parser.parse_args()

    # process arguments
    save_dir = Path(args.save_dir)
    if save_dir.exists():
        shutil.rmtree(str(save_dir))
    save_dir.mkdir(exist_ok=True)

    src_obj_scene_dir = Path(args.obj_scene_dir)
    src_scene_dir = Path(args.scene_dir)
    src_obj_dir = Path(args.obj_dir)

    dst_obj_scene_dir = save_dir / "obj_scene_eval"
    dst_scene_dir = save_dir / "scene_eval"
    dst_obj_dir = save_dir / "obj"

    dst_obj_scene_dir.mkdir(exist_ok=True)
    # dst_scene_dir.mkdir(exist_ok=True)
    dst_obj_dir.mkdir(exist_ok=True)

    # copy images from the obj_scene
    os.symlink(str(src_obj_scene_dir / "images"), str(dst_obj_scene_dir / "images"))
    os.symlink(str(src_obj_scene_dir / "sparse_pc.ply"), str(dst_obj_scene_dir / "sparse_pc.ply"))
    if not (src_obj_scene_dir / "masks").exists():
        image_paths = list((src_obj_scene_dir / "images").glob("*"))
        default_image_path = image_paths[0]
        default_image = cv2.imread(str(default_image_path))
        default_mask = np.full_like(default_image, fill_value=255)[..., 0]
        (dst_obj_scene_dir / "masks").mkdir()
        for i in range(len(image_paths)):
            cv2.imwrite(str(dst_obj_scene_dir / "masks" / f"mask_{i:05}.png"), default_mask)
    else:
        os.symlink(str(src_obj_scene_dir / "masks"), str(dst_obj_scene_dir / "masks"))


    with open(str(src_obj_scene_dir / "transforms.json")) as f:
        trnsfms_obj_scene = json.load(f)
    # they are all already ~centered
    trnsfms_obj_scene["euler_rotation"] = [0.0, 0.0, 0.0]
    trnsfms_obj_scene["object_center"] = [0.0, 0.0, 0.0]
    with open(str(dst_obj_scene_dir / "transforms.json"), "w") as f:
        json.dump(trnsfms_obj_scene, f, indent=4)
    print("[INFO] Saved obj_scene")
    
    # just copy obj_scene
    os.symlink(str(save_dir / "obj_scene_eval"), str(save_dir / "obj_scene"))

    os.symlink(str(src_scene_dir), str(dst_scene_dir))
    print("[INFO] Saved scene")

    # not great but ok
    os.symlink(str(src_obj_dir / "images"), str(dst_obj_dir / "images"))

    # create pointcloud
    if args.obj_splat_ckpt is None:
        os.symlink(str(src_obj_dir / "sparse_pc.ply"), str(dst_obj_dir / "sparse_pc.ply"))
    else:
        model = torch.load(args.obj_splat_ckpt, map_location="cpu")
        means = model["pipeline"]["_model.gauss_params.means"]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(means.cpu().numpy())
        o3d.io.write_point_cloud(str(dst_obj_dir / "sparse_pc.ply"), pcd)
    print("[INFO] Saved obj point cloud")
    
    os.symlink(str(src_obj_dir / "masks"), str(dst_obj_dir / "masks"))
    os.symlink(str(src_obj_dir / "transforms.json"), str(dst_obj_dir / "transforms.json"))

    print("[INFO] Done!")


if __name__ == "__main__":
    main()