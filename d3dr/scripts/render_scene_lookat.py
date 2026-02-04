import argparse
import subprocess
import sys
from pathlib import Path

sys.path.append(".")
from train_everything import (
    _render_rgb,
)


def main():
    parser = argparse.ArgumentParser(description="Render scene lookat script")
    parser.add_argument(
        "--init_obj_path",
        type=str,
        required=True,
        help="Path to the initial object file",
    )
    parser.add_argument(
        "--init_scene_path",
        type=str,
        required=True,
        help="Path to the initial scene file",
    )
    parser.add_argument(
        "--transforms_obj",
        type=str,
        default=None,
        help="Path to transforms of object",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Path to where to store files",
    )
    parser.add_argument(
        "--num_poses", type=int, default=100, help="How many poses to use"
    )
    parser.add_argument(
        "--ns_config_path",
        type=str,
        required=True,
        help="Path to nerfstudio config to use",
    )
    parser.add_argument(
        "--ignore_scene",
        type=int,
        default=0,
        help="Whether to ignore scene to calculate stuff",
    )
    parser.add_argument(
        "--render_name", type=str, default="rgb", help="What to render"
    )

    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    images_dir = save_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Run generate_poses_look_at_obj.py
    cmd_1 = [
        "python3",
        "d3dr/scripts/generate_poses_look_at_obj.py",
        "--init_obj_path",
        args.init_obj_path,
        "--init_scene_path",
        args.init_scene_path,
        "--save_path",
        str(save_dir / "transforms_lao.json"),
        "--for_video",
        "1",
        "--ignore_scene",
        str(args.ignore_scene),
    ]
    if args.transforms_obj is not None:
        cmd_1.extend(["--transforms_obj", args.transforms_obj])
    subprocess.run(cmd_1)

    # Run transform_2_camera_path.py
    cmd_2 = [
        "python3",
        "d3dr/scripts/transforms_2_camera_path.py",
        "--tr_path",
        str(save_dir / "transforms_lao.json"),
    ]
    subprocess.run(cmd_2)

    # generate rgb images of the scene
    _render_rgb(
        config_path=args.ns_config_path,
        camera_path=str(save_dir / "camera_path.json"),
        output_dir=str(images_dir),
        name="rgb",
    )


if __name__ == "__main__":
    main()
