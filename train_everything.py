from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

# ------------------------------


def get_workspace_dir(root, exp_name, use_prev_dir):
    desired = os.path.join(root, exp_name)
    cnt = 0

    all_dirs = os.listdir(root)
    for d in all_dirs:
        if d.startswith(exp_name):
            cnt += 1

    # maybe take the previous one
    if use_prev_dir:
        cnt -= 1
    if cnt < 0:
        raise RuntimeError("No previous directory found!")

    desired = os.path.join(root, f"{exp_name}_{cnt:03}")
    return desired


# ------------------------------


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="./outputs/")
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default="./processed")
    parser.add_argument("--gaussian_splatting_root", type=str, default=None)
    parser.add_argument("--scenes_info", type=str, default="./scenes_info.json")
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--use_prev_dir", action="store_true")
    parser.add_argument("--skip_steps", type=int, default=[], nargs="*")

    parser.add_argument("--init_obj_path", type=str, default=None)
    parser.add_argument("--init_scene_path", type=str, default=None)
    parser.add_argument("--transforms_obj", type=str, default=None)

    parser.add_argument("--scene_desc", type=str, default=None)
    parser.add_argument("--obj_desc", type=str, default=None)
    parser.add_argument("--prompt_initial", type=str, default=None)
    parser.add_argument("--prompt_desired", type=str, default=None)
    parser.add_argument("--load_ckpt", type=str, default=None)

    parser.add_argument("--ic_light_prompt", type=str, default=None)
    parser.add_argument("--betas_str", type=str, default="0.1 0.6")
    parser.add_argument("--betas_refine_str", type=str, default="0.3 0.4")
    parser.add_argument("--betas_str_generation", type=str, default="0.3 0.4")
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--angles_z_str", type=str, default="0 45")
    parser.add_argument("--use_scene_eval", type=int, default=1)
    parser.add_argument("--generate_num", type=int, default=32)
    parser.add_argument(
        "--use_min_for_generation",
        type=int,
        default=0,
        help="1 - use, 0 - use mean, -1 - ignore",
    )
    parser.add_argument("--num_fixed_train_angles", type=int, default=100)

    parser.add_argument("--optimize_latent_for", type=int, default=16)
    parser.add_argument("--optimize_image_for", type=int, default=64)
    parser.add_argument("--optimize_image_for_refine", type=int, default=512)
    parser.add_argument("--num_together_images", type=int, default=8)
    parser.add_argument("--num_together_images_refine", type=int, default=8)
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--prob_mvc", type=float, default=-1.0)
    parser.add_argument("--sh_degree_inverval", type=int, default=500)
    parser.add_argument("--max_num_iterations_voi", type=int, default=3000)
    parser.add_argument("--refine_after", type=int, default=1000)
    parser.add_argument(
        "--refine_linear",
        type=int,
        default=1,
        help="whether to refine using the linear decrease of t_range",
    )
    parser.add_argument("--t_range_str", type=str, default="(0.02, 0.5)")
    parser.add_argument("--t_range_refine_str", type=str, default="(0.02 0.5)")
    parser.add_argument("--obj_initialization", type=str, default="mean")
    parser.add_argument("--use_conv_in", type=int, default=0)

    parser.add_argument(
        "--sd_version", type=str, default="2.1", help="what sd model to use"
    )
    parser.add_argument(
        "--use_controlnet",
        type=int,
        default=1,
        help="whether to use controlnet",
    )

    # personalization parameters
    parser.add_argument("--person_prob", type=float, default=0.7)
    parser.add_argument("--generate_using_iclight", type=int, default=1)
    parser.add_argument("--use_personalization", type=int, default=1)
    parser.add_argument(
        "--use_personalization_from",
        type=str,
        default=None,
        help="Path from where to take the personalizations",
    )
    parser.add_argument(
        "--use_unet",
        type=int,
        default=0,
        help="use unet or lora for personalization",
    )
    parser.add_argument(
        "--num_personalization_iterations",
        type=int,
        default=1000,
        help="use unet or lora for personalization",
    )
    parser.add_argument("--lora_or_unet_path", type=str, default=None)
    parser.add_argument("--double_personalizations", type=int, default=1)
    parser.add_argument("--refine_obj", type=int, default=0)
    parser.add_argument("--use_wandb", type=int, default=1)
    parser.add_argument(
        "--scene_shadows",
        type=str,
        default="sub-part",
        choices=["none", "div", "sub", "sub-part"],
        help="How to make shadows on a scene",
    )
    parser.add_argument(
        "--keep_scene_shadows_top",
        type=float,
        default=0.15,
        help="What percentage of gaussians should be leaved to dark",
    )
    parser.add_argument("--crop_ratio", type=float, default=0.7)
    parser.add_argument("--datamanager_seed", type=int, default=42)
    parser.add_argument("--fixed_place_prompt", type=str, default=None)
    parser.add_argument(
        "--leave_highest_areas",
        type=int,
        default=1,
        help="whether to leave only highest",
    )
    parser.add_argument(
        "--not_visible_const",
        type=float,
        default=0.5,
        help="<not_visible_const => not visible",
    )

    args = parser.parse_args()
    os.makedirs(args.root, exist_ok=True)
    args.root = os.path.abspath(args.root)
    args.workspace = get_workspace_dir(
        args.root, args.exp_name + "_" + args.scene_name, args.use_prev_dir
    )
    args.wandb_exp_name = Path(args.workspace).name
    print("workspace:", args.workspace)

    with open(args.scenes_info, "r") as f:
        _scenes_info = json.load(f)
        if args.scene_name not in _scenes_info:
            raise RuntimeError(
                f"Scene {args.scene_name} not found in {args._scenes_info}!"
            )
        scene_info = _scenes_info[args.scene_name]

    args.dataset_root = scene_info.get("dataset_root", args.dataset_root)
    args.gaussian_splatting_root = scene_info.get(
        "gaussian_splatting_root", args.gaussian_splatting_root
    )
    args.curr_data_root = os.path.join(args.dataset_root, args.scene_name)
    os.makedirs(args.workspace, exist_ok=args.use_prev_dir)

    # find initial paths
    args.init_obj_path = scene_info.get("init_obj_path", args.init_obj_path)
    args.init_scene_path = scene_info.get(
        "init_scene_path", args.init_scene_path
    )
    args.transforms_obj = scene_info.get("transforms_obj", args.transforms_obj)
    args.fixed_place_prompt = scene_info.get(
        "fixed_place_prompt", args.fixed_place_prompt
    )
    args.num_fixed_train_angles = scene_info.get(
        "num_fixed_train_angles", args.num_fixed_train_angles
    )
    args.not_visible_const = scene_info.get(
        "not_visible_const", args.not_visible_const
    )
    args.keep_scene_shadows_top = scene_info.get(
        "keep_scene_shadows_top", args.keep_scene_shadows_top
    )

    if args.init_obj_path is None:
        args.init_obj_path = find_store_dir_path(
            os.path.join(args.gaussian_splatting_root, args.scene_name + "-obj")
        )
    if args.init_scene_path is None:
        args.init_scene_path = find_store_dir_path(
            os.path.join(
                args.gaussian_splatting_root,
                args.scene_name
                + ("-scene_eval" if args.use_scene_eval else "-scene"),
            )
        )

    # find prompts
    if args.scene_desc is None:
        args.scene_desc = scene_info.get("scene_desc", args.scene_desc)
    if args.obj_desc is None:
        args.obj_desc = scene_info.get("obj_desc", args.obj_desc)
    if args.prompt_initial is None:
        args.prompt_initial = scene_info.get(
            "prompt_initial", f"a {args.scene_desc}"
        )
    if args.prompt_desired is None:
        args.prompt_desired = scene_info.get(
            "prompt_desired",
            f"a <rare_token> {args.obj_desc} in a {args.scene_desc}",
        ).replace("<rare_token>", "<ktn>")
    if args.ic_light_prompt is None:
        args.ic_light_prompt = scene_info.get(
            "ic_light_prompt", args.ic_light_prompt
        )

    args.max_num_iterations_voi = scene_info.get(
        "max_num_iterations_voi", args.max_num_iterations_voi
    )
    args.refine_after = scene_info.get("refine_after", args.refine_after)
    args.optimize_image_for = scene_info.get(
        "optimize_image_for", args.optimize_image_for
    )
    args.optimize_latent_for = scene_info.get(
        "optimize_latent_for", args.optimize_latent_for
    )
    if args.use_personalization == 0:
        args.prompt_desired = args.prompt_desired.replace("<ktn> ", "")

    # change betas
    args.betas_str = scene_info.get("betas_str", args.betas_str)
    args.betas_str_generation = scene_info.get(
        "betas_str_generation", args.betas_str_generation
    )
    args.betas_refine_str = scene_info.get(
        "betas_refine_str", args.betas_refine_str
    )
    args.use_min_for_generation = scene_info.get(
        "use_min_for_generation", args.use_min_for_generation
    )

    if any(
        [
            args.scene_desc is None,
            args.obj_desc is None,
            args.ic_light_prompt is None,
        ]
    ):
        raise RuntimeError(
            "scene_desc, obj_desc, ic_light_prompt must be provided!"
        )

    args.angles_z_str = scene_info.get("angles_z_str", args.angles_z_str)
    args.angles_z = [int(el) for el in args.angles_z_str.split(" ")]

    # save arguments
    num_same = len(list(Path(args.workspace).glob("args*.json")))
    save_path_args = Path(args.workspace) / f"args_{num_same:03}.json"
    print(f"Save args at: {save_path_args}")
    with open(str(save_path_args), "w") as f:
        json.dump(vars(args), f, indent=4)

    print(args)
    return args


# ------------------------------


def find_store_dir_path(begin_path):
    begin_path = Path(begin_path)
    # e.g.:
    # {begin_path}/dn-splatter/2025-01-10_013845/nerfstudio_models/step-000029999.ckpt'
    # {begin_path}/dn-splatter/2024-12-25_235425/nerfstudio_models/step-000029999.ckpt'
    all_pos = list(begin_path.rglob("*nerfstudio_models/?*"))
    # f.e. x.parent.parent.name is 2025-01-10_013845
    if len(all_pos) == 0:
        return None
    all_pos = sorted(all_pos, key=lambda x: x.parent.parent.name)
    return str(all_pos[-1].parent.parent)


# ------------------------------


def generate_object_images(
    curr_data_root,
    workspace,
    init_obj_path,
    betas_str,
    scene_name,
    angles_z_str,
    generate_num,
    leave_highest_areas,
):

    save_dir = os.path.join(workspace, "obj_sample_views")
    os.makedirs(save_dir, exist_ok=True)
    save_poses_path = os.path.join(save_dir, "transforms_circle.json")

    with open(
        os.path.join(curr_data_root, "obj_scene_eval/transforms.json"), "r"
    ) as f:
        data = json.load(f)
        camera_angle_x = data["camera_angle_x"]

    if leave_highest_areas != 0:
        print("Leave highest areas is True!")
        generate_num = generate_num * 2

    os.system(
        "python3 d3dr/scripts/generate_camera_on_sphere.py "
        + f"--betas_str '{betas_str}' "
        + f"--angles_z_str '{angles_z_str}' "
        + f"--num {generate_num} "
        + f"--fov_degrees {np.rad2deg(camera_angle_x)} "
        + f"--data_root {curr_data_root} "
        + f"--store_path {save_poses_path} "
    )

    save_camera_path = os.path.join(save_dir, "camera_path.json")
    os.system(
        "python3 d3dr/scripts/transforms_2_camera_path.py "
        + f"--tr_path {save_poses_path} "
        + f"--save_name {save_camera_path} "
    )

    _render_rgb(
        config_path=_get_config(init_obj_path),
        camera_path=save_camera_path,
        output_dir=os.path.join(save_dir, "rgb"),
        name="rgb",
    )

    _render_mask(
        config_path=_get_config(init_obj_path),
        camera_path=save_camera_path,
        output_dir=os.path.join(save_dir, "mask"),
        name="accumulation",
        img_format="jpeg",
    )

    # leave only with higher areas
    if leave_highest_areas:
        files_mask = sorted(
            list(Path(os.path.join(save_dir, "mask")).glob("*.jpg"))
        )
        files_rgb = sorted(
            list(Path(os.path.join(save_dir, "rgb")).glob("*.png"))
        )

        areas = []
        for mask_file in files_mask:
            mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
            mask = mask[:, :, 0]
            areas.append([mask.sum(), str(mask_file)])

        areas.sort(key=lambda x: x[0], reverse=True)
        to_delete_masks = set([el[1] for el in areas[generate_num // 2 :]])

        for mask_file, mask_rgb in zip(files_mask, files_rgb):
            mask_file = str(mask_file)
            mask_rgb = str(mask_rgb)
            if mask_file in to_delete_masks:
                os.remove(mask_file)
                os.remove(mask_rgb)

        print("Removed masks:", len(to_delete_masks))


def personalize_object(
    obj_desc,
    workspace,
    ic_light_prompt,
    generate_using_iclight,
    prob,
    use_unet,
    num_personalization_iterations,
    use_conv_in,
    double_personalizations,
    fixed_place_prompt,
    sd_version,
):

    if fixed_place_prompt is not None:
        fixed_place_prompt_arg = f'--fixed_place_prompt "{fixed_place_prompt}" '
    else:
        fixed_place_prompt_arg = " "

    if double_personalizations != 0:
        image_dir = os.path.join(workspace, "obj_sample_views/rgb")
        save_dir_1 = os.path.join(workspace, "personalization_object_1")

        # personalize on the full images
        cmd = (
            "python3 d3dr/diffusion/personalization/lora_rough_personalization.py "
            + f"--prompt 'a <rare_token> {obj_desc}' "
            + "--rare_token '<ktn>' "
            + f"--ic_light_prompt '{ic_light_prompt}' "
            + f"--image_dir {image_dir} "
            + f"--save_dir {save_dir_1} "
            + "--mixed_precision fp16 "
            + f"--generate_using_iclight {generate_using_iclight} "
            + f"--model_name {sd_version} "
            + f"--prob {prob} "
            + "--debug 0 "
            + fixed_place_prompt_arg
            + f"--num_train_iterations {num_personalization_iterations}"
        )
        print(f"Running \n{cmd}")
        os.system(cmd)

        # preserve textures
        save_dir_2 = os.path.join(workspace, "personalization_object_2")
        generate_cp_iclight = os.path.join(
            workspace, "personalization_object_1/generated_images_iclight"
        )
        os.system(
            "python3 d3dr/diffusion/personalization/lora_texture_preserving_personalization.py "
            + f"--prompt 'a <rare_token> {obj_desc}' "
            + "--rare_token '<ktn>' "
            + f"--ic_light_prompt '{ic_light_prompt}' "
            + f"--image_dir {image_dir} "
            + f"--save_dir {save_dir_2} "
            + "--mixed_precision fp16 "
            + f"--generate_using_iclight {generate_using_iclight} "
            + f"--generate_cp_iclight {generate_cp_iclight} "
            + f"--model_name {sd_version} "
            + "--prob 1.5 "
            + "--debug 0 "
            + "--optimize_both_convs "
            + "--crop_ratio_max 1.0 "
            + fixed_place_prompt_arg
            + f"--num_train_iterations {1000} "
        )

    else:
        image_dir = os.path.join(workspace, "obj_sample_views/rgb")
        save_dir = os.path.join(workspace, "personalization_object")
        run_file = (
            "diffusion_personalization.py"
            if use_unet
            else "lora_personalization.py"
        )
        run_file = "lora_personalization_wimg.py" if use_conv_in else run_file
        mixed_precision = "no" if use_unet else "fp16"

        if use_conv_in:
            assert prob >= 1.0, "do not use preservation loss when conv_in"

        os.system(
            f"python3 d3dr/diffusion/personalization/{run_file} "
            + f"--prompt 'a <rare_token> {obj_desc}' "
            + "--rare_token '<ktn>' "
            + f"--ic_light_prompt '{ic_light_prompt}' "
            + f"--image_dir {image_dir} "
            + f"--save_dir {save_dir} "
            + f"--mixed_precision {mixed_precision} "
            + f"--generate_using_iclight {generate_using_iclight} "
            + f"--model_name {sd_version} "
            + f"--prob {prob} "
            + "--debug 0 "
            + fixed_place_prompt_arg
            + f"--num_train_iterations {num_personalization_iterations} "
        )


# ------------------------------


def perform_voi(
    workspace,
    init_obj_path,
    init_scene_path,
    prompt_initial,
    prompt_desired,
    max_num_iterations,
    sh_degree_interval,
    optimize_latent_for,
    optimize_image_for,
    angles_z,
    betas_str,
    betas_refine_str,
    voxel_size,
    use_personalization,
    obj_initialization,
    t_range_str,
    t_range_refine_str,
    refine_after,
    use_unet,
    num_together_images,
    num_together_images_refine,
    wandb_exp_name,
    prob_mvc,
    obj_desc,
    lora_or_unet_path,
    use_conv_in,
    double_personalizations,
    refine_obj,
    guidance_scale,
    load_ckpt,
    use_min_for_generation,
    optimize_image_for_refine,
    scene_shadows,
    keep_scene_shadows_top,
    use_wandb,
    crop_ratio,
    datamanager_seed,
    refine_linear,
    transforms_obj,
    sd_version,
    use_controlnet,
    use_personalization_from: str | None,
    not_visible_const: float,
    num_fixed_train_angles: int,
):
    t_range = t_range_str.replace("(", "").replace(")", "").replace(", ", " ")
    t_range_refine = (
        t_range_refine_str.replace("(", "").replace(")", "").replace(", ", " ")
    )
    output_dir = os.path.join(workspace, "3dgs_voi")
    prompt_obj = f"a <ktn> {obj_desc}"
    # maybe_load_ckpt = f"--pipeline.model.load_checkpoint {load_ckpt} " if not load_ckpt is None else " "
    maybe_load_ckpt = (
        f"--load_checkpoint {load_ckpt} " if load_ckpt is not None else " "
    )
    vis_wandb = "--vis wandb " if use_wandb != 0 else " "
    refine_linear_str = (
        f"--pipeline.model.refine-range-linear {(max_num_iterations - refine_after)} "
        if refine_linear != 0
        else "--pipeline.model.refine-range-linear -1 "
    )
    print("refine_linear:", refine_linear_str)

    maybe_transforms_obj = (
        " "
        if transforms_obj is None
        else f"--pipeline.model.transforms_obj '{transforms_obj}' "
        + f"--pipeline.datamanager.transforms_obj '{transforms_obj}' "
    )

    if sd_version == "2.1":
        controlnet_model_name = "thibaud/controlnet-sd21-depth-diffusers"
    elif sd_version == "2.0":
        controlnet_model_name = "thibaud/controlnet-sd21-depth-diffusers"
    elif sd_version == "1.5":
        controlnet_model_name = "lllyasviel/sd-controlnet-depth"

    personalization_folders_root = (
        workspace
        if use_personalization_from is None
        else use_personalization_from
    )

    if double_personalizations and use_personalization != 0:
        lora_unet_path_1 = os.path.join(
            personalization_folders_root, "personalization_object_1"
        )
        lora_unet_path_2 = os.path.join(
            personalization_folders_root, "personalization_object_2"
        )
        conv_in_path = os.path.join(
            personalization_folders_root, "personalization_object_2/conv_in.pth"
        )

        print("Run ns-train")

        os.system(
            "ns-train d3dr "
            + f"--output_dir {output_dir} "
            + "--project-name voi-3dgs "
            + f"--experiment-name {wandb_exp_name} "
            + vis_wandb
            + f"--pipeline.datamanager.init-scene-path {init_scene_path} "
            + f"--pipeline.datamanager.init-obj-path {init_obj_path} "
            + f"--pipeline.datamanager.seed {datamanager_seed} "
            + f"--pipeline.model.init-scene-path {init_scene_path} "
            + f"--pipeline.model.init-obj-path {init_obj_path} "
            + f"--pipeline.model.prompt-desired '{prompt_desired}' "
            + maybe_transforms_obj
            + f"--pipeline.model.prompt-initial '{prompt_initial}' "
            + f"--pipeline.model.guidance_scale '{guidance_scale}' "
            + f"--pipeline.model.prompt-obj '{prompt_obj}' "
            + "--pipeline.model.background-color black "
            + f"--pipeline.model.refine_after {refine_after} "
            + f"--pipeline.model.lora-object-path {lora_unet_path_1} "
            + f"--pipeline.model.lora-object-texture-path {lora_unet_path_2} "
            + f"--pipeline.model.sd-version {sd_version} "
            + f"--pipeline.model.controlnet-model-name {controlnet_model_name} "
            + f"--pipeline.model.use_controlnet {use_controlnet} "
            + "--viewer.make-share-url False "
            + f"--pipeline.datamanager.vertical-rotation-range {-angles_z[1]} {-angles_z[0]} "
            + f"--pipeline.model.obj-initialization {obj_initialization} "
            + maybe_load_ckpt
            + f"--pipeline.model.conv_in_path {conv_in_path} "
            + f"--pipeline.model.optimize_latent_for {optimize_latent_for} "
            + f"--pipeline.model.optimize_image_for {optimize_image_for} "
            + f"--pipeline.model.optimize_image_for_refine {optimize_image_for_refine} "
            + f"--pipeline.model.num_together_images {num_together_images} "
            + f"--pipeline.model.num_together_images_refine {num_together_images_refine} "
            + f"--pipeline.model.sh-degree-interval {sh_degree_interval} "
            + f"--pipeline.model.t-range {t_range} "
            + f"--pipeline.model.t-range-refine {t_range_refine} "
            + refine_linear_str
            + f"--pipeline.model.prob_mvc {prob_mvc} "
            + f"--pipeline.model.refine_obj {refine_obj} "
            + f"--pipeline.model.scene_shadows {scene_shadows} "
            + f"--pipeline.model.keep_scene_shadows_top {keep_scene_shadows_top} "
            + f"--pipeline.model.crop_ratio {crop_ratio} "
            + f"--pipeline.model.not_visible_const {not_visible_const} "
            + f"--pipeline.datamanager.betas {betas_str} "
            + f"--pipeline.datamanager.use_min_for_generation {use_min_for_generation} "
            + f"--pipeline.datamanager.betas_refine {betas_refine_str} "
            + f"--pipeline.datamanager.num_fixed_train_angles {num_fixed_train_angles} "
            + f"--pipeline.datamanager.voxel-size {voxel_size} "
            + "--viewer.quit-on-train-completion True "
            + f"--max_num_iterations {max_num_iterations} "
        )

    else:
        if use_unet:
            if not lora_or_unet_path:
                sd_unet_path = os.path.join(
                    personalization_folders_root, "personalization_object/unet"
                )
            else:
                sd_unet_path = lora_or_unet_path
            sd_param = f"--pipeline.model.sd-unet-path {sd_unet_path} "
        else:
            if not lora_or_unet_path:
                lora_unet_path = os.path.join(
                    personalization_folders_root, "personalization_object"
                )
            else:
                lora_unet_path = lora_or_unet_path

            if not os.path.exists(lora_unet_path):
                lora_unet_path = lora_unet_path + "_1"
            sd_param = f"--pipeline.model.lora-object-path {lora_unet_path} "

        if use_conv_in:
            conv_in_path = os.path.join(
                personalization_folders_root,
                "personalization_object/conv_in.pth",
            )
        maybe_conv_in_path = (
            f"--pipeline.model.conv_in_path {conv_in_path} "
            if use_conv_in
            else " "
        )
        maybe_use_sd = f"{sd_param} " if use_personalization != 0 else " "

        os.system(
            "ns-train d3dr "
            + f"--output_dir {output_dir} "
            + "--project-name voi-3dgs "
            + f"--experiment-name {wandb_exp_name} "
            + vis_wandb
            + f"--pipeline.datamanager.init-scene-path {init_scene_path} "
            + f"--pipeline.datamanager.init-obj-path {init_obj_path} "
            + f"--pipeline.datamanager.seed {datamanager_seed} "
            + f"--pipeline.model.init-scene-path {init_scene_path} "
            + f"--pipeline.model.init-obj-path {init_obj_path} "
            + f"--pipeline.model.prompt-desired '{prompt_desired}' "
            + f"--pipeline.model.prompt-initial '{prompt_initial}' "
            + f"--pipeline.model.prompt-obj '{prompt_obj}' "
            + "--pipeline.model.background-color black "
            + f"--pipeline.model.refine_after {refine_after} "
            + f"--pipeline.model.controlnet-model-name {controlnet_model_name} "
            + f"--pipeline.model.use_controlnet {use_controlnet} "
            + "--viewer.make-share-url False "
            + f"--pipeline.datamanager.vertical-rotation-range {-angles_z[1]} {-angles_z[0]} "
            + maybe_use_sd
            + f"--pipeline.model.obj-initialization {obj_initialization} "
            + maybe_conv_in_path
            + maybe_load_ckpt
            + f"--pipeline.model.optimize_latent_for {optimize_latent_for} "
            + maybe_transforms_obj
            + f"--pipeline.model.optimize_image_for {optimize_image_for} "
            + f"--pipeline.model.optimize_image_for_refine {optimize_image_for_refine} "
            + f"--pipeline.model.num_together_images {num_together_images} "
            + f"--pipeline.model.num_together_images_refine {num_together_images_refine} "
            + f"--pipeline.model.sh-degree-interval {sh_degree_interval} "
            + f"--pipeline.model.t-range {t_range} "
            + f"--pipeline.model.t-range-refine {t_range_refine} "
            + refine_linear_str
            + f"--pipeline.model.prob_mvc {prob_mvc} "
            + f"--pipeline.model.refine_obj {refine_obj} "
            + f"--pipeline.model.scene_shadows {scene_shadows} "
            + f"--pipeline.model.keep_scene_shadows_top {keep_scene_shadows_top} "
            + f"--pipeline.model.crop_ratio {crop_ratio} "
            + f"--pipeline.model.not_visible_const {not_visible_const} "
            + f"--pipeline.datamanager.betas {betas_str} "
            + f"--pipeline.datamanager.betas_refine {betas_refine_str} "
            + f"--pipeline.datamanager.use_min_for_generation {use_min_for_generation} "
            + f"--pipeline.datamanager.voxel-size {voxel_size} "
            + f"--pipeline.datamanager.num_fixed_train_angles {num_fixed_train_angles} "
            + "--viewer.quit-on-train-completion True "
            + f"--max_num_iterations {max_num_iterations} "
        )


# ------------------------------


def _get_config(path):
    store_dir_path = find_store_dir_path(path)
    if store_dir_path is None:
        return None
    return Path(store_dir_path) / "config.yml"


def _render_rgb(
    config_path, camera_path, output_dir, name="rgb", img_format="png"
):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    cmd = (
        "ns-render camera-path "
        + f"--load-config {config_path} "
        + f"--camera-path-filename {camera_path} "
        + f"--rendered-output-names {name} "
        + f"--output-path {output_dir} "
        + f"--image-format {img_format} "
        + "--output-format images "
    )
    print(cmd)
    os.system(cmd)


def _render_mask(
    config_path, camera_path, output_dir, name="mask", img_format="jpeg"
):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.system(
        "ns-render camera-path "
        + f"--load-config {config_path} "
        + f"--camera-path-filename {camera_path} "
        + f"--rendered-output-names {name} "
        + f"--output-path {output_dir} "
        + "--output-format images "
        + f"--image-format {img_format} "
        + "--colormap-options.colormap gray "
    )


def render_voi(
    dataset_root,
    scene_name,
    workspace,
    gaussian_splatting_root,
    init_scene_path,
):
    transforms_obj_scene_path = Path(workspace) / "transforms_obj_scene.json"
    camera_path = Path(workspace) / "camera_path.json"
    shutil.copy2(
        Path(dataset_root) / scene_name / "obj_scene_eval/transforms.json",
        transforms_obj_scene_path,
    )
    os.system(
        "python3 d3dr/scripts/transforms_2_camera_path.py "
        + f"--tr_path {transforms_obj_scene_path} "
        + f"--save_name {camera_path} "
    )

    output_rendering_dir = Path(workspace) / "rendering"
    config_voi_path = _get_config(workspace)

    # render voi
    _render_rgb(
        config_voi_path,
        camera_path,
        output_rendering_dir / "voi_rgb_obj_scene",
        name="rgb_obj_scene",
    )
    _render_mask(
        config_voi_path,
        camera_path,
        output_rendering_dir / "voi_mask_obj_scene",
        name="mask_obj",
    )

    # render scene real
    _render_rgb(
        os.path.join(init_scene_path, "config.yml"),
        camera_path,
        output_rendering_dir / "real_rgb_scene",
        name="rgb",
    )

    if gaussian_splatting_root is None:
        return

    config_combined_initial_path = _get_config(
        Path(gaussian_splatting_root) / f"{scene_name}-combined-initial"
    )
    # render obj_scene initial
    if config_combined_initial_path is None:
        return

    _render_rgb(
        config_combined_initial_path,
        camera_path,
        output_rendering_dir / "initial_rgb_obj_scene",
        name="rgb",
    )
    _render_mask(
        config_combined_initial_path,
        camera_path,
        output_rendering_dir / "initial_mask_obj_scene",
    )


def evaluate_voi(
    scene_name,
    dataset_root,
    workspace,
    gaussian_splatting_root,
    prompt_initial,
    prompt_desired,
):
    config_voi_path = _get_config(workspace)
    output_path_voi = Path(workspace) / "metrics.json"

    os.system(
        "python3 d3dr/validation/eval.py "
        + f"--load-config {config_voi_path} "
        + f"--output-path {output_path_voi} "
    )

    if gaussian_splatting_root is None:
        return
    config_combined_initial_path = _get_config(
        Path(gaussian_splatting_root) / f"{scene_name}-combined-initial"
    )
    if config_combined_initial_path is None:
        return
    output_path_combined_initial = Path(workspace) / "metrics_comb_init.json"

    os.system(
        "python3 d3dr/validation/eval_combined.py "
        + f"--load-config {config_combined_initial_path} "
        + f"--output_path {output_path_combined_initial} "
    )


def main():
    args = get_args()

    exit_code = os.system("ffmpeg --help > /dev/null")
    if exit_code != 0:
        raise RuntimeError("ffmpeg is not installed!")

    if 0 not in args.skip_steps and args.use_personalization_from is None:
        start_time = time.time()
        generate_object_images(
            curr_data_root=args.curr_data_root,
            workspace=args.workspace,
            init_obj_path=args.init_obj_path,
            betas_str=args.betas_str_generation,
            scene_name=args.scene_name,
            angles_z_str=args.angles_z_str,
            generate_num=args.generate_num,
            leave_highest_areas=args.leave_highest_areas,
        )
        print(
            f"generate_object_images() took {time.time() - start_time:.2f} seconds"
        )

    if (
        1 not in args.skip_steps
        and args.use_personalization != 0
        and args.use_personalization_from is None
    ):
        start_time = time.time()
        print("Start personalizing")
        personalize_object(
            obj_desc=args.obj_desc,
            workspace=args.workspace,
            ic_light_prompt=args.ic_light_prompt,
            generate_using_iclight=args.generate_using_iclight,
            prob=args.person_prob,
            use_unet=args.use_unet,
            num_personalization_iterations=args.num_personalization_iterations,
            use_conv_in=args.use_conv_in,
            double_personalizations=args.double_personalizations,
            fixed_place_prompt=args.fixed_place_prompt,
            sd_version=args.sd_version,
        )
        print(
            f"personalize_object() took {time.time() - start_time:.2f} seconds"
        )

    if 2 not in args.skip_steps:
        print("Run VOI")
        start_time = time.time()

        # too many arguments :(
        perform_voi(
            workspace=args.workspace,
            init_obj_path=args.init_obj_path,
            prompt_desired=args.prompt_desired,
            prompt_initial=args.prompt_initial,
            init_scene_path=args.init_scene_path,
            max_num_iterations=args.max_num_iterations_voi,
            sh_degree_interval=args.sh_degree_inverval,
            optimize_latent_for=args.optimize_latent_for,
            optimize_image_for=args.optimize_image_for,
            angles_z=args.angles_z,
            betas_str=args.betas_str,
            betas_refine_str=args.betas_refine_str,
            voxel_size=args.voxel_size,
            use_personalization=args.use_personalization,
            obj_initialization=args.obj_initialization,
            t_range_str=args.t_range_str,
            t_range_refine_str=args.t_range_refine_str,
            refine_after=args.refine_after,
            use_unet=args.use_unet,
            num_together_images=args.num_together_images,
            num_together_images_refine=args.num_together_images_refine,
            wandb_exp_name=args.wandb_exp_name,
            prob_mvc=args.prob_mvc,
            obj_desc=args.obj_desc,
            lora_or_unet_path=args.lora_or_unet_path,
            use_conv_in=args.use_conv_in,
            double_personalizations=args.double_personalizations,
            refine_obj=args.refine_obj,
            guidance_scale=args.guidance_scale,
            load_ckpt=args.load_ckpt,
            use_min_for_generation=args.use_min_for_generation,
            optimize_image_for_refine=args.optimize_image_for_refine,
            scene_shadows=args.scene_shadows,
            keep_scene_shadows_top=args.keep_scene_shadows_top,
            use_wandb=args.use_wandb,
            crop_ratio=args.crop_ratio,
            datamanager_seed=args.datamanager_seed,
            refine_linear=args.refine_linear,
            transforms_obj=args.transforms_obj,
            sd_version=args.sd_version,
            use_controlnet=args.use_controlnet,
            use_personalization_from=args.use_personalization_from,
            not_visible_const=args.not_visible_const,
            num_fixed_train_angles=args.num_fixed_train_angles,
        )
        print(f"perform_voi() took {time.time() - start_time:.2f} seconds")

    if 3 not in args.skip_steps:
        render_voi(
            dataset_root=args.dataset_root,
            scene_name=args.scene_name,
            workspace=args.workspace,
            gaussian_splatting_root=args.gaussian_splatting_root,
            init_scene_path=args.init_scene_path,
        )

    if 4 not in args.skip_steps:
        evaluate_voi(
            scene_name=args.scene_name,
            dataset_root=args.dataset_root,
            workspace=args.workspace,
            gaussian_splatting_root=args.gaussian_splatting_root,
            prompt_initial=args.prompt_initial,
            prompt_desired=args.prompt_desired,
        )


if __name__ == "__main__":
    main()
