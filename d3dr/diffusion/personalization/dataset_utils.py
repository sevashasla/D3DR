"""
This module provides utilities for enriching image datasets for personalization tasks.
It supports generating additional images using external scripts, copying images between directories,
and augmenting datasets with both generated and raw images.
"""

import gc
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

CURRENT_PATH = Path(__file__).parent.resolve()

PLACES_PATH = str(CURRENT_PATH / "places.json")
IC_LIGHT_RUN = str(CURRENT_PATH / "ic_light_run.py")
PRESERVATION_RUN_PATH = str(CURRENT_PATH / "generate_preservation.py")

def copy_all_from_to(from_, to):
    """Copies all files from one directory to another"""
    for file in Path(from_).glob("*"):
        shutil.copy2(file, to)

def enrich_dataset(
        image_dir,
        save_dir,
        
        generate_iclight_n=32,
        iclight_prompt=None,
        just_cp_iclight_dir=None,
        do_generate_iclight=True,
        add_raw_images_ratio=0.15,

        generate_preservation_n=32,
        preservation_prompt=None,
        just_cp_preservation_dir=None,

        prob=0.9,
        with_preservation=None,
        fixed_place_prompt=None,
):
    """
    Creates a dataset of:
     - IC-Light processed images of `image_dir` (and possibly original images)
     - preservation images
    
    The `add_raw_images_ratio` controls how many original images are added to the 
    dataset.
    """
    rng = np.random.default_rng(0)
    image_dir = Path(image_dir)
    save_dir = Path(save_dir)
    
    save_iclight_dir = save_dir / "generated_images_iclight"
    save_preservation_dir = save_dir / "generated_images_obj"

    if with_preservation is None:
        with_preservation = (prob < 1.0)
    
    print(f"[INFO] Build dataset: with_preservation = {with_preservation}, prob = {prob}")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_iclight_dir, exist_ok=True)

    if with_preservation:
        os.makedirs(save_preservation_dir, exist_ok=True)

        if just_cp_preservation_dir is not None:
            # copy all images and maybe text descriptions
            copy_all_from_to(just_cp_preservation_dir, save_preservation_dir)
            print("[INFO] Done copying preservation images")
        else:
            # generate object images from scratch
            cmd = [
                "python3", PRESERVATION_RUN_PATH,
                "--prompt", preservation_prompt,
                "--num_images", generate_preservation_n,
                "--output_dir", str(save_preservation_dir),
            ]
            cmd = [str(el) for el in cmd]
            subprocess.run(cmd)
            print("[INFO] Done generating preservation images")

        # free resources
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("[INFO] Do not use preservation")

    if just_cp_iclight_dir is not None:
        # just copy images
        print("[INFO] Copy preservation images")
        print("from:", just_cp_iclight_dir)
        print("to:", save_iclight_dir)
        if str(Path(just_cp_iclight_dir)) != str(save_iclight_dir):
            copy_all_from_to(just_cp_iclight_dir, save_iclight_dir)

        if len(os.listdir(just_cp_iclight_dir)) - 1 < generate_iclight_n:
            print(f"[WARN] {just_cp_iclight_dir} is small")
    else:
        if not do_generate_iclight:
            # just copy images without iclight image generation
            copy_all_from_to(image_dir, save_iclight_dir)
            # generate a json file for it
            with open(str(save_iclight_dir / "prompts.json"), "w") as f:
                prompts = []
                iclight_files = save_iclight_dir.rglob("*")
                iclight_files = sorted(iclight_files)
                for file in iclight_files:
                    prompt_curr = {
                            "prompt": iclight_prompt,
                            "fg_name": file.name,
                            "bg_name": "",
                            "result_image": file.name,
                        }
                    prompts.append(prompt_curr)
                json.dump(prompts, f, indent=4)
        else:
            # generate more
            if fixed_place_prompt is not None:
                fixed_place_prompt = ["--fixed_place_prompt", fixed_place_prompt]
            else:
                fixed_place_prompt = []

            cmd = [
                "python3", IC_LIGHT_RUN,
                "--process_dir",
                "--input_fg_dir", str(image_dir),
                "--prompt", iclight_prompt,
                "--prompts_dir", PLACES_PATH,
                *fixed_place_prompt,
                "--num_images", generate_iclight_n,
                "--output_dir", str(save_iclight_dir),
                "--position_light", "left",
            ]
            cmd = [str(el) for el in cmd]
            
            subprocess.run(cmd)
            gc.collect()
            torch.cuda.empty_cache()

            print("[INFO] Done generating preservation images")

            if add_raw_images_ratio >= 0.0:
                with open(save_iclight_dir / "prompts.json", "r") as f:
                    curr_prompts = json.load(f)
                n = len(curr_prompts) # should be generate_iclight_n
                # copy some random images to the initial dataset
                raw_images = list(image_dir.rglob("*"))
                add_raw_images_num = int(np.ceil(add_raw_images_ratio / (1 - add_raw_images_ratio) * n))
                for j in range(add_raw_images_num):
                    random_img_path_in = rng.choice(raw_images)
                    random_img_path_out = save_iclight_dir / f"{j + generate_iclight_n:05}.png"
                    shutil.copy2(random_img_path_in, random_img_path_out)
                    curr_prompts.append({
                        "prompt": iclight_prompt,
                        "fg_name": random_img_path_in.name,
                        "bg_name": "",
                        "result_image": random_img_path_out.name,
                    })
                
                with open(str(save_iclight_dir / "prompts.json"), "w") as f:
                    json.dump(curr_prompts, f, indent=4)
                
                print("[INFO] Done copying raw images")
    
    # free resources
    gc.collect()
    torch.cuda.empty_cache()

    print("[INFO] Done dataset preparation!")
    return save_dir, save_iclight_dir, save_preservation_dir
