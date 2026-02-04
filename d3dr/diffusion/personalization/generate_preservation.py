"""
Generate preservation images (batched)
"""

import argparse
import gc
import os

import cv2
import numpy as np

from d3dr.diffusion.sd_utils import StableDiffusion


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fp16", type=int, default=1)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--num_images", type=int, default=10)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    return parser.parse_args()

def main():
    args = get_args()

    torch_device = "cuda"
    os.makedirs(args.output_dir, exist_ok=True)
    print(args.output_dir)

    guidance = StableDiffusion(
        device=torch_device,
        sd_version=args.model_name,
        height=args.height,
        width=args.width,
        fp16=args.fp16,
    )

    curr_image_idx = 0
    for batch_idx in range(int(np.ceil(args.num_images / args.batch_size).item())):
        if (batch_idx + 1) * args.batch_size <= args.num_images:
            curr_num_images = args.batch_size
        else:
            curr_num_images = args.num_images % args.batch_size

        images_generated = guidance.generate_images_by_prompts(
            [args.prompt], 
            num_same=curr_num_images,
            num_inference_steps=args.num_inference_steps, 
            guidance_scale=args.guidance_scale, 
            seed=args.seed + batch_idx,
        )

        for i in range(len(images_generated)):
            cv2.imwrite(
                os.path.join(args.output_dir, f"generated_{curr_image_idx:05}.png"),
                cv2.cvtColor(images_generated[i], cv2.COLOR_RGB2BGR), 
            )
            curr_image_idx += 1

        gc.collect()


if __name__ == "__main__":
    main()
