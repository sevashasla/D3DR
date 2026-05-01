"""
DDS image inpaitning
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from d3dr.diffusion.sd_utils import StableDiffusion
from d3dr.diffusion.sds.utils import (
    center_crop_to_square_with_ratio,
    read_image,
)


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_1", type=str, default="", help="desired prompt"
    )
    parser.add_argument(
        "--prompt_2", type=str, default="", help="initial prompt"
    )
    parser.add_argument("--exp_desc", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model_name",
        type=str,
        default="Manojb/stable-diffusion-2-1-base",
    )
    parser.add_argument("--sd_unet_path", type=str, default=None)
    parser.add_argument(
        "--lora_adapters_paths", type=str, action="append", default=[]
    )
    parser.add_argument("--conv_in_path", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--image_obj", type=str, default=None)

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, required=None)

    parser.add_argument("--num_train_iterations", type=int, default=400)
    parser.add_argument("--show_iter", type=int, default=50)
    parser.add_argument("--acc_step", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--power", type=float, default=0.0)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./dds/",
    )

    parser.add_argument("--add_random_noise_mask", action="store_true")
    parser.add_argument("--mask_grad", type=int, default=1)
    parser.add_argument("--use_random_noise", type=int, default=1)
    parser.add_argument("--use_step_ratio", type=int, default=0)
    parser.add_argument("--initial_step", type=float, default=0.0)
    parser.add_argument("--fp16", type=int, default=1)
    parser.add_argument("--t_range_max", type=float, default=0.98)
    parser.add_argument("--crop_ratio", type=float, default=0.7)

    return parser.parse_args()


def main():
    args = get_args()
    torch_device = "cuda"
    guidance = StableDiffusion(
        device=torch_device,
        sd_version=args.model_name,
        height=args.height,
        width=args.width,
        sd_unet_path=args.sd_unet_path,
        fp16=args.fp16,
        lora_adapters_paths=args.lora_adapters_paths,
        conv_in_path=args.conv_in_path,
        t_range=(0.02, args.t_range_max),
    )

    init_image, init_image_torch, init_latent = read_image(
        args.image_path,
        height=args.height,
        width=args.width,
        guidance=guidance,
        crop_ratio=args.crop_ratio,
    )

    if args.mask_path is not None:
        init_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        init_mask = center_crop_to_square_with_ratio(
            init_mask, crop_ratio=args.crop_ratio
        )
        init_mask = (
            cv2.resize(init_mask, (args.height, args.width))[None, None, ...]
            / 255.0
            > 0.95
        ).astype(np.float32)
        with torch.no_grad():
            mask = torch.from_numpy(init_mask).to(
                device=torch_device, dtype=guidance.precision_t
            )
            mask_small = F.interpolate(
                mask,
                size=(args.height // 8, args.width // 8),
                mode="bilinear",
                align_corners=False,
            )
    else:
        init_mask = None
        mask = None
        mask_small = None

    if args.image_obj is None:
        rgb_obj_pred = None
    else:
        rgb_obj_pred = cv2.cvtColor(
            cv2.imread(args.image_obj), cv2.COLOR_BGR2RGB
        )
        rgb_obj_pred = cv2.resize(rgb_obj_pred, (args.width, args.height))
        rgb_obj_pred = guidance.torch2latents(guidance.np2torch(rgb_obj_pred))

    torch.manual_seed(args.seed)

    sds_image = init_latent.detach().clone().requires_grad_(True)
    if args.add_random_noise_mask:
        with torch.no_grad():
            sds_image.data = guidance.torch2latents(
                init_image_torch * (1.0 - mask)
                + torch.randn_like(init_image_torch) * mask
            )

    optimizer = torch.optim.SGD([sds_image], lr=args.lr)
    opt_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=args.num_train_iterations // args.acc_step,
        power=args.power,
    )

    args.save_dir = guidance.get_save_dir(args.save_dir)
    print("Save dir:", args.save_dir)

    with open(Path(args.save_dir) / "args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    prompt_desired = guidance.get_text_embeds(args.prompt_1)
    prompt_initial = guidance.get_text_embeds(args.prompt_2)
    uncond_emb = guidance.get_text_embeds("")
    text_embeddings_desired = torch.cat([uncond_emb, prompt_desired])
    text_embeddings_initial = torch.cat([uncond_emb, prompt_initial])

    step_ratio = None
    noise = None
    if args.use_random_noise == 0:  # use random noise
        noise = torch.randn_like(init_latent)

    for i in tqdm(range(args.num_train_iterations)):
        # zero_grad
        optimizer.zero_grad()
        if args.use_step_ratio != 0:
            step_ratio = min(
                1,
                (1 - args.initial_step)
                + args.initial_step * i / args.num_train_iterations,
            )

        loss = guidance.train_step_dds(
            text_embeddings_initial=text_embeddings_initial,
            text_embeddings_desired=text_embeddings_desired,
            latents_initial=init_latent,
            pred_rgb_obj=rgb_obj_pred,
            rgb_pred=sds_image,
            guidance_scale=args.guidance_scale,
            as_latent=True,
            step_ratio=step_ratio,
            noise=noise,
            use_weights=False,
        )

        loss.backward()

        if args.mask_grad != 0 and mask_small is not None:
            sds_image.grad = sds_image.grad * mask_small

        optimizer.step()
        opt_scheduler.step()

        if (i + 1) % args.show_iter == 0:
            result_image = guidance.torch2np(guidance.latents2torch(sds_image))
            guidance.save_images(
                images=result_image,
                save_name=f"dds_image_{i + 1}.jpg",
                prompt=args.prompt_1 + "|" + args.prompt_2,
                save_dir=args.save_dir,
                exp_desc=args.exp_desc,
            )


if __name__ == "__main__":
    main()
