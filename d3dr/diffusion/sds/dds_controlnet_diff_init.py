import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from d3dr.diffusion.sd_utils_controlnet import SDControlNet
from d3dr.diffusion.sds.utils import (
    center_crop_to_square_with_ratio,
    masked_mean,
    read_image,
)


def get_args(do_parse=True):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt_1", type=str, default="", help="desired prompt"
    )
    parser.add_argument(
        "--prompt_2", type=str, default="", help="initial prompt"
    )
    parser.add_argument(
        "--exp_desc",
        type=str,
        default="",
        help="the description of the experiment",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sd_model_name", type=str, default="Manojb/stable-diffusion-2-1-base"
    )
    parser.add_argument(
        "--sd_unet_path",
        type=str,
        default=None,
        help="path to the unet model (for personalization)",
    )
    parser.add_argument(
        "--controlnet_model_name",
        type=str,
        default="thibaud/controlnet-sd21-depth-diffusers",
    )
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--controlnet_conditioning_scale",
        type=float,
        default=1.0,
        help="the scale of the controlnet conditioning. Something bigger that 1.0 WORKS POOR!",
    )

    parser.add_argument(
        "--image_comp_path",
        type=str,
        default=None,
        help="path to the obj + scene rgb",
    )
    parser.add_argument(
        "--image_nocomp_path",
        type=str,
        default=None,
        help="path to the ONLY scene rgb",
    )
    parser.add_argument(
        "--controlnet_comp_path",
        type=str,
        default=None,
        help="path to the obj + scene rgb depth/normal/etc for controlnet",
    )
    parser.add_argument(
        "--controlnet_nocomp_path",
        type=str,
        default=None,
        help="path to the ONLY scene rgb depth/normal/etc for controlnet",
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        default=None,
        help="path to the mask of obj + scene rgb. 1 - obj, 0 - scene",
    )
    parser.add_argument("--crop_ratio", type=float, default=0.7)

    parser.add_argument("--num_train_iterations", type=int, default=400)
    parser.add_argument(
        "--show_iter",
        type=int,
        default=50,
        help="how often to show the generated image",
    )
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument(
        "--power",
        type=float,
        default=0.01,
        help="power for the polynomial decay",
    )
    parser.add_argument("--save_dir", type=str, default="./dds/")

    parser.add_argument(
        "--exp",
        type=str,
        default="implicit_dds",
        choices=["optimize_alpha", "implicit_dds", "explicit_dds"],
        help="experiment type. For some reason optimize_alpha doesn't work",
    )
    parser.add_argument(
        "--add_random_noise_mask",
        type=int,
        default=0,
        help="initialize with random noise",
    )
    parser.add_argument(
        "--add_mean_init", type=int, default=1, help="initialize with mean"
    )
    parser.add_argument(
        "--mask_grad",
        type=int,
        default=1,
        help="mask the gradient? (for latent space it will not be correct...)",
    )
    parser.add_argument(
        "--use_random_noise",
        type=int,
        default=1,
        help="in SDS one can take random noise or a FIXED noise to predict",
    )
    parser.add_argument(
        "--use_step_ratio",
        type=int,
        default=0,
        help="Larger steps at the beginning of the optimization",
    )
    parser.add_argument(
        "--initial_step",
        type=float,
        default=0.0,
        help="if use_step_ratio, then we might want to begin from 0.5 (-> timestep 500) instead of 0.0",
    )
    parser.add_argument("--fp16", type=int, default=1)
    parser.add_argument("--t_range_max", type=float, default=0.98)
    parser.add_argument(
        "--lora_adapters_paths", type=str, action="append", default=[]
    )
    if do_parse:
        return parser.parse_args()

    return parser


def preprocess_depth_image(
    image_path, height, width, torch_device, guidance=None
):
    controlnet_image = cv2.imread(image_path)
    controlnet_image = center_crop_to_square_with_ratio(
        controlnet_image
    )  # depth in my case
    controlnet_image = cv2.resize(controlnet_image, (height, width)) / 255.0
    controlnet_image_emb = guidance.get_image_embeds(controlnet_image).to(
        torch_device
    )
    return controlnet_image_emb


def main():
    args = get_args()

    torch_device = "cuda"

    # Load the model
    guidance = SDControlNet(
        device="cuda",
        sd_version=args.sd_model_name,
        controlnet_name=args.controlnet_model_name,
        height=args.height,
        width=args.width,
        sd_unet_path=args.sd_unet_path,
        fp16=args.fp16,
        lora_adapters_paths=args.lora_adapters_paths,
        t_range=(0.02, args.t_range_max),
    )

    # Load images
    init_image_nocomp, init_image_torch_nocomp, init_latent_nocomp = read_image(
        args.image_nocomp_path,
        args.height,
        args.width,
        guidance,
        crop_ratio=args.crop_ratio,
    )
    init_image_comp, init_image_torch_comp, init_latent_comp = read_image(
        args.image_comp_path,
        args.height,
        args.width,
        guidance,
        crop_ratio=args.crop_ratio,
    )

    # Load mask
    init_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
    init_mask = center_crop_to_square_with_ratio(
        init_mask, crop_ratio=args.crop_ratio
    )
    init_mask = (
        cv2.resize(init_mask, (args.height, args.width))[None, None, ...]
        / 255.0
        > 0.5
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

    # Load controlnet embeddings
    controlnet_nocomp_emb = preprocess_depth_image(
        args.controlnet_nocomp_path,
        args.height,
        args.width,
        torch_device,
        guidance,
    )
    controlnet_comp_emb = preprocess_depth_image(
        args.controlnet_comp_path,
        args.height,
        args.width,
        torch_device,
        guidance,
    )

    torch.manual_seed(args.seed)

    # initialize the sds_image (the correct name is dds)
    with torch.no_grad():
        if args.add_mean_init > 0:
            init_image_torch_comp = init_image_torch_comp.to(torch.float32) * (
                1.0 - mask.to(torch.float32)
            ) + masked_mean(
                init_image_torch_comp.to(torch.float32),
                mask.to(torch.float32),
                dim=(2, 3),
            )[..., None, None] * mask.to(torch.float32)
            init_image_torch_comp = init_image_torch_comp.to(
                guidance.precision_t
            )
        if args.add_random_noise_mask > 0:
            init_image_torch_comp = (
                init_image_torch_comp * (1.0 - mask)
                + torch.randn_like(init_image_torch_comp) * mask
            )

        init_latent_comp = guidance.torch2latents(init_image_torch_comp)

    sds_image = init_latent_comp.detach().clone().requires_grad_(True)

    # optimizer and scheduler
    optimizer = torch.optim.SGD([sds_image], lr=args.lr)
    opt_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=args.num_train_iterations, power=args.power
    )

    args.save_dir = guidance.get_save_dir(args.save_dir)
    print("Save dir:", args.save_dir)
    guidance.save_images(
        images=guidance.torch2np(init_image_torch_comp),
        save_name="initial.png",
        prompt=f"{args.prompt_2} -> {args.prompt_1}",
        save_dir=args.save_dir,
        exp_desc=args.exp_desc,
    )

    # save the parameters of the experiment
    with open(Path(args.save_dir) / "args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    # Load prompts
    prompt_initial = guidance.get_text_embeds(args.prompt_2)
    prompt_desired = guidance.get_text_embeds(args.prompt_1)
    uncond_emb = guidance.get_text_embeds("")
    text_embeddings_initial = torch.cat([uncond_emb, prompt_initial])
    text_embeddings_desired = torch.cat([uncond_emb, prompt_desired])

    step_ratio = None
    noise = None
    if args.use_random_noise == 0:  # use random noise
        noise = torch.randn_like(init_latent_comp)

    # The main train cycle!
    for i in tqdm(range(args.num_train_iterations)):
        # zero_grad
        optimizer.zero_grad()
        if args.use_step_ratio != 0:
            step_ratio = min(1, i / args.num_train_iterations)

        # the main idea is to use dds, but the initialization of the
        # optimized image is different. Here we use the controlnet
        # and the conditions from the controlnet are different.
        loss = guidance.train_step_dds(
            text_embeddings_initial=text_embeddings_initial,
            text_embeddings_desired=text_embeddings_desired,
            image_embeddings_initial=controlnet_nocomp_emb,
            image_embeddings_desired=controlnet_comp_emb,  # they are the same
            latents_initial=init_latent_nocomp,  # I call it a "pulling latent"
            rgb_pred=sds_image,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_conditioning_scale,
            as_latent=True,
            step_ratio=step_ratio,
            noise=noise,
            use_weights=False,
        )

        loss.backward()

        # one should mask the gradient
        if args.mask_grad != 0:
            sds_image.grad = sds_image.grad * mask_small

        optimizer.step()
        opt_scheduler.step()

        if (i + 1) % args.show_iter == 0:
            result_image = guidance.torch2np(guidance.latents2torch(sds_image))
            guidance.save_images(
                images=result_image,
                save_name=f"dds_image_{i + 1}.png",
                prompt=f"{args.prompt_2} -> {args.prompt_1}",
                save_dir=args.save_dir,
                exp_desc=args.exp_desc,
            )


if __name__ == "__main__":
    main()
