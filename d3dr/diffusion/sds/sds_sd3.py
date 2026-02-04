"""
SDS
(I have a little of comments here. Please check out try_dds_controlnet_diff_init.py for more comments)
"""

import argparse

import cv2
import torch
from tqdm.auto import tqdm

from d3dr.diffusion.sd3_utils import StableDiffusion3
from d3dr.diffusion.sds.utils import center_crop_to_square


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A cat")
    parser.add_argument("--exp_desc", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image_obj", type=str, default=None)
    parser.add_argument(
        "--model_name",
        type=str,
        default="stabilityai/stable-diffusion-3-medium-diffusers",
    )
    parser.add_argument("--sd_unet_path", type=str, default=None)
    parser.add_argument(
        "--lora_adapters_paths", type=str, action="append", default=[]
    )
    parser.add_argument("--conv_in_path", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)

    parser.add_argument("--num_train_iterations", type=int, default=500)
    parser.add_argument("--show_iter", type=int, default=50)
    parser.add_argument("--acc_step", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--power", type=float, default=0.1)
    parser.add_argument("--t_range_max", type=float, default=0.98)
    parser.add_argument("--save_dir", type=str, default="./sds/")
    parser.add_argument("--use_ratio", type=int, default=0)
    parser.add_argument("--fp16", type=int, default=1)
    return parser.parse_args()


def main():
    args = get_args()
    torch_device = "cuda"
    guidance = StableDiffusion3(
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

    torch.manual_seed(args.seed)

    if args.image_obj is not None:
        init_image = cv2.imread(args.image_obj)
        init_image = cv2.cvtColor(init_image, cv2.COLOR_BGR2RGB)
        init_image = center_crop_to_square(init_image)
        init_image = cv2.resize(init_image, (args.height, args.width))[
            None, ...
        ]
        with torch.no_grad():
            init_image_torch = guidance.np2torch(init_image)
            sds_image = guidance.torch2latents(init_image_torch)
    else:
        sds_image = torch.randn(
            (
                1,
                guidance.unet.config.in_channels,
                int(args.height) // guidance.vae_scale_factor,
                int(args.width) // guidance.vae_scale_factor,
            ),
            device=guidance.device,
        )

    sds_image = sds_image.detach().clone().requires_grad_(True)

    optimizer = torch.optim.SGD([sds_image], lr=args.lr)
    opt_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer,
        total_iters=args.num_train_iterations // args.acc_step,
        power=args.power,
    )

    if args.image_obj is None:
        rgb_obj_pred = None
    else:
        rgb_obj_pred = cv2.cvtColor(
            cv2.imread(args.image_obj), cv2.COLOR_BGR2RGB
        )
        rgb_obj_pred = cv2.resize(rgb_obj_pred, (args.width, args.height))
        rgb_obj_pred = guidance.torch2latents(guidance.np2torch(rgb_obj_pred))

    pos_embeds, pos_embeds_pooled = guidance.get_text_embeds(args.prompt)
    neg_embeds, neg_embeds_pooled = guidance.get_text_embeds("")
    text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0)
    text_embeds_pooled = torch.cat(
        [neg_embeds_pooled, pos_embeds_pooled], dim=0
    )

    args.save_dir = guidance.get_save_dir(args.save_dir)
    print("Save dir:", args.save_dir)

    guidance.save_images(
        images=guidance.torch2np(guidance.latents2torch(sds_image)),
        save_name="initial.jpg",
        prompt=args.prompt,
        save_dir=args.save_dir,
        exp_desc=args.exp_desc,
    )

    ratio = None

    for i in tqdm(range(args.num_train_iterations + 1)):
        # zero_grad
        if args.use_ratio != 0:
            ratio = min(1, i / args.num_train_iterations)

        optimizer.zero_grad()
        loss = guidance.train_step(
            text_embeds,
            text_embeds_pooled,
            sds_image,
            guidance_scale=args.guidance_scale,
            as_latent=True,
            step_ratio=ratio,
            pred_rgb_obj=rgb_obj_pred,
        )

        loss.backward()

        optimizer.step()
        opt_scheduler.step()

        if (i + 1) % args.show_iter == 0:
            result_image = guidance.torch2np(guidance.latents2torch(sds_image))
            guidance.save_images(
                images=result_image,
                save_name=f"sds_image_{i + 1}.jpg",
                prompt=args.prompt,
                save_dir=args.save_dir,
                exp_desc=args.exp_desc,
            )


if __name__ == "__main__":
    main()
