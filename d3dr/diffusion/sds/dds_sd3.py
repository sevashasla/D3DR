import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from d3dr.diffusion.sd3_utils import StableDiffusion3
from d3dr.diffusion.sds.sds_sd3 import center_crop_to_square


def get_args(should_parse=True):
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
        default="stabilityai/stable-diffusion-3-medium-diffusers",
    )
    parser.add_argument("--sd_unet_path", type=str, default=None)
    parser.add_argument(
        "--lora_adapters_paths", type=str, action="append", default=[]
    )
    parser.add_argument("--conv_in_path", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=15.0)
    parser.add_argument("--guidance_scale_src", type=float, default=6.0)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)

    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--image_obj", type=str, default=None)

    parser.add_argument("--num_train_iterations", type=int, default=400)
    parser.add_argument("--show_iter", type=int, default=50)
    parser.add_argument("--acc_step", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--power", type=float, default=0.0)
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./dds/",
    )

    parser.add_argument("--mask_grad", type=int, default=1)
    parser.add_argument("--initial_step", type=float, default=0.0)
    parser.add_argument("--fp16", type=int, default=1)
    parser.add_argument("--t_range_max", type=float, default=0.98)
    parser.add_argument("--use_ratio", type=int, default=0)
    parser.add_argument("--use_eta", type=int, default=0)

    if should_parse:
        args = parser.parse_args()
        return args

    return parser


# very hardcoded fn
def learning_rate_fn(t: int, max_step: int, max_lr: float):
    # 0.0 -> 0.6: linear (0 -> max_lr * 0.25)
    # 0.6 -> 0.8: linear (max_lr * 0.25 -> max_lr)
    # 0.8 -> 1.0: linear (max_lr -> max_lr * 0.75)

    if 0 <= t < max_step * 0.6:
        return (t / (max_step * 0.6)) * (max_lr * 0.25)

    elif max_step * 0.6 <= t < max_step * 0.8:
        progress = (t - max_step * 0.6) / (max_step * 0.2)
        return (max_lr * 0.25) + progress * (max_lr * 0.75)

    else:
        progress = (t - max_step * 0.8) / (max_step * 0.2)
        return max_lr - progress * (max_lr * 0.25)


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

    init_image = cv2.imread(args.image_path)
    init_image = cv2.cvtColor(init_image, cv2.COLOR_BGR2RGB)
    init_image = center_crop_to_square(init_image)
    init_image = cv2.resize(init_image, (args.height, args.width))[None, ...]
    with torch.no_grad():
        init_image_torch = guidance.np2torch(init_image)
        init_latent = guidance.torch2latents(init_image_torch)

    if args.mask_path is not None:
        init_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        init_mask = center_crop_to_square(init_mask)
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

    # rgb_obj_pred = None

    torch.manual_seed(args.seed)

    sds_image = init_latent.detach().clone().requires_grad_(True)

    optimizer = torch.optim.SGD([sds_image], lr=args.lr)

    args.save_dir = guidance.get_save_dir(args.save_dir)
    print("Save dir:", args.save_dir)

    with open(Path(args.save_dir) / "args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

    emb_desired, emb_desired_pooled = guidance.get_text_embeds(args.prompt_1)
    emb_initial, emb_initial_pooled = guidance.get_text_embeds(args.prompt_2)
    emb_uncond, emb_uncond_pooled = guidance.get_text_embeds("")

    text_embeddings_desired = torch.cat([emb_uncond, emb_desired])
    text_embeddings_desired_pooled = torch.cat(
        [emb_uncond_pooled, emb_desired_pooled]
    )
    text_embeddings_initial = torch.cat([emb_uncond, emb_initial])
    text_embeddings_initial_pooled = torch.cat(
        [emb_uncond_pooled, emb_initial_pooled]
    )

    ratio = None
    eta_i = 0.0
    for i in tqdm(range(args.num_train_iterations)):
        # zero_grad
        optimizer.zero_grad()

        if args.use_ratio != 0:
            ratio = min(1, i / args.num_train_iterations)

        if args.use_eta != 0:
            eta_i = min(1, i / args.num_train_iterations)

        loss = guidance.train_step_dds(
            text_embeddings_initial=text_embeddings_initial,
            text_embeddings_initial_pooled=text_embeddings_initial_pooled,
            text_embeddings_desired=text_embeddings_desired,
            text_embeddings_desired_pooled=text_embeddings_desired_pooled,
            latents_initial=init_latent,
            pred_rgb_obj=rgb_obj_pred,
            rgb_pred=sds_image,
            guidance_scale_tgt=args.guidance_scale,
            guidance_scale_src=args.guidance_scale_src,
            step_ratio=ratio,
            as_latent=True,
            eta=eta_i,
        )

        loss.backward()

        if args.mask_grad != 0 and mask_small is not None:
            sds_image.grad = sds_image.grad * mask_small

        current_lr = learning_rate_fn(i, args.num_train_iterations, args.lr)
        optimizer.param_groups[0]["lr"] = current_lr

        optimizer.step()

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
