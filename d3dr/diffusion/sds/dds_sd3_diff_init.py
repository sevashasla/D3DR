"""
python3 d3dr/diffusion/sds/dds_sd3_diff_init.py \
    --prompt_1 "a cup on a plate" \
    --prompt_2 "a plate" \
    --image_path "./data/cups/no_cup_l1.jpg" \
    --image_comp_path "./data/cups/cupl2l1.jpg" \
    --mask_path "./data/cups/mask_composition.png" \
    --num_train_iteration 200 \
    --show_iter 25 \
    --use_ratio 1 \
    --add_mean_init 0
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from d3dr.diffusion.sd3_utils import StableDiffusion3
from d3dr.diffusion.sds.dds_controlnet_diff_init import read_image
from d3dr.diffusion.sds.dds_sd3 import center_crop_to_square
from d3dr.diffusion.sds.dds_sd3 import get_args as _get_args


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


def masked_mean(x, mask, dim=None):
    if dim is None:
        dim = tuple(range(0, len(x.shape)))
    return torch.sum(x * mask, dim=dim) / torch.sum(mask, dim=dim)


def get_args():
    parser = _get_args(should_parse=False)

    parser.add_argument("--image_comp_path", type=str, required=True)
    parser.add_argument(
        "--add_random_noise_mask",
        type=int,
        default=0,
        help="initialize with random noise",
    )
    parser.add_argument(
        "--add_mean_init", type=int, default=1, help="initialize with mean"
    )

    args = parser.parse_args()
    return args


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

    # Load images
    _, init_image_torch_nocomp, init_latent_nocomp = read_image(
        args.image_path, args.height, args.width, guidance
    )
    _, init_image_torch_comp, init_latent_comp = read_image(
        args.image_comp_path, args.height, args.width, guidance
    )

    # Load mask
    init_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
    init_mask = center_crop_to_square(init_mask)
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

    rgb_obj_pred = None
    torch.manual_seed(args.seed)

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

    for i in tqdm(range(args.num_train_iterations)):
        # zero_grad
        optimizer.zero_grad()

        if args.use_ratio != 0:
            ratio = min(1, i / args.num_train_iterations)

        loss = guidance.train_step_dds(
            text_embeddings_initial=text_embeddings_initial,
            text_embeddings_initial_pooled=text_embeddings_initial_pooled,
            text_embeddings_desired=text_embeddings_desired,
            text_embeddings_desired_pooled=text_embeddings_desired_pooled,
            latents_initial=init_latent_nocomp,
            pred_rgb_obj=rgb_obj_pred,
            rgb_pred=sds_image,
            guidance_scale_tgt=args.guidance_scale,
            guidance_scale_src=args.guidance_scale_src,
            step_ratio=ratio,
            as_latent=True,
            eta=0.0,
        )

        loss.backward()

        if args.mask_grad != 0 and mask_small is not None:
            sds_image.grad = sds_image.grad * mask_small

        current_lr = learning_rate_fn(i, args.num_train_iterations, args.lr)
        optimizer.param_groups[0]["lr"] = current_lr

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
