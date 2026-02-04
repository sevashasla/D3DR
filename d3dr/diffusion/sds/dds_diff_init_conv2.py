import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from d3dr.diffusion.sd_utils import StableDiffusion
from d3dr.diffusion.sds.dds_controlnet_diff_init import (
    center_crop_to_square,
    masked_mean,
    read_image,
)
from d3dr.diffusion.sds.dds_controlnet_diff_init import get_args as _get_args


def get_args(do_parse=True):
    parser = _get_args(do_parse=False)
    parser.add_argument("--image_obj_path", type=str, required=True)
    parser.add_argument("--conv_in_path", type=str, required=True)

    if do_parse:
        return parser.parse_args()
    return parser


def main():
    args = get_args()

    if (args.add_random_noise_mask + args.add_mean_init) > 1:
        raise ValueError(
            "Only one of add_random_noise_mask and add_mean_init can be set to True"
        )

    torch_device = "cuda"
    guidance = StableDiffusion(
        device="cuda",
        sd_version=args.sd_model_name,
        height=args.height,
        width=args.width,
        sd_unet_path=args.sd_unet_path,
        fp16=args.fp16,
        lora_object_texture_path=args.lora_adapters_paths[0],
        conv_in_path=args.conv_in_path,
        t_range=(0.02, args.t_range_max),
    )
    guidance.refining_phase()

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
    init_mask = center_crop_to_square(init_mask, crop_ratio=args.crop_ratio)
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

    init_image_obj, init_image_obj_torch, init_latent_obj = read_image(
        args.image_obj_path,
        args.height,
        args.width,
        guidance,
        mask,
        crop_ratio=args.crop_ratio,
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
            latents_initial=init_latent_nocomp,
            rgb_pred=sds_image,
            pred_rgb_obj=init_latent_obj,
            guidance_scale=args.guidance_scale,
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
