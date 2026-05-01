import argparse

import cv2
import numpy as np
import torch

from d3dr.diffusion.sd_utils import StableDiffusion

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default=None, required=True)
parser.add_argument("--exp_desc", type=str, default="")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--sd_model_name", type=str, default="Manojb/stable-diffusion-2-1-base"
)
parser.add_argument("--sd_unet_path", type=str, default=None)
parser.add_argument("--guidance_scale", type=float, default=7.5)
parser.add_argument("--num_inference_steps", type=int, default=25)
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--image_path", type=str, default=None)
parser.add_argument("--mask_path", type=str, default=None)

parser.add_argument("--num_same", type=int, default=4)
parser.add_argument("--save_dir", type=str, default="./diffusion_relighting")
parser.add_argument("--initial_step", type=float, default=0.0)
parser.add_argument(
    "--add_mean_init", action="store_true", help="initialize with mean"
)
parser.add_argument(
    "--lora_adapters_paths", type=str, action="append", default=[]
)
parser.add_argument("--conv_in_path", type=str, default=None)
parser.add_argument("--image_obj_path", type=str, required=True)
parser.add_argument("--fp16", type=int, default=1)
parser.add_argument("--random_light", type=int, default=0)
parser.add_argument("--steps_range_str", type=str, default="400 600")
parser.add_argument("--pure_noise", action="store_true")
parser.add_argument("--crop_ratio", type=float, default=1.0)
parser.add_argument("--crop_by_mask", action="store_true")

args = parser.parse_args()
args.steps_range = eval("(" + args.steps_range_str.replace(" ", ", ") + ")")
args.crop_ratio = min(args.crop_ratio, 1.0)

if args.image_path is None:
    args.image_path = args.image_obj_path

torch_device = "cuda"

guidance = StableDiffusion(
    device="cuda",
    sd_version=args.sd_model_name,
    height=args.height,
    width=args.width,
    sd_unet_path=args.sd_unet_path,
    lora_adapters_paths=None,
    lora_object_texture_path=args.lora_adapters_paths[0],
    fp16=args.fp16,
    conv_in_path=args.conv_in_path,
)
guidance.refining_phase()

torch.manual_seed(args.seed)
np.random.seed(args.seed)


def masked_mean(x, mask, axis=-1):
    return (x * mask).sum(axis=axis, keepdims=True) / mask.sum(
        axis=axis, keepdims=True
    )


# The functions to preprocess the images
def center_crop_to_square(cv2_image):
    h, w = cv2_image.shape[:2]
    if h > w:
        start = (h - w) // 2
        cv2_image = cv2_image[start : h - start, ...]
    elif w > h:
        start = (w - h) // 2
        cv2_image = cv2_image[:, start : w - start, ...]

    wh, _ = cv2_image.shape[:2]
    wh_small = int(wh * args.crop_ratio)
    wh_small = wh
    cv2_image = cv2_image[
        wh // 2 - wh_small // 2 : wh // 2 + wh_small // 2,
        wh // 2 - wh_small // 2 : wh // 2 + wh_small // 2,
        ...,
    ]
    return cv2_image


# # The functions to preprocess the images
# def center_crop_to_square(cv2_image):
#     h, w = cv2_image.shape[:2]
#     h_4 = h // 16
#     w_4 = w // 16
#     cv2_image = cv2_image[h//2-h_4 : h//2+h_4, w//2-w_4 : w//2+w_4, ...]
#     return cv2_image


def simple_crop_by_mask(image, mask, out_w, out_h, thresh=64):
    """Crop image to the largest mask bbox, then resize."""
    H, W = image.shape[:2]

    # Make sure mask matches image size and is binary uint8 (0/255)
    if mask.shape[:2] != (H, W):
        mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
    mask_u8 = ((mask > thresh) * 255).astype(np.uint8)

    # Find largest contour
    cnts = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = cnts[0] if len(cnts) == 2 else cnts[1]

    # If nothing found, just resize whole image/mask
    if not contours:
        img_out = cv2.resize(
            image, (out_w, out_h), interpolation=cv2.INTER_LINEAR
        )
        m_out = (
            cv2.resize(mask_u8, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
            > 0
        )
        return img_out, m_out.astype(np.uint8)

    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

    # Crop and resize (cv2.resize expects (width, height))
    s = min(max(w, h), min(W, H)) * args.crop_ratio
    s = int(s)
    x1 = max(0, min(W - s, x + w // 2 - s // 2))
    y1 = max(0, min(H - s, y + h // 2 - s // 2))
    crop_img = image[y1 : y1 + s, x1 : x1 + s]
    crop_mask = mask_u8[y1 : y1 + s, x1 : x1 + s]

    img_out = cv2.resize(
        crop_img, (out_w, out_h), interpolation=cv2.INTER_LINEAR
    )
    m_out = (
        cv2.resize(crop_mask, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
        > 0
    )
    return img_out, m_out.astype(np.uint8)


@torch.no_grad()
def read_image_mask(
    image_path,
    mask_path,
    random_light=False,
    apply_mask=False,
    add_mean_init=False,
    crop_by_mask=False,
):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = center_crop_to_square(image)
    # image = cv2.resize(image, (args.height, args.width))

    if mask_path is not None:
        init_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        init_mask = center_crop_to_square(init_mask)

        if crop_by_mask:
            image, init_mask = simple_crop_by_mask(
                image, init_mask, args.width, args.height
            )
        else:
            image = cv2.resize(image, (args.height, args.width))
            init_mask = (
                cv2.resize(init_mask, (args.height, args.width)) > 64
            ).astype(np.uint8)

        if add_mean_init:
            mask = init_mask[..., None]
            obj_mean = masked_mean(image, mask, axis=(0, 1))
            image = image * (1.0 - mask) + obj_mean * mask
            image = image.clip(0, 255).astype(np.uint8)

        if apply_mask:
            mask = init_mask[..., None]
            image = image * mask
            image = image.clip(0, 255).astype(np.uint8)

    else:
        init_mask = None

    if random_light != 0:
        contours, _ = cv2.findContours(
            init_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        N = 2
        rad_min = max(h, w) // 10
        rad_max = max(h, w) // 3
        alpha_min = 0.1
        alpha_max = 0.4

        for i in range(N):
            black_image = np.zeros_like(image)
            position = (
                np.random.randint(x, x + w),
                np.random.randint(y, y + h),
            )
            # position = (np.random.randint(image.shape[0]), np.random.randint(image.shape[1]))
            rad = np.random.randint(rad_min, rad_max)
            alpha = np.random.uniform(alpha_min, alpha_max)
            cv2.circle(black_image, position, rad, (255, 255, 255), -1)
            image = cv2.addWeighted(image, 1, black_image, alpha, 0)

    image = image[None, ...]
    # guidance.save_images(
    #     images=image,
    #     save_name="sdedit_light_raw.png",
    #     prompt=args.prompt,
    #     save_dir=args.save_dir,
    # )

    image_torch = guidance.np2torch(image)
    latent = guidance.torch2latents(image_torch)
    return image, image_torch, latent, init_mask


emb_uncond = guidance.get_text_embeds("")
emb_cond = guidance.get_text_embeds(args.prompt)

init_image, init_image_torch, init_latent, _ = read_image_mask(
    args.image_path,
    args.mask_path,
    random_light=args.random_light,
    add_mean_init=args.add_mean_init,
    crop_by_mask=args.crop_by_mask,
)
init_image_obj, init_image_obj_torch, init_latent_obj, _ = read_image_mask(
    args.image_obj_path,
    args.mask_path,
    random_light=0,
    apply_mask=False,
    crop_by_mask=args.crop_by_mask,
)
init_latent = init_latent.to(torch_device)
init_latent_obj = init_latent_obj.to(torch_device)

latents = guidance.perform_sdedit(
    init_latent,
    rgb_obj_pred=init_latent_obj,
    emb_cond=emb_cond,
    emb_uncond=emb_uncond,
    guidance_scale=args.guidance_scale,
    batch_size=args.num_same,
    timestep_range=args.steps_range,
    num_inference_steps=args.num_inference_steps,
    pure_noise=args.pure_noise,
)

images_torch = guidance.latents2torch(latents)
images_np = guidance.torch2np(images_torch)
guidance.save_images(
    images=images_np,
    save_name="sdedit_2.png",
    prompt=args.prompt,
    save_dir=args.save_dir,
)
