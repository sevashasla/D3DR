import cv2


import torch
from d3dr.diffusion.sd_utils import StableDiffusion

import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default=None, required=True)
parser.add_argument("--exp_desc", type=str, default="")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--sd_model_name", type=str, default="stabilityai/stable-diffusion-2-1-base")
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
parser.add_argument("--add_mean_init", action="store_true", help="initialize with mean")
parser.add_argument("--lora_adapters_paths", type=str, action="append", default=[])
parser.add_argument("--fp16", type=int, default=1)
parser.add_argument("--random_light", type=int, default=0)
parser.add_argument("--steps_range_str", type=str, default="400 600")
args = parser.parse_args()
args.steps_range = eval("(" + args.steps_range_str.replace(" ", ", ") + ")")


torch_device = "cuda"

guidance = StableDiffusion(
    device="cuda",
    sd_version=args.sd_model_name,
    height=args.height,
    width=args.width,
    sd_unet_path=args.sd_unet_path,
    lora_adapters_paths=args.lora_adapters_paths,
    fp16=args.fp16,
)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# The functions to preprocess the images
def center_crop_to_square(cv2_image):
    h, w = cv2_image.shape[:2]
    if h > w:
        start = (h - w) // 2
        return cv2_image[start: h - start, ...]
    elif w > h:
        start = (w - h) // 2
        return cv2_image[:, start: w - start, ...]
    else:
        return cv2_image


@torch.no_grad()
def read_image_mask(
    image_path,
    mask_path,
    random_light=False,
):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = center_crop_to_square(image)
    image = cv2.resize(image, (args.height, args.width))

    if not mask_path is None:
        init_mask = (cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE))
        init_mask = center_crop_to_square(init_mask)
        init_mask = (cv2.resize(init_mask, (args.height, args.width)) > 64).astype(np.uint8)
    else:
        init_mask = None

    if random_light != 0:
        contours, _ = cv2.findContours(init_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)

        N = 2
        rad_min = max(h, w) // 10
        rad_max = max(h, w) // 3
        alpha_min = 0.1
        alpha_max = 0.4

        for i in range(N):
            black_image = np.zeros_like(image)
            position = (np.random.randint(x, x + w), np.random.randint(y, y + h))
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

init_image, init_image_torch, init_latent, _ = read_image_mask(args.image_path, args.mask_path, random_light=args.random_light)
init_latent = init_latent.to(torch_device)

latents = guidance.perform_sdedit(
    init_latent, 
    emb_cond=emb_cond,
    emb_uncond=emb_uncond,
    guidance_scale=args.guidance_scale, 
    batch_size=args.num_same,
    timestep_range=args.steps_range,
    num_inference_steps=args.num_inference_steps,
)

images_torch = guidance.latents2torch(latents)
images_np = guidance.torch2np(images_torch)
guidance.save_images(
    images=images_np,
    save_name="sdedit_light.png",
    prompt=args.prompt,
    save_dir=args.save_dir,
)
