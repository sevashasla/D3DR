from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image as PILImage
from tqdm.auto import tqdm


def get_save_dir(args):
    """
    Defines where to save the current personalization experiment
    """
    if args.debug:
        idx = 0
        while True:
            curr_dir = os.path.join(str(args.save_dir), f"{idx:03}")
            if not os.path.exists(curr_dir):
                return curr_dir
            if args.skip_used:
                idx += 1
                continue
            # the directory exists
            all_files = os.listdir(curr_dir)

            if (
                "unet" in all_files
                or "pytorch_lora_weights.safetensors" in all_files
                or "conv_in.pth" in all_files
            ):
                idx += 1
                continue
            else:
                # this directory is useless
                return curr_dir
    else:
        return args.save_dir


@torch.no_grad()
def latents2np(latents, vae, device):
    with torch.no_grad():
        latents = 1 / vae.config.scaling_factor * latents
        images = vae.decode(latents).sample
    images = (images / 2 + 0.5).clamp(0, 1)
    images = (images.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
    return images


@torch.no_grad()
def np2latents(images, vae, device, weight_dtype):
    images = torch.tensor(images, device=device).to(weight_dtype)
    images = images.permute(0, 3, 1, 2) / 255
    images = 2 * images - 1
    latents = vae.encode(images).latent_dist.sample()
    latents = vae.config.scaling_factor * latents
    return latents


@torch.no_grad()
def torch2latents(images, vae, device):
    images = 2 * images - 1
    latents = vae.encode(images).latent_dist.sample()
    latents = vae.config.scaling_factor * latents
    return latents


@torch.no_grad()
def encode_text(text, text_encoder, tokenizer, device):
    prompt_input = tokenizer(
        text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    prompt_emb = text_encoder(prompt_input.input_ids.to(device))[0]
    return prompt_emb


def encode_text_grad(text, text_encoder, tokenizer, device):
    prompt_input = tokenizer(
        text,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        return_tensors="pt",
    )
    prompt_emb = text_encoder(prompt_input.input_ids.to(device))[0]
    return prompt_emb


def save_images(
    images,
    save_name: str | None = None,
    save_dir: str = None,
    prompt: str = None,
):
    """
    Optionally saves images at `save_name` if given.
    """
    if save_name is not None:
        save_path = Path(save_dir) / save_name

        all_files = list(Path(save_dir).glob(save_path.stem + "*"))
        save_path = save_path.with_name(
            save_path.stem + f"_{len(all_files):03}" + save_path.suffix
        )

        hor_num_sq = int(np.sqrt(len(images)))
        ver_num_sq = int(np.ceil(len(images) / hor_num_sq))

        plt.figure(figsize=(hor_num_sq * 5, ver_num_sq * 5))
        supt = prompt if prompt is not None else ""
        plt.suptitle(supt)

        for i, image in enumerate(images):
            plt.subplot(ver_num_sq, hor_num_sq, i + 1)
            image = PILImage.fromarray(image)
            plt.imshow(image)
            plt.axis("off")
        plt.subplots_adjust(wspace=0, hspace=0)
        print(f"savefig: {save_path}")
        plt.savefig(save_path)


@torch.no_grad()
def generate_image(
    prompt: str = None,
    emb=None,
    num_same: int = 1,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    height: int = 512,
    width: int = 512,
    save_name: str = None,
    save_dir: str = None,
    seed: int = 0,
    scheduler=None,
    tokenizer=None,
    text_encoder=None,
    vae=None,
    unet=None,
    device: str = None,
):
    """Generates image given `prompt` or text embedding `emb`"""

    unet_fp32 = deepcopy(unet).to(torch.float32)
    vae_fp32 = deepcopy(vae).to(torch.float32)
    text_encoder_fp32 = deepcopy(text_encoder).to(torch.float32)

    rng = torch.Generator(device="cuda")
    rng.manual_seed(seed)

    if emb is not None:
        text_embeddings = emb
        if num_same > 1:
            text_embeddings = torch.cat([text_embeddings] * num_same)
    else:
        prompts = [prompt] * num_same
        text_embeddings = encode_text(
            prompts, text_encoder_fp32, tokenizer, device
        )

    uncond_embeddings = encode_text(
        [""] * num_same, text_encoder_fp32, tokenizer, device
    )
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    latents = torch.randn(
        (num_same, unet_fp32.config.in_channels, height // 8, width // 8),
        device=device,
        generator=rng,
    )

    scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(
            latent_model_input, timestep=t
        )

        # predict the noise residual
        noise_pred = unet_fp32(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents).prev_sample

    # scale and decode the image latents with vae
    images = latents2np(latents, vae=vae_fp32, device=device)

    # save images
    if save_name is not None:
        print(f"save images to {save_name}")
        save_images(
            images, save_name=save_name, save_dir=save_dir, prompt=prompt
        )

    del unet_fp32, vae_fp32, text_encoder_fp32
    return latents, images


def cast_training_params(model, dtype=torch.float32):
    """
    Casts the training parameters of the model to the specified data type.

    Args:
        model: The PyTorch model whose parameters will be cast.
        dtype: The data type to which the model parameters will be cast.
    """
    if not isinstance(model, list):
        model = [model]
    for m in model:
        for param in m.parameters():
            # only upcast trainable parameters into fp32
            if param.requires_grad:
                param.data = param.to(dtype)


def _center_crop(image, size=512):
    h, w, _ = image.shape
    return image[
        h // 2 - size // 2 : h // 2 - size // 2 + size,
        w // 2 - size // 2 : w // 2 - size // 2 + size,
        :,
    ]


def get_random_images(image_dir, num_images=1, size=512):
    files = os.listdir(image_dir)
    files = np.random.choice(files, num_images)
    images = [cv2.imread(os.path.join(image_dir, f)) for f in files]
    images = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images]
    # images = [_center_crop(im, size) for im in images]
    images = [cv2.resize(im, (size, size)) for im in images]
    images = np.stack(images, axis=0)
    return images
