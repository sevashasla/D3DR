# The code is borrowed from https://github.com/ashawkey/stable-dreamfusion/blob/main/guidance/sd_utils.py
# Thanks ashawkey for the code!
# It is a little bit changed to fit the current project.

import os
from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image as PILImage
from tqdm import tqdm
from transformers import logging

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True


class ConvWImage(nn.Module):
    def __init__(self, conv_in):
        super().__init__()

        self.conv_in_real = conv_in
        self.conv_in_opt = nn.Conv2d(
            in_channels=self.conv_in_real.in_channels,  # should be 8
            out_channels=self.conv_in_real.out_channels,
            kernel_size=self.conv_in_real.kernel_size,
            stride=self.conv_in_real.stride,
            padding=self.conv_in_real.padding,
            dilation=self.conv_in_real.dilation,
        )
        with torch.no_grad():
            self.conv_in_opt.weight.zero_()
            self.conv_in_opt.bias.zero_()

        self.register_buffer(
            "conv_in_real_weight", conv_in.weight.clone(), persistent=False
        )
        self.register_buffer(
            "conv_in_real_bias", conv_in.bias.clone(), persistent=False
        )

        self.only_orig = False

    def activate_only_orig(self):
        self.only_orig = True

    def deactivate_only_orig(self):
        self.only_orig = False

    def forward(self, x):
        if self.only_orig:
            out1 = F.conv2d(
                x,
                weight=self.conv_in_real_weight,
                bias=self.conv_in_real_bias,
                stride=self.conv_in_real.stride,
                padding=self.conv_in_real.padding,
            )
            return out1

        x_noisy, x_obj = torch.chunk(x, chunks=2, dim=1)
        out1 = self.conv_in_real(x_noisy)
        out2 = self.conv_in_opt(x_obj)
        out = out1 + out2
        return out


class StableDiffusion(nn.Module):
    """
    numpy: [B, H, W, 3] from 0 to 255
    torch: [B, 3, H, W] from -1 to 1
    latents: [B, C, h, w] from -? to ?, already processed to the vae format!
    """

    def __init__(
        self,
        device: str = "cuda",
        fp16: bool = False,
        vram_O: bool = False,
        sd_version: str = "2.1",
        hf_key: str = None,
        t_range: tuple = (0.02, 0.98),
        height: int = 512,
        width: int = 512,
        sd_unet_path: str = None,
        lora_adapters_paths: Optional[List[str]] = None,
        lora_object_texture_path: Optional[str] = None,
        conv_in_path: str = None,
        text_encoder_path: str = None,
    ):
        """
        :param vram_O
            optimization for low VRAM usage
        :param sd_version
            stable diffusion version
        :param hf_key
            hugging face Stable diffusion model key
        :param t_range
            range of timesteps
        :param height, width
            image size
        :param sd_unet_path
            path to the unet model (optional)
        """

        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.height = height
        self.width = width
        self.lora_adapters_paths = (
            lora_adapters_paths
            if lora_adapters_paths is not None and len(lora_adapters_paths) > 0
            else None
        )
        self.lora_object_texture_path = lora_object_texture_path
        self.use_double_personalization = lora_object_texture_path is not None
        self.conv_in_path = conv_in_path

        print("[INFO] loading stable diffusion...")

        if hf_key is not None:
            print(f"[INFO] using hugging face custom model key: {hf_key}")
            model_key = hf_key
        elif (
            self.sd_version == "2.1"
            or self.sd_version == "Manojb/stable-diffusion-2-1-base"
        ):
            model_key = "Manojb/stable-diffusion-2-1-base"
        elif (
            self.sd_version == "2.0"
            or self.sd_version == "stabilityai/stable-diffusion-2-base"
        ):
            model_key = "stabilityai/stable-diffusion-2-base"
        elif (
            self.sd_version == "1.5"
            or self.sd_version == "runwayml/stable-diffusion-v1-5"
        ):
            model_key = "runwayml/stable-diffusion-v1-5"
        elif (
            self.sd_version == "1.4"
            or self.sd_version == "CompVis/stable-diffusion-v1-4"
        ):
            model_key = self.sd_version
        elif (
            self.sd_version == "sdxl"
            or self.sd_version == "stabilityai/stable-diffusion-xl-base-1.0"
        ):
            model_key = self.sd_version
        elif os.path.exists(self.sd_version):
            model_key = self.sd_version
        else:
            raise ValueError(
                f"Stable-diffusion version {self.sd_version} not supported."
            )

        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.precision_t
        )

        if vram_O:
            pipe.enable_sequential_cpu_offload()
            pipe.enable_vae_slicing()
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(1)
            # pipe.enable_model_cpu_offload()
        else:
            pipe.to(device)

        #
        self.vae = pipe.vae
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet
        self.pipe = pipe
        self.sd_unet_path = sd_unet_path
        self.text_encoder_path = text_encoder_path

        self.conv_in_loaded = False
        self.loaded_lighting_phase = False
        self.loaded_refining_phase = False
        self.lighting_phase()

        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.precision_t
        )
        self.inference_scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.precision_t
        )

        self.unet.requires_grad_(False)
        self.unet.eval()
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.vae.requires_grad_(False)
        self.vae.eval()

        self.unet_original_forward = self.unet.forward
        self.unet.forward = self.unet_call

        # del pipe # BUG: maybe bug here because of LoRA

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(
            self.device
        )  # for convenience

        print("[INFO] loaded stable diffusion!")

    def load_conv_in(self):
        # already loaded
        if self.conv_in_loaded:
            raise RuntimeError("Want to load conv in again!")

        # else load
        with torch.no_grad():
            self.unet.conv_in = ConvWImage(self.unet.conv_in)
            conv_in_state_dict = torch.load(
                self.conv_in_path, map_location=self.device, weights_only=True
            )
            self.unet.conv_in.load_state_dict(conv_in_state_dict)
            self.unet.conv_in = self.unet.conv_in.to(
                dtype=self.precision_t, device=self.device
            )
        self.conv_in_loaded = True
        print("Conv in Loaded!")

    @torch.no_grad()
    def lighting_phase(self):
        """
        Load all unet/loras for lighting phase
        (object & maybe scene)
        """
        if self.loaded_refining_phase:
            raise RuntimeError("Cant use lighting phase after refining")
        if self.loaded_lighting_phase:
            return

        # maybe load the unet (if use personalization)
        if self.sd_unet_path is not None:
            self.unet = self.unet.from_pretrained(
                self.sd_unet_path,
                torch_dtype=self.precision_t,
            ).to(self.device)
            print(f"Unet from {self.sd_unet_path} loaded!")

        if self.text_encoder_path is not None:
            self.text_encoder = self.text_encoder.from_pretrained(
                self.text_encoder_path,
                torch_dtype=self.precision_t,
            ).to(self.device)
            print(f"Text encoder from {self.text_encoder_path} loaded!")

        if self.lora_object_texture_path is not None:
            # 1. use loras
            # 2. use optimized texture preserver
            assert self.conv_in_path is not None, (
                "conv_in_path should not be None!"
            )
        else:
            # 1. use just loras => maybe need to load conv_in
            if self.conv_in_path is not None:
                self.load_conv_in()

        # load al loras
        if self.lora_adapters_paths is not None:
            print("load loras!")
            for lora_adapter_path in self.lora_adapters_paths:
                print(f"loaded {lora_adapter_path}")
                self.pipe.load_lora_weights(
                    lora_adapter_path, adapter_name=lora_adapter_path
                )
            self.unet.set_adapters(
                self.lora_adapters_paths,
                [1.0 / len(self.lora_adapters_paths)]
                * len(self.lora_adapters_paths),
            )

        self.loaded_lighting_phase = True
        self.loaded_refining_phase = False

    @torch.no_grad()
    def refining_phase(self):
        # already loaded
        if self.loaded_refining_phase:
            return

        # the strategy is not 2-step
        if self.lora_object_texture_path is None:
            return

        # delete all loras
        if self.lora_adapters_paths is not None:
            self.pipe.delete_adapters(self.lora_adapters_paths)

        # load object texture
        self.pipe.load_lora_weights(
            self.lora_object_texture_path,
            adapter_name=self.lora_object_texture_path,
        )
        self.unet.set_adapters(self.lora_object_texture_path, 1.0)
        self.load_conv_in()

        self.loaded_refining_phase = True
        self.loaded_lighting_phase = False

    def unet_call(self, sample, timestep, encoder_hidden_states, **kwargs):
        """
        this is a forward hook to deal with
        """
        if kwargs.get("only_orig", False):
            kwargs.pop("only_orig")
            if self.conv_in_loaded:
                self.unet.conv_in.activate_only_orig()
            self.pipe.disable_lora()
        else:
            if self.conv_in_loaded:
                self.unet.conv_in.deactivate_only_orig()
            self.pipe.enable_lora()

        return self.unet_original_forward(
            sample, timestep, encoder_hidden_states, **kwargs
        )

    def train_step(
        self,
        text_embeddings,
        pred_rgb,  # torch_images
        pred_rgb_obj=None,
        guidance_scale: float = 100.0,
        as_latent: bool = False,
        step_ratio: float = None,
        noise: torch.Tensor = None,
        return_grad: bool = False,
    ):
        """
        Please check the documentation for sd_utils_controlnet/train_step_dds if you need,
        the parameters and the logic is the same here
        """

        batch_size = pred_rgb.shape[0]
        assert text_embeddings.dtype == self.precision_t, (
            f"text_embeddings {text_embeddings.dtype} != {self.precision_t}"
        )
        pred_rgb = pred_rgb.to(self.precision_t)
        if noise is not None:
            noise = noise.to(self.precision_t)

        if as_latent:
            latents = pred_rgb
            latents_rgb_obj = pred_rgb_obj
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(
                pred_rgb,
                (self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )
            # encode image into latents with vae, requires grad!
            latents = self.torch2latents(pred_rgb_512)
            latents_rgb_obj = self.torch2latents_resize(pred_rgb_obj)

        latents_rgb_obj2 = self._maybe_repeat(latents_rgb_obj, 2)

        if step_ratio is None:
            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                (batch_size,),
                dtype=torch.long,
                device=self.device,
            )
        else:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            # sqrt looks better for 2d images
            t = np.round(
                np.sqrt(1 - step_ratio) * self.num_train_timesteps
            ).clip(self.min_step, self.max_step)
            t = torch.full(
                (batch_size,), t, dtype=torch.long, device=self.device
            )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            if noise is None:
                noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([t] * 2)

            # if use conv_in
            if latents_rgb_obj2 is not None:
                latent_model_input = torch.cat(
                    [latent_model_input, latents_rgb_obj2], dim=1
                )

            unet_output = self.unet(
                latent_model_input, tt, encoder_hidden_states=text_embeddings
            ).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_text = unet_output.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond
            )

        # w(t), sigma_t^2
        w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)

        grad = w * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        if return_grad:
            return grad

        # targets = (latents - grad).detach()
        # loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]

        # another way to compute the loss
        loss = (grad * latents).sum(axis=(1, 2, 3)).mean()

        return loss

    def _maybe_repeat(self, x, times=1):
        if x is None:
            return None
        else:
            return torch.cat([x] * times)

    def train_step_dds(
        self,
        text_embeddings_initial,  # "", "initial prompt"
        text_embeddings_desired,  # "", "desired prompt"
        latents_initial,  # torch_images
        rgb_pred,  # torch_images
        pred_rgb_obj=None,
        guidance_scale: float = 100.0,
        as_latent: bool = False,
        step_ratio: float = None,
        noise: torch.Tensor = None,
        use_weights=False,
        return_grad: bool = False,
    ):
        """
        Please check the documentation for sd_utils_controlnet/train_step_dds if you need,
        the parameters and the logic is the same here

        latents_initial: [B, 4, 64, 64]
        pred_rgb_obj: [B, 4, 64, 64]
        """

        batch_size = rgb_pred.shape[0]
        rgb_pred = rgb_pred.to(self.precision_t)
        assert text_embeddings_initial.dtype == self.precision_t, (
            f"text_embeddings_initial {text_embeddings_initial.dtype} != {self.precision_t}"
        )
        assert text_embeddings_desired.dtype == self.precision_t, (
            f"text_embeddings_desired {text_embeddings_desired.dtype} != {self.precision_t}"
        )
        assert latents_initial.dtype == self.precision_t, (
            f"latents_initial {latents_initial.dtype} != {self.precision_t}"
        )

        if noise is not None:
            noise = noise.to(self.precision_t)

        if as_latent:
            latents = rgb_pred
            latents_rgb_obj = pred_rgb_obj
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(
                rgb_pred,
                (self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )
            # encode image into latents with vae, requires grad!
            latents = self.torch2latents(pred_rgb_512)
            latents_rgb_obj = self.torch2latents_resize(pred_rgb_obj)

        latents_rgb_obj2 = self._maybe_repeat(latents_rgb_obj, 2)

        if step_ratio is None:
            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                (batch_size,),
                dtype=torch.long,
                device=self.device,
            )
        else:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            # sqrt looks better for 2d images
            t = np.round(
                np.sqrt(1 - step_ratio) * self.num_train_timesteps
            ).clip(self.min_step, self.max_step)
            t = torch.full(
                (batch_size,), t, dtype=torch.long, device=self.device
            )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # with torch.autocast(device_type="cuda", dtype=self.precision_t):
            # add noise
            if noise is None:
                noise = torch.randn_like(latents)
            noise2 = torch.cat([noise] * 2)
            t2 = torch.cat([t] * 2)

            latents_initial2 = torch.cat(
                [latents_initial, latents_initial], dim=0
            )
            latents2 = torch.cat([latents, latents], dim=0)
            latents_initial2_noisy = self.scheduler.add_noise(
                latents_initial2, noise2, t2
            )
            latents2_noisy = self.scheduler.add_noise(latents2, noise2, t2)

            # if use conv_in
            if latents_rgb_obj2 is not None:
                latents2_noisy = torch.cat(
                    [latents2_noisy, latents_rgb_obj2], dim=1
                )
            # pred noise
            unet_output_1 = self.unet(
                latents_initial2_noisy,
                t2,
                encoder_hidden_states=text_embeddings_initial,
                only_orig=True,
            ).sample
            unet_output_2 = self.unet(
                latents2_noisy,
                t2,
                encoder_hidden_states=text_embeddings_desired,
            ).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond_initial, noise_pred_text_initial = (
                unet_output_1.chunk(2)
            )
            noise_pred_uncond_desired, noise_pred_text_desired = (
                unet_output_2.chunk(2)
            )

            noise_pred_initial = noise_pred_uncond_initial + guidance_scale * (
                noise_pred_text_initial - noise_pred_uncond_initial
            )
            noise_pred_desired = noise_pred_uncond_desired + guidance_scale * (
                noise_pred_text_desired - noise_pred_uncond_desired
            )

        # w(t), sigma_t^2
        if use_weights:
            w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)
        else:
            w = 1.0

        grad = w * (noise_pred_desired - noise_pred_initial)
        grad = torch.nan_to_num(grad)

        if return_grad:
            return grad

        # another way to compute the loss
        loss = (grad * latents).sum(axis=(1, 2, 3)).mean()

        return loss

    def train_step_dds_2(
        self,
        text_embeddings_initial,  # "", "initial prompt"
        text_embeddings_desired,  # "", "desired prompt"
        latents_initial,  # torch_images
        rgb_pred,  # torch_images
        pred_rgb_obj=None,
        guidance_scale: float = 100.0,
        as_latent: bool = False,
        step_ratio: float = None,
        noise: torch.Tensor = None,
        use_weights=False,
        return_grad: bool = False,
    ):
        """
        Please check the documentation for sd_utils_controlnet/train_step_dds if you need,
        the parameters and the logic is the same here

        latents_initial: [B, 4, 64, 64]
        pred_rgb_obj: [B, 4, 64, 64]
        """

        batch_size = rgb_pred.shape[0]
        rgb_pred = rgb_pred.to(self.precision_t)
        assert text_embeddings_initial.dtype == self.precision_t, (
            f"text_embeddings_initial {text_embeddings_initial.dtype} != {self.precision_t}"
        )
        assert text_embeddings_desired.dtype == self.precision_t, (
            f"text_embeddings_desired {text_embeddings_desired.dtype} != {self.precision_t}"
        )
        assert latents_initial.dtype == self.precision_t, (
            f"latents_initial {latents_initial.dtype} != {self.precision_t}"
        )

        if noise is not None:
            noise = noise.to(self.precision_t)

        if as_latent:
            latents = rgb_pred
            latents_rgb_obj = pred_rgb_obj
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb_512 = F.interpolate(
                rgb_pred,
                (self.height, self.width),
                mode="bilinear",
                align_corners=False,
            )
            # encode image into latents with vae, requires grad!
            latents = self.torch2latents(pred_rgb_512)
            latents_rgb_obj = self.torch2latents_resize(pred_rgb_obj)

        ted_uncond, ted_cond = text_embeddings_desired.chunk(2)
        tei_uncond, tei_cond = text_embeddings_desired.chunk(2)

        te_uncond_obj = torch.cat([ted_uncond, tei_uncond, tei_cond])

        if step_ratio is None:
            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                (batch_size,),
                dtype=torch.long,
                device=self.device,
            )
        else:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            # sqrt looks better for 2d images
            t = np.round(
                np.sqrt(1 - step_ratio) * self.num_train_timesteps
            ).clip(self.min_step, self.max_step)
            t = torch.full(
                (batch_size,), t, dtype=torch.long, device=self.device
            )

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # with torch.autocast(device_type="cuda", dtype=self.precision_t):
            # add noise
            if noise is None:
                noise = torch.randn_like(latents)
            noise3 = torch.cat([noise] * 3)
            t3 = torch.cat([t] * 3)

            latents3_uncond_obj = torch.cat(
                [latents, latents_initial, latents_initial], dim=0
            )
            latents3_uncond_obj_noisy = self.scheduler.add_noise(
                latents3_uncond_obj, noise3, t3
            )
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # if use conv_in
            if latents_rgb_obj is not None:
                latents_noisy = torch.cat(
                    [latents_noisy, latents_rgb_obj], dim=1
                )
            # pred noise
            unet_output_1 = self.unet(
                latents3_uncond_obj_noisy,
                t3,
                encoder_hidden_states=te_uncond_obj,
                only_orig=True,
            ).sample
            unet_output_2 = self.unet(
                latents_noisy, t, encoder_hidden_states=ted_cond
            ).sample

            # perform guidance (high scale from paper!)
            (
                noise_pred_uncond_desired,
                noise_pred_uncond_initial,
                noise_pred_text_initial,
            ) = unet_output_1.chunk(3)
            noise_pred_text_desired = unet_output_2

            noise_pred_initial = noise_pred_uncond_initial + guidance_scale * (
                noise_pred_text_initial - noise_pred_uncond_initial
            )
            noise_pred_desired = noise_pred_uncond_desired + guidance_scale * (
                noise_pred_text_desired - noise_pred_uncond_desired
            )

        # w(t), sigma_t^2
        if use_weights:
            w = (1 - self.alphas[t]).view(batch_size, 1, 1, 1)
        else:
            w = 1.0

        grad = w * (noise_pred_desired - noise_pred_initial)
        grad = torch.nan_to_num(grad)

        if return_grad:
            return grad

        # another way to compute the loss
        loss = (grad * latents).sum(axis=(1, 2, 3)).mean()

        return loss

    @staticmethod
    def get_save_dir(save_dir):
        """
        Method useful to get the next save directory.
        """
        os.makedirs(save_dir, exist_ok=True)
        all_dirs = list(Path(save_dir).glob("*"))
        all_dirs = sorted([d for d in all_dirs if d.is_dir()])
        for d in all_dirs[:-1]:
            if len(list(d.glob("*"))) == 0:
                return str(d)

        curr_idx = len(all_dirs) + 1
        while True:
            save_dir = Path(save_dir) / f"{curr_idx:03}"
            if save_dir.exists():
                curr_idx += 1
            else:
                break
        save_dir = str(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    @staticmethod
    @torch.no_grad()
    def save_images(
        images,
        save_name: str = "out.png",
        prompt: str = None,
        save_dir: str = "./",
        exp_desc: str = "",
        only_orig=False,
    ):
        save_path = Path(save_dir) / save_name

        all_files = list(Path(save_dir).glob(save_path.stem + "*"))
        save_path = save_path.with_name(
            save_path.stem + f"_{len(all_files):03}" + save_path.suffix
        )

        if not (only_orig and (len(images) == 1)):
            hor_num_sq = int(np.sqrt(len(images)))
            ver_num_sq = int(np.ceil(len(images) / hor_num_sq))

            plt.figure(figsize=(hor_num_sq * 5, ver_num_sq * 5))
            supt = (
                prompt if prompt is not None else f"Learned prompt for {prompt}"
            )
            supt = f"{supt} {exp_desc}"
            plt.suptitle(supt)

            for i, image in enumerate(images):
                plt.subplot(ver_num_sq, hor_num_sq, i + 1)
                plt.imshow(image)
                plt.axis("off")
            plt.subplots_adjust(wspace=0, hspace=0)
            plt.savefig(save_path)

        if len(images) == 1:
            save_path = save_path.with_name(
                save_path.stem + "_orig" + save_path.suffix
            )
            PILImage.fromarray(images[0]).save(save_path)

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        embeddings = embeddings.to(self.precision_t)
        return embeddings

    @torch.no_grad()
    def perform_sdedit(
        self,
        image,
        emb_uncond,
        emb_cond,
        rgb_obj_pred=None,
        guidance_scale: float = 7.5,
        as_latent=True,
        timestep_range=(400, 600),
        t_step=None,
        num_inference_steps=25,
        batch_size=1,
        pure_noise=False,
    ):
        image = image.to(self.precision_t)

        if as_latent:
            latents = image
            latents_rgb_obj = rgb_obj_pred
        else:
            latents = self.torch2latents_resize(image)
            latents_rgb_obj = self.torch2latents_resize(rgb_obj_pred)

        if latents.shape[0] != batch_size:
            latents = latents.repeat(batch_size, 1, 1, 1)

        latents_rgb_obj2 = self._maybe_repeat(latents_rgb_obj, 2)
        if (
            latents_rgb_obj2 is not None
            and latents_rgb_obj2.shape[0] != batch_size * 2
        ):
            latents_rgb_obj2 = self._maybe_repeat(latents_rgb_obj2, batch_size)

        self.inference_scheduler.set_timesteps(num_inference_steps)
        inference_timesteps = self.inference_scheduler.timesteps.detach().to(
            self.device
        )

        if not pure_noise:
            # equal timesteps
            if t_step is None:
                t_add_noise = torch.randint(
                    timestep_range[0],
                    timestep_range[1] + 1,
                    (1,),
                    dtype=torch.long,
                    device=self.device,
                ).repeat(batch_size)
            else:
                t_add_noise = torch.tensor(
                    [t_step] * batch_size, dtype=torch.long, device=self.device
                )
            noise_added = torch.randn_like(latents)

            latents = self.inference_scheduler.add_noise(
                latents, noise_added, t_add_noise
            )
            # latents = noise_added
            start_step = inference_timesteps[
                (inference_timesteps > t_add_noise[0])
            ].argmin()
            start_step = max(start_step - 1, 0)
        else:
            latents = torch.randn_like(latents)
            start_step = 0

        text_embeddings = torch.cat(
            [emb_uncond] * batch_size + [emb_cond] * batch_size, dim=0
        )

        # for i, t in tqdm(enumerate(inference_timesteps[start_step:]), total=len(inference_timesteps[start_step:])):
        for i, t in enumerate(inference_timesteps[start_step:]):
            latent2 = torch.cat([latents] * 2)
            t2 = torch.cat([t.repeat(batch_size)] * 2)
            text_embeddings2 = text_embeddings  # it is already repeated

            # if use conv_in
            if latents_rgb_obj2 is not None:
                latent2 = torch.cat([latent2, latents_rgb_obj2], dim=1)

            noise_pred = self.unet(
                latent2,
                t2,
                encoder_hidden_states=text_embeddings2,
            ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.inference_scheduler.step(noise_pred, t, latents)[
                "prev_sample"
            ]

        return latents

    @torch.no_grad()
    def perform_sdedit_2(
        self,
        image,
        image_obj,
        emb_uncond,
        emb_cond,
        guidance_scale: float = 7.5,
        as_latent=True,
        timestep_range=(400, 600),
        num_inference_steps=25,
        batch_size=1,
    ):
        image = image.to(self.precision_t)

        if as_latent:
            latents = image
            latents_obj = image_obj
        else:
            latents = self.torch2latents_resize(image)
            latents_obj = self.torch2latents_resize(image_obj)

        if latents.shape[0] != batch_size:
            latents = latents.repeat(batch_size, 1, 1, 1)
            latents_obj = latents_obj.repeat(batch_size, 1, 1, 1)

        self.inference_scheduler.set_timesteps(num_inference_steps)
        # equal timesteps
        t_add_noise = torch.randint(
            timestep_range[0],
            timestep_range[1] + 1,
            (1,),
            dtype=torch.long,
            device=self.device,
        ).repeat(batch_size)
        noise_added = torch.randn_like(latents)

        latents = self.inference_scheduler.add_noise(
            latents, noise_added, t_add_noise
        )
        # latents = noise_added
        inference_timesteps = self.inference_scheduler.timesteps.detach().to(
            self.device
        )
        start_step = inference_timesteps[
            (inference_timesteps > t_add_noise[0])
        ].argmin()
        start_step = max(start_step - 1, 0)

        text_embeddings = torch.cat(
            [emb_uncond] * batch_size + [emb_cond] * batch_size, dim=0
        )

        latents_obj2 = torch.cat([latents_obj] * 2)
        for i, t in tqdm(
            enumerate(inference_timesteps[start_step:]),
            total=len(inference_timesteps[start_step:]),
        ):
            latent2 = torch.cat([latents] * 2)
            latent_in = torch.cat([latent2, latents_obj2], dim=1)
            t2 = torch.cat([t.repeat(batch_size)] * 2)
            text_embeddings2 = text_embeddings  # it is already repeated

            noise_pred = self.unet(
                latent_in,
                t2,
                encoder_hidden_states=text_embeddings2,
            ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.inference_scheduler.step(noise_pred, t, latents)[
                "prev_sample"
            ]

        return latents

    @torch.no_grad()
    def perform_inpainting(
        self,
        rgb_pred,
        emb_cond,
        emb_uncond,
        rgb_obj_pred=None,
        guidance_scale: float = 7.5,
        as_latent=True,
        batch_size=9,
        timestep_range=(900, 950),
        num_inference_steps=25,
        mask_small=None,
        latents_nocomp=None,
    ):
        rgb_pred = rgb_pred.to(self.precision_t)
        assert emb_cond.dtype == self.precision_t, (
            f"emb_cond {emb_cond.dtype} != {self.precision_t}"
        )
        assert emb_uncond.dtype == self.precision_t, (
            f"emb_uncond {emb_uncond.dtype} != {self.precision_t}"
        )
        assert mask_small.dtype == self.precision_t, (
            f"mask_small {mask_small.dtype} != {self.precision_t}"
        )

        if as_latent:
            latents = rgb_pred
            latents_rgb_obj = rgb_obj_pred
        else:
            latents = self.torch2latents_resize(rgb_pred)
            latents_rgb_obj = self.torch2latents_resize(rgb_obj_pred)

        if latents_nocomp is None:
            latents_nocomp = latents.clone()

        if latents.shape[0] != batch_size:
            latents = latents.repeat(batch_size, 1, 1, 1)

        if latents_nocomp.shape[0] != batch_size:
            latents_nocomp = latents_nocomp.repeat(batch_size, 1, 1, 1)

        latents_rgb_obj2 = self._maybe_repeat(latents_rgb_obj, 2)
        if (
            latents_rgb_obj2 is not None
            and latents_rgb_obj2.shape[0] != batch_size * 2
        ):
            latents_rgb_obj2 = self._maybe_repeat(latents_rgb_obj2, batch_size)

        self.inference_scheduler.set_timesteps(num_inference_steps)
        # equal timesteps
        t_add_noise = torch.randint(
            timestep_range[0],
            timestep_range[1] + 1,
            (1,),
            dtype=torch.long,
            device=self.device,
        ).repeat(batch_size)
        noise_added = torch.randn_like(latents)
        latents = self.inference_scheduler.add_noise(
            latents, noise_added, t_add_noise
        )
        inference_timesteps = self.inference_scheduler.timesteps.detach().to(
            self.device
        )
        start_step = inference_timesteps[
            (inference_timesteps > t_add_noise[0])
        ].argmin()
        text_embeddings2 = torch.cat(
            [emb_uncond] * batch_size + [emb_cond] * batch_size, dim=0
        )

        for i, t in tqdm(
            enumerate(inference_timesteps[start_step:]),
            total=len(inference_timesteps[start_step:]),
        ):
            latent2 = torch.cat([latents] * 2)
            t2 = torch.cat([t.repeat(batch_size)] * 2)
            text_embeddings2 = text_embeddings2

            if latents_rgb_obj2 is not None:
                latent2 = torch.cat([latent2, latents_rgb_obj2], axis=1)

            noise_pred = self.unet(
                latent2,
                t2,
                encoder_hidden_states=text_embeddings2,
            ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.inference_scheduler.step(noise_pred, t, latents)[
                "prev_sample"
            ]

            j = start_step + i
            if mask_small is not None:
                if j + 1 < len(inference_timesteps):
                    next_timestep = inference_timesteps[j + 1].repeat(
                        batch_size
                    )
                    rgb_pred_noisy = self.inference_scheduler.add_noise(
                        latents_nocomp, noise_added, next_timestep
                    )
                    latents = latents * mask_small + rgb_pred_noisy * (
                        1 - mask_small
                    )
                else:
                    latents = latents * mask_small + latents_nocomp * (
                        1 - mask_small
                    )
        # latents = latents * mask_small + rgb_pred * (1 - mask_small)
        return latents

    @torch.no_grad()
    def generate_images_latents_by_embeds(
        self,
        text_embeddings,
        num_inference_steps=25,
        guidance_scale=7.5,
        latents=None,
        start_step=0,
        seed=0,
    ):

        torch.manual_seed(seed)
        if latents is None:
            latents = torch.randn(
                (
                    text_embeddings.shape[0] // 2,
                    self.unet.in_channels,
                    self.height // 8,
                    self.width // 8,
                ),
                device=self.device,
                dtype=self.precision_t,
            )
        text_embeddings = text_embeddings.to(self.precision_t)

        self.scheduler.set_timesteps(num_inference_steps)

        for i, t in tqdm(enumerate(self.scheduler.timesteps[start_step:])):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input, t, encoder_hidden_states=text_embeddings
            )["sample"]

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents)["prev_sample"]
        return latents

    @torch.no_grad()
    def generate_images_by_prompts(
        self,
        prompts,
        negative_prompts="",
        num_same=1,
        num_inference_steps=25,
        guidance_scale=7.5,
        seed=0,
        latents=None,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if len(prompts) == 1:
            prompts = prompts * num_same
        if len(prompts) != num_same:
            raise RuntimeError(
                "Number of prompts should be 1 or equal to num_same"
            )

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts] * len(prompts)

        # Prompts -> text embeds
        pos_embeds = self.get_text_embeds(prompts)  # [1, 77, 768]
        neg_embeds = self.get_text_embeds(negative_prompts)
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0)  # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.generate_images_latents_by_embeds(
            text_embeds,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            latents=latents,
            seed=seed,
        )  # [1, 4, 64, 64]

        # Img latents -> imgs
        torch_images = self.latents2torch(latents)  # [1, 3, 512, 512]
        np_images = self.torch2np(torch_images)

        return np_images

    def np2torch(self, np_images):
        if np_images is None:
            return None
        torch_images = torch.tensor(np_images, device=self.device).to(
            self.precision_t
        )
        if torch_images.ndim == 3:
            torch_images = torch_images.unsqueeze(0)
        images = torch_images.permute(0, 3, 1, 2) / 255
        images = 2 * images - 1
        return images

    def torch2np(self, torch_images):
        if torch_images is None:
            return None
        torch_images = (torch_images / 2 + 0.5).clamp(0, 1)
        np_images = (
            (torch_images.permute(0, 2, 3, 1) * 255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )
        return np_images

    def latents2torch(self, latents):
        if latents is None:
            return None
        latents = latents.to(self.precision_t)
        latents = 1 / self.vae.config.scaling_factor * latents
        torch_images = self.vae.decode(latents).sample
        return torch_images

    def torch2latents(self, torch_images):
        if torch_images is None:
            return None
        torch_images = torch_images.to(self.precision_t)
        posterior = self.vae.encode(torch_images).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents

    def torch2latents_resize(self, torch_images):
        if torch_images is None:
            return None
        torch_images = torch_images.to(self.precision_t)
        torch_images_512 = F.interpolate(
            torch_images,
            (self.height, self.width),
            mode="bilinear",
            align_corners=False,
        )
        latents = self.torch2latents(torch_images_512)
        return latents


if __name__ == "__main__":
    import argparse

    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", type=str)
    parser.add_argument("--negative", default="", type=str)
    parser.add_argument(
        "--sd_version",
        type=str,
        default="2.1",
        choices=["1.5", "2.0", "2.1"],
        help="stable diffusion version",
    )
    parser.add_argument(
        "--hf_key",
        type=str,
        default=None,
        help="hugging face Stable diffusion model key",
    )
    parser.add_argument(
        "--fp16", action="store_true", help="use float16 for training"
    )
    parser.add_argument(
        "--vram_O", action="store_true", help="optimization for low VRAM usage"
    )
    parser.add_argument("-H", type=int, default=512)
    parser.add_argument("-W", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device("cuda")

    sd = StableDiffusion(
        device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key
    )

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
