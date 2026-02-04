# The code is borrowed from https://github.com/ashawkey/stable-dreamfusion/blob/main/guidance/sd_utils.py
# Thanks ashawkey for the code!
# It is a little bit changed to fit the current project.

from transformers import CLIPTextModel, CLIPTokenizer, logging
from diffusers import (
    FlowMatchEulerDiscreteScheduler, 
    StableDiffusion3Pipeline
)


import os
import numpy as np
from pathlib import Path
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from tqdm import tqdm
import itertools
from typing import List, Optional

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from torch.cuda.amp import custom_bwd, custom_fwd

from d3dr.diffusion.sd_utils import (
    StableDiffusion,
    ConvWImage,
    seed_everything
)


class StableDiffusion3(nn.Module):
    '''
    numpy: [B, H, W, 3] from 0 to 255
    torch: [B, 3, H, W] from -1 to 1
    latents: [B, C, h, w] from -? to ?, already processed to the vae format!
    '''
    def __init__(
        self, 
        device: str = 'cuda', 
        fp16: bool = False, 
        sd_version: str = '3.0', 
        hf_key: str = None, 
        t_range: tuple = (0.02, 0.98), 
        height: int = 512, width: int = 512,
        sd_unet_path: str = None,
        lora_adapters_paths: Optional[List[str]] = None,
        lora_object_texture_path: Optional[str] = None,
        conv_in_path: str = None,
        text_encoder_path: str = None,
    ):
        '''
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
        '''

        super().__init__()

        self.device = device
        self.sd_version = sd_version
        self.height = height
        self.width = width
        self.lora_adapters_paths = lora_adapters_paths if not lora_adapters_paths is None and len(lora_adapters_paths) > 0 else None
        self.lora_object_texture_path = lora_object_texture_path
        self.use_double_personalization = (not lora_object_texture_path is None)
        self.conv_in_path = conv_in_path

        print(f'[INFO] loading stable diffusion...')

        if hf_key is not None:
            print(f'[INFO] using hugging face custom model key: {hf_key}')
            model_key = hf_key
        elif self.sd_version == '3.0' or self.sd_version == "stabilityai/stable-diffusion-3-medium-diffusers":
            model_key = self.sd_version
        elif self.sd_version == '3.5' or self.sd_version == "stabilityai/stable-diffusion-3.5-medium":
            model_key = self.sd_version
        else:
            raise ValueError(f'Stable-diffusion version {self.sd_version} not supported.')

        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableDiffusion3Pipeline.from_pretrained(model_key, torch_dtype=self.precision_t)
        pipe.to(device)

        self.vae = pipe.vae
        self.vae_scale_factor = (
            2 ** (len(self.vae.config.block_out_channels) - 1)
        )

        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.tokenizer_3 = pipe.tokenizer_3
        self.tokenizer_max_length = self.tokenizer.model_max_length

        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2
        self.text_encoder_3 = pipe.text_encoder_3
        
        self.unet = pipe.transformer

        self.pipe = pipe
        self.sd_unet_path = sd_unet_path
        self.text_encoder_path = text_encoder_path

        self.conv_in_loaded = False
        self.loaded_lighting_phase = False
        self.loaded_refining_phase = False
        self.lighting_phase()

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.precision_t)
        self.inference_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(model_key, subfolder="scheduler", torch_dtype=self.precision_t)
        
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])

        self.unet.requires_grad_(False)
        self.unet.eval()
        self.text_encoder.requires_grad_(False)
        self.text_encoder.eval()
        self.vae.requires_grad_(False)
        self.vae.eval()

        # self.unet_original_forward = self.unet.forward
        # self.unet.forward = self.unet_call

        # del pipe # BUG: maybe bug here because of LoRA
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps

        print(f'[INFO] loaded stable diffusion!')

    def load_conv_in(self):
        # already loaded
        if self.conv_in_loaded:
            raise RuntimeError("Want to load conv in again!")

        # else load
        with torch.no_grad():
            self.unet.conv_in = ConvWImage(self.unet.conv_in)
            conv_in_state_dict = torch.load(self.conv_in_path, map_location=self.device, weights_only=True)
            self.unet.conv_in.load_state_dict(conv_in_state_dict)
            self.unet.conv_in = self.unet.conv_in.to(dtype=self.precision_t, device=self.device)
        self.conv_in_loaded = True
        print("Conv in Loaded!")

    @torch.no_grad()
    def lighting_phase(self):
        '''
        Load all unet/loras for lighting phase
        (object & maybe scene)
        '''
        if self.loaded_refining_phase:
            raise RuntimeError("Cant use lighting phase after refining")
        if self.loaded_lighting_phase:
            return
        
        # maybe load the unet (if use personalization)
        if not self.sd_unet_path is None:
            self.unet = self.unet.from_pretrained(
                self.sd_unet_path, torch_dtype=self.precision_t,
            ).to(self.device)
            print(f"Unet from {self.sd_unet_path} loaded!")

        if not self.text_encoder_path is None:
            self.text_encoder = self.text_encoder.from_pretrained(
                self.text_encoder_path, torch_dtype=self.precision_t,
            ).to(self.device)
            print(f"Text encoder from {self.text_encoder_path} loaded!")

        if not self.lora_object_texture_path is None:
            # 1. use loras
            # 2. use optimized texture preserver
            assert not self.conv_in_path is None, "conv_in_path should not be None!"
        else:
            # 1. use just loras => maybe need to load conv_in
            if not self.conv_in_path is None:
                self.load_conv_in()

        # load al loras
        if not self.lora_adapters_paths is None:
            print("load loras!")
            for lora_adapter_path in self.lora_adapters_paths:
                print(f"loaded {lora_adapter_path}")
                self.pipe.load_lora_weights(lora_adapter_path, adapter_name=lora_adapter_path)
            self.unet.set_adapters(self.lora_adapters_paths, [1.0/len(self.lora_adapters_paths)] * len(self.lora_adapters_paths))
        
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
        if not self.lora_adapters_paths is None:
            self.pipe.delete_adapters(self.lora_adapters_paths)

        # load object texture
        self.pipe.load_lora_weights(self.lora_object_texture_path, adapter_name=self.lora_object_texture_path)
        self.unet.set_adapters(self.lora_object_texture_path, 1.0)
        self.load_conv_in()

        self.loaded_refining_phase = True
        self.loaded_lighting_phase = False

    def unet_call(self, sample, timestep, encoder_hidden_states, **kwargs):
        '''
        this is a forward hook to deal with        
        '''
        if kwargs.get("only_orig", False):
            kwargs.pop("only_orig")
            if self.conv_in_loaded:
                self.unet.conv_in.activate_only_orig()
            self.pipe.disable_lora()
        else:
            if self.conv_in_loaded:
                self.unet.conv_in.deactivate_only_orig()
            self.pipe.enable_lora()
        
        return self.unet_original_forward(sample, timestep, encoder_hidden_states, **kwargs)

    def train_step(
            self, 
            text_embeddings, 
            text_embeddings_pooled, 
            pred_rgb, # torch_images
            pred_rgb_obj=None,
            guidance_scale: float=20.0, 
            as_latent: bool=False, 
            step_ratio: float = None,
            noise: torch.Tensor = None,
            return_grad: bool = False,
        ):
        '''
        Please check the documentation for sd_utils_controlnet/train_step_dds if you need, 
        the parameters and the logic is the same here
        '''

        # x_t = tx + (1-t)eps
        # v ~= x - eps
        # x_t + v * (1 - t) = x + (1 - t)(-x) + (1-t)eps + (1-t)v = x + (1-t)(-x + eps + v)
        # x_new - x_old ~ (-x + eps + v)
        # => loss = x * (x - eps - v)

        assert text_embeddings.dtype == self.precision_t, f"text_embeddings {text_embeddings.dtype} != {self.precision_t}"
        pred_rgb = pred_rgb.to(self.precision_t)
        if not noise is None:
            noise = noise.to(self.precision_t)

        if as_latent:
            latents = pred_rgb
            latents_rgb_obj = pred_rgb_obj
        else:
            # interp to w x h to be fed into vae.
            pred_rgb = F.interpolate(pred_rgb, (self.height, self.width), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.torch2latents(pred_rgb)
            latents_rgb_obj = self.torch2latents_resize(pred_rgb_obj)
        
        latents_rgb_obj2 = self._maybe_repeat(latents_rgb_obj, 2)

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        if step_ratio is None:
            indices = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],))
        else:
            index = int(self.num_train_timesteps * step_ratio)
            index = max(min(index, self.max_step), self.min_step)
            indices = torch.full((latents.shape[0],), fill_value=index, dtype=torch.long)
        timesteps = self.scheduler.timesteps[indices].to(device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.inference_mode():
            # add noise
            if noise is None:
                noise = torch.randn_like(latents)
            
            sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
            latents_noisy = sigmas * noise + (1.0 - sigmas) * latents
                        
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            tt = torch.cat([timesteps] * 2)

            # if use conv_in
            if not latents_rgb_obj2 is None:
                latent_model_input = torch.cat([latent_model_input, latents_rgb_obj2], dim=1)
            
            unet_output = self.unet(
                hidden_states=latent_model_input,
                timestep=tt,
                encoder_hidden_states=text_embeddings,
                pooled_projections=text_embeddings_pooled,
            ).sample

            # perform guidance (high scale from paper!)
            v_pred_uncond, v_pred_text = unet_output.chunk(2)
            v_pred = v_pred_uncond + guidance_scale * (v_pred_text - v_pred_uncond)

            direction = (latents - noise + v_pred)

        grad = direction
        grad = torch.nan_to_num(grad)

        if return_grad:
            return grad

        loss = (grad * latents).sum(axis=(1, 2, 3)).mean()

        return loss
    
    def get_sigmas(self, timesteps, n_dim=4, dtype=torch.float32):
        sigmas = self.scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.scheduler.timesteps.to(self.device)
        timesteps = timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def _maybe_repeat(self, x, times=1):
        if x is None:
            return None
        else:
            return torch.cat([x] * times)

    def train_step_dds(
            self, 
            text_embeddings_initial, # "", "initial prompt"
            text_embeddings_initial_pooled,
            text_embeddings_desired, # "", "desired prompt"
            text_embeddings_desired_pooled, 
            latents_initial, # torch_images
            rgb_pred, # torch_images
            pred_rgb_obj=None,
            guidance_scale_tgt: float = 15.0, 
            guidance_scale_src: float = 6.0,
            as_latent: bool=False, 
            step_ratio: float = None,
            noise: torch.Tensor = None,
            return_grad: bool = False,
            eta: float = 1.0, # from https://www.arxiv.org/pdf/2509.05342
        ):
        '''
        latents_initial: [B, 16, 128, 128]
        pred_rgb_obj: [B, 16, 128, 128]
        '''

        batch_size = rgb_pred.shape[0]
        rgb_pred = rgb_pred.to(self.precision_t)
        assert text_embeddings_initial.dtype == self.precision_t, f"text_embeddings_initial {text_embeddings_initial.dtype} != {self.precision_t}"
        assert text_embeddings_initial_pooled.dtype == self.precision_t, f"text_embeddings_initial_pooled {text_embeddings_initial_pooled.dtype} != {self.precision_t}"
        assert text_embeddings_desired.dtype == self.precision_t, f"text_embeddings_desired {text_embeddings_desired.dtype} != {self.precision_t}"
        assert text_embeddings_desired_pooled.dtype == self.precision_t, f"text_embeddings_desired_pooled {text_embeddings_desired_pooled.dtype} != {self.precision_t}"
        assert latents_initial.dtype == self.precision_t, f"latents_initial {latents_initial.dtype} != {self.precision_t}"

        if not noise is None:
            noise = noise.to(self.precision_t)

        if as_latent:
            latents = rgb_pred
            latents_rgb_obj = pred_rgb_obj
        else:
            # interp to 512x512 to be fed into vae.
            pred_rgb = F.interpolate(rgb_pred, (self.height, self.width), mode='bilinear', align_corners=False)
            # encode image into latents with vae, requires grad!
            latents = self.torch2latents(pred_rgb)
            latents_rgb_obj = self.torch2latents_resize(pred_rgb_obj)

        latents_rgb_obj2 = self._maybe_repeat(latents_rgb_obj, 2)

        if step_ratio is None:
            indices = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],))
        else:
            index = int(self.num_train_timesteps * step_ratio)
            index = max(min(index, self.max_step), self.min_step)
            indices = torch.full((latents.shape[0],), fill_value=index, dtype=torch.long)
        timesteps = self.scheduler.timesteps[indices].to(device=self.device)

        with torch.inference_mode():
            # add noise
            if noise is None:
                noise = torch.randn_like(latents)
            noise2 = torch.cat([noise] * 2)
            t2 = torch.cat([timesteps] * 2)

            latents_initial2 = torch.cat([latents_initial, latents_initial], dim=0)
            latents2 = torch.cat([latents, latents], dim=0)

            # add noise
            sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
            latents_initial2_noisy = sigmas * noise2 + (1.0 - sigmas) * latents_initial2
            c_t_i = eta * (t2 / self.num_train_timesteps).view(-1, 1, 1, 1).to(self.precision_t)
            latents2_noisy = sigmas * noise2 + (1.0 - sigmas) * latents2 + c_t_i * (latents2 - latents_initial2)
            # print(f"timestep: {timesteps.item()}, sigma: {sigmas.item()}, eta {eta}, c_t_i {c_t_i[0].item()} ")

            # if use conv_in
            if not latents_rgb_obj2 is None:
                latents2_noisy = torch.cat([latents2_noisy, latents_rgb_obj2], dim=1)
            # pred noise
            unet_output_1 = self.unet(    
                hidden_states=latents_initial2_noisy,
                timestep=t2,
                encoder_hidden_states=text_embeddings_initial,
                pooled_projections=text_embeddings_initial_pooled,
            ).sample

            unet_output_2 = self.unet(
                hidden_states=latents2_noisy,
                timestep=t2,
                encoder_hidden_states=text_embeddings_desired,
                pooled_projections=text_embeddings_desired_pooled,
            ).sample

            # perform guidance (high scale from paper!)
            v_pred_uncond_initial, v_pred_text_initial = unet_output_1.chunk(2)
            v_pred_uncond_desired, v_pred_text_desired = unet_output_2.chunk(2)

            v_pred_initial = v_pred_uncond_initial + guidance_scale_src * (v_pred_text_initial - v_pred_uncond_initial)
            v_pred_desired = v_pred_uncond_desired + guidance_scale_tgt * (v_pred_text_desired - v_pred_uncond_desired)

            direction = (1 - eta) * (latents - latents_initial) + (v_pred_desired - v_pred_initial)
            # DDS: (noise_pred_desired - noise) - (noise_pred_initial - noise)
            # DDS_rf: (latents_desired - noise + v_pred_desired) - (latents_initial - noise + v_pred_initial) = 
            # (latents_desired  - latents_initial) + (v_pred_desired - v_pred_initial)

        grad = direction
        grad = torch.nan_to_num(grad)

        if return_grad:
            return grad

        # another way to compute the loss
        loss = (grad * latents).sum(axis=(1, 2, 3)).mean()

        return loss

    @staticmethod
    def get_save_dir(*args, **kwargs):
        return StableDiffusion.get_save_dir(*args, **kwargs)

    @staticmethod
    def save_images(*args, **kwargs):
        return StableDiffusion.save_images(*args, **kwargs)

    def _get_clip_prompt_embeds(
        self,
        prompt,
        clip_model_index: int = 0,
    ):
        clip_tokenizers = [self.tokenizer, self.tokenizer_2]
        clip_text_encoders = [self.text_encoder, self.text_encoder_2]

        tokenizer = clip_tokenizers[clip_model_index]
        text_encoder = clip_text_encoders[clip_model_index]

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_embeds = text_encoder(text_input_ids.to(self.device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]
        prompt_embeds = prompt_embeds.hidden_states[-2]
        
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=self.device)
        
        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(batch_size * 1, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, 1, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * 1, -1)

        return prompt_embeds, pooled_prompt_embeds

        return prompt_embeds, pooled_prompt_embeds
    
    def _get_t5_prompt_embeds(
        self,
        prompt,
        max_sequence_length: int = 256,
    ):  
        device = self.device
        dtype = self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        if self.text_encoder_3 is None:
            return torch.zeros(
                (
                    batch_size,
                    self.tokenizer_max_length,
                    self.unet.config.joint_attention_dim,
                ),
                device=device,
                dtype=dtype,
            )

        text_inputs = self.tokenizer_3(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids

        prompt_embeds = self.text_encoder_3(text_input_ids.to(device))[0]

        dtype = self.text_encoder_3.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(batch_size * 1, seq_len, -1)

        return prompt_embeds

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        prompt_2 = prompt
        prompt_3 = prompt

        prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
            prompt=prompt,
            clip_model_index=0,
        )
        prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
            prompt=prompt_2,
            clip_model_index=1,
        )
        clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

        t5_prompt_embed = self._get_t5_prompt_embeds(prompt_3)

        clip_prompt_embeds = torch.nn.functional.pad(
            clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
        )

        prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
        pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        return prompt_embeds, pooled_prompt_embeds
    
    @torch.no_grad()
    def perform_sdedit(
        self, 
        image, 
        emb_uncond,
        emb_cond, 
        rgb_obj_pred=None,
        guidance_scale: float=7.5, 
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
        if not latents_rgb_obj2 is None and latents_rgb_obj2.shape[0] != batch_size * 2:
            latents_rgb_obj2 = self._maybe_repeat(latents_rgb_obj2, batch_size)

        self.inference_scheduler.set_timesteps(num_inference_steps)
        inference_timesteps = self.inference_scheduler.timesteps.detach().to(self.device)

        if not pure_noise:
            # equal timesteps
            if t_step is None:
                t_add_noise = torch.randint(
                    timestep_range[0], 
                    timestep_range[1] + 1, 
                    (1,), 
                    dtype=torch.long, 
                    device=self.device
                ).repeat(batch_size)
            else:
                t_add_noise = torch.tensor(
                    [t_step] * batch_size, 
                    dtype=torch.long, 
                    device=self.device
                )
            noise_added = torch.randn_like(latents)

            latents = self.inference_scheduler.add_noise(latents, noise_added, t_add_noise)
            # latents = noise_added
            start_step = inference_timesteps[(inference_timesteps > t_add_noise[0])].argmin()
            start_step = max(start_step - 1, 0)
        else:
            latents = torch.randn_like(latents)
            start_step = 0

        text_embeddings = torch.cat([emb_uncond] * batch_size + [emb_cond] * batch_size, dim=0)

        # for i, t in tqdm(enumerate(inference_timesteps[start_step:]), total=len(inference_timesteps[start_step:])):
        for i, t in enumerate(inference_timesteps[start_step:]):
            latent2 = torch.cat([latents] * 2)
            t2 = torch.cat([t.repeat(batch_size)] * 2)
            text_embeddings2 = text_embeddings # it is already repeated
            
            # if use conv_in
            if not latents_rgb_obj2 is None:
                latent2 = torch.cat([latent2, latents_rgb_obj2], dim=1)

            noise_pred = self.unet(
                latent2, 
                t2, 
                encoder_hidden_states=text_embeddings2,
            ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.inference_scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents
    
    @torch.no_grad()
    def generate_images_latents_by_embeds(
        self, 
        text_embeddings, 
        text_embeddings_pooled,
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
                        text_embeddings.shape[0] // 2, self.unet.config.in_channels, 
                        int(self.height) // self.vae_scale_factor,
                        int(self.width) // self.vae_scale_factor,
                    ), 
                    device=self.device,
                    dtype=self.precision_t,
                )
        text_embeddings = text_embeddings.to(self.precision_t)
        text_embeddings_pooled = text_embeddings_pooled.to(self.precision_t)

        self.inference_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.inference_scheduler.timesteps

        for i, t in tqdm(enumerate(timesteps)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            timestep = t.expand(latent_model_input.shape[0])
            # predict the noise residual
            noise_pred = self.unet(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=text_embeddings,
                pooled_projections=text_embeddings_pooled,
                return_dict=False,
            )[0]

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.inference_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            
        return latents
    
    @torch.no_grad()
    def generate_images_by_prompts(
            self, 
            prompts, 
            negative_prompts='', 
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
            raise RuntimeError("Number of prompts should be 1 or equal to num_same")

        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts] * len(prompts)

        # Prompts -> text embeds
        pos_embeds, pos_embeds_pooled = self.get_text_embeds(prompts) # [1, 77, 768]
        neg_embeds, neg_embeds_pooled = self.get_text_embeds(negative_prompts)        
        text_embeds = torch.cat([neg_embeds, pos_embeds], dim=0) # [2, 77, 768]
        text_embeds_pooled = torch.cat([neg_embeds_pooled, pos_embeds_pooled], dim=0) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.generate_images_latents_by_embeds(
            text_embeds, 
            text_embeds_pooled,
            num_inference_steps=num_inference_steps, 
            guidance_scale=guidance_scale,
            latents=latents, 
            seed=seed,
        ) # [1, 16, 128, 128]

        # Img latents -> imgs
        torch_images = self.latents2torch(latents) # [1, 3, 1024, 1024]
        np_images = self.torch2np(torch_images)

        return np_images
    
    def np2torch(self, np_images):
        if np_images is None:
            return None
        torch_images = torch.tensor(np_images, device=self.device).to(self.precision_t)
        if torch_images.ndim == 3:
            torch_images = torch_images.unsqueeze(0)
        images = torch_images.permute(0, 3, 1, 2) / 255
        images = 2 * images - 1
        return images

    def torch2np(self, torch_images):
        if torch_images is None:
            return None
        torch_images = (torch_images / 2 + 0.5).clamp(0, 1)
        np_images = (torch_images.permute(0, 2, 3, 1) * 255).to(torch.uint8).cpu().numpy()
        return np_images

    def latents2torch(self, latents):
        if latents is None:
            return None
        
        latents = latents.to(self.precision_t)
        latents = (latents / self.vae.config.scaling_factor) + self.vae.config.shift_factor
        torch_images = self.vae.decode(latents).sample
        return torch_images

    def torch2latents(self, torch_images):
        if torch_images is None:
            return None
        torch_images = torch_images.to(self.precision_t)
        posterior = self.vae.encode(torch_images).latent_dist.sample()
        latents = (posterior - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return latents

    def torch2latents_resize(self, torch_images):
        if torch_images is None:
            return None
        torch_images = torch_images.to(self.precision_t)
        torch_images_512 = F.interpolate(
            torch_images, 
            (self.height, self.width), 
            mode='bilinear', align_corners=False
        )
        latents = self.torch2latents(torch_images_512)
        return latents

if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.1', choices=['1.5', '2.0', '2.1'], help="stable diffusion version")
    parser.add_argument('--hf_key', type=str, default=None, help="hugging face Stable diffusion model key")
    parser.add_argument('--fp16', action='store_true', help="use float16 for training")
    parser.add_argument('--vram_O', action='store_true', help="optimization for low VRAM usage")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.fp16, opt.vram_O, opt.sd_version, opt.hf_key)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()

