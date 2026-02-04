from transformers import logging
from diffusers import (
    ControlNetModel,
)
from diffusers.image_processor import VaeImageProcessor

from diffusers.utils.import_utils import is_xformers_available
from os.path import isfile
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
# import warnings
# warnings.warn("Should fix this import!!")
# import sys
# sys.path.append(os.path.dirname(__file__))

# from sd_utils import StableDiffusion
from d3dr.diffusion.sd_utils import StableDiffusion

class SDControlNet(StableDiffusion):
    def __init__(
            self, 
            device, 
            fp16=False, 
            vram_O=False, 
            sd_version='2.1', 
            controlnet_name='thibaud/controlnet-sd21-depth-diffusers',
            hf_key=None, 
            t_range=[0.02, 0.98], 
            height=512, width=512,
            sd_unet_path: str = None,
            lora_adapters_paths: Optional[List[str]] = None,
            lora_object_texture_path: Optional[str] = None,
            conv_in_path: Optional[str] = None
        ):
        '''
        Check out the parameters from StableDiffusion class in sd_utils.py
        :param controlnet_name
            name of the controlnet model
        '''
        super().__init__(
            device=device, 
            fp16=fp16, vram_O=vram_O, 
            sd_version=sd_version, hf_key=hf_key, 
            t_range=t_range, 
            height=height, width=width,
            sd_unet_path=sd_unet_path,
            lora_adapters_paths=lora_adapters_paths,
            lora_object_texture_path=lora_object_texture_path,
            conv_in_path=conv_in_path,
        )

        # build a controlnet
        self.controlnet_name = controlnet_name
        self.controlnet = ControlNetModel.from_pretrained(self.controlnet_name, torch_dtype=self.precision_t).to(self.device)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        # build a controlnet image processor
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )

    def train_step(
            self, 
            text_embeddings, 
            image_embeddings, # for controlnets
            pred_rgb, # torch_images
            pred_rgb_obj=None, 
            guidance_scale: float=7.5, 
            controlnet_conditioning_scale: float=1.0, # one should use this, because other scales don't work well :/
            as_latent: bool = False, 
            step_ratio: float = None,
            noise: torch.Tensor = None,
            return_grad: bool = False,
        ):
        '''
        Please check the documentation for train_step_dds if you need, 
        the parameters and the logic is the same here
        '''

        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.precision_t)
        assert text_embeddings.dtype == self.precision_t, f"text_embeddings {text_embeddings.dtype} != {self.precision_t}"
        assert image_embeddings.dtype == self.precision_t, f"image_embeddings {image_embeddings.dtype} != {self.precision_t}"

        if not noise is None:
            noise = noise.to(self.precision_t)

        if as_latent:
            latents = pred_rgb
            latents_rgb_obj = pred_rgb_obj
        else:
            latents = self.torch2latents_resize(pred_rgb)
            latents_rgb_obj = self.torch2latents_resize(pred_rgb_obj)
        
        # just to speed up computation
        latents_rgb_obj2 = self._maybe_repeat(latents_rgb_obj, 2)

        if step_ratio is None:
            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)
        else:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            # sqrt looks better for 2d images
            t = np.round(np.sqrt(1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
            # add noise
            if noise is None:
                noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            image_embeddings = torch.cat([image_embeddings] * 2)
            tt = torch.cat([t] * 2)

            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent_model_input,
                tt,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=image_embeddings,
                conditioning_scale=controlnet_conditioning_scale,
                return_dict=False,
            )
            
            # if use conv_in
            if not latents_rgb_obj2 is None:
                latents_noisy = torch.cat([latents_noisy, latents_rgb_obj2], dim=1)
            
            unet_output = self.unet(
                latent_model_input, 
                tt, 
                encoder_hidden_states=text_embeddings,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_text = unet_output.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

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
    
    def train_step_dds(
            self, 
            text_embeddings_initial, # "", "initial prompt"
            text_embeddings_desired, # "", "desired prompt"
            image_embeddings_initial, # torch_images
            image_embeddings_desired, # torch_images
            latents_initial, # torch_images
            rgb_pred, # torch_images
            pred_rgb_obj = None,
            guidance_scale: float=7.5, 
            controlnet_conditioning_scale: float=1.0,
            as_latent: bool=False, # we optimize image or latents
            step_ratio: float = None,
            noise: torch.Tensor = None,
            use_weights=False,
            return_grad=False,
            use_controlnet=True,
        ):
        '''
        Makes one step of DDS training procedure.

        :param text_embeddings_initial
            initial text embeddings, "", "initial prompt"
        :param text_embeddings_desired
            desired text embeddings, "", "desired prompt"
        :param image_embeddings_initial
            initial image embeddings (for controlnet)
        :param image_embeddings_desired
            desired image embeddings (for controlnet)
        :param latents_initial
            initial latents of the image (only scene)
        :param rgb_pred
            predicted image (obj + scene)
        :use_weights: bool
            whether to use weights for the loss or assume w = 1
        '''

        if not use_controlnet:
            return StableDiffusion.train_step_dds(
                self,
                text_embeddings_initial=text_embeddings_initial,
                text_embeddings_desired=text_embeddings_desired,
                latents_initial=latents_initial,
                rgb_pred=rgb_pred,
                pred_rgb_obj=pred_rgb_obj,
                guidance_scale=guidance_scale, 
                as_latent=as_latent,
                step_ratio=step_ratio,
                noise=noise,
                use_weights=use_weights,
                return_grad=return_grad,
            )

        batch_size = rgb_pred.shape[0]
        rgb_pred = rgb_pred.to(self.precision_t)
        assert text_embeddings_initial.dtype == self.precision_t, f"text_embeddings_initial {text_embeddings_initial.dtype} != {self.precision_t}"
        assert text_embeddings_desired.dtype == self.precision_t, f"text_embeddings_desired {text_embeddings_desired.dtype} != {self.precision_t}"
        assert image_embeddings_initial.dtype == self.precision_t, f"image_embeddings_initial {image_embeddings_initial.dtype} != {self.precision_t}"
        assert image_embeddings_desired.dtype == self.precision_t, f"image_embeddings_desired {image_embeddings_desired.dtype} != {self.precision_t}"

        if not noise is None:
            noise = noise.to(self.precision_t)

        if as_latent:
            latents = rgb_pred
            latents_rgb_obj = pred_rgb_obj
        else:
            latents = self.torch2latents_resize(rgb_pred)
            latents_rgb_obj = self.torch2latents_resize(pred_rgb_obj)

        latents_rgb_obj2 = self._maybe_repeat(latents_rgb_obj, 2)

        if step_ratio is None:
            # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
            t = torch.randint(self.min_step, self.max_step + 1, (batch_size,), dtype=torch.long, device=self.device)
        else:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            # sqrt looks better for 2d images
            t = np.round(np.sqrt(1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((batch_size,), t, dtype=torch.long, device=self.device)

        if image_embeddings_desired is None:
            image_embeddings_desired = image_embeddings_initial
        
        image_embeddings_initial2 = torch.cat([
            image_embeddings_initial, image_embeddings_initial, 
        ])

        image_embeddings_desired2 = torch.cat([
            image_embeddings_desired, image_embeddings_desired, 
        ])

        # predict the noise residual with unet, NO grad!
        with torch.no_grad():
        # with torch.autocast(device_type="cuda", dtype=self.precision_t):
            # add noise
            if noise is None:
                noise = torch.randn_like(latents)
            noise2 = torch.cat([noise] * 2)
            t2 = torch.cat([t] * 2)

            # latents (to avoid double unet call for classifier free guidance) 
            # and add noise to latents
            latents_initial2 = torch.cat([latents_initial, latents_initial], dim=0)
            latents_initial2_noisy = self.scheduler.add_noise(latents_initial2, noise2, t2)
            
            latents2 = torch.cat([latents, latents], dim=0)
            latents2_noisy = self.scheduler.add_noise(latents2, noise2, t2)

            # pred of controlnet
            down_block_res_samples_initial, mid_block_res_sample_initial = self.controlnet(
                latents_initial2_noisy,
                t2,
                encoder_hidden_states=text_embeddings_initial,
                controlnet_cond=image_embeddings_initial2,
                conditioning_scale=controlnet_conditioning_scale,
                return_dict=False,
            )

            down_block_res_samples_desired, mid_block_res_sample_desired = self.controlnet(
                latents2_noisy,
                t2,
                encoder_hidden_states=text_embeddings_desired,
                controlnet_cond=image_embeddings_desired2,
                conditioning_scale=controlnet_conditioning_scale,
                return_dict=False,
            )

            # if use conv_in
            if not latents_rgb_obj2 is None:
                latents2_noisy = torch.cat([latents2_noisy, latents_rgb_obj2], dim=1)

            # pred noise
            unet_output_1 = self.unet(
                latents_initial2_noisy, 
                t2, 
                encoder_hidden_states=text_embeddings_initial,
                down_block_additional_residuals=down_block_res_samples_initial,
                mid_block_additional_residual=mid_block_res_sample_initial,
                only_orig=True,
            ).sample

            unet_output_2 = self.unet(
                latents2_noisy, 
                t2, 
                encoder_hidden_states=text_embeddings_desired,
                down_block_additional_residuals=down_block_res_samples_desired,
                mid_block_additional_residual=mid_block_res_sample_desired,
            ).sample

            # perform guidance (high scale from paper!)
            noise_pred_uncond_initial, noise_pred_text_initial = unet_output_1.chunk(2)
            noise_pred_uncond_desired, noise_pred_text_desired = unet_output_2.chunk(2)

            noise_pred_initial = noise_pred_uncond_initial + guidance_scale * (noise_pred_text_initial - noise_pred_uncond_initial)
            noise_pred_desired = noise_pred_uncond_desired + guidance_scale * (noise_pred_text_desired - noise_pred_uncond_desired)
            
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
    
    @torch.no_grad()
    def perform_sdedit(
        self, 
        image, 
        emb_uncond,
        emb_cond, 
        controlnet_comp_emb,
        pred_rgb_obj=None,
        guidance_scale: float=7.5, 
        as_latent=True,
        timestep_range=(400, 600),
        t_step=None,
        num_inference_steps=25,
        batch_size=1,
        use_controlnet=True,
    ):
        
        if not use_controlnet:
            return StableDiffusion.perform_sdedit(
                self,
                image=image, 
                emb_uncond=emb_uncond,
                emb_cond=emb_cond, 
                rgb_obj_pred=pred_rgb_obj,
                guidance_scale=guidance_scale, 
                as_latent=as_latent,
                timestep_range=timestep_range,
                t_step=t_step,
                num_inference_steps=num_inference_steps,
                batch_size=batch_size,
            )
        image = image.to(self.precision_t)

        if as_latent:
            latents = image
            latents_rgb_obj = pred_rgb_obj
        else:
            latents = self.torch2latents_resize(image)
            latents_rgb_obj = self.torch2latents_resize(pred_rgb_obj)

        if latents.shape[0] != batch_size:
            latents = latents.repeat(batch_size, 1, 1, 1)

        latents_rgb_obj2 = self._maybe_repeat(latents_rgb_obj, 2)
        if not latents_rgb_obj2 is None and latents_rgb_obj2.shape[0] != batch_size * 2:
            latents_rgb_obj2 = self._maybe_repeat(latents_rgb_obj2, batch_size)

        self.inference_scheduler.set_timesteps(num_inference_steps)
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
        inference_timesteps = self.inference_scheduler.timesteps.detach().to(self.device)
        start_step = inference_timesteps[(inference_timesteps > t_add_noise[0])].argmin()
        start_step = max(start_step - 1, 0)

        text_embeddings = torch.cat([emb_uncond] * batch_size + [emb_cond] * batch_size, dim=0)
        controlnet_comp_emb2 = torch.cat([controlnet_comp_emb] * 2 * batch_size) # it is already repeated
        text_embeddings2 = text_embeddings # it is already repeated

        # for i, t in tqdm(enumerate(inference_timesteps[start_step:]), total=len(inference_timesteps[start_step:])):
        for i, t in enumerate(inference_timesteps[start_step:]):
            latent2 = torch.cat([latents] * 2)
            t2 = torch.cat([t.repeat(batch_size)] * 2)

            # pred of controlnet
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                latent2,
                t2,
                encoder_hidden_states=text_embeddings2,
                controlnet_cond=controlnet_comp_emb2,
                conditioning_scale=1.0,
                return_dict=False,
            )

            if not latents_rgb_obj2 is None:
                latent2 = torch.cat([latent2, latents_rgb_obj2], dim=1)
            
            # if use conv_in
            noise_pred = self.unet(
                latent2, 
                t2, 
                encoder_hidden_states=text_embeddings2,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            ).sample

            # perform guidance
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.inference_scheduler.step(noise_pred, t, latents)['prev_sample']

        return latents

    
    def get_image_embeds(self, image_controlnet, height=None, width=None):
        '''
        Calculates image embeddings from the controlnet.

        :param image_controlnet
            np.array (important!) of images
        :param height, width
            height and width of the desired image
        '''
        image = self.control_image_processor.preprocess(
            image_controlnet, 
            height=self.height if height is None else height, 
            width=self.width if width is None else width,
        ).to(self.precision_t)
        return image

