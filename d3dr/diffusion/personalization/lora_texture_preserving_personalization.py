import json
import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from dataset_wimg import ImageDatasetWImg
from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import convert_state_dict_to_diffusers
from peft import LoraConfig
from peft.utils import get_peft_model_state_dict
from tqdm.auto import tqdm
from utils import (
    cast_training_params,
    encode_text,
    get_random_images,
    latents2np,
    np2latents,
    save_images,
    torch2latents,
)

from d3dr.diffusion.personalization.lora_rough_personalization import (
    get_args as _get_args,
)
from d3dr.diffusion.personalization.utils import get_save_dir


class CropStrategy:
    """
    Defines how to adjust cropping
    """

    def __init__(
        self,
        num_train_iterations: int,
        crop_strategy: str,
        crop_ratio: float,
        crop_ratio_max: float,
    ):
        self.max_iter = num_train_iterations
        self.crop_ratio = crop_ratio
        self.crop_ratio_max = crop_ratio_max
        self.crop_strategy = crop_strategy
        if self.crop_strategy == "linear":
            self.crop_fn = self._linear
        elif self.crop_strategy == "random":
            self.crop_fn = self._random
        elif self.crop_strategy == "linear_half":
            self.crop_fn = self._linear_half
        elif self.crop_strategy == "poly":
            self.crop_fn = self._poly
        elif self.crop_strategy == "none":
            self.crop_fn = self._none

    def _none(self, curr_iter):
        return self.crop_ratio_max

    def _linear(self, curr_iter):
        a = (1.0 - self.crop_ratio) / self.max_iter
        b = self.crop_ratio
        return a * curr_iter + b

    def _linear_half(self, curr_iter):
        if curr_iter < self.max_iter // 2:
            b = self.crop_ratio
            a = (0.5 - self.crop_ratio) / (self.max_iter / 2.0)
        else:
            # a * max_iter / 2 + b = 0.5
            # a * max_iter + b = max_crop_ratio
            # a * max_iter / 2 = max_crop_ratio - 0.5

            a = (self.crop_ratio_max - 0.5) / self.max_iter * 2
            b = 0.5 - a * self.max_iter / 2

        return max(min(a * curr_iter + b, 1.0), self.crop_ratio)

    def _poly(self, curr_iter):
        return max(self._linear(curr_iter) ** 2.0, self.crop_ratio)

    def _random(self, curr_iter):
        return np.random.uniform(self.crop_ratio, self.crop_ratio_max)

    def __call__(self, *args, **kwargs):
        return self.crop_fn(*args, **kwargs)


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

    def forward(self, x):
        x_noisy, x_obj = torch.chunk(x, chunks=2, dim=1)
        out1 = self.conv_in_real(x_noisy)
        out2 = self.conv_in_opt(x_obj)
        out = out1 + out2
        return out


def get_args(should_parse=True):
    parser = _get_args(should_parse=False)
    parser.add_argument("--make_dark_background", action="store_true")
    parser.add_argument("--optimize_both_convs", action="store_true")
    parser.add_argument("--grad_mask", action="store_true")
    parser.add_argument(
        "--crop_ratio",
        type=float,
        default=0.3,
        help="whether to train on random crops",
    )
    parser.add_argument(
        "--crop_ratio_max",
        type=float,
        default=1.0,
        help="maximal crop ratio during training",
    )
    parser.add_argument(
        "--crop_type",
        type=str,
        default="mask",
        choices=["center", "mask"],
        help="how to crop during training",
    )
    parser.add_argument(
        "--take_only",
        type=int,
        default=-1,
        help="USELESS?; how much images to take from the dataset",
    )
    parser.add_argument(
        "--crop_strategy",
        type=str,
        default="linear_half",
        choices=["linear", "random", "linear_half", "poly", "linear_2", "none"],
        help="crop strategy",
    )
    parser.add_argument(
        "--set_to_0",
        type=float,
        default=-1.0,
        help="how often to use 0 instead of input image",
    )
    parser.add_argument("--guidance_scale_img", type=float, default=1.0)
    parser.add_argument("--add_raw_images_ratio", type=float, default=0.15)
    parser.add_argument("--ckpt_lora_path", type=str, default=None)

    if not should_parse:
        return parser

    args = parser.parse_args()
    return args


@torch.no_grad()
def generate_image_texture_preserving(
    prompt: str = None,
    obj_latents=None,
    emb=None,
    num_same: int = 1,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    guidance_scale_img: float = 1.0,
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

    obj_latents2 = torch.cat([obj_latents] * 2)

    scheduler.set_timesteps(num_inference_steps)

    for t in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(
            latent_model_input, timestep=t
        )
        # concatenate with original images
        latent_model_input = torch.cat(
            [latent_model_input, obj_latents2], dim=1
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


def main():
    args = get_args()
    if args.model_name == "2.1":
        args.model_name = "stabilityai/stable-diffusion-2-1-base"
    elif args.model_name == "2.0":
        args.model_name = "stabilityai/stable-diffusion-2-base"
    elif args.model_name == "1.5":
        args.model_name = "runwayml/stable-diffusion-v1-5"

    # replce <rare_token> in prompt
    prompt_object = args.prompt.replace("<rare_token> ", "")
    args.prompt = args.prompt.replace("<rare_token>", args.rare_token)

    print("prompt_object:", prompt_object)
    print("prompt_train :", args.prompt)

    os.makedirs(args.save_dir, exist_ok=True)
    args.save_dir = get_save_dir(args)
    print("Save dir:", args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(Path(args.save_dir) / "my_args.json", "w") as f:
        json.dump(vars(args), f)

    accelerator = Accelerator(
        gradient_accumulation_steps=args.num_acc_step,
        project_dir=os.path.join(args.save_dir, "logs"),
        mixed_precision=args.mixed_precision,
    )
    torch_device = accelerator.device
    if accelerator.is_main_process:
        accelerator.init_trackers("train_example")

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_name, torch_dtype=weight_dtype
    )
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    if args.sd_unet_path is not None:
        unet = unet.from_pretrained(
            args.sd_unet_path, device=torch_device, torch_dtype=weight_dtype
        )
        unet = unet.to(torch_device)  # idk why but I have to do it again 0_o
        print(f"Unet from {args.sd_unet_path} loaded!")
    _scheduler = DDIMScheduler.from_pretrained(
        args.model_name, subfolder="scheduler"
    )
    _train_scheduler = DDIMScheduler.from_pretrained(
        args.model_name, subfolder="scheduler"
    )

    # freeze unet, vae and text_encoder
    unet.conv_in = ConvWImage(unet.conv_in)
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    if args.ckpt_lora_path is None:
        unet_lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            init_lora_weights="gaussian",
            target_modules=[
                "to_k",
                "to_q",
                "to_v",
                "to_out.0",
                "add_k_proj",
                "add_v_proj",
            ],
        )
        unet.add_adapter(unet_lora_config)
    else:
        pipe.load_lora_weights(args.ckpt_lora_path, adapter_name="default")
        unet.set_adapters("default")

    if args.optimize_both_convs:
        unet.conv_in.requires_grad_(True)
    else:
        unet.conv_in.conv_in_opt.requires_grad_(True)

    # only upcast trainable parameters (LoRA) into fp32
    if args.mixed_precision == "fp16":
        cast_training_params(unet, dtype=torch.float32)

    lora_layers = filter(lambda p: p.requires_grad, unet.parameters())
    torch.manual_seed(args.seed)

    model, text_encoder, vae = accelerator.prepare(unet, text_encoder, vae)

    dataset = ImageDatasetWImg(
        make_dark_background=args.make_dark_background,
        crop_type=args.crop_type,
        take_only=args.take_only,
        image_dir=args.image_dir,
        width=args.width,
        height=args.height,
        generate_n=args.generate_n,
        orig_obj_prompt=prompt_object,
        ic_light_prompt=args.ic_light_prompt,
        save_dir=args.save_dir,
        generate_using_iclight=(args.generate_using_iclight != 0),
        generate_cp_dir=args.generate_cp_dir,
        generate_cp_iclight=args.generate_cp_iclight,
        add_raw_images_ratio=args.add_raw_images_ratio,
        prob=args.prob,
        fixed_place_prompt=args.fixed_place_prompt,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )

    optimizer = torch.optim.AdamW(
        lora_layers,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.num_train_iterations,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    optimizer, dataloader, lr_scheduler = accelerator.prepare(
        optimizer, dataloader, lr_scheduler
    )

    prompt_emb = encode_text(
        [args.prompt], text_encoder, tokenizer, device=torch_device
    )
    prompt_object_emb = encode_text(
        [prompt_object], text_encoder, tokenizer, device=torch_device
    )
    embs = torch.cat([prompt_object_emb, prompt_emb], axis=0)

    generate_image_texture_preserving(
        prompt=args.prompt,
        obj_latents=np2latents(
            get_random_images(args.image_dir, num_images=1),
            vae,
            torch_device,
            weight_dtype,
        ),
        num_same=1,
        save_name="initial.png",
        seed=args.seed,
        save_dir=args.save_dir,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        guidance_scale_img=args.guidance_scale_img,
        height=args.height,
        width=args.width,
        scheduler=_scheduler,
        tokenizer=tokenizer,
        text_encoder=accelerator.unwrap_model(text_encoder),
        vae=accelerator.unwrap_model(vae),
        unet=accelerator.unwrap_model(model),
        device=accelerator.device,
    )

    iter_dataloader = iter(dataloader)

    crop_strategy = CropStrategy(
        num_train_iterations=args.num_train_iterations,
        crop_ratio=args.crop_ratio,
        crop_ratio_max=args.crop_ratio_max,
        crop_strategy=args.crop_strategy,
    )
    crop_strategy_to_view = CropStrategy(
        num_train_iterations=args.num_train_iterations,
        crop_ratio=args.crop_ratio,
        crop_ratio_max=args.crop_ratio_max,
        crop_strategy="linear_half",
    )

    for i in tqdm(range(args.num_train_iterations)):
        # get batch
        model.train()
        curr_crop = crop_strategy(i)
        dataset.store_crop(curr_crop)

        try:
            images_obj, images_real, batch_emb_ids, mask_obj = next(
                iter_dataloader
            )
        except StopIteration:
            iter_dataloader = iter(dataloader)
            images_obj, images_real, batch_emb_ids, mask_obj = next(
                iter_dataloader
            )

        latents_real = images_real.to(device=torch_device, dtype=weight_dtype)
        latents_real = torch2latents(
            latents_real, vae, torch_device
        )  # get latents
        latents_obj = images_obj.to(device=torch_device, dtype=weight_dtype)
        latents_obj = torch2latents(
            latents_obj, vae, torch_device
        )  # get latents
        curr_bs = latents_real.size(0)

        mask_obj = mask_obj.to(device=torch_device, dtype=weight_dtype)
        mask_obj = F.interpolate(mask_obj, latents_obj.shape[2:4])

        if args.set_to_0 > 0.0:
            zeros_els = (torch.rand(curr_bs) < args.set_to_0).to(latents_obj)
            with torch.no_grad():
                latents_obj = latents_obj * zeros_els.reshape(-1, 1, 1, 1)

        timesteps = torch.randint(
            0,
            _train_scheduler.config.num_train_timesteps,
            size=(curr_bs,),
            device=torch_device,
        )
        prompt_emb_batch = embs[batch_emb_ids][:, 0, :, :]

        noise = torch.randn_like(latents_real)
        noised_images = _train_scheduler.add_noise(
            latents_real, noise, timesteps
        )
        latents_model_input = torch.cat([noised_images, latents_obj], dim=1)

        with accelerator.accumulate(model):
            noise_pred = model(
                latents_model_input,
                timesteps,
                encoder_hidden_states=prompt_emb_batch,
            ).sample

            loss = F.mse_loss(noise_pred.float(), noise.float())
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.is_main_process and (
            (i + 1) % args.show_iter == 0 or i == args.num_train_iterations - 1
        ):
            model.eval()
            unwrapped_unet = deepcopy(model).to(torch.float32)
            unet_lora_state_dict = convert_state_dict_to_diffusers(
                get_peft_model_state_dict(unwrapped_unet)
            )
            StableDiffusionPipeline.save_lora_weights(
                save_directory=args.save_dir,
                unet_lora_layers=unet_lora_state_dict,
                safe_serialization=True,
            )
            torch.save(
                accelerator.unwrap_model(model).conv_in.state_dict(),
                os.path.join(args.save_dir, "conv_in.pth"),
            )

            dataset.store_crop(crop_strategy_to_view(i))
            obj_images_generate, _, _, _ = dataset[0]
            obj_images_generate = (
                (obj_images_generate.permute(1, 2, 0)[None, ...] * 255)
                .numpy()
                .astype(np.uint8)
            )
            obj_images_generate = obj_images_generate.repeat(9, axis=0)
            obj_latents_generate = np2latents(
                obj_images_generate, vae, torch_device, weight_dtype
            )

            save_images(
                obj_images_generate,
                save_name="obj_training.png",
                save_dir=args.save_dir,
                prompt=args.prompt,
            )
            generate_image_texture_preserving(
                prompt=args.prompt,
                obj_latents=obj_latents_generate,
                num_same=9,
                save_name="training.png",
                seed=args.seed,
                save_dir=args.save_dir,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                scheduler=_scheduler,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                vae=vae,
                unet=accelerator.unwrap_model(model),
                device=accelerator.device,
            )


if __name__ == "__main__":
    main()
