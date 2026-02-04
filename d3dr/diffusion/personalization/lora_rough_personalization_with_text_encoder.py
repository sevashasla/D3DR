import argparse
import itertools
import json
import os
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from dataset import ImageDataset
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
    encode_text_grad,
    generate_image,
    torch2latents,
)


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="a <rare_token> caution wet floor sign")
    parser.add_argument("--ic_light_prompt", type=str, default="a yellow stand sign")
    # dca
    # take something from https://github.com/2kpr/dreambooth-tokens
    # <doh> is okay
    parser.add_argument("--rare_token", type=str, default="<ktn>")
    parser.add_argument("--exp_desc", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model_name", type=str, default="stabilityai/stable-diffusion-2-1-base")
    parser.add_argument("--sd_unet_path", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--num_inference_steps", type=int, default=25)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--prob", type=float, default=0.7)

    parser.add_argument("--lora_rank", type=int, default=4)

    parser.add_argument("--num_train_iterations", type=int, default=1000)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--generate_cp_dir", type=str, default=None)
    parser.add_argument("--generate_cp_iclight", type=str, default=None)
    parser.add_argument("--generate_using_iclight", type=int, default=1)
    parser.add_argument("--show_iter", type=int, default=100)
    parser.add_argument("--generate_n", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_acc_step", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    # parser.add_argument("--lambd_preservation_loss", type=float, default=0.0)
    parser.add_argument("--save_dir", type=str, default="./personalization_play/")
    parser.add_argument("--debug", type=int, default=1)
    parser.add_argument("--skip_used", action="store_true")

    args = parser.parse_args()

    return args

def main():
    args = get_args()

    # replce <rare_token> in prompt
    prompt_object = args.prompt.replace("<rare_token> ", "")
    args.prompt = args.prompt.replace("<rare_token>", args.rare_token)

    print("prompt_object:", prompt_object)
    print("prompt_train :", args.prompt)

    os.makedirs(args.save_dir, exist_ok=True)
    if args.debug:
        args.save_dir = str(Path(args.save_dir) / f"{len(list(Path(args.save_dir).glob('*'))):03}")
    print("Save dir:", args.save_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    with open(Path(args.save_dir) / "my_args.json", "w") as f:
        json.dump(vars(args), f, indent=4)

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

    pipe = StableDiffusionPipeline.from_pretrained(args.model_name, torch_dtype=weight_dtype)
    vae = pipe.vae
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    if args.sd_unet_path is not None:
        unet = unet.from_pretrained(args.sd_unet_path, device=torch_device, torch_dtype=weight_dtype)
        unet = unet.to(torch_device) # idk why but I have to do it again 0_o
        print(f"Unet from {args.sd_unet_path} loaded!")
    _scheduler = DDIMScheduler.from_pretrained(args.model_name, subfolder="scheduler")
    _train_scheduler = DDIMScheduler.from_pretrained(args.model_name, subfolder="scheduler")

    # freeze unet, vae and text_encoder
    unet.requires_grad_(False)
    vae.requires_grad_(False)

    unet_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )

    text_encoder_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        init_lora_weights="gaussian",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    )

    unet.add_adapter(unet_lora_config)
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(unet, dtype=torch.float32)

    text_encoder.add_adapter(text_encoder_lora_config)
    if args.mixed_precision == "fp16":
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(text_encoder, dtype=torch.float32)

    lora_layers = filter(lambda p: p.requires_grad, itertools.chain(unet.parameters(), text_encoder.parameters()))
    torch.manual_seed(args.seed)

    model, text_encoder, vae = accelerator.prepare(
        unet, text_encoder, vae
    )

    dataset = ImageDataset(
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
        prob=args.prob,
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

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

    generate_image(
        prompt=prompt_object,
        num_same=1,
        save_name="initial.png",
        seed=args.seed,
        save_dir=args.save_dir,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height, width=args.width,
        scheduler=_scheduler,
        tokenizer=tokenizer,
        text_encoder=accelerator.unwrap_model(text_encoder),
        vae=accelerator.unwrap_model(vae),
        unet=accelerator.unwrap_model(model),
        device=accelerator.device,
    )

    iter_dataloader = iter(dataloader)
    for i in tqdm(range(args.num_train_iterations)):
        # get batch
        model.train()
        
        # keep gradient
        prompt_emb = encode_text_grad([args.prompt], text_encoder, tokenizer, device=torch_device)
        prompt_object_emb = encode_text_grad([prompt_object], text_encoder, tokenizer, device=torch_device)
        embs = torch.cat([prompt_object_emb, prompt_emb], axis=0)

        try:
            batch_images, batch_emb_ids = next(iter_dataloader)
        except StopIteration:
            iter_dataloader = iter(dataloader)
            batch_images, batch_emb_ids = next(iter_dataloader)

        clean_images = batch_images.to(device=torch_device, dtype=weight_dtype)
        clean_images = torch2latents(clean_images, vae, torch_device) # get latents
        curr_bs = clean_images.size(0)

        timesteps = torch.randint(0, _train_scheduler.config.num_train_timesteps, size=(curr_bs,), device=torch_device)
        prompt_emb_batch = embs[batch_emb_ids][:, 0, :, :]

        noise = torch.randn_like(clean_images)
        noised_images = _train_scheduler.add_noise(clean_images, noise, timesteps)

        with accelerator.accumulate(model):
            noise_pred = model(noised_images, timesteps, encoder_hidden_states=prompt_emb_batch).sample
            loss = F.mse_loss(noise_pred.float(), noise.float())
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        if accelerator.is_main_process and ((i + 1) % args.show_iter == 0 or i == args.num_train_iterations - 1):
            model.eval()
            unwrapped_unet = deepcopy(model).to(torch.float32)
            unet_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(unwrapped_unet))
            text_encoder_lora_layers_to_save = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder))
            
            StableDiffusionPipeline.save_lora_weights(
                save_directory=args.save_dir,
                unet_lora_layers=unet_lora_layers_to_save,
                text_encoder_lora_layers=text_encoder_lora_layers_to_save,
                safe_serialization=True,
            )
            
            generate_image(
                prompt=args.prompt, 
                num_same=9,
                save_name="training.png",
                seed=args.seed,
                save_dir=args.save_dir,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.height, width=args.width,
                scheduler=_scheduler,
                tokenizer=tokenizer,
                text_encoder=text_encoder,
                vae=vae,
                unet=accelerator.unwrap_model(model),
                device=accelerator.device,
            )

if __name__ == "__main__":
    main()
