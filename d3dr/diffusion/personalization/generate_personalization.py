"""
Generate images of Rough Diffusion Personalization.
"""

import argparse
import os

import torch
from diffusers import (
    DDIMScheduler,
    StableDiffusionPipeline,
)
from utils import generate_image

parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompt", type=str, default="A <rare_token> caution wet floor sign"
)
# dca
# take something from https://github.com/2kpr/dreambooth-tokens
parser.add_argument("--rare_prompt", type=str, default="<ktn>")
parser.add_argument("--exp_desc", type=str, default="")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--model_name", type=str, default="Manojb/stable-diffusion-2-1-base"
)
parser.add_argument("--sd_unet_path", type=str, default=None)
parser.add_argument("--guidance_scale", type=float, default=7.5)
parser.add_argument("--num_inference_steps", type=int, default=25)
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--height", type=int, default=512)

parser.add_argument(
    "--checkpoint_root", type=str, default="./personalization_play/"
)
parser.add_argument("--checkpoint_dir", type=str, default="004")
parser.add_argument("--mixed_precision", type=str, default="fp16")
parser.add_argument(
    "--lora_adapters_paths", type=str, action="append", default=[]
)

args = parser.parse_args()

# flip <rare_token> in prompt
args.prompt = args.prompt.replace("<rare_token>", args.rare_prompt)
print(args.prompt)

if not os.path.exists(args.checkpoint_dir):
    args.checkpoint_dir = os.path.join(
        args.checkpoint_root, args.checkpoint_dir
    )
if not os.path.exists(args.checkpoint_dir):
    raise RuntimeError(
        f"Checkpoint directory {args.checkpoint_dir} does not exist!"
    )

args.save_dir = args.checkpoint_dir
torch_device = "cuda"

weight_dtype = torch.float32
if args.mixed_precision == "fp16":
    weight_dtype = torch.float16

pipe = StableDiffusionPipeline.from_pretrained(
    args.model_name, torch_dtype=weight_dtype
)
pipe = pipe.to(torch_device)
vae = pipe.vae
tokenizer = pipe.tokenizer
text_encoder = pipe.text_encoder
unet = pipe.unet
if args.sd_unet_path is not None:
    unet = unet.from_pretrained(
        args.sd_unet_path, torch_dtype=weight_dtype, device=torch_device
    )
    unet = unet.to(torch_device)
    print(f"Unet from {args.sd_unet_path} loaded!")
_scheduler = DDIMScheduler.from_pretrained(
    args.model_name, subfolder="scheduler"
)

# add lora adapters
for lora_adapter_path in args.lora_adapters_paths:
    pipe.load_lora_weights(lora_adapter_path, adapter_name=lora_adapter_path)

if len(args.lora_adapters_paths) > 0:
    unet.set_adapters(
        args.lora_adapters_paths,
        [1 / len(args.lora_adapters_paths)] * len(args.lora_adapters_paths),
    )

torch.manual_seed(args.seed)
generate_image(
    prompt=args.prompt,
    num_same=16,
    save_name="personalized.png",
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
    unet=unet,
    device=torch_device,
)
