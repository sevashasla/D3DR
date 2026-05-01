import argparse
import os

from d3dr.diffusion.sd_utils import StableDiffusion

parser = argparse.ArgumentParser(help="")

parser.add_argument("--sd_unet_path", type=str, default=None)
parser.add_argument("--text_encoder_path", type=str, default=None)
parser.add_argument(
    "--lora_adapters_paths", type=str, action="append", default=[]
)
parser.add_argument(
    "--fp16",
    type=int,
    default=1,
    help="whether to inference a diffusion model in fp16",
)
parser.add_argument("--prompt", type=str, required=True, help="e.g. a cat")
parser.add_argument("--output_name", type=str, default="simple.png")
parser.add_argument("--exp_desc", type=str, default="")
parser.add_argument("--num_inference_steps", type=int, default=25)
parser.add_argument("--guidance_scale", type=float, default=7.5)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--model_name", type=str, default="Manojb/stable-diffusion-2-1-base"
)
parser.add_argument("--width", type=int, default=512)
parser.add_argument("--height", type=int, default=512)
parser.add_argument("--save_dir", type=str, default="./diffusion_play_output/")
parser.add_argument(
    "--num_same", type=int, default=9, help="how many images to generate"
)
args = parser.parse_args()

torch_device = "cuda"
os.makedirs(args.save_dir, exist_ok=True)
print(args.save_dir)

guidance = StableDiffusion(
    device=torch_device,
    hf_key=args.model_name,
    height=args.height,
    width=args.width,
    fp16=args.fp16,
    sd_unet_path=args.sd_unet_path,
    lora_adapters_paths=args.lora_adapters_paths,
    text_encoder_path=args.text_encoder_path,
)

images = guidance.generate_images_by_prompts(
    [args.prompt],
    num_same=args.num_same,
    num_inference_steps=args.num_inference_steps,
    guidance_scale=args.guidance_scale,
    seed=args.seed,
)

print(f"Save at: {args.save_dir}")

guidance.save_images(
    images,
    save_name=args.output_name,
    prompt=args.prompt,
    exp_desc=args.exp_desc,
    save_dir=args.save_dir,
)
