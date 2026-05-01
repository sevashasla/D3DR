import argparse

import torch
from tqdm.auto import tqdm

from d3dr.diffusion.sd_utils import StableDiffusion


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="A cat")
    parser.add_argument("--exp_desc", type=str, default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--model_name", type=str, default="Manojb/stable-diffusion-2-1-base"
    )
    parser.add_argument("--sd_unet_path", type=str, default=None)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument(
        "--lora_adapters_paths", type=str, action="append", default=[]
    )

    parser.add_argument("--num_train_iterations", type=int, default=1000)
    parser.add_argument("--show_iter", type=int, default=100)
    parser.add_argument("--acc_step", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-1)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--save_dir", type=str, default="./sds/")
    parser.add_argument("--sds_step", type=float, default=0.1)
    parser.add_argument("--sds_step_num", type=int, default=16)
    parser.add_argument("--sds_full_step_num", type=int, default=16)
    parser.add_argument("--use_2_step_sds", type=int, default=1)

    parser.add_argument("--fp16", type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    torch_device = "cuda"

    guidance = StableDiffusion(
        device="cuda",
        sd_version=args.model_name,
        height=args.height,
        width=args.width,
        sd_unet_path=args.sd_unet_path,
        lora_adapters_paths=args.lora_adapters_paths,
        fp16=args.fp16,
    )

    torch.manual_seed(args.seed)

    sds_image = torch.randn(
        (1, 3, args.height, args.width),
        device=torch_device,
    )
    sds_image = sds_image.detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([sds_image], lr=args.lr)
    opt_scheduler = torch.optim.lr_scheduler.PolynomialLR(
        optimizer, total_iters=args.num_train_iterations, power=args.power
    )

    real_emb = guidance.get_text_embeds(args.prompt)
    uncond_emb = guidance.get_text_embeds("")
    text_embs = torch.cat([uncond_emb, real_emb])
    noise = None

    args.save_dir = guidance.get_save_dir(args.save_dir)
    print("Save dir:", args.save_dir)

    if args.use_2_step_sds != 0:
        # sum_steps = args.sds_step_num + args.sds_full_step_num
        sum_steps = args.sds_full_step_num
        for i in tqdm(range(args.num_train_iterations // sum_steps)):
            with torch.no_grad():
                latent_0 = guidance.torch2latents(sds_image)
                for j in range(args.sds_step_num):
                    grad = guidance.train_step(
                        text_embs,
                        latent_0,
                        guidance_scale=args.guidance_scale,
                        as_latent=True,
                        step_ratio=None,
                        noise=noise,
                        return_grad=True,
                    ).to(torch.float32)

                    latent_0 -= args.sds_step * grad
                sds_image_1 = guidance.latents2torch(latent_0)
                # sds_image.data = sds_image_1.data

            for j in range(args.sds_full_step_num):
                # zero_grad
                optimizer.zero_grad()

                loss = (
                    (sds_image_1 - sds_image).pow(2).sum(dim=(1, 2, 3)).mean()
                )
                loss.backward()

                optimizer.step()
                opt_scheduler.step()

            if ((i + 1) * sum_steps) // args.show_iter > (
                (i) * sum_steps
            ) // args.show_iter:
                result_image = guidance.torch2np(sds_image)
                guidance.save_images(
                    images=result_image,
                    save_name=f"sds_image_{i + 1}.png",
                    prompt=args.prompt,
                    save_dir=args.save_dir,
                    exp_desc=args.exp_desc,
                )
    else:
        for i in tqdm(range(args.num_train_iterations)):
            optimizer.zero_grad()
            latent_0 = guidance.torch2latents(sds_image)
            loss = guidance.train_step(
                text_embs,
                latent_0,
                guidance_scale=args.guidance_scale,
                as_latent=True,
                step_ratio=None,
                noise=noise,
                return_grad=False,
            ).to(torch.float32)
            loss.backward()
            optimizer.step()
            opt_scheduler.step()

            if (i + 1) % args.show_iter == 0:
                result_image = guidance.torch2np(sds_image)
                guidance.save_images(
                    images=result_image,
                    save_name=f"sds_image_{i + 1}.png",
                    prompt=args.prompt,
                    save_dir=args.save_dir,
                    exp_desc=args.exp_desc,
                )


if __name__ == "__main__":
    main()
