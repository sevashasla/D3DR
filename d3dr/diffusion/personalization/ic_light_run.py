"""
Inference IC-Light on an image or on a set of images.

The code is taken from:
1. https://github.com/lllyasviel/IC-Light/blob/main/gradio_demo_bg.py
2. https://github.com/lllyasviel/IC-Light/blob/main/briarmbg.py
3. https://github.com/lllyasviel/IC-Light/issues/117
Thanks @lllyasviel for the code!
"""

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import safetensors.torch as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import (
    AutoencoderKL,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from huggingface_hub import PyTorchModelHubMixin
from PIL import Image
from torch.hub import download_url_to_file
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_store_path",
    type=str,
    default="./checkpoints/iclight_sd15_fbc.safetensors",
)
parser.add_argument("--input_fg_path", type=str, default=None)
parser.add_argument("--mask_fg_path", type=str, default=None)
parser.add_argument("--input_bg_path", type=str, default=None)
parser.add_argument(
    "--position_light",
    type=str,
    default=None,
    help="if not None then will use this",
)
parser.add_argument("--store_dir", type=str, default="./iclight_exps/")
parser.add_argument("--output_path", type=str, default=None)
parser.add_argument("--prompt", type=str, default=None, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--a_prompt", type=str, default="best quality")
parser.add_argument(
    "--n_prompt",
    type=str,
    default="lowres, bad anatomy, bad hands, cropped, worst quality",
)
parser.add_argument("--cfg", type=float, default=7.0)
parser.add_argument("--highres_scale", type=float, default=1.5)
parser.add_argument("--highres_denoise", type=float, default=0.5)
parser.add_argument("--lowres_denoise", type=float, default=0.9)
parser.add_argument("--num_samples", type=int, default=1)

parser.add_argument("--process_dir", action="store_true")
parser.add_argument("--input_fg_dir", type=str, default=None)
parser.add_argument("--input_bg_dir", type=str, default=None)
parser.add_argument("--bg_desc_json", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)
parser.add_argument("--prompts_dir", type=str, default=None)
parser.add_argument("--num_images", type=int, default=10)
parser.add_argument("--fixed_place_prompt", type=str, default=None)

args = parser.parse_args()

USE_POSITION = args.position_light is not None
print("USE_POSITION:", USE_POSITION)

if USE_POSITION:
    print("Change parameters to default for use_position")
    args.model_store_path = "./checkpoints/iclight_sd15_fc.safetensors"
    args.steps = 25
    args.cfg = 2.0


class REBNCONV(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dirate=1, stride=1):
        super(REBNCONV, self).__init__()

        self.conv_s1 = nn.Conv2d(
            in_ch,
            out_ch,
            3,
            padding=1 * dirate,
            dilation=1 * dirate,
            stride=stride,
        )
        self.bn_s1 = nn.BatchNorm2d(out_ch)
        self.relu_s1 = nn.ReLU(inplace=True)

    def forward(self, x):
        hx = x
        xout = self.relu_s1(self.bn_s1(self.conv_s1(hx)))

        return xout


def _upsample_like(src, tar):
    src = F.interpolate(src, size=tar.shape[2:], mode="bilinear")
    return src


### RSU-7 ###
class RSU7(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3, img_size=512):
        super(RSU7, self).__init__()

        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)  ## 1 -> 1/2

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        b, c, h, w = x.shape

        hx = x
        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)
        hx = self.pool5(hx5)

        hx6 = self.rebnconv6(hx)

        hx7 = self.rebnconv7(hx6)

        hx6d = self.rebnconv6d(torch.cat((hx7, hx6), 1))
        hx6dup = _upsample_like(hx6d, hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6dup, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-6 ###
class RSU6(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU6, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)

        hx5d = self.rebnconv5d(torch.cat((hx6, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-5 ###
class RSU5(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU5, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4 ###
class RSU4(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=1)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup, hx1), 1))

        return hx1d + hxin


### RSU-4F ###
class RSU4F(nn.Module):
    def __init__(self, in_ch=3, mid_ch=12, out_ch=3):
        super(RSU4F, self).__init__()

        self.rebnconvin = REBNCONV(in_ch, out_ch, dirate=1)

        self.rebnconv1 = REBNCONV(out_ch, mid_ch, dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch, mid_ch, dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch, mid_ch, dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch, mid_ch, dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch * 2, mid_ch, dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch * 2, mid_ch, dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch * 2, out_ch, dirate=1)

    def forward(self, x):
        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4, hx3), 1))
        hx2d = self.rebnconv2d(torch.cat((hx3d, hx2), 1))
        hx1d = self.rebnconv1d(torch.cat((hx2d, hx1), 1))

        return hx1d + hxin


class myrebnconv(nn.Module):
    def __init__(
        self,
        in_ch=3,
        out_ch=1,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
    ):
        super(myrebnconv, self).__init__()

        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.rl = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.rl(self.bn(self.conv(x)))


class BriaRMBG(nn.Module, PyTorchModelHubMixin):
    def __init__(self, config: dict = {"in_ch": 3, "out_ch": 1}):
        super(BriaRMBG, self).__init__()
        in_ch = config["in_ch"]
        out_ch = config["out_ch"]
        self.conv_in = nn.Conv2d(in_ch, 64, 3, stride=2, padding=1)
        self.pool_in = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage1 = RSU7(64, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        # self.outconv = nn.Conv2d(6*out_ch,out_ch,1)

    def forward(self, x):
        hx = x

        hxin = self.conv_in(hx)
        # hx = self.pool_in(hxin)

        # stage 1
        hx1 = self.stage1(hxin)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)
        d1 = _upsample_like(d1, x)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, x)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, x)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, x)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, x)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, x)

        return [
            F.sigmoid(d1),
            F.sigmoid(d2),
            F.sigmoid(d3),
            F.sigmoid(d4),
            F.sigmoid(d5),
            F.sigmoid(d6),
        ], [hx1d, hx2d, hx3d, hx4d, hx5d, hx6]


def hooked_unet_forward(sample, timestep, encoder_hidden_states, **kwargs):
    c_concat = kwargs["cross_attention_kwargs"]["concat_conds"].to(sample)
    c_concat = torch.cat(
        [c_concat] * (sample.shape[0] // c_concat.shape[0]), dim=0
    )
    new_sample = torch.cat([sample, c_concat], dim=1)
    kwargs["cross_attention_kwargs"] = {}
    return unet_original_forward(
        new_sample, timestep, encoder_hidden_states, **kwargs
    )


@torch.inference_mode()
def encode_prompt_inner(txt: str):
    max_length = tokenizer.model_max_length
    chunk_length = tokenizer.model_max_length - 2
    id_start = tokenizer.bos_token_id
    id_end = tokenizer.eos_token_id
    id_pad = id_end

    def pad(x, p, i):
        return x[:i] if len(x) >= i else x + [p] * (i - len(x))

    tokens = tokenizer(txt, truncation=False, add_special_tokens=False)[
        "input_ids"
    ]
    chunks = [
        [id_start] + tokens[i : i + chunk_length] + [id_end]
        for i in range(0, len(tokens), chunk_length)
    ]
    chunks = [pad(ck, id_pad, max_length) for ck in chunks]

    token_ids = torch.tensor(chunks).to(device=device, dtype=torch.int64)
    conds = text_encoder(token_ids).last_hidden_state

    return conds


@torch.inference_mode()
def encode_prompt_pair(positive_prompt, negative_prompt):
    c = encode_prompt_inner(positive_prompt)
    uc = encode_prompt_inner(negative_prompt)

    c_len = float(len(c))
    uc_len = float(len(uc))
    max_count = max(c_len, uc_len)
    c_repeat = int(math.ceil(max_count / c_len))
    uc_repeat = int(math.ceil(max_count / uc_len))
    max_chunk = max(len(c), len(uc))

    c = torch.cat([c] * c_repeat, dim=0)[:max_chunk]
    uc = torch.cat([uc] * uc_repeat, dim=0)[:max_chunk]

    c = torch.cat([p[None, ...] for p in c], dim=1)
    uc = torch.cat([p[None, ...] for p in uc], dim=1)

    return c, uc


@torch.inference_mode()
def pytorch2numpy(imgs, quant=True):
    results = []
    for x in imgs:
        y = x.movedim(0, -1)

        if quant:
            y = y * 127.5 + 127.5
            y = y.detach().float().cpu().numpy().clip(0, 255).astype(np.uint8)
        else:
            y = y * 0.5 + 0.5
            y = y.detach().float().cpu().numpy().clip(0, 1).astype(np.float32)

        results.append(y)
    return results


@torch.inference_mode()
def numpy2pytorch(imgs):
    h = (
        torch.from_numpy(np.stack(imgs, axis=0)).float() / 127.0 - 1.0
    )  # so that 127 must be strictly 0.0
    h = h.movedim(-1, 1)
    return h


def resize_and_center_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    original_width, original_height = pil_image.size
    scale_factor = max(
        target_width / original_width, target_height / original_height
    )
    resized_width = int(round(original_width * scale_factor))
    resized_height = int(round(original_height * scale_factor))
    resized_image = pil_image.resize(
        (resized_width, resized_height), Image.LANCZOS
    )
    left = (resized_width - target_width) / 2
    top = (resized_height - target_height) / 2
    right = (resized_width + target_width) / 2
    bottom = (resized_height + target_height) / 2
    cropped_image = resized_image.crop((left, top, right, bottom))
    return np.array(cropped_image)


def resize_without_crop(image, target_width, target_height):
    pil_image = Image.fromarray(image)
    resized_image = pil_image.resize(
        (target_width, target_height), Image.LANCZOS
    )
    return np.array(resized_image)


@torch.inference_mode()
def run_rmbg(img, sigma=0.0):
    H, W, C = img.shape
    assert C == 3
    k = (256.0 / float(H * W)) ** 0.5
    feed = resize_without_crop(
        img, int(64 * round(W * k)), int(64 * round(H * k))
    )
    feed = numpy2pytorch([feed]).to(device=device, dtype=torch.float32)
    alpha = rmbg(feed)[0][0]
    alpha = torch.nn.functional.interpolate(alpha, size=(H, W), mode="bilinear")
    alpha = alpha.movedim(1, -1)[0]
    alpha = alpha.detach().float().cpu().numpy().clip(0, 1)
    result = 127 + (img.astype(np.float32) - 127 + sigma) * alpha
    return result.clip(0, 255).astype(np.uint8), alpha


# 'stablediffusionapi/realistic-vision-v51'
# 'runwayml/stable-diffusion-v1-5'
sd15_name = "stablediffusionapi/realistic-vision-v51"
tokenizer = CLIPTokenizer.from_pretrained(sd15_name, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    sd15_name, subfolder="text_encoder"
)
vae = AutoencoderKL.from_pretrained(sd15_name, subfolder="vae")
unet = UNet2DConditionModel.from_pretrained(sd15_name, subfolder="unet")
rmbg = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

# Change UNet

with torch.no_grad():
    new_conv_in = torch.nn.Conv2d(
        8 if USE_POSITION else 12,
        unet.conv_in.out_channels,
        unet.conv_in.kernel_size,
        unet.conv_in.stride,
        unet.conv_in.padding,
    )
    new_conv_in.weight.zero_()
    new_conv_in.weight[:, :4, :, :].copy_(unet.conv_in.weight)
    new_conv_in.bias = unet.conv_in.bias
    unet.conv_in = new_conv_in

unet_original_forward = unet.forward

unet.forward = hooked_unet_forward

# Load

model_path = args.model_store_path
if USE_POSITION:
    MODEL_NAME = "iclight_sd15_fc.safetensors"
else:
    MODEL_NAME = "iclight_sd15_fbc.safetensors"
assert os.path.basename(model_path) == MODEL_NAME

if not os.path.exists(model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    download_url_to_file(
        url=f"https://huggingface.co/lllyasviel/ic-light/resolve/main/{MODEL_NAME}",
        dst=model_path,
    )

sd_offset = sf.load_file(model_path)
sd_origin = unet.state_dict()
keys = sd_origin.keys()
sd_merged = {k: sd_origin[k] + sd_offset[k] for k in sd_origin.keys()}
unet.load_state_dict(sd_merged, strict=True)
del sd_offset, sd_origin, sd_merged, keys

# Device

device = torch.device("cuda")
text_encoder = text_encoder.to(device=device, dtype=torch.float16)
vae = vae.to(device=device, dtype=torch.bfloat16)
unet = unet.to(device=device, dtype=torch.float16)
rmbg = rmbg.to(device=device, dtype=torch.float32)

# SDP

unet.set_attn_processor(AttnProcessor2_0())
vae.set_attn_processor(AttnProcessor2_0())

# Samplers

ddim_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)

euler_a_scheduler = EulerAncestralDiscreteScheduler(
    num_train_timesteps=1000, beta_start=0.00085, beta_end=0.012, steps_offset=1
)

dpmpp_2m_sde_karras_scheduler = DPMSolverMultistepScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    algorithm_type="sde-dpmsolver++",
    use_karras_sigmas=True,
    steps_offset=1,
)

# Pipelines

t2i_pipe = StableDiffusionPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None,
)

i2i_pipe = StableDiffusionImg2ImgPipeline(
    vae=vae,
    text_encoder=text_encoder,
    tokenizer=tokenizer,
    unet=unet,
    scheduler=dpmpp_2m_sde_karras_scheduler,
    safety_checker=None,
    requires_safety_checker=False,
    feature_extractor=None,
    image_encoder=None,
)


@torch.inference_mode()
def process(
    input_fg,
    input_bg,
    prompt,
    image_width,
    image_height,
    num_samples,
    seed,
    steps,
    a_prompt,
    n_prompt,
    cfg,
    highres_scale,
    highres_denoise,
):

    rng = torch.Generator(device=device).manual_seed(seed)

    fg = resize_and_center_crop(input_fg, image_width, image_height)
    bg = resize_and_center_crop(input_bg, image_width, image_height)
    concat_conds = numpy2pytorch([fg, bg]).to(
        device=vae.device, dtype=vae.dtype
    )
    concat_conds = (
        vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    )
    concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    conds, unconds = encode_prompt_pair(
        positive_prompt=prompt + ", " + a_prompt, negative_prompt=n_prompt
    )

    latents = (
        t2i_pipe(
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=steps,
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type="latent",
            guidance_scale=cfg,
            cross_attention_kwargs={"concat_conds": concat_conds},
        ).images.to(vae.dtype)
        / vae.config.scaling_factor
    )

    pixels = vae.decode(latents).sample

    pixels = pytorch2numpy(pixels)
    pixels = [
        resize_without_crop(
            image=p,
            target_width=int(round(image_width * highres_scale / 64.0) * 64),
            target_height=int(round(image_height * highres_scale / 64.0) * 64),
        )
        for p in pixels
    ]

    pixels = numpy2pytorch(pixels).to(device=vae.device, dtype=vae.dtype)
    latents = vae.encode(pixels).latent_dist.mode() * vae.config.scaling_factor
    latents = latents.to(device=unet.device, dtype=unet.dtype)

    image_height, image_width = latents.shape[2] * 8, latents.shape[3] * 8
    fg = resize_and_center_crop(input_fg, image_width, image_height)
    bg = resize_and_center_crop(input_bg, image_width, image_height)
    concat_conds = numpy2pytorch([fg, bg]).to(
        device=vae.device, dtype=vae.dtype
    )
    concat_conds = (
        vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    )
    concat_conds = torch.cat([c[None, ...] for c in concat_conds], dim=1)

    latents = (
        i2i_pipe(
            image=latents,
            strength=highres_denoise,
            prompt_embeds=conds,
            negative_prompt_embeds=unconds,
            width=image_width,
            height=image_height,
            num_inference_steps=int(round(steps / highres_denoise)),
            num_images_per_prompt=num_samples,
            generator=rng,
            output_type="latent",
            guidance_scale=cfg,
            cross_attention_kwargs={"concat_conds": concat_conds},
        ).images.to(vae.dtype)
        / vae.config.scaling_factor
    )

    pixels = vae.decode(latents).sample
    pixels = pytorch2numpy(pixels, quant=False)

    return pixels, [fg, bg]


@torch.inference_mode()
def process_relight(
    input_fg,
    input_bg,
    prompt,
    image_width,
    image_height,
    num_samples,
    seed,
    steps,
    a_prompt,
    n_prompt,
    cfg,
    highres_scale,
    highres_denoise,
):
    input_fg, matting = run_rmbg(input_fg)

    results, extra_images = process(
        input_fg,
        input_bg,
        prompt,
        image_width,
        image_height,
        num_samples,
        seed,
        steps,
        a_prompt,
        n_prompt,
        cfg,
        highres_scale,
        highres_denoise,
    )

    results = [(x * 255.0).clip(0, 255).astype(np.uint8) for x in results]
    return results + extra_images


@torch.inference_mode()
def process_position(
    input_fg,
    prompt,
    image_width,
    image_height,
    num_samples,
    seed,
    steps,
    a_prompt,
    n_prompt,
    cfg,
    highres_scale,
    highres_denoise,
    lowres_denoise,
    bg_source,
):
    print("process position")
    if bg_source == "left":
        gradient = np.linspace(255, 0, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == "right":
        gradient = np.linspace(0, 255, image_width)
        image = np.tile(gradient, (image_height, 1))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == "top":
        gradient = np.linspace(255, 0, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    elif bg_source == "bottom":
        gradient = np.linspace(0, 255, image_height)[:, None]
        image = np.tile(gradient, (1, image_width))
        input_bg = np.stack((image,) * 3, axis=-1).astype(np.uint8)
    else:
        raise "Wrong initial latent!"

    rng = torch.Generator(device=device).manual_seed(int(seed))

    fg = resize_and_center_crop(input_fg, image_width, image_height)

    concat_conds = numpy2pytorch([fg]).to(device=vae.device, dtype=vae.dtype)
    concat_conds = (
        vae.encode(concat_conds).latent_dist.mode() * vae.config.scaling_factor
    )

    conds, unconds = encode_prompt_pair(
        positive_prompt=prompt + ", " + a_prompt, negative_prompt=n_prompt
    )

    if input_bg is None:
        latents = (
            t2i_pipe(
                prompt_embeds=conds,
                negative_prompt_embeds=unconds,
                width=image_width,
                height=image_height,
                num_inference_steps=steps,
                num_images_per_prompt=num_samples,
                generator=rng,
                output_type="latent",
                guidance_scale=cfg,
                cross_attention_kwargs={"concat_conds": concat_conds},
            ).images.to(vae.dtype)
            / vae.config.scaling_factor
        )
    else:
        bg = resize_and_center_crop(input_bg, image_width, image_height)
        bg_latent = numpy2pytorch([bg]).to(device=vae.device, dtype=vae.dtype)
        bg_latent = (
            vae.encode(bg_latent).latent_dist.mode() * vae.config.scaling_factor
        )
        latents = (
            i2i_pipe(
                image=bg_latent,
                strength=lowres_denoise,
                prompt_embeds=conds,
                negative_prompt_embeds=unconds,
                width=image_width,
                height=image_height,
                num_inference_steps=int(round(steps / lowres_denoise)),
                num_images_per_prompt=num_samples,
                generator=rng,
                output_type="latent",
                guidance_scale=cfg,
                cross_attention_kwargs={"concat_conds": concat_conds},
            ).images.to(vae.dtype)
            / vae.config.scaling_factor
        )

    pixels = vae.decode(latents).sample
    return pytorch2numpy(pixels)


@torch.inference_mode()
def process_relight_position(
    input_fg,
    prompt,
    image_width,
    image_height,
    num_samples,
    seed,
    steps,
    a_prompt,
    n_prompt,
    cfg,
    highres_scale,
    highres_denoise,
    lowres_denoise,
    bg_source,
):
    input_fg, matting = run_rmbg(input_fg)
    results = process_position(
        input_fg,
        prompt,
        image_width,
        image_height,
        num_samples,
        seed,
        steps,
        a_prompt,
        n_prompt,
        cfg,
        highres_scale,
        highres_denoise,
        lowres_denoise,
        bg_source,
    )
    return results


def adjust_dimensions(width, height, max_dim=1024, divisible_by=8):
    """
    Adjust width and height to maintain the original aspect ratio,
    cap at max_dim, and make them divisible by a specified value.
    """
    # Calculate aspect ratio
    aspect_ratio = width / height

    # Determine scaling factor to cap at max_dim
    if width > height:
        scaled_width = min(width, max_dim)
        scaled_height = scaled_width / aspect_ratio
    else:
        scaled_height = min(height, max_dim)
        scaled_width = scaled_height * aspect_ratio

    # Ensure divisibility by the specified value
    scaled_width = int((scaled_width // divisible_by) * divisible_by)
    scaled_height = int((scaled_height // divisible_by) * divisible_by)

    return scaled_width, scaled_height


def process_one_image(
    input_fg_path,
    input_bg_path,
    prompt,
    steps=20,
    a_prompt="best quality",
    n_prompt="lowres, bad anatomy, bad hands, cropped, worst quality",
    cfg=7.0,
    highres_scale=1.5,
    highres_denoise=0.5,
    num_samples=1,
    seed=42,
    position_light=None,
    lowres_denoise=0.9,
    mask_fg_path=None,
):
    # Load input images
    input_fg = np.array(Image.open(input_fg_path).convert("RGB"))
    if mask_fg_path is not None:
        mask_fg = np.array(Image.open(mask_fg_path))
        mask_fg_01 = (mask_fg > 64).astype(np.uint8)[..., 0:1]
        input_fg = input_fg * mask_fg_01
        input_fg = 127 + (input_fg.astype(np.float32) - 127) * mask_fg_01
        input_fg = input_fg.clip(0, 255).astype(np.uint8)

    image_height, image_width = input_fg.shape[:2]
    image_width, image_height = adjust_dimensions(
        image_width, image_height, max_dim=1024, divisible_by=8
    )
    print(f"Adjusted dimensions: {image_width} x {image_height}")

    if position_light is not None:
        print("Run position light!")
        results = process_relight_position(
            input_fg=input_fg,
            prompt=prompt,
            image_width=image_width,
            image_height=image_height,
            num_samples=num_samples,
            seed=seed,
            steps=steps,
            a_prompt=a_prompt,
            n_prompt=n_prompt,
            cfg=cfg,
            highres_scale=highres_scale,
            highres_denoise=highres_denoise,
            lowres_denoise=lowres_denoise,
            bg_source=position_light,
        )

    else:
        input_bg = np.array(Image.open(input_bg_path).convert("RGB"))

        # Process and save the result
        results = process_relight(
            input_fg=input_fg,
            input_bg=input_bg,
            prompt=prompt,
            image_width=image_width,
            image_height=image_height,
            num_samples=num_samples,
            seed=seed,
            steps=steps,
            a_prompt=a_prompt,
            n_prompt=n_prompt,
            cfg=cfg,
            highres_scale=highres_scale,
            highres_denoise=highres_denoise,
        )

    return results[0]


def main():

    common_args = dict(
        steps=args.steps,
        a_prompt=args.a_prompt,
        n_prompt=args.n_prompt,
        cfg=args.cfg,
        highres_scale=args.highres_scale,
        highres_denoise=args.highres_denoise,
        num_samples=args.num_samples,
        seed=args.seed,
        lowres_denoise=args.lowres_denoise,
    )

    torch.manual_seed(args.seed)

    # https://github.com/Lightning-AI/litgpt/issues/327
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

    np.random.seed(args.seed)
    if args.process_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        if args.input_fg_dir is None:
            all_fg_names = [args.input_fg_path]
        else:
            all_fg_names = sorted(os.listdir(args.input_fg_dir))
            all_fg_names = [
                os.path.join(args.input_fg_dir, img_name)
                for img_name in all_fg_names
            ]
        random_ids = np.random.choice(len(all_fg_names), args.num_images)
        # random_ids = [np.random.choice(len(all_fg_names), 1).item()] * args.num_images
        input_fg = [all_fg_names[i] for i in random_ids]

        if USE_POSITION:
            all_position_lights = np.random.choice(
                ["left", "right", "top", "bottom"], args.num_images
            ).tolist()
            if args.fixed_place_prompt is None:
                with open(args.prompts_dir) as f:
                    all_prompts = json.load(f)
            else:
                all_prompts = [args.fixed_place_prompt] * args.num_images

            random_ids = np.random.choice(len(all_prompts), args.num_images)
            space_prompts = [all_prompts[i] for i in random_ids]

            for i, fg, pl in tqdm(
                zip(range(args.num_images), input_fg, all_position_lights),
                total=args.num_images,
            ):
                common_args["seed"] = args.seed + i
                results = process_one_image(
                    input_fg_path=fg,
                    input_bg_path=None,
                    position_light=pl,
                    prompt=args.prompt + f" in {space_prompts[i]}",
                    **common_args,
                )
                Image.fromarray(results).save(
                    os.path.join(args.output_dir, f"{i:05}.png")
                )

            with open(os.path.join(args.output_dir, "prompts.json"), "w") as f:
                prompts = [
                    {
                        "prompt": args.prompt + f" in {space_prompts[i]}",
                        "fg_name": Path(input_fg[i]).name,
                        "position_light": all_position_lights[i],
                        "result_image": f"{i:05}.png",
                    }
                    for i in range(args.num_images)
                ]
                json.dump(prompts, f, indent=4)

        else:
            all_bg_names = sorted(
                [
                    f
                    for f in os.listdir(args.input_bg_dir)
                    if f.endswith(".png")
                    or f.endswith(".jpg")
                    or f.endswith(".webp")
                ]
            )
            random_ids = np.random.choice(len(all_bg_names), args.num_images)
            input_bg_names = [all_bg_names[i] for i in random_ids]
            input_bg = [
                os.path.join(args.input_bg_dir, img_name)
                for img_name in input_bg_names
            ]

            for i, fg, bg in tqdm(
                zip(range(args.num_images), input_fg, input_bg),
                total=args.num_images,
            ):
                results = process_one_image(
                    input_fg_path=fg,
                    input_bg_path=bg,
                    prompt=args.prompt,
                    position_light=None,
                    **common_args,
                )
                Image.fromarray(results).save(
                    os.path.join(args.output_dir, f"{i:05}.png")
                )

            with open(args.bg_desc_json) as f:
                bg_desc = json.load(f)

            with open(os.path.join(args.output_dir, "prompts.json"), "w") as f:
                prompts = [
                    {
                        "prompt": f"{args.prompt} in {bg_desc[input_bg_names[i]]}",
                        "fg_name": Path(input_fg[i]).name,
                        "bg_name": input_bg_names[i],
                        "result_image": f"{i:05}.png",
                    }
                    for i in range(args.num_images)
                ]
                json.dump(prompts, f)

    else:
        results = process_one_image(
            input_fg_path=args.input_fg_path,
            input_bg_path=args.input_bg_path,
            prompt=args.prompt,
            position_light=args.position_light,
            mask_fg_path=args.mask_fg_path,
            **common_args,
        )
        if args.output_path is None:
            num_files = len(os.listdir(args.store_dir))
            args.output_path = os.path.join(
                args.store_dir, f"output_{num_files:03}.png"
            )
        Image.fromarray(results).save(args.output_path)


if __name__ == "__main__":
    main()
