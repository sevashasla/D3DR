'''
This file is used to extract image features from CLIP model for real images.
'''

from transformers import CLIPTextModel, CLIPTokenizer

import os
from pathlib import Path
import pickle
import cv2
import numpy as np

from transformers import CLIPProcessor, CLIPModel
from PIL import Image as PILImage
import json

import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@torch.no_grad()
def batched_call(model, processor, images, batch_size=16):
    images_processed = processor(images=images, return_tensors="pt")
    image_features = []
    for i in tqdm(range(0, len(images), batch_size), desc="Extracting image features (batched)"):
        batch = {k: v[i:i+batch_size].to(device) for k, v in images_processed.items()}
        image_features.append(model.get_image_features(**batch).cpu())
    image_features = torch.cat(image_features)
    return image_features

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="openai/clip-vit-base-patch32", help="Pretrained model name or path")
    parser.add_argument("--prompt_before", type=str, required=True, help="Prompt before the change")
    parser.add_argument("--prompt_after", type=str, required=True, help="Prompt after the change")
    parser.add_argument("--images_dir_before", type=str, required=True, help="Directory of images before the change")
    parser.add_argument("--images_dir_after", type=str, required=True, help="Directory of images after the change")
    parser.add_argument("--masks_dir", type=str, required=True, help="Directory of masks")
    parser.add_argument("--store_path", type=str, required=True, help="A file to save the output files")

    args = parser.parse_args()
    return args

def main():
    args = get_args()

    model = CLIPModel.from_pretrained(args.model_name)
    model = model.to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(args.model_name)

    processed_texts = processor(text=[args.prompt_before, args.prompt_after], padding=True, return_tensors="pt")

    images_before = [PILImage.open(str(f)) for f in sorted(list(Path(args.images_dir_before).glob("*.png")))]
    print("images_before", len(images_before))
    images_after = [PILImage.open(str(f)) for f in sorted(list(Path(args.images_dir_after).glob("*")))]
    print("images_after", len(images_after))
    images_masks = [cv2.imread(str(f), cv2.IMREAD_GRAYSCALE).astype(np.float32) for f in sorted(list(Path(args.masks_dir).glob("*")))]
    images_ids_good = [i for i, m in enumerate(images_masks) if np.sum(m > 0.1) > 150]
    # TODO: bad fix
    images_ids_good = [i for i in images_ids_good if (i < len(images_after) and i < len(images_ids_good))]
    # leave only images with good masks
    images_before = [images_before[i] for i in images_ids_good]
    images_after = [images_after[i] for i in images_ids_good]

    # get features for images
    image_features_before = batched_call(model, processor, images_before)
    image_features_after = batched_call(model, processor, images_after)

    # get features for text
    with torch.no_grad():
        processed_texts = {k: v.to(device) for k, v in processed_texts.items()}
        text_features = model.get_text_features(**processed_texts).cpu()

    # normalize features
    image_features_before = F.normalize(image_features_before, dim=-1) # [N, 512]
    image_features_after = F.normalize(image_features_after, dim=-1) # [N, 512]
    text_features = F.normalize(text_features, dim=-1) # [2, 512]

    # scores to 0-1
    CTIS_score = (image_features_after @ text_features[1].reshape(-1, 1))
    CTIS_score = CTIS_score.mean().item() * 0.5 + 0.5
    DTIS_score = (image_features_after - image_features_before) @ (text_features[1] - text_features[0]).reshape(-1, 1)
    DTIS_score = DTIS_score.mean().item() * 0.5 + 0.5

    with open(args.store_path, "w") as f: 
        json.dump({
            "CTIS": CTIS_score, 
            "DTIS": DTIS_score
        }, f)

if __name__ == "__main__":
    main()

