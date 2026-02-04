from PIL import Image
import os
import argparse
import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True)

args = parser.parse_args()
input_dir = args.input_dir

for filename in tqdm.tqdm(os.listdir(input_dir)):
    if filename.endswith(".png"):
        img = Image.open(os.path.join(input_dir, filename))
        # Convert to RGB (drops the alpha channel)
        rgb_img = img.convert("RGB")
        rgb_img.save(os.path.join(input_dir, filename))

