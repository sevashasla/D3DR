from pathlib import Path

import numpy as np
import torch
from dataset_utils import enrich_dataset
from PIL import Image
from torchvision import transforms as T


class ImageDataset(torch.utils.data.Dataset):
    """
    A dataset for Rough Diffusion Personalization.
    """
    def __init__(
            self, 
            image_dir,
            width, height,
            orig_obj_prompt=None,
            ic_light_prompt=None,
            save_dir=None,
            generate_using_iclight=True,
            generate_cp_iclight=None,
            generate_n=32,
            generate_obj_similar=32,
            generate_cp_dir=None,
            prob=0.9,
            with_preservation=None,
            add_raw_images_ratio=0.0,
            fixed_place_prompt=None
        ):
        """
        Creates the dataset and enriches `image_dir` with more IC-Light generated images.
        """

        self.save_dir, self.save_iclight_dir, self.save_preservation_dir = enrich_dataset(
            image_dir=image_dir,
            save_dir=save_dir,
            
            generate_iclight_n=generate_n,
            iclight_prompt=ic_light_prompt,
            just_cp_iclight_dir=generate_cp_iclight,
            do_generate_iclight=(generate_using_iclight != 0),
            add_raw_images_ratio=add_raw_images_ratio,

            generate_preservation_n=generate_obj_similar,
            preservation_prompt=orig_obj_prompt,
            just_cp_preservation_dir=generate_cp_dir,

            prob=prob,
            with_preservation=with_preservation,
            fixed_place_prompt=fixed_place_prompt,
        )

        self.prob = prob
        self.image_dir = Path(image_dir)
        self.width = width
        self.height = height

        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize((self.width, self.height)),
        ])

        self.images_iclight = self.read_all_images(self.save_iclight_dir)
        self.images_preservation = self.read_all_images(self.save_preservation_dir)

    def read_all_images(self, d, exts=("png", "jpg")):
        d = Path(d)
        paths = []
        for ext in exts:
            paths.extend(d.glob(f"*.{ext}"))
        paths.sort()
        images = []
        for p in paths:
            images.append(Image.open(p))
        return images

    def __len__(self):
        return len(self.images_iclight)

    def __getitem__(self, idx):
        """
        Returns either object image (probability self.prob) or preservation image
        (probability 1.0 - self.prob).
        """
        if np.random.rand() < self.prob:
            image = self.images_iclight[idx]
            emb_idx = torch.tensor([1])
        else:
            random_idx = np.random.choice(len(self.images_preservation))
            image = self.images_preservation[random_idx]
            emb_idx = torch.tensor([0])
        image = self.transform(image)
        return image, emb_idx
