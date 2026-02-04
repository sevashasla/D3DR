import cv2
import torch


def center_crop_to_square(cv2_image):
    h, w = cv2_image.shape[:2]
    if h > w:
        start = (h - w) // 2
        return cv2_image[start: h - start, ...]
    elif w > h:
        start = (w - h) // 2
        return cv2_image[:, start: w - start, ...]
    else:
        return cv2_image
    
def center_crop_to_square_with_ratio(cv2_image, crop_ratio=1.0):
    h, w = cv2_image.shape[:2]
    result_image = cv2_image
    if h > w:
        start = (h - w) // 2
        result_image = cv2_image[start: h - start, ...]
    elif w > h:
        start = (w - h) // 2
        result_image = cv2_image[:, start: w - start, ...]
    
    wh_orig = result_image.shape[0] 
    wh_small = int(wh_orig * crop_ratio)

    result_image = result_image[
        (wh_orig//2 - wh_small//2): (wh_orig//2 - wh_small//2) + wh_small,
        (wh_orig//2 - wh_small//2): (wh_orig//2 - wh_small//2) + wh_small,
        ...
    ]
    
    return result_image

def masked_mean(x, mask, dim=None):
    if dim is None:
        dim = tuple(range(0, len(x.shape)))
    return torch.sum(x * mask, dim=dim) / torch.sum(mask, dim=dim)

@torch.no_grad()
def read_image(image_path, height, width, guidance=None, mask=None, crop_ratio=1.0):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = center_crop_to_square_with_ratio(image, crop_ratio)
    image = cv2.resize(image, (height, width))[None, ...] # (1, C, H, W)
    image_torch = guidance.np2torch(image)
    if mask is not None:
        image_torch = image_torch * mask
    latent = guidance.torch2latents(image_torch)
    return image, image_torch, latent
