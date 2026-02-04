import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Optional
from gsplat import spherical_harmonics
from d3dr.utils.rotate_splats import load_gauss_params

MAGIC_CONST = 0.2820947917738781

@torch.no_grad()
def average_sh_by_dir(
    degrees_to_use: int,
    coeffs: Tensor,  # [..., K, 3]
    masks: Optional[Tensor] = None,
    num_dirs: int = 128,
):
    '''Find an average color by averaging on the sphere.

    Args:
        degrees_to_use: The degree to be used.
        dirs: Directions. [..., 3]
        coeffs: Coefficients. [..., K, 3]
        masks: Optional boolen masks to skip some computation. [...,] Default: None.
        num_dirs: Number of directions to be used. Default: 128.

    Returns:
        Spherical harmonics. [..., 3]
    '''

    # generate directions
    # dirs: Tensor,  # [..., 3]
    device = coeffs.device
    N = coeffs.shape[:-2]
    uniform_rand_1 = torch.rand(size=(*N, num_dirs,), device=device)
    uniform_rand_2 = torch.rand(size=(*N, num_dirs,), device=device)
    theta = 2 * torch.pi * uniform_rand_1
    phi = torch.arccos(1 - 2 * uniform_rand_2)

    dirs = torch.stack(
        [
            torch.sin(phi) * torch.cos(theta),
            torch.sin(phi) * torch.sin(theta),
            torch.cos(phi),
        ],
        dim=-1,
    ) # [*N, num_dirs, 3]
    print(dirs.norm(dim=-1))

    coeffs_big = coeffs.unsqueeze(-3).expand(*N, num_dirs, *coeffs.shape[-2:])
    print("dirs", dirs.shape)
    print("coeffs_big", coeffs_big.shape)

    if masks is not None:
        masks = masks.unsqueeze(-1).expand(*masks.shape, num_dirs) 

    # compute SH
    colors = spherical_harmonics(
        degrees_to_use,
        dirs,
        coeffs_big,
        masks=masks,
    )

    # need to divide by this coeff
    colors = colors.mean(dim=-2) / MAGIC_CONST
    return colors
    