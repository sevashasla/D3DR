"""
Code borrowed from: https://github.com/yes-its-shivam/gaussian_splat_rotation
And from here: https://github.com/graphdeco-inria/gaussian-splatting/issues/176
"""

import torch
import torch.nn.functional as F

from e3nn import o3

# Code borrowed from: pytorch3d
# https://github.com/facebookresearch/pytorch3d/blob/81d82980bc82fd605f27cca87f89ba08af94db3d/pytorch3d/transforms/rotation_conversions.py#L107
# --------------------------------------------------------

def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    if torch.is_grad_enabled():
        ret[positive_mask] = torch.sqrt(x[positive_mask])
    else:
        ret = torch.where(positive_mask, torch.sqrt(x), ret)
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)
    out = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return standardize_quaternion(out)

# --------------------------------------------------------

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    # pyre-fixme[58]: `/` is not supported for operand types `float` and `Tensor`.
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

# --------------------------------------------------------

def quat_multiply(quaternion0, quaternion1):
    w0, x0, y0, z0 = torch.split(quaternion0, 1, dim=-1)
    w1, x1, y1, z1 = torch.split(quaternion1, 1, dim=-1)
    return torch.concat((
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
    ), dim=-1)

def transform_shs(shs_feat, rotation_matrix, comp_device="cuda"):
    ## rotate shs
    shs_feat = shs_feat.to(comp_device)
    P = torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0]]).float().to(comp_device) # switch axes: yzx -> xyz
    permuted_rotation_matrix = torch.linalg.inv(P) @ rotation_matrix.to(comp_device) @ P
    rot_angles = o3._rotation.matrix_to_angles(permuted_rotation_matrix)
    rot_angles = [r.cpu() for r in rot_angles]
    # Construction coefficient
    D_1 = o3.wigner_D(1, rot_angles[0], -rot_angles[1], rot_angles[2]).to(comp_device)
    D_2 = o3.wigner_D(2, rot_angles[0], -rot_angles[1], rot_angles[2]).to(comp_device)
    D_3 = o3.wigner_D(3, rot_angles[0], -rot_angles[1], rot_angles[2]).to(comp_device)

    # rotation of the shs features
    shs_feat[:, 0:3] = D_1 @ shs_feat[:, 0:3]
    shs_feat[:, 3:8] = D_2 @ shs_feat[:, 3:8]
    shs_feat[:, 8:15] = D_3 @ shs_feat[:, 8:15]
    return shs_feat

def rotate_quats(quats, rotation_matrix, comp_device="cuda"):
    quats_new = torch.nn.functional.normalize(quat_multiply(
        quats, 
        torch.tensor(matrix_to_quaternion(rotation_matrix), device=comp_device, dtype=torch.float),
    ))
    return quats_new

# --------------------------------------------------------

@torch.no_grad()
def affine_transform_dn_splats(gauss_params, rotation_matrix, movement_vector, comp_device="cuda"):
    '''
    Performs affine rotation of gaussian splatting coefficients (if necessary). 

    features_dc -> remain the same
    features_rest
    means
    normals
    opacities   -> remain the same
    quats
    scales      -> remain the same
    ''' 
    # place on devices
    if rotation_matrix.device != comp_device:
        rotation_matrix = rotation_matrix.to(comp_device)
    if movement_vector.device != comp_device:
        movement_vector = movement_vector.to(comp_device)
    
    gauss_params_device = gauss_params["means"].device

    features_rest = gauss_params['features_rest'].to(comp_device)
    means = gauss_params['means'].to(comp_device)
    if "normals" in gauss_params:
        normals = gauss_params['normals'].to(comp_device)
    quats = gauss_params['quats'].to(comp_device)

    # rotate shs
    features_rest_new = transform_shs(features_rest, rotation_matrix, comp_device=comp_device)
    gauss_params['features_rest'] = features_rest_new.to(gauss_params_device)

    # rotate means
    means_new = means @ rotation_matrix.t() + movement_vector
    gauss_params['means'] = means_new.to(gauss_params_device)

    # rotate normals
    if "normals" in gauss_params:
        normals_new = normals @ rotation_matrix.t()
        gauss_params['normals'] = normals_new.to(gauss_params_device)

    # rotate quats
    quats_new = torch.nn.functional.normalize(quat_multiply(
        quats, 
        torch.tensor(matrix_to_quaternion(rotation_matrix), device=comp_device, dtype=torch.float),
    ))
    gauss_params['quats'] = quats_new.to(gauss_params_device)

    return gauss_params
