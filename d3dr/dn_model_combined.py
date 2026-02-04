"""
Depth + normal splatter for combined gaussian splatting - scene and object
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Type

import torch
from torch.nn import Parameter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from d3dr.losses import DepthLoss, DepthLossType, TVLoss
from d3dr.metrics import DepthMetrics, NormalMetrics, RGBMetrics
from d3dr.regularization_strategy import (
    DNRegularization,
)

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")
from gsplat.cuda_legacy._wrapper import num_sh_bases
from nerfstudio.cameras.camera_optimizers import CameraOptimizer
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.models.splatfacto import (
    RGB2SH,
    get_viewmat,
)
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE

# -----------------------------------------------------------

from d3dr.dn_model import DNSplatterModelConfig, DNSplatterModel
from d3dr.utils.read_and_rotate import read_from_file_and_rotate

# -----------------------------------------------------------

@dataclass
class DNSplatterModelCombinedConfig(DNSplatterModelConfig):
    _target: Type = field(default_factory=lambda: DNSplatterModelCombined)

    regularization_strategy: Literal["dn-splatter"] = "dn-splatter"

    init_scene_path: Optional[str] = None
    """Path to the checkpoint to load the initial scene from"""

    init_obj_path: Optional[str] = None
    """Path to the checkpoint to load the initial object from"""

    transforms_obj: Optional[str] = None
    """Path to the json file with the transforms for the object"""

    mean_init: bool = False
    """Initialize the object with the mean of the object. Usually better for convergence"""

class DNSplatterModelCombined(DNSplatterModel):
    """Depth + Normal splatter for combined scene + object"""

    config: DNSplatterModelCombinedConfig

    def populate_modules(self):
        # TODO: Can be rewritten using super().populate_modules()

        # load gauss parameters from checkpoints
        init_gauss_params = read_from_file_and_rotate(
            self.config.init_obj_path, 
            self.config.init_scene_path,
            self.config.transforms_obj,
        )

        if init_gauss_params.get("obj_ids", None) is not None:
            self.obj_ids = init_gauss_params["obj_ids"]

        if init_gauss_params.get('means', None) is not None:
            means = torch.nn.Parameter(init_gauss_params['means'])
        elif self.seed_points is not None and not self.config.random_init:
            means = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
        else:
            means = torch.nn.Parameter((torch.rand((500000, 3)) - 0.5) * 10)
        CONSOLE.log(f"Number of initial seed points {means.shape[0]}")
        self.xys_grad_norm = None
        self.max_2Dsize = None
        dim_sh = num_sh_bases(self.config.sh_degree)
        num_points = means.shape[0]

        if (
            init_gauss_params.get('features_dc', None) is not None and \
            init_gauss_params.get('features_rest', None) is not None
        ):
            features_dc_init = init_gauss_params['features_dc']
            features_rest_init = init_gauss_params['features_rest']
            if self.config.mean_init:
                features_dc_init[self.obj_ids] = features_dc_init[self.obj_ids].mean(dim=0, keepdim=True).repeat(sum(self.obj_ids), 1)
                features_rest_init[self.obj_ids] = 0.0
            
            features_dc = torch.nn.Parameter(features_dc_init)
            features_rest = torch.nn.Parameter(features_rest_init)

        elif self.seed_points is not None and not self.config.random_init:
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :])
            features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            raise RuntimeError("features_dc or features_rest not found in the init_gauss_params")

        self.step = 0
        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)

        self.mse_loss = torch.nn.MSELoss()

        # Depth Losses
        if self.config.use_depth_loss:
            self.depth_loss = DepthLoss(self.config.depth_loss_type)
            assert self.config.depth_lambda > 0, "depth_lambda should be > 0"

        if self.config.use_depth_smooth_loss:
            if self.config.smooth_loss_type == DepthLossType.EdgeAwareTV:
                self.smooth_loss = DepthLoss(depth_loss_type=DepthLossType.EdgeAwareTV)
            else:
                self.smooth_loss = DepthLoss(depth_loss_type=DepthLossType.TV)

        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0, kernel_size=11)
        self.lpips = LearnedPerceptualImagePatchSimilarity()
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.rgb_metrics = RGBMetrics()
        self.depth_metrics = DepthMetrics()
        self.normal_metrics = NormalMetrics()

        if init_gauss_params.get('opacities', None) is not None:
            opacities = torch.nn.Parameter(init_gauss_params['opacities'])
        else:    
            raise RuntimeError("opacities not found in the init_gauss_params")
        
        if not init_gauss_params.get('scales', None) is None:
            scales = torch.nn.Parameter(init_gauss_params['scales'])
            print("Loaded scales")
        else:
            raise RuntimeError("scales not found in the init_gauss_params")
        
        if not init_gauss_params.get('quats', None) is None:
            quats = torch.nn.Parameter(init_gauss_params['quats'])
            print("Loaded quats")
        else:
            raise RuntimeError("quats not found in the init_gauss_params")
        
        if not init_gauss_params.get('normals', None) is None:
            normals = torch.nn.Parameter(init_gauss_params['normals'])
            print("Loaded normals")
        else:
            print("normals not found in the init_gauss_params, initialiaze random")
            normals = torch.nn.Parameter(torch.randn(num_points, 3))

        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
                "normals": normals,
            }
        )

        self.camera_idx = 0
        self.camera = None
        if self.config.use_normal_tv_loss:
            self.tv_loss = TVLoss()

        # camera optimizer
        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        if self.config.regularization_strategy == "dn-splatter":
            self.regularization_strategy = DNRegularization()
        else:
            raise NotImplementedError

        if self.config.use_depth_loss:
            self.regularization_strategy.depth_loss_type = self.config.depth_loss_type
            self.regularization_strategy.depth_loss = self.depth_loss
            self.regularization_strategy.depth_lambda = self.config.depth_lambda
        else:
            self.regularization_strategy.depth_loss_type = None
            self.regularization_strategy.depth_loss = None

        if not self.config.use_normal_loss:
            self.regularization_strategy.normal_loss = None

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in [
                "means",
                "scales",
                "quats",
                "features_dc",
                "features_rest",
                "opacities",
                "normals",
            ]
        }

    def get_outputs(self, camera):
        parent_outputs = DNSplatterModel.get_outputs(self, camera)
        
        # calculate object mask
        if not self.training:
            optimized_camera_to_world = camera.camera_to_worlds
            BLOCK_WIDTH = (
                16  # this controls the tile size of rasterization, 16 is a good default
            )
            camera_scale_fac = self._get_downscale_factor()
            camera.rescale_output_resolution(1 / camera_scale_fac)
            viewmat = get_viewmat(optimized_camera_to_world)
            K = camera.get_intrinsics_matrices().cuda()
            W, H = int(camera.width.item()), int(camera.height.item())
            self.last_size = (H, W)
            camera.rescale_output_resolution(camera_scale_fac)  # type: ignore
            render_mode = "RGB"

            means_obj = self.means[self.obj_ids]
            quats_obj = self.quats[self.obj_ids]
            scales_obj = self.scales[self.obj_ids]
            opacities_obj = torch.ones_like(self.opacities[self.obj_ids])
            colors_obj = torch.ones_like(self.colors[self.obj_ids])

            render, alpha, info = rasterization(
                means=means_obj,
                quats=quats_obj / quats_obj.norm(dim=-1, keepdim=True),
                scales=torch.exp(scales_obj),
                opacities=opacities_obj.squeeze(-1),
                colors=colors_obj,
                viewmats=viewmat,  # [1, 4, 4]
                Ks=K,  # [1, 3, 3]
                width=W,
                height=H,
                tile_size=BLOCK_WIDTH,
                packed=False,
                near_plane=0.01,
                far_plane=1e10,
                render_mode=render_mode,
                sh_degree=None,
                sparse_grad=False,
                absgrad=True,
                rasterize_mode=self.config.rasterize_mode,
                # set some threshold to disregrad small gaussians for faster rendering.
                # radius_clip=3.0,
            )

            mask = torch.clamp(render[:, ..., :3], 0.0, 1.0)
            parent_outputs['mask'] = mask.squeeze(0)
        return parent_outputs

# -----------------------------------------------------------
