"""
Gaussian Splatting + Generative Object Insertion
"""

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

from pyparsing import col
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

from nerfstudio.utils import writer
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import RGB2SH, get_viewmat
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE

import cv2
import numpy as np

try:
    from gsplat.rendering import rasterization
except ImportError:
    print("Please install gsplat>=1.0.0")
from gsplat.cuda_legacy._wrapper import num_sh_bases


# -----------------------------------------------------------

from d3dr.dn_model import DNSplatterModelConfig, DNSplatterModel
from d3dr.utils.read_and_rotate import read_from_file_and_rotate

# -----------------------------------------------------------

# diffusion
from d3dr.guidance_controller import GuidanceController

# -----------------------------------------------------------

@dataclass
class D3DRModelConfig(DNSplatterModelConfig):
    _target: Type = field(default_factory=lambda: D3DRModel)

    init_scene_path: Optional[str] = None
    """Path to the checkpoint to load the initial scene from"""

    init_obj_path: Optional[str] = None
    """Path to the checkpoint to load the initial object from"""

    transforms_obj: Optional[str] = None
    """Path to the json file with the transforms for the object"""

    sd_version: str = "2.1"
    """Version of the Stable Diffusion model to use"""

    sd_unet_path: Optional[str] = None
    """Path to the checkpoint of the UNet model to use (if use personalization)"""

    lora_object_path: Optional[str] = None
    """Path to the checkpoint of the LoRA parameters (for object personalization)"""

    lora_object_texture_path: Optional[str] = None
    """Path to the checkpoint of the LoRA parameters specifically for texture preservation"""

    conv_in_path: Optional[str] = None
    """Path to the checkpoint of the new convolution (for object personalization)"""
    
    lora_scene_path: Optional[str] = None
    """Path to the checkpoint of the LoRA parameters (for scene personalization)"""

    sd_width: int = 512
    """Width of the images used in Stable Diffusion model"""
    
    sd_height: int = 512
    """Height of the images used in Stable Diffusion model"""

    controlnet_model_name: str = "thibaud/controlnet-sd21-depth-diffusers"
    """Name of the controlnet model to use"""

    use_controlnet: int = 1
    """Whether to use ControlNet in diffusion"""

    mask_grad: bool = True
    """Whether to mask the gradient"""

    prompt_desired: str = ""
    """Desired prompt"""

    prompt_obj: str = ""
    """A prompt describing the object, should contain the token used in DreamBooth personalization"""

    prompt_initial: str = ""
    """Initial prompt"""

    use_weights_sds: bool = False
    """Whether to use weights in SDS"""

    mean_init: bool = True
    """LEGACY: Whether to initialize the object with mean"""  

    guidance_scale: float = 10.0
    """Scale for the classifier free guidance"""

    controlnet_conditioning_scale: float = 1.0
    """Scale of the controlnet conditioning (like classifier free guidance)"""

    fp16: bool = True
    """Whether to use fp16 for Diffusion Models"""

    now_debug: bool = True
    """Debug mode"""

    optimize_latent_for: int = 4
    """Number of latent steps (for 2-step-DDS) """

    optimize_image_for: int = 64
    """Number of image steps (for 2-step-DDS) """

    optimize_image_for_refine: int = 512
    """
    For how many steps the GS parameters are being optimized after SDEdit 
    is performed on `num_together_images_refine` images
    """

    num_together_images: int = 8
    """How many images are used simultaneously during optimization"""

    num_together_images_refine: int = 16
    """How many images are used simultaneously during the refinement phase"""

    lr_latent: float = 1e-1
    """Learning rate for latent optimization"""

    mask_const: float = 0.5
    """A constant to binarize the mask. The float mask values above this become 1.0"""

    t_range: Tuple[float, float] = (0.02, 0.5)
    """Range of timestep values for the diffusion model used for DDS"""

    t_range_refine: Tuple[float, float] = (0.02, 0.5)
    """Range of timestep values for the diffusion model during the SDEdit (refinement)"""

    refine_range_linear: int = -1
    """
    Linearly decrease the steps during the refinement phase in t_range_refine;
    if > 0 then it represents the number of refinement steps.
    """

    refine_after: int = 2000
    """Use SDEdit refinement instead of 2D DDS after this step"""

    prob_mvc: float = 0.0
    """Make images produced by Diffusion Model multiview consistent with this probability"""

    not_visible_const: float = 0.5
    """
    During the optimization the object is rendered along with its mask. Some pixels might 
    be hidden because of the scene. The `mask_real` is the real mask of the object 
    (without taking into account scene pixels), while `mask_obj` is the rendered mask of the object.
    If `mask_real` differs significantly from `mask_obj` then we treat that the object is not 
    visible and try another camera pose..
    """

    refine_obj: int = 1
    """Whether refine only obj during refining phase and detach other parameters such as shadows."""

    scene_shadows: Literal["none", "div", "sub", "sub-part"] = "none"
    """Whether to model scene shadows"""

    keep_scene_shadows_every: int = 500
    """Keep only the darkest gaussians in scene shadows"""

    keep_scene_shadows_top: float = 0.15
    """Keep only top 15% of darkest gaussians and make others white"""

    obj_initialization: Literal["none", "mean", "kmeans", "random"] = "mean"
    """How to initialize object"""

    crop_ratio: float = 0.7
    """Crop ratio during the refinement phase"""
    

class D3DRModel(DNSplatterModel):
    """Depth + Normal splatter"""

    config: D3DRModelConfig
    guidance_controller: GuidanceController

    def populate_modules(self):
        # load modules for the parent class
        DNSplatterModel.populate_modules(self)

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
            self.bb_min = means[self.obj_ids].amin(dim=0)
            self.bb_max = means[self.obj_ids].amax(dim=0)
        else:
            raise RuntimeError("means not found in the init_gauss_params")
        CONSOLE.log(f"Number of initial points {means.shape[0]}")
        num_points = means.shape[0]

        if (
            init_gauss_params.get('features_dc', None) is not None and \
            init_gauss_params.get('features_rest', None) is not None
        ):  
            features_dc_init = init_gauss_params['features_dc']
            features_rest_init = init_gauss_params['features_rest']
            self.orig_obj_dc = features_dc_init[self.obj_ids].detach().clone()
            if self.config.obj_initialization == "mean":
                features_dc_init[self.obj_ids] = features_dc_init[self.obj_ids].mean(dim=0, keepdim=True).repeat(sum(self.obj_ids), 1)
                features_rest_init[self.obj_ids] = 0.0
            elif self.config.obj_initialization == "kmeans":
                raise NotImplementedError("kmeans not implemented")
            elif self.config.obj_initialization == "random":
                features_dc_init[self.obj_ids] = torch.rand_like(features_dc_init[self.obj_ids]).abs()
                features_rest_init[self.obj_ids] = 0.0
                features_dc_init[self.obj_ids] = RGB2SH(features_dc_init[self.obj_ids])
            
            features_dc = torch.nn.Parameter(features_dc_init)
            features_rest = torch.nn.Parameter(features_rest_init)
            CONSOLE.log("Loaded features_dc, features_rest successfully!")
        else:
            raise RuntimeError("features_dc or features_rest not found in the init_gauss_params")

        if not init_gauss_params.get('opacities', None) is None:
            opacities = torch.nn.Parameter(init_gauss_params['opacities'])
            CONSOLE.log("Loaded opacities successfully!")
        else:    
            raise RuntimeError("The `opacities` not found in the init_gauss_params")

        if not init_gauss_params.get('normals', None) is None:
            normals = torch.nn.Parameter(init_gauss_params['normals'])
            CONSOLE.log("Loaded normals successfully!")
        else:
            CONSOLE.log("The `normals` not found in the init_gauss_params, initialiaze randomly")
            normals = torch.nn.Parameter(torch.randn(num_points, 3))
    
        if not init_gauss_params.get('scales', None) is None:
            scales = torch.nn.Parameter(init_gauss_params['scales'])
            CONSOLE.log("Loaded scales successfully!")
        else:
            raise RuntimeError("scales not found in the init_gauss_params")

        if not init_gauss_params.get('quats', None) is None:
            quats = torch.nn.Parameter(init_gauss_params['quats'])
            CONSOLE.log("Loaded quats successfully!")
        else:
            raise RuntimeError("quats not found in the init_gauss_params")

        # a new optimizable parameter that controls scene shadows
        scene_shadows = torch.ones_like(features_dc[~self.obj_ids][..., 0:1]) * 1e-2
        # take closest scene gaussians instead of all
        self.mask_close_object = None
        if "part" in self.config.scene_shadows:
            self.mask_close_object = self._closest_gaussians(means[~self.obj_ids])
            scene_shadows = scene_shadows[self.mask_close_object]
        scene_shadows = torch.nn.Parameter(scene_shadows)

        self.gauss_params = torch.nn.ParameterDict(
            {
                "means": means,
                "scales": scales,
                "quats": quats,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
                "normals": normals,
                "scene_shadows": scene_shadows,
            }
        )

        self.guidance_controller = GuidanceController(config=self.config)

    @torch.no_grad()
    def _closest_gaussians(self, scene_gaussians):
        bb_center = (self.bb_min + self.bb_max) / 2.0
        # increase in 3 times
        bb_min = (bb_center) + (self.bb_min - bb_center) * 3.0
        bb_max = (bb_center) + (self.bb_max - bb_center) * 3.0
        
        mask = (scene_gaussians >= bb_min) & (scene_gaussians <= bb_max)
        mask = mask.all(dim=1)
        return mask

    @property
    def scene_shadows(self):
        return self.gauss_params["scene_shadows"]
    
    @torch.no_grad()
    def keep_darkest_shadows(self):
        """Keep only the darkes shadows alive and simply remove all the others"""
        top_value = torch.quantile(self.scene_shadows, 1.0 - self.config.keep_scene_shadows_top)
        mask = (self.scene_shadows < top_value)
        new_tensor = self.scene_shadows.clone()
        new_tensor[mask] = torch.full_like(new_tensor[mask], 1e-5)
        self.scene_shadows.data = new_tensor

    def get_scene_shadow_colors(self, obj_scene_colors: Tensor) -> Tensor:
        """
        The `colors_all` is a combination of self.features_dc and self.features_rest. 
        In our case the shape is usually [N_scene + N_obj, 15, 3]. 
        This function changes the features_dc component of scene gaussians by applying the 
        self.scene_shadows parameter using the chosen strategy. 
        """
        obj_colors_all = obj_scene_colors[self.obj_ids]
        scene_colors_all = obj_scene_colors[~self.obj_ids]

        scene_fc = scene_colors_all[:, 0, :] # [N_scene, 3]
        scene_rest = scene_colors_all[:, 1:, :]
        
        # All of the scene_shadows coefficients should participate in 
        # gradient computation
        self.scene_shadows.data = self.scene_shadows.data.clip(1e-5, 1.0)
        scene_shadows = self.scene_shadows.clip(0, 1)
        if self.config.scene_shadows == "none":
            pass
        elif self.config.scene_shadows == "div":
            scene_fc = scene_fc * (1.0 - scene_shadows)
        elif self.config.scene_shadows == "sub":
            scene_fc = scene_fc - scene_shadows
        elif self.config.scene_shadows == "sub-part":
            scene_fc[self.mask_close_object] = scene_fc[self.mask_close_object] - scene_shadows
        
        scene_colors_all = torch.cat([scene_fc[:, None, :], scene_rest], dim=1)
        colors_all_new = torch.cat((scene_colors_all, obj_colors_all), dim=0)
        return colors_all_new

    def detach_parameters(self, obj_scene_params: Tensor, how: str = "scene") -> Tensor:
        """
        :param obj_scene_parameters: Parameters (colors, opacities and etc) of the object and the scene
        :param detach_how: How to detach the parameters
            - scene: detach the scene parameters
            - obj: detach the object parameters
            - both: detach both
            - none: do not detach
        """
        detach_scene = how in ["scene", "both"]
        detach_obj = how in ["obj", "both"]

        return torch.cat([
            obj_scene_params[~self.obj_ids].detach() if detach_scene else obj_scene_params[~self.obj_ids],
            obj_scene_params[self.obj_ids].detach() if detach_obj else obj_scene_params[self.obj_ids],
        ], axis=0)

    def get_loss_dict(
        self, outputs, batch, metrics_dict=None
    ) -> Dict[str, torch.Tensor]:
        # we do not need to calculate PSNR, SSIM, LPIPS for optimization anymore?
        # we need to use SDS/DDS right now
        optimized_image = batch["optimized_image"]

        # [1, 800, 800, 3] -> [1, 3, 800, 800]
        rgb_obj_scene = outputs["rgb"].unsqueeze(0).permute(0, 3, 1, 2)
        rgb_obj_scene_sdhw = F.interpolate(
            rgb_obj_scene, 
            size=(self.config.sd_height, self.config.sd_width), 
            mode="bilinear"
        )
        # maybe save the images
        if self.config.now_debug:
            if self.step % (self.config.optimize_image_for // 2) == 0:
                writer.put_image(name="train/rgb_obj_scene", image=rgb_obj_scene_sdhw[0].permute(1, 2, 0), step=self.step)
                
            if (self.step + 5) % self.config.optimize_image_for == 0:
                all_optimized_images = self.guidance_controller.get_all_images()
                img = torch.cat(all_optimized_images, dim=3)
                img = img[0].permute(1, 2, 0)
                img = (img.clip(0, 1) * 255.0).to(torch.uint8)
                writer.put_image(name=f"train/optimized_image", image=img, step=self.step)
                

        loss_lx = torch.abs(optimized_image - rgb_obj_scene_sdhw).mean() # loss_l1
        # loss_lx = F.mse_loss(optimized_image, rgb_obj_scene_sdhw)
        simloss = 1 - self.ssim(optimized_image, rgb_obj_scene_sdhw)

        main_loss = \
            (1 - self.config.ssim_lambda) * loss_lx + \
            self.config.ssim_lambda * simloss
        
        
        return {
            "main_loss": main_loss,
        }

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        """
        The only optimization parameters in our case are: features_dc, features_rest 
        and scene_shadows.
        """
        return {
            name: [self.gauss_params[name]]
            for name in [
                "features_dc",
                "features_rest",
                "scene_shadows",
            ]
        }

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        return {}
    
    def load_state_dict(self, dict, **kwargs):  # type: ignore
        for name in self.gauss_params.keys():
            self.gauss_params[name].data = dict[f'gauss_params.{name}'].to(self.device)
            CONSOLE.log(f"Successfully loaded {name} from ckpt")
    
    def __get_outputs_helper(
            self,
            means,
            quats,
            scales,
            opacities,
            colors,
            sh_degree_to_use,
            rasterization_kwargs,
            background,
            need_rgb=False,
            need_depth=False,
            normalized=False,
        ):

        if need_rgb and need_depth:
            render_mode = "RGB+ED"
        elif need_rgb:
            render_mode = "RGB"
        elif need_depth:
            render_mode = "ED"
        else:
            return None, None, None
        
        if not normalized:
            scales = torch.exp(self.scales)
            opacities = torch.sigmoid(self.opacities)
            quats = self.quats / self.quats.norm(dim=-1, keepdim=True)

        render, alpha, _ = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities.squeeze(-1),
            colors=colors,
            render_mode=render_mode,
            sh_degree=sh_degree_to_use,
            **rasterization_kwargs,
        )

        rgb = None
        if render_mode in ["RGB", "RGB+ED"]:
            rgb = render[:, ..., :3] + (1 - alpha) * background
            rgb = torch.clamp(rgb, 0.0, 1.0).squeeze(0)

        depth_im = None
        if render_mode in ["RGB+ED"]:
            depth_im = render[:, ..., 3:4]
        elif render_mode in ["ED"]:
            depth_im = render[:, ..., 0:1]
        
        if render_mode in ["RGB+ED", "ED"]:
            depth_im = torch.where(
                alpha > 0, depth_im, depth_im.detach().max()
            ).squeeze(0)
            depth_im = depth_im.detach()

        mask = (alpha > 0).to(torch.float32).squeeze(0)
        mask = mask.detach()

        return rgb, depth_im, mask

    def get_outputs(
            self, 
            camera, 
            required_params=(
                "rgb_obj", "rgb_scene", "rgb_obj_scene", "rgb_obj_orig",
                "depth_obj", "depth_obj_scene", "depth_scene",
                "mask_obj_real", 
            ),
            detach_scene_shadows=False,
        ):
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            rgb_obj: FloatTensor[H, W, 3]
                ONLY Rendered object

            rgb_obj_orig: FloatTensor[H, W, 3]
                ONLY Rendered object with original colors

            rgb_scene: FloatTensor[H, W, 3] 
                ONLY Rendered scene

            rgb_obj_scene: FloatTensor[H, W, 3] 
                Rendered object + scene

            depth_obj: FloatTensor[H, W, 1] 
                Rendered depth object

            depth: FloatTensor[H, W, 1] 
                Rendered depth object + scene (due to floaters the depth is combined from object_depth and scene_depth)

            depth_worse: FloatTensor[H, W, 1] 
                Rendered depth object + scene (do not take into account scene floaters)

            depth_scene: FloatTensor[H, W, 1]
                Rendered depth scene

            mask_obj_real: FloatTensor[H, W, 1]
                Rendered mask object (might be hidden because of scene gaussians floaters)
                1 - object
                0 - not object

            mask_obj: FloatTensor[H, W, 1]
                Rendered mask object (do not take into account scene floaters)
                1 - object
                0 - not object
        """

        if isinstance(required_params, str):
            required_params = (required_params,)
        required_params = set(required_params)

        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        if self.training:
            assert camera.shape[0] == 1, "Only one camera at a time"
            optimized_camera_to_world = self.camera_optimizer.apply_to_camera(camera)
        else:
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

        colors_all = torch.cat(
            (self.features_dc[:, None, :], self.features_rest), dim=1
        )

        if self.config.sh_degree > 0:
            sh_degree_to_use = self.config.sh_degree
            # and detach some features_rest which are not used
            sh_degree_to_optimize = min(
                self.step // self.config.sh_degree_interval, self.config.sh_degree
            )
            if sh_degree_to_optimize != sh_degree_to_use:
                ids_to_optimize = num_sh_bases(sh_degree_to_optimize) + 1
                colors_all = torch.cat([
                    colors_all[:, :ids_to_optimize, :],
                    colors_all[:, ids_to_optimize:, :].detach()
                ], dim=1)
        else:
            colors_all = torch.sigmoid(colors_all)
            sh_degree_to_use = None

        # We do not need normals. But the depth might be useful.
        common_rasterization_kwargs = dict(
            viewmats=viewmat,  # [1, 4, 4]
            Ks=K,  # [1, 3, 3]
            width=W,
            height=H,
            tile_size=BLOCK_WIDTH,
            packed=False,
            near_plane=0.01,
            far_plane=1e10,
            sparse_grad=False,
            absgrad=False,
            rasterize_mode=self.config.rasterize_mode,
            # set some threshold to disregrad small gaussians for faster rendering.
            # radius_clip=3.0,
        )

        background = self._get_background_color()

        scales_exp = torch.exp(self.scales)
        opacities_sigmoid = torch.sigmoid(self.opacities)
        quats_norm = self.quats / self.quats.norm(dim=-1, keepdim=True)

        # 0. Calculate real object's mask
        mask_obj_real = None
        if set(["mask_obj_real", "depth_obj_scene"]) & required_params:
            with torch.no_grad():
                colors_mask_obj_scene = torch.zeros_like(self.features_dc)
                colors_mask_obj_scene[self.obj_ids] = 1.0

                mask_obj_real, _, _ = self.__get_outputs_helper(
                    means=self.means,
                    quats=quats_norm,
                    scales=scales_exp,
                    opacities=opacities_sigmoid,
                    colors=colors_mask_obj_scene,
                    sh_degree_to_use=None,
                    rasterization_kwargs=common_rasterization_kwargs,
                    background=background,
                    need_rgb=True,
                    need_depth=False,
                    normalized=True,
                )

                mask_obj_real = (mask_obj_real[..., 0:1] > self.config.mask_const).to(torch.float32)

        rgb_obj_orig = None
        if ("rgb_obj_orig") in required_params:
            with torch.no_grad():
                rgb_obj_orig, _, _ = self.__get_outputs_helper(
                    means=self.means[self.obj_ids],
                    quats=quats_norm[self.obj_ids],
                    scales=scales_exp[self.obj_ids],
                    opacities=opacities_sigmoid[self.obj_ids],
                    colors=torch.sigmoid(self.orig_obj_dc),
                    sh_degree_to_use=None,
                    rasterization_kwargs=common_rasterization_kwargs,
                    background=background,
                    need_rgb=("rgb_obj" in required_params),
                    need_depth=set(["depth_obj", "depth_obj_scene"]) & required_params,
                    normalized=True,
                )

        # 1. Calculate object's colors and masks
        rgb_obj = None
        depth_im_obj = None
        mask_obj = None
        if set(["rgb_obj", "depth_obj", "depth_obj_scene"]) & required_params:
            rgb_obj, depth_im_obj, mask_obj = self.__get_outputs_helper(
                means=self.means[self.obj_ids],
                quats=quats_norm[self.obj_ids],
                scales=scales_exp[self.obj_ids],
                opacities=opacities_sigmoid[self.obj_ids],
                colors=colors_all[self.obj_ids],
                sh_degree_to_use=sh_degree_to_use,
                rasterization_kwargs=common_rasterization_kwargs,
                background=background,
                need_rgb=("rgb_obj" in required_params),
                need_depth=set(["depth_obj", "depth_obj_scene"]) & required_params,
                normalized=True,
            )

        # 2. Get the prediction of the scene
        rgb_scene = None
        depth_im_scene = None
        if set(["rgb_scene", "depth_scene", "depth_obj_scene"]) & required_params:
            with torch.no_grad():
                rgb_scene, depth_im_scene, _ = self.__get_outputs_helper(
                    means=self.means[~self.obj_ids],
                    quats=quats_norm[~self.obj_ids],
                    scales=scales_exp[~self.obj_ids],
                    opacities=opacities_sigmoid[~self.obj_ids],
                    colors=colors_all[~self.obj_ids],
                    sh_degree_to_use=sh_degree_to_use,
                    rasterization_kwargs=common_rasterization_kwargs,
                    background=background,
                    need_rgb=("rgb_scene" in required_params),
                    need_depth=set(["depth_scene", "depth_obj_scene"]) & required_params,
                    normalized=True,
                )

        # 3. Get the prediction of the scene + object
        rgb_obj_scene = None
        depth_im_obj_scene = None
        depth_im_obj_scene_better = None
        # detach scene (shadows parameters will be applied later)
        # then apply shadows parameters
        if detach_scene_shadows: 
            colors_all_shadows = self.detach_parameters(self.get_scene_shadow_colors(colors_all), how="scene")
        else:
            colors_all_shadows = self.get_scene_shadow_colors(self.detach_parameters(colors_all, how="scene"))
        if set(["rgb_obj_scene", "depth_obj_scene"]) & required_params:
            rgb_obj_scene, depth_im_obj_scene, _ = self.__get_outputs_helper(
                means=self.means.detach(),
                quats=quats_norm.detach(),
                scales=scales_exp.detach(),
                opacities=opacities_sigmoid.detach(), 
                colors=colors_all_shadows,
                sh_degree_to_use=sh_degree_to_use,
                rasterization_kwargs=common_rasterization_kwargs,
                background=background,
                need_rgb=("rgb_obj_scene" in required_params),
                need_depth=("depth_obj_scene" in required_params),
                normalized=True,
            )

            # combine object's depth with scene's depth
            if "depth_obj_scene" in required_params:
                depth_im_obj_scene_better = mask_obj_real * depth_im_obj + (1 - mask_obj_real) * depth_im_scene

        result = {
            "rgb_obj_scene": rgb_obj_scene,
            "rgb_obj_orig": rgb_obj_orig,
            "rgb_obj": rgb_obj,
            "rgb_scene": rgb_scene,

            "depth": depth_im_obj_scene_better,
            "depth_worse": depth_im_obj_scene,
            "depth_im_scene": depth_im_scene,
            "depth_im_obj": depth_im_obj,
            
            "mask_obj": mask_obj,
            "mask_obj_real": mask_obj_real,
        }

        return result
    
    @torch.no_grad()
    def _psnr_masked(self, x, y, mask):
        if torch.any(mask):
            x = x.permute(2, 3, 1, 0) # [1, C, H, W] -> [H, W, C, 1]
            y = y.permute(2, 3, 1, 0)
            mse = ((x - y)[mask] ** 2).sum() / mask.float().sum()
            return -10.0 * torch.log10(mse)
        else:
            raise RuntimeError("Mask is empty! in _masked_mse")

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """

        gt_rgb = self.get_gt_img(batch["image"])
        gt_mask = self.get_gt_img(batch["mask"])
        gt_mask_01 = (gt_mask > self.config.mask_const)
        x, y, w, h = cv2.boundingRect(gt_mask_01.squeeze().cpu().numpy().astype(np.uint8))

        predicted_rgb = outputs["rgb_obj_scene"]
        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        cropped_gt_rgb = gt_rgb[:, :, y:y+h, x:x+w]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]
        cropped_predicted_rgb = predicted_rgb[:, :, y:y+h, x:x+w]
        
        psnr = self.psnr(gt_rgb, predicted_rgb)
        psnr_cropped = self.psnr(cropped_gt_rgb, cropped_predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        psnr_part = self._psnr_masked(gt_rgb, predicted_rgb, gt_mask_01)
        
        near_object_mask = torch.zeros_like(gt_mask_01)
        psnr_shadows = self._psnr_masked(gt_rgb, predicted_rgb, ~gt_mask_01)
        try:
            # print(f"cropped pred shape: {cropped_predicted_rgb.shape}")
            ssim_part = self.ssim(cropped_gt_rgb, cropped_predicted_rgb)
        except Exception as e:
            print(f"[WARNING] Failed to compute SSIM: {e}")
            return {}, {"img": combined_rgb}

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)
        metrics_dict["psnr_part"] = float(psnr_part.item())
        metrics_dict["psnr_cropped"] = float(psnr_cropped.item())
        metrics_dict["ssim_part"] = float(ssim_part)
        metrics_dict["psnr_shadows"] = float(psnr_shadows.item())

        images_dict = {"img": combined_rgb}
        
        return metrics_dict, images_dict

# -----------------------------------------------------------
