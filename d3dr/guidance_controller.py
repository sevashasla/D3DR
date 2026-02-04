import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from torch import Tensor

import os
from copy import deepcopy

# diffusion
from d3dr.diffusion import SDControlNet
from d3dr.utils.multi_view_consistency import MultiViewConsistency

class GuidanceController:
    def __init__(
        self,
        config,
    ):
        self.config = config

        self.refine_after = self.config.refine_after
        self.sd_height = self.config.sd_height
        self.sd_width = self.config.sd_width
        self.mask_const = self.config.mask_const

        self.optimize_latent_for = self.config.optimize_latent_for
        self.curr_optimize_image_for = self.config.optimize_image_for
        self.curr_num_together_images = self.config.num_together_images
        self.use_controlnet = self.config.use_controlnet

        self.lr_latent = self.config.lr_latent
        self.guidance_scale = self.config.guidance_scale
        self.controlnet_conditioning_scale = self.config.controlnet_conditioning_scale
        self.use_weights_sds = self.config.use_weights_sds
        self.not_visible_const = self.config.not_visible_const

        self.refine_obj = self.config.refine_obj

        lora_adapters_paths = [self.config.lora_object_path, self.config.lora_scene_path]
        lora_adapters_paths = [path for path in lora_adapters_paths if path is not None]
        self.lora_adapters_paths = lora_adapters_paths if len(lora_adapters_paths) > 0 else None
        self.mvc = MultiViewConsistency(device="cuda")

        # load the diffusion model
        self.guidance = SDControlNet(
            device="cuda",
            sd_version=self.config.sd_version,
            controlnet_name=self.config.controlnet_model_name,
            sd_unet_path=self.config.sd_unet_path,
            height=self.config.sd_height,
            width=self.config.sd_width,
            fp16=self.config.fp16,
            t_range=self.config.t_range,
            lora_adapters_paths=self.lora_adapters_paths,
            lora_object_texture_path=self.config.lora_object_texture_path,
            conv_in_path=self.config.conv_in_path,
        )

        # optimization variables
        self.optimized_images = []
        self.cameras = []
        self.crops = []
        self.num_images_used = 0
        self.masks = []
        self.depths = []

        # prompts
        self.prompt_desired = self.config.prompt_desired
        self.prompt_initial = self.config.prompt_initial
        self.prompt_obj = self.config.prompt_obj

        self.emb_uncond = self.guidance.get_text_embeds("")
        self.emb_initial = self.guidance.get_text_embeds(self.prompt_initial)
        self.emb_desired = self.guidance.get_text_embeds(self.prompt_desired)
        self.emb_obj = self.guidance.get_text_embeds(self.prompt_obj)

        self.text_embeddings_initial = torch.cat([self.emb_uncond, self.emb_initial]) 
        self.text_embeddings_desired = torch.cat([self.emb_uncond, self.emb_desired])
        
        self.is_lighting_phase = False
        self.is_refining_phase = False

    def need_new_cameras(self) -> bool:
        need_new_cameras_flag = (len(self.cameras) == 0)
        return need_new_cameras_flag
    
    @torch.no_grad()
    def check_visibility(self, outputs) -> bool:
        mask_obj = outputs["mask_obj"]
        mask_obj_real = outputs["mask_obj_real"]

        if (mask_obj_real * mask_obj).sum() / (mask_obj.sum() + 1e-7) < self.not_visible_const:
            return False
        else:
            return True
        
    def lighting_phase(self):
        """
        move guidance to lighting phase
        turn on multi view consistency
        """
        if self.is_lighting_phase:
            return
        self.is_lighting_phase = True
        self.is_refining_phase = False
        self.curr_prob_mvc = self.config.prob_mvc
        self.curr_optimize_image_for = self.config.optimize_image_for
        self.curr_t_range = self.config.t_range
        self.curr_num_together_images = self.config.num_together_images
        self.guidance.lighting_phase()
    
    def refining_phase(self):
        """
        move guidance to refining phase (load new adapters)
        turn off multi view consistency
        """
        if self.is_refining_phase:
            return
        self.is_lighting_phase = False
        self.is_refining_phase = True
        self.curr_prob_mvc = -1.0 # turn off MVC during refiningt_range
        self.guidance.refining_phase()
        self.curr_t_range = self.config.t_range_refine
        self.curr_optimize_image_for = self.config.optimize_image_for_refine
        self.curr_num_together_images = self.config.num_together_images_refine
        self._clean_up()

    @property
    def use_double_personalization(self):
        return self.guidance.use_double_personalization

    def add_new_camera(self, outputs, camera, step) -> bool:
        """
        Add a new camera to the current cameras
        :param outputs: outputs from the model
        :param camera: camera to add
        :param step: current step

        returns: whether to need to generate a new camera
        """

        if not self.check_visibility(outputs):
            return True
        
        optimized_image = self.optimize(outputs, step)
        self.optimized_images.append(optimized_image)
        self.masks.append(outputs["mask_obj_real"][..., 0])
        self.depths.append(outputs["depth_im_obj"][..., 0])
        self.cameras.append(camera)
        self.crops.append(outputs.get("crop_params", None))

        if len(self.cameras) >= self.curr_num_together_images:
            # make them consistent
            if torch.rand(1).item() < self.curr_prob_mvc:
                optimized_images_big = [F.interpolate(img, size=self.depths[0].shape[:2], mode="bilinear") for img in self.optimized_images]
                # (1, 3, H, W) -> (H, W, 3)
                optimized_images_big = [img[0].permute(1, 2, 0) for img in optimized_images_big]

                self.mvc.make_consistent(
                        optimized_images_big,
                        self.depths,
                        [(m > self.config.mask_const) for m in self.masks],
                        None,
                        self.cameras,
                        return_mesh=False,       
                )
                # (H, W, 3) -> (1, 3, H, W)
                optimized_images_big = [img.permute(2, 0, 1).unsqueeze(0) for img in optimized_images_big]
                # (1, 3, H, W) -> (1, 3, H*, W*)
                optimized_images_big = [F.interpolate(
                    img, 
                    size=self.optimized_images[0].shape[2:], 
                    mode="bilinear"
                ) for img in optimized_images_big]
                self.optimized_images = optimized_images_big

            return False
        return True
    
    def _clean_up(self) -> None:
        """
        Clean up the images and cameras
        """

        self.cameras = []
        self.crops = []
        self.optimized_images = []
        self.num_images_used = 0
        self.masks = []
        self.depths = []

    def next_train(self, step):
        """
        Get the next camera for training
        :param step: current step
        """

        return_idx = step % self.curr_num_together_images

        camera = self.cameras[return_idx]
        crop = self.crops[return_idx]
        optimized_image = self.optimized_images[return_idx]

        self.num_images_used += 1
        if self.num_images_used >= self.curr_optimize_image_for:
            self._clean_up()

        return camera, {"optimized_image": optimized_image, "crop_params": crop}

    @staticmethod
    def depth_to_01(depth: Tensor) -> Tensor:
        """
        Normalize the depth to [0, 1]

        :param depth: Depth tensor [B, C, H, W]
        """

        with torch.no_grad():
            mmax = depth.amax(dim=(1, 2, 3), keepdims=True)
            mmin = depth.amin(dim=(1, 2, 3), keepdims=True)

            # closer <=> higher value
            return (mmax - depth) / (mmax - mmin)

    @torch.no_grad()
    def _optimize_dds_2d(self, outputs):        
        # else optimize for a few steps
        # [800, 800, 1] -> [1, 1, 800, 800]
        mask_obj = outputs["mask_obj"][..., 0:1].unsqueeze(0).permute(0, 3, 1, 2)
        mask_obj = (mask_obj > self.mask_const).to(torch.float32)
        mask_obj_sdhw = F.interpolate(
            mask_obj, 
            size=(self.sd_height, self.sd_width), 
            mode="bilinear"
        )
        mask_obj_small = F.interpolate(
            mask_obj, 
            size=(self.sd_height // 8, self.sd_width // 8), 
            mode="bilinear"
        )
        # [1, 800, 800, 3] -> [1, 3, 800, 800]
        rgb_scene = outputs["rgb_scene"].unsqueeze(0).permute(0, 3, 1, 2) * 2.0 - 1.0
        # print(f"rgb_scene max:{rgb_scene.max().item()}, min:{rgb_scene.min().item()}")
        # [800, 800, 1] -> [1, 1, 800, 800] -> [1, 3, 800, 800]
        depth_im_scene = outputs["depth_im_scene"].unsqueeze(0)\
            .permute(0, 3, 1, 2).repeat(1, 3, 1, 1)
        # [1, 800, 800, 3] -> [1, 3, 800, 800]
        rgb_obj_scene = outputs["rgb_obj_scene"].unsqueeze(0).permute(0, 3, 1, 2) * 2.0 - 1.0
        # print(f"rgb_obj_scene max:{rgb_obj_scene.max().item()}, min:{rgb_obj_scene.min().item()}")
        # [800, 800, 1] -> [1, 1, 800, 800] -> [1, 3, 800, 800]
        depth_im_obj_scene = outputs["depth"].unsqueeze(0).permute(0, 3, 1, 2).repeat(1, 3, 1, 1)

        # process the depth images
        controlnet_nocomp_emb = self.guidance.get_image_embeds(
            self.depth_to_01(depth_im_scene), 
            height=self.sd_height, width=self.sd_width,
        )
        controlnet_comp_emb = self.guidance.get_image_embeds(
            self.depth_to_01(depth_im_obj_scene), 
            height=self.sd_height, width=self.sd_width,
        )

        # get the latent for the pulling image
        init_latent_nocomp = self.guidance.torch2latents_resize(rgb_scene)
        latent_comp = self.guidance.torch2latents_resize(rgb_obj_scene)

        if not self.config.conv_in_path is None:
            if (self.use_double_personalization and self.is_refining_phase) or (not self.use_double_personalization):
                rgb_obj_orig = outputs["rgb_obj_orig"].unsqueeze(0).permute(0, 3, 1, 2) * 2.0 - 1.0
                latent_obj_orig = self.guidance.torch2latents_resize(rgb_obj_orig)
            else:
                latent_obj_orig = None
        else:
            latent_obj_orig = None

        for _i in range(self.optimize_latent_for):
            grad_dds = self.guidance.train_step_dds(
                text_embeddings_initial=self.text_embeddings_initial, 
                text_embeddings_desired=self.text_embeddings_desired, 
                image_embeddings_initial=controlnet_nocomp_emb, 
                image_embeddings_desired=controlnet_comp_emb,
                latents_initial=init_latent_nocomp,
                rgb_pred=latent_comp, 
                pred_rgb_obj=latent_obj_orig,
                as_latent=True, 
                guidance_scale=self.guidance_scale,
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                use_weights=self.use_weights_sds,
                return_grad=True,
                use_controlnet=self.use_controlnet != 0,
            )
            grad_dds = grad_dds.to(torch.float32)
            if self.config.scene_shadows == "none":
                grad_dds = grad_dds * mask_obj_small
            # it was stated that SGD works better than Adam for DDS Loss
            latent_comp -= self.lr_latent * grad_dds

        optimized_image = self.guidance.latents2torch(latent_comp).to(torch.float32)
        optimized_image = optimized_image * 0.5 + 0.5

        return optimized_image

    @torch.no_grad()
    def _optimize_sdedit(self, outputs, optimize_obj=False, t_step=None):
        # [1, 800, 800, 3] -> [1, 3, 800, 800]
        if optimize_obj:
            key_rgb = "rgb_obj"
            key_depth = "depth_im_obj"
            emb_required = self.emb_obj
        else:
            key_rgb = "rgb_obj_scene"
            key_depth = "depth"
            emb_required = self.emb_desired

        rgb_obj_scene = outputs[key_rgb].unsqueeze(0).permute(0, 3, 1, 2) * 2.0 - 1.0
        # [800, 800, 1] -> [1, 1, 800, 800] -> [1, 3, 800, 800]
        depth_im_obj_scene = outputs[key_depth].unsqueeze(0).permute(0, 3, 1, 2).repeat(1, 3, 1, 1)

        controlnet_comp_emb = self.guidance.get_image_embeds(
            self.depth_to_01(depth_im_obj_scene), 
            height=self.sd_height, width=self.sd_width,
        )

        if not self.config.conv_in_path is None:
            if (self.use_double_personalization and self.is_refining_phase) or (not self.use_double_personalization):
                rgb_obj_orig = outputs["rgb_obj_orig"].unsqueeze(0).permute(0, 3, 1, 2) * 2.0 - 1.0
            else:
                rgb_obj_orig = None
        else:
            rgb_obj_orig = None

        latent_comp = self.guidance.perform_sdedit(
            image=rgb_obj_scene, 
            emb_uncond=self.emb_uncond,
            emb_cond=emb_required, 
            controlnet_comp_emb=controlnet_comp_emb,
            pred_rgb_obj=rgb_obj_orig,
            as_latent=False,
            timestep_range=[int(el * 1000) for el in self.curr_t_range],
            t_step=int(t_step * 1000) if t_step is not None else None,
            use_controlnet=self.use_controlnet != 0,
        )

        optimized_image = self.guidance.latents2torch(latent_comp).to(torch.float32)
        optimized_image = optimized_image * 0.5 + 0.5
        return optimized_image
    
    @staticmethod
    def _make_square(y_min, x_min, y_max, x_max):
        h = y_max - y_min
        w = x_max - x_min
        y_c = y_min + h / 2
        x_c = x_min + w / 2

        wh = max(h, w)
        return (
            int(y_c - wh / 2), int(x_c - wh / 2), 
            int(y_c + wh / 2), int(x_c + wh / 2)
        )


    def crop_image_dict(self, image_dict, mask_obj, crop_params=None):
        '''
        Expects obj_mask as bool

        image dict of {k: [H_k, W_k, ...]}
        mask_obj: Hm, Wm
        '''

        Hm, Wm = mask_obj.shape
        if not crop_params is None:
            x_min, x_max, y_min, y_max = crop_params
        else:
            # find bounding box
            mask_obj = mask_obj.bool()
            nonzero = torch.nonzero(mask_obj)
            if len(nonzero) == 0:
                print("Nothing to crop :(")
                return {k: v for k, v in image_dict.items()}
            y_min, x_min = nonzero.min(dim=0).values.tolist()
            y_max, x_max = nonzero.max(dim=0).values.tolist()
            # now we find random crops
            if self.config.crop_ratio < 1.0:
                y_wh_new = int((y_max - y_min) * self.config.crop_ratio)
                y_min = torch.randint(low=y_min, high=y_max - y_wh_new, size=(1,)).item()
                y_max = y_min + y_wh_new
                x_wh_new = int((x_max - x_min) * self.config.crop_ratio)
                x_min = torch.randint(low=x_min, high=x_max - x_wh_new, size=(1,)).item()
                x_max = x_min + x_wh_new
            y_min, x_min, y_max, x_max = self._make_square(y_min, x_min, y_max, x_max)
        
        # crop images
        result = {}
        for k in image_dict:
            if image_dict[k] is None:
                continue
            H, W, _ = image_dict[k].shape

            y_min_i, x_min_i = int(H / Hm * y_min), int(W / Wm * x_min) 
            y_max_i, x_max_i = int(H / Hm * y_max), int(W / Wm * x_max)
            y_min_i, x_min_i, y_max_i, x_max_i = self._make_square(y_min_i, x_min_i, y_max_i, x_max_i)
            result[k] = image_dict[k][y_min_i:y_max_i, x_min_i:x_max_i, ...]
        
        result["crop_params"] = (x_min, x_max, y_min, y_max)
        return result
    
    def get_all_images(self):
        return [img.clone() for img in self.optimized_images]

    def optimize(self, outputs, curr_iteration):
        """
        (Maybe) optimize the current image using the diffusion models.
        """

        if curr_iteration >= self.refine_after:
            t_step = None
            if self.config.refine_range_linear > 0:
                alpha = (curr_iteration - self.refine_after) / self.config.refine_range_linear
                t_step = self.curr_t_range[0] * alpha + self.curr_t_range[1] * (1 - alpha)
                print(f"alpha = {alpha}, t_step = {t_step}")

            optimized_image = self._optimize_sdedit(
                outputs, 
                optimize_obj=(self.refine_obj != 0),
                t_step=t_step,
            )
        else:
            optimized_image = self._optimize_dds_2d(outputs)

        optimized_image.clamp_(0.0, 1.0)
        
        return optimized_image
