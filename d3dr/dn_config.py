from __future__ import annotations

from d3dr.data.normal_nerfstudio import NormalNerfstudioConfig
from d3dr.dn_datamanager import DNSplatterManagerConfig
from d3dr.dn_model import DNSplatterModelConfig
from d3dr.dn_model_combined import DNSplatterModelCombinedConfig
from d3dr.dn_model_voi import D3DRModelConfig
from d3dr.dn_pipeline import DNSplatterPipelineConfig
from d3dr.d3dr_pipeline import DNSplatterVOIPipelineConfig
from d3dr.random_cameras_datamanager_obj_scene import RandomCamerasDataManagerObjSceneConfig

from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.engine.trainer import TrainerConfig
from nerfstudio.plugins.types import MethodSpecification

dn_splatter = MethodSpecification(
    config=TrainerConfig(
        method_name="dn-splatter",
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        steps_per_save=1000000,
        steps_per_eval_all_images=1000000,
        max_num_iterations=30000,
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 100, "color": 10, "shs": 10},
        pipeline=DNSplatterPipelineConfig(
            datamanager=DNSplatterManagerConfig(
                dataparser=NormalNerfstudioConfig(load_3D_points=True)
            ),
            model=DNSplatterModelConfig(),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6, max_steps=30000
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5, max_steps=30000
                ),
            },
            "normals": {
                "optimizer": AdamOptimizerConfig(
                    lr=1e-3, eps=1e-15
                ),  # this does nothing, its just here to make the trainer happy
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="DN-Splatter: depth and normal priors for 3DGS",
)

dn_splatter_big = MethodSpecification(
    config=TrainerConfig(
        method_name="dn-splatter-big",
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        steps_per_save=1000000,
        steps_per_eval_all_images=1000000,
        max_num_iterations=30000,
        mixed_precision=False,
        pipeline=DNSplatterPipelineConfig(
            datamanager=DNSplatterManagerConfig(
                dataparser=NormalNerfstudioConfig(load_3D_points=True)
            ),
            model=DNSplatterModelConfig(
                cull_alpha_thresh=0.005,
                continue_cull_post_densification=False,
            ),
        ),
        optimizers={
            "means": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {
                "optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5, max_steps=30000
                ),
            },
            "normals": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="DN-Splatter Big variant",
)


dn_splatter_combined = MethodSpecification(
    config=TrainerConfig(
        method_name="dn-splatter-combined",
        steps_per_eval_image=500,
        steps_per_eval_batch=500,
        steps_per_save=1000000, # TODO: WTF?
        steps_per_eval_all_images=1000000,
        max_num_iterations=10000,
        mixed_precision=False,
        gradient_accumulation_steps={
            "camera_opt": 100, 
            "color": 10, 
            "shs": 10,
        },
        pipeline=DNSplatterPipelineConfig(
            datamanager=DNSplatterManagerConfig(
                dataparser=NormalNerfstudioConfig(load_3D_points=True)
            ),
            model=DNSplatterModelCombinedConfig(
                refine_every=1000000, # do not refining!
                reset_alpha_every=1000000,
                continue_cull_post_densification=False,
                stop_screen_size_at=0,
                stop_split_at=0,
                sh_degree_interval=1,
                # turn of all losses
                use_depth_smooth_loss=False,
                use_depth_loss=False,
                use_normal_loss=False,
                use_normal_cosine_loss=False,
                use_normal_tv_loss=False,
                use_sparse_loss =False,
                use_binary_opacities=False,
            ),
        ),
        optimizers={
            "means": {   # this does nothing, its just here to make the trainer happy
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": None,
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {   # this does nothing, its just here to make the trainer happy
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": None,
            },
            "scales": { # this does nothing, its just here to make the trainer happy
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": None,
            },
            "quats": {  # this does nothing, its just here to make the trainer happy
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-5, max_steps=30000
                ),
            },
            "normals": { # this does nothing, its just here to make the trainer happy
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="DN-Splatter: depth and normal priors for 3DGS",
)

d3dr = MethodSpecification(
    config=TrainerConfig(
        method_name="d3dr",
        steps_per_eval_image=1000000,
        steps_per_eval_batch=1000000,
        steps_per_save=200,
        steps_per_eval_all_images=1000000,
        max_num_iterations=10000,
        mixed_precision=False,
        gradient_accumulation_steps={
            "camera_opt": 100, 
            "color": 10, 
            "shs": 10
        },
        pipeline=DNSplatterVOIPipelineConfig(
            datamanager=RandomCamerasDataManagerObjSceneConfig(
                vertical_rotation_range=(-45, 0),
                position_jitter=0.1,
                betas=(0.1, 0.6),
            ),
            model=D3DRModelConfig(
                refine_every=1000000, # do not refine!
                reset_alpha_every=1000000,
                continue_cull_post_densification=False,
                stop_screen_size_at=0,
                stop_split_at=0,
                sh_degree_interval=1,
                # turn of all regularization losses
                use_depth_smooth_loss=False,
                use_depth_loss=False,
                use_normal_loss=False,
                use_normal_cosine_loss=False,
                use_normal_tv_loss=False,
                use_sparse_loss =False,
                use_binary_opacities=False,
                # optimization parameters
                optimize_latent_for=4,
                optimize_image_for=64,
                num_together_images=4,
            ),
        ),
        optimizers={
            "means": {   # this does nothing, its just here to make the trainer happy
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": None,
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 32, eps=1e-15),
                "scheduler": None,
            },
            "scene_shadows": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {   # this does nothing, its just here to make the trainer happy
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": None,
            },
            "scales": { # this does nothing, its just here to make the trainer happy
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": None,
            },
            "quats": {  # this does nothing, its just here to make the trainer happy
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": None,
            },
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": None,
            },
            "normals": { # this does nothing, its just here to make the trainer happy
                "optimizer": AdamOptimizerConfig(lr=0.0, eps=1e-15),
                "scheduler": None,
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="Object Insertion using Diffusion Models",
)
