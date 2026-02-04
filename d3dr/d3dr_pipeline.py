from dataclasses import dataclass, field
from typing import Literal, Optional, Type
import torch

from torch.cuda.amp.grad_scaler import GradScaler
from d3dr.dn_model_voi import D3DRModelConfig
from nerfstudio.data.datamanagers.base_datamanager import (
    DataManagerConfig,
)
from d3dr.random_cameras_datamanager_obj_scene import RandomCamerasDataManagerObjSceneConfig
from nerfstudio.models.base_model import ModelConfig
from nerfstudio.pipelines.base_pipeline import (
    VanillaPipeline,
    VanillaPipelineConfig,
)
from nerfstudio.utils import profiler
from nerfstudio.utils.rich_utils import CONSOLE



@dataclass
class DNSplatterVOIPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: DNSplatterVOIPipeline)
    datamanager: RandomCamerasDataManagerObjSceneConfig = field(default_factory=lambda: DataManagerConfig())
    model: ModelConfig = field(default_factory=lambda: D3DRModelConfig())


class DNSplatterVOIPipeline(VanillaPipeline):
    """Pipeline for convenient eval metrics across model types"""

    def __init__(
        self,
        config: DNSplatterVOIPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(
            config=config,
            device=device,
            test_mode=test_mode,
            world_size=world_size,
            local_rank=local_rank,
            grad_scaler=grad_scaler,
        )
        self._is_refining_phase = False 

    def _set_phase(self, step):
        if step > self.model.config.refine_after:
            self._is_refining_phase = True
        else:
            self._is_refining_phase = False

    @profiler.time_function
    def get_train_loss_dict(self, step: int):
        """This function gets your training loss dict. This will be responsible for
        getting the next batch of data from the DataManager and interfacing with the
        Model class, feeding the data to the model's forward function.

        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """

        self._set_phase(step)

        if self._is_refining_phase:
            self.datamanager.refining_phase()
            self.model.guidance_controller.refining_phase()
        else:
            self.datamanager.lighting_phase()
            self.model.guidance_controller.lighting_phase()

        if (step + 1) % self.model.config.keep_scene_shadows_every == 0:
            self.model.keep_darkest_shadows()
        
        # TODO: Maybe change logic here as:
        # while self.model.guidance_controller.need_new_cameras(): ...?
        if self.model.guidance_controller.need_new_cameras():
            with torch.no_grad():
                _i = 0
                while True:
                    camera, batch = self.datamanager.next_train(step + _i)
                    _i += 1
                    model_outputs = self._model.get_outputs(
                        camera, 
                        detach_scene_shadows=(self._is_refining_phase != 0),
                    )
                    if self._is_refining_phase:
                        # need to crop them
                        model_outputs = self.model.guidance_controller.crop_image_dict(
                            # TODO: Don't need to clone the mask, right?
                            model_outputs, model_outputs["mask_obj"][..., 0] 
                        )

                    need_more = self.model.guidance_controller.add_new_camera(
                        model_outputs, camera, step
                    )
                    if not need_more:
                        break
        
        camera, batch = self.model.guidance_controller.next_train(step)
        if self._is_refining_phase and (self.model.config.refine_obj != 0):
            required_params = ["rgb_obj"]
        else:
            required_params = ["rgb_obj_scene"]
        
        if not self.model.config.conv_in_path is None:
            required_params = required_params + ["rgb_obj_orig"]

        if self._is_refining_phase:
            required_params = required_params + ["rgb_obj"] # we need it anyway to crop

        model_outputs = self._model.get_outputs(
            camera, 
            required_params=required_params,
            detach_scene_shadows=(self._is_refining_phase != 0),
        )  # train distributed data parallel model if world_size > 1
        if self._is_refining_phase:
            model_outputs = self.model.guidance_controller.crop_image_dict(
                model_outputs, model_outputs["mask_obj"][..., 0],
                batch["crop_params"]
            )
        key_rgb_0 = required_params[0]
        model_outputs["rgb"] = model_outputs[key_rgb_0] # required_params[0] contains the required key

        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)

        return model_outputs, loss_dict, metrics_dict
