# dark_to_light
curr_scene_name=dark_to_light
obj_init_path="/scratch/izar/skorokho/data/toster2/splat_dark_toster/ns_processed_dark_toster/splatfacto/2025-08-13_214741/"
scene_init_path="/scratch/izar/skorokho/data/toster2/splat_light_notoster/ns_processed_light_notoster/splatfacto/2025-08-13_220402/"
transforms_obj="/scratch/izar/skorokho/data/toster2/dark_to_light/obj_scene_eval/"
OUTPUT_ROOT="/scratch/izar/skorokho/data/toster2/"

ns-train dn-splatter-combined \
    --optimizers.features_dc.optimizer.lr 0.0 \
    --optimizers.features_rest.optimizer.lr 0.0 \
    --optimizers.camera_opt.optimizer.lr 0.0 \
    --optimizers.opacities.optimizer.lr 0.0 \
    --optimizers.scales.optimizer.lr 0.0 \
    --optimizers.quats.optimizer.lr 0.0 \
    --optimizers.camera-opt.optimizer.lr 0.0 \
    --optimizers.normals.optimizer.lr 0.0 \
    --max-num-iterations 1 \
    --pipeline.model.init-obj-path $obj_init_path \
    --pipeline.model.init-scene-path $scene_init_path \
    --pipeline.model.use-depth-loss False \
    --pipeline.model.depth-lambda 0.2 \
    --pipeline.model.use-normal-loss False \
    --pipeline.model.use-normal-tv-loss False \
    --pipeline.model.normal-supervision mono \
    --pipeline.model.two-d-gaussians False \
    --pipeline.model.background-color black \
    --pipeline.model.transforms_obj $transforms_obj \
    --output-dir $OUTPUT_ROOT \
    --experiment-name $curr_scene_name-combined-initial \
    --viewer.quit-on-train-completion True \
    nsdn \
    --data /scratch/izar/skorokho/processed_3/bathroom_1/obj_scene_eval/ \
    --normal-format opencv
echo "FINISHED SAVING INITIAL MODEL $curr_scene_name/obj_scene_eval"

##################### ---------------- #####################

curr_scene_name=ns_processed_stone_in_garden
obj_init_path="/scratch/izar/skorokho/data/recordings_09082025/splat_stone/ns_processed_stone/splatfacto/2025-08-12_002126/"
scene_init_path="/scratch/izar/skorokho/data/recordings_09082025/splat_outdoor_green/ns_processed_outdoor_green/splatfacto/2025-08-12_001058/"
transforms_obj="/scratch/izar/skorokho/data/recordings_09082025/ns_processed_stone_in_garden/obj_scene_eval"
OUTPUT_ROOT="/scratch/izar/skorokho/data/recordings_09082025"

ns-train dn-splatter-combined \
    --optimizers.features_dc.optimizer.lr 0.0 \
    --optimizers.features_rest.optimizer.lr 0.0 \
    --optimizers.camera_opt.optimizer.lr 0.0 \
    --optimizers.opacities.optimizer.lr 0.0 \
    --optimizers.scales.optimizer.lr 0.0 \
    --optimizers.quats.optimizer.lr 0.0 \
    --optimizers.camera-opt.optimizer.lr 0.0 \
    --optimizers.normals.optimizer.lr 0.0 \
    --max-num-iterations 1 \
    --pipeline.model.init-obj-path $obj_init_path \
    --pipeline.model.init-scene-path $scene_init_path \
    --pipeline.model.use-depth-loss False \
    --pipeline.model.depth-lambda 0.2 \
    --pipeline.model.use-normal-loss False \
    --pipeline.model.use-normal-tv-loss False \
    --pipeline.model.normal-supervision mono \
    --pipeline.model.background-color black \
    --pipeline.model.transforms_obj $transforms_obj \
    --output-dir $OUTPUT_ROOT \
    --experiment-name $curr_scene_name-combined-initial \
    --viewer.quit-on-train-completion True \
    nsdn \
    --data /scratch/izar/skorokho/processed_3/bathroom_1/obj_scene_eval/ \
    --normal-format opencv
echo "FINISHED SAVING INITIAL MODEL $curr_scene_name/obj_scene_eval"

##################### ---------------- #####################

curr_scene_name=ns_processed_dustbins_ifo_container
obj_init_path="/scratch/izar/skorokho/data/recordings_09082025/splat_dustbin/ns_processed_dustbin/splatfacto/2025-08-12_001958/"
scene_init_path="/scratch/izar/skorokho/data/recordings_09082025/splat_outdoor_container/ns_processed_outdoor_container/splatfacto/2025-08-12_000028/"
transforms_obj="/scratch/izar/skorokho/data/recordings_09082025/ns_processed_dustbins_ifo_container/obj_scene_eval"
OUTPUT_ROOT="/scratch/izar/skorokho/data/recordings_09082025"

ns-train dn-splatter-combined \
    --optimizers.features_dc.optimizer.lr 0.0 \
    --optimizers.features_rest.optimizer.lr 0.0 \
    --optimizers.camera_opt.optimizer.lr 0.0 \
    --optimizers.opacities.optimizer.lr 0.0 \
    --optimizers.scales.optimizer.lr 0.0 \
    --optimizers.quats.optimizer.lr 0.0 \
    --optimizers.camera-opt.optimizer.lr 0.0 \
    --optimizers.normals.optimizer.lr 0.0 \
    --max-num-iterations 1 \
    --pipeline.model.init-obj-path $obj_init_path \
    --pipeline.model.init-scene-path $scene_init_path \
    --pipeline.model.use-depth-loss False \
    --pipeline.model.depth-lambda 0.2 \
    --pipeline.model.use-normal-loss False \
    --pipeline.model.use-normal-tv-loss False \
    --pipeline.model.normal-supervision mono \
    --pipeline.model.background-color black \
    --pipeline.model.transforms_obj $transforms_obj \
    --output-dir $OUTPUT_ROOT \
    --experiment-name $curr_scene_name-combined-initial \
    --viewer.quit-on-train-completion True \
    nsdn \
    --data /scratch/izar/skorokho/processed_3/bathroom_1/obj_scene_eval/ \
    --normal-format opencv
echo "FINISHED SAVING INITIAL MODEL $curr_scene_name/obj_scene_eval"

##################### ---------------- #####################

curr_scene_name=ns_processed_fabric_bin_in_corridor
obj_init_path="/scratch/izar/skorokho/data/recordings_09082025/splat_fabric_bin_living_room/ns_processed_fabric_bin_living_room/splatfacto/2025-08-12_000702/"
scene_init_path="/scratch/izar/skorokho/data/recordings_09082025/splat_corridor/ns_processed_corridor/splatfacto/2025-08-12_001325/"
transforms_obj="/scratch/izar/skorokho/data/recordings_09082025/ns_processed_fabric_bin_in_corridor/obj_scene_eval"
OUTPUT_ROOT="/scratch/izar/skorokho/data/recordings_09082025"

ns-train dn-splatter-combined \
    --optimizers.features_dc.optimizer.lr 0.0 \
    --optimizers.features_rest.optimizer.lr 0.0 \
    --optimizers.camera_opt.optimizer.lr 0.0 \
    --optimizers.opacities.optimizer.lr 0.0 \
    --optimizers.scales.optimizer.lr 0.0 \
    --optimizers.quats.optimizer.lr 0.0 \
    --optimizers.camera-opt.optimizer.lr 0.0 \
    --optimizers.normals.optimizer.lr 0.0 \
    --max-num-iterations 1 \
    --pipeline.model.init-obj-path $obj_init_path \
    --pipeline.model.init-scene-path $scene_init_path \
    --pipeline.model.use-depth-loss False \
    --pipeline.model.depth-lambda 0.2 \
    --pipeline.model.use-normal-loss False \
    --pipeline.model.use-normal-tv-loss False \
    --pipeline.model.normal-supervision mono \
    --pipeline.model.background-color black \
    --pipeline.model.transforms_obj $transforms_obj \
    --output-dir $OUTPUT_ROOT \
    --experiment-name $curr_scene_name-combined-initial \
    --viewer.quit-on-train-completion True \
    nsdn \
    --data /scratch/izar/skorokho/processed_3/bathroom_1/obj_scene_eval/ \
    --normal-format opencv
echo "FINISHED SAVING INITIAL MODEL $curr_scene_name/obj_scene_eval"


curr_scene_name=ns_processed_chair_in_living_room
obj_init_path="/scratch/izar/skorokho/data/recordings_09082025/splat_chair_my_room/ns_processed_chair_my_room/splatfacto/2025-08-12_000834/"
scene_init_path="/scratch/izar/skorokho/data/recordings_09082025/splat_living_room/ns_processed_living_room/splatfacto/2025-08-12_001156/"
transforms_obj="/scratch/izar/skorokho/data/recordings_09082025/ns_processed_chair_in_living_room/obj_scene_eval"
OUTPUT_ROOT="/scratch/izar/skorokho/data/recordings_09082025"

ns-train dn-splatter-combined \
    --optimizers.features_dc.optimizer.lr 0.0 \
    --optimizers.features_rest.optimizer.lr 0.0 \
    --optimizers.camera_opt.optimizer.lr 0.0 \
    --optimizers.opacities.optimizer.lr 0.0 \
    --optimizers.scales.optimizer.lr 0.0 \
    --optimizers.quats.optimizer.lr 0.0 \
    --optimizers.camera-opt.optimizer.lr 0.0 \
    --optimizers.normals.optimizer.lr 0.0 \
    --max-num-iterations 1 \
    --pipeline.model.init-obj-path $obj_init_path \
    --pipeline.model.init-scene-path $scene_init_path \
    --pipeline.model.use-depth-loss False \
    --pipeline.model.depth-lambda 0.2 \
    --pipeline.model.use-normal-loss False \
    --pipeline.model.use-normal-tv-loss False \
    --pipeline.model.normal-supervision mono \
    --pipeline.model.background-color black \
    --pipeline.model.transforms_obj $transforms_obj \
    --output-dir $OUTPUT_ROOT \
    --experiment-name $curr_scene_name-combined-initial \
    --viewer.quit-on-train-completion True \
    nsdn \
    --data /scratch/izar/skorokho/processed_3/bathroom_1/obj_scene_eval/ \
    --normal-format opencv
echo "FINISHED SAVING INITIAL MODEL $curr_scene_name/obj_scene_eval"
