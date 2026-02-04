#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time 03:00:00
#SBATCH --mem=32G
#SBATCH -J run_all_synthetic
#SBATCH --output=/home/skorokho/coding/slurm_output/slurm-%j-%x.out

cd /home/skorokho/coding/voi_gs/dn-splatter
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dn-splatter

SCENES=(bathroom_1 bathroom_2 bedroom_1 bedroom_2 kitchen_1 kitchen_2 living_room_1 living_room_2 office_1 office_2)
# DIRS=(obj scene)
# DIRS=(obj)

SCENES_ROOT=/scratch/izar/skorokho/processed_3/
OUTPUT_ROOT=/scratch/izar/skorokho/output_processed_3/


# train the method
for i in "$@"; do
    curr_scene_name=${SCENES[i]}
    echo "RUNNNING ON $curr_scene_name/obj_scene"    
    obj_init_path=$(ls -1a $OUTPUT_ROOT/$curr_scene_name-obj/dn-splatter | tail -1)
    obj_init_path=$OUTPUT_ROOT/$curr_scene_name-obj/dn-splatter/$obj_init_path
    scene_init_path=$(ls -1a $OUTPUT_ROOT/$curr_scene_name-scene_eval/dn-splatter | tail -1)
    scene_init_path=$OUTPUT_ROOT/$curr_scene_name-scene_eval/dn-splatter/$scene_init_path

    # I do not want to write own trainer :/
    # srun --ntasks=1 ns-train dn-splatter-combined \
    ns-train dn-splatter-combined \
        --optimizers.features_dc.optimizer.lr 0.0 \
        --optimizers.features_rest.optimizer.lr 0.0 \
        --optimizers.camera_opt.optimizer.lr 0.0 \
        --max-num-iterations 1 \
        --pipeline.model.init-obj-path $obj_init_path \
        --pipeline.model.init-scene-path $scene_init_path \
        --pipeline.model.use-depth-loss False \
        --pipeline.model.depth-lambda 0.2 \
        --pipeline.model.use-normal-loss False \
        --pipeline.model.use-normal-tv-loss False \
        --pipeline.model.normal-supervision mono \
        --pipeline.model.background-color black \
        --output-dir $OUTPUT_ROOT \
        --experiment-name $curr_scene_name-combined-initial \
        --viewer.quit-on-train-completion True \
        nsdn \
        --data $SCENES_ROOT/$curr_scene_name/obj_scene_eval \
        --normal-format opencv
    echo "FINISHED SAVING INITIAL MODEL $curr_scene_name/obj_scene_eval"

    # ns-train dn-splatter-combined \
    # srun --ntasks=1 ns-train dn-splatter-combined \
    #     --pipeline.model.init-obj-path $obj_init_path \
    #     --pipeline.model.init-scene-path $scene_init_path \
    #     --pipeline.model.use-depth-loss False \
    #     --pipeline.model.depth-lambda 0.2 \
    #     --pipeline.model.use-normal-loss False \
    #     --pipeline.model.use-normal-tv-loss False \
    #     --pipeline.model.normal-supervision mono \
    #     --pipeline.model.background-color black \
    #     --pipeline.model.mean-init True \
    #     --output-dir $OUTPUT_ROOT \
    #     --experiment-name $curr_scene_name-combined \
    #     --viewer.quit-on-train-completion True \
    #     nsdn \
    #     --data $SCENES_ROOT/$curr_scene_name/obj_scene \
    #     --normal-format opencv
    # echo "FINISHED $curr_scene_name/obj_scene"
done
