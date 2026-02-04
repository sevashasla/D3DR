#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time 02:00:00
#SBATCH --mem=16G
#SBATCH -J blender_to_train
#SBATCH --output=/home/skorokho/coding/slurm_output/slurm-%j-%x.out

cd /home/skorokho/coding/voi_gs
source ~/miniforge3/etc/profile.d/conda.sh
conda activate dn-splatter
SCENES=(bathroom_1 bathroom_2 bedroom_1 bedroom_2 kitchen_1 kitchen_2 living_room_1 living_room_2 office_1 office_2)


for i in "$@"; do
    scene_name=${SCENES[i]}
    echo "will run on $scene_name"
done

for i in "$@"; do
    scene_name=${SCENES[i]}
    echo $scene_name
    # srun --ntasks=1 python3 render_dataset/my_blender_to_train.py \
    #     --input_dir /scratch/izar/skorokho/blender_attmpt_3/$scene_name/ \
    #     --output_dir /scratch/izar/skorokho/processed_3/$scene_name/ \
    #     --type all

    python3 render_dataset/my_blender_to_train.py \
        --input_dir /scratch/izar/skorokho/blender_attmpt_5/$scene_name/ \
        --output_dir /scratch/izar/skorokho/processed_3/$scene_name/ \
        --help_eval_dir /scratch/izar/skorokho/blender_attmpt_3/$scene_name/ \
        --run_depth 1 --run_normals 1 \
        --type "obj_scene_eval" "scene_eval"
done


# ns-train dn-splatter \
#     --pipeline.model.use-depth-loss False \
#     --pipeline.model.depth-lambda 0.2 \
#     --pipeline.model.use-normal-loss True \
#     --pipeline.model.use-normal-tv-loss True \
#     --pipeline.model.normal-supervision mono \
#     --pipeline.model.background-color black \
#     --output-dir /scratch/izar/skorokho/output_processed_1/ \
#     --experiment-name bathroom_1-obj-mask \
#     --viewer.quit-on-train-completion True \
#     nsdn \
#     --data /scratch/izar/skorokho/processed_1/bathroom_1/obj \
#     --normal-format opencv