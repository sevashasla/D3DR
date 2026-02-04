#!/bin/bash
#SBATCH --nodes 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time 06:00:00
#SBATCH --mem=32G
#SBATCH -J blender
#SBATCH --output=/home/skorokho/coding/slurm_output/slurm-%j-%x.out

SCENES=(bathroom_1.blend bathroom_2.blend bedroom_1.blend bedroom_2.blend kitchen_1.blend kitchen_2.blend living_room_1.blend living_room_2.blend office_1.blend office_2.blend)

for i in "$@"; do
    scene_name=${SCENES[i]}
    echo "will run on $scene_name"
done

cd /home/skorokho/blender/blender-4.2.1-linux-x64
for i in "$@"; do
    scene_name=${SCENES[i]}
    echo $scene_name
    # srun --ntasks=1 ./blender -b /scratch/izar/skorokho/data/my_blend_3/$scene_name --python ~/coding/voi_gs/render_dataset/render_indoor.py
    ./blender -b /scratch/izar/skorokho/data/my_blend_3/$scene_name --python ~/coding/voi_gs/render_dataset/render_indoor.py
done
