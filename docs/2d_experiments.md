# 2D Diffusion Models Experiments

We provide a large number of scripts to perform experiments with SDS and DDS.

## SDS

1. SDS generation:

    ```bash
    python3 d3dr/diffusion/sds/sds.py --prompt "a cat"
    ```

2. SDS generation with SD3

    ```bash
    python3 d3dr/diffusion/sds/sds_sd3.py --prompt "a cat"
    ```

3. SDS in pixel space:

    for 2-step-SDS run:
    
    ```bash
    python3 d3dr/diffusion/sds/sds_pixels.py
    ```
    
    for regular SDS run (the results are noisy):
    
    ```bash
    python3 d3dr/diffusion/sds/sds_pixels.py --use_2_step_sds 0
    ```

## DDS

1. DDS

    ```bash
    python3 d3dr/diffusion/sds/dds.py  \
        --prompt_1 "a statue head on a plate" \
        --prompt_2 "a cup on a plate" \
        --image_path "./data/cups/cupl1l2.jpg" \
        --mask_path "./data/cups/mask_composition.png"
    ```

2. DDS with different initialization:

    DDS
    ```bash
    python3 d3dr/diffusion/sds/dds_diff_init.py \
        --prompt_1 "a cup on a plate" \
        --prompt_2 "a plate" \
        --image_nocomp_path "./data/cups/no_cup_l1.jpg" \
        --image_comp_path "./data/cups/cupl2l1.jpg" \
        --mask_path "./data/cups/mask_composition.png"
    ```

    DDS-SD3
    ```bash
    python3 d3dr/diffusion/sds/dds_sd3_diff_init.py \
        --prompt_1 "a cup on a plate" \
        --prompt_2 "a plate" \
        --image_path "./data/cups/no_cup_l1.jpg" \
        --image_comp_path "./data/cups/cupl2l1.jpg" \
        --mask_path "./data/cups/mask_composition.png" \
        --num_train_iteration 200 \
        --show_iter 25 \
        --use_ratio 1 \
        --add_mean_init 0
    ```