#!/bin/bash

#SBATCH -p gpu
#SBATCH --cpus-per-task 8
#SBATCH --gres gpu:A100:1
#SBATCH -n 3
#SBATCH --mem 40G
#SBATCH --job-name med_sam_cryopp 
#SBATCH -o ./output_random_ablation/10028_tail_1_adapter_5pic_test_11.txt

## change weights, folder name in 2 places task and folder

python test.py -net 'sam' -mod 'sam_adpt' -exp_name '10028_tail_1_adapter_5pic_test_11' -sam_ckpt ./checkpoint/sam/sam_vit_h_4b8939.pth -weights ./logs_random_ablation/10028_tail_1_adapter_5pic_train_11/Model/checkpoint_best.pth -image_size 1024 -dataset CryoPPP -b 1 -data_path ./dataset/random_train_data/10028 -prompt_approach 'points_grids'

