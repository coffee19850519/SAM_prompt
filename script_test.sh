#!/bin/bash

#SBATCH -p gpu
#SBATCH --cpus-per-task 8
##SBATCH -n 4  
#SBATCH --mem 20G
#SBATCH --job-name med_sam_cryopp 
#SBATCH -o ./output_t3/cryopp_output_t3_10028c_10017d_test.txt

## change weights, folder name in 2 places task and folder

python test.py -net 'sam' -mod 'sam_adpt' -exp_name 'protein_test_10028c_10017d' -sam_ckpt ./checkpoint/sam/sam_vit_h_4b8939.pth -weights ./logs_task_2/protein_train_10028/Model/checkpoint_best.pth -image_size 1024 -dataset CryoPPP -b 1 -data_path ./dataset/protein_data_2/10017 -prompt_approach 'points_grids'