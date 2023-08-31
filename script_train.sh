#!/bin/bash

#SBATCH -p gpu
#SBATCH --cpus-per-task 8
##SBATCH -n 4  
##SBATCH --mem 100G
#SBATCH --job-name med_sam_cryopp 
#SBATCH -o ./output_ablation_testing/10028_3_fl_0_m.txt

python train.py -net 'sam' -mod 'sam_adpt' -exp_name 'ab_3_fl_0_m_train_10028' -sam_ckpt ./checkpoint/sam/sam_vit_h_4b8939.pth -image_size 1024 -dataset CryoPPP -b 1 -data_path ./dataset/protein_data_2/10028 -prompt_approach 'points_grids'
