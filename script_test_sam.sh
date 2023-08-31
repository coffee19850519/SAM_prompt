#!/bin/bash

#SBATCH -p gpu
#SBATCH --cpus-per-task 8
##SBATCH -n 4  
#SBATCH --mem 20G
#SBATCH --job-name med_sam_cryopp 
#SBATCH -o ./output_test/cryopp_output_10028_test.txt

## change weights, folder name in 2 places task and folder

python test_sam.py -net 'sam' -mod 'sam_adpt' -exp_name 'protein_test_10028' -sam_ckpt ./checkpoint/sam/sam_vit_h_4b8939.pth -image_size 1024 -dataset CryoPPP -b 1 -data_path ./dataset/protein_data_2/10028  -prompt_approach 'points_grids'