#!/bin/bash

#SBATCH -p gpu
#SBATCH --cpus-per-task 8
#SBATCH --gres gpu:A100:1
#SBATCH -n 3  
#SBATCH --mem 40G
#SBATCH --job-name med_sam_cryopp 
#SBATCH -o ./output_random_ablation/10028_tail_1_adapter_5pic_train_11.txt
##SBATCH --exclude g003

##export a ='10028_3a4c_l_0_m'

##${a}

##export CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

##nvidia-smi --query-gpu=timestamp,temperature.gpu,pci.bus_id,utilization.gpu,utilization.memory,memory.total,memory.free,memory.used --format=csv -l 5  -i $CUDA_VISIBLE_DEVICES --filename=$PWD/gpu_util_${SLURM_JOB_ID}-${SLURM_JOB_NAME}.csv &

##NVIDIA_PID=`echo $!`

python train_1.py -net 'sam' -mod 'sam_adpt' -exp_name '10028_tail_1_adapter_5pic_train_11' -sam_ckpt ./checkpoint/sam/sam_vit_h_4b8939.pth -image_size 1024 -dataset CryoPPP -b 1 -data_path ./dataset/random_train_data/10028 -prompt_approach 'points_grids'

##kill $NVIDIA_PID
