#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -p gpu-private
#SBATCH --job-name=STDC
#SBATCH --mem=32GB
#SBATCH --time=7-23:59:59 

source /etc/profile
# Initialize conda for the current shell session
eval "$(conda shell.bash hook)"
conda activate openmmlab

module load cuda/11.8
# python tools/train.py /home2/lmfm45/mmsegmentation/configs/dmnet/dmnet_r50-d8_512x1024_40k_mydata.py
python /home2/lmfm45/mmsegmentation/mmseg/utils/mean_std.py