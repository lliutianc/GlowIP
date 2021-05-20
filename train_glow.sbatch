#!/bin/bash

#SBATCH --job-name=train_glow
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=20G

#SBATCH --time=2:00:00
#SBATCH --output=%x-%j.SLURM

echo $CUDA_VISIBLE_DEVICES


#module load CUDA/9.0 cuDNN/7.4.1-CUDA-10.0
module load Conda/3
conda activate glow

python train_glow.py -dataset celeba
