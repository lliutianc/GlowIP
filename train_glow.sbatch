#!/bin/bash

#SBATCH --job-name=train_glow
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=20G

#SBATCH --time=2:00:00
#SBATCH --output=%x-%j.SLURM

echo $CUDA_VISIBLE_DEVICES

module purge
module load GCC/6.4.0-2.28  OpenMPI/2.1.2
module load CUDA/9.0 cuDNN/7.5.0.56-CUDA-9.0
module load Conda/3
conda activate glow

python train_glow.py -dataset celeba
