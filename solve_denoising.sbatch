#!/bin/bash

#SBATCH --job-name=solve_denosing
#SBATCH --gres=gpu:v100:1
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END

#SBATCH --time=5:00:00
#SBATCH --output=sbatch-logs/%x-%j.SLURM

echo $CUDA_VISIBLE_DEVICES

#module load CUDA/9.0 cuDNN/7.4.1-CUDA-10.0
module load Conda/3
conda activate glow

# use glow to add image noise
#python solve_denoising.py -prior glow -experiment denoise -dataset celeba -batchsize 4 -noise_scale 0.8 -noise glow

# dcgan noise
#python solve_denoising.py -prior glow -experiment denoise -dataset celeba -batchsize 4 -noise_scale 0.8 -noise dcgan


# gaussian noise
python solve_denoising.py -prior glow -experiment denoise -dataset celeba -batchsize 4 -noise_scale 0.1 -noise gaussian
python solve_denoising.py -prior glow -experiment denoise -dataset celeba -batchsize 4 -noise_scale 0.1 -noise gaussian -lr 0.5 -steps 100

## poisson noise
#python solve_denoising.py -prior glow -experiment denoise -dataset celeba -batchsize 4 -noise_scale 0.1 -noise poisson -gamma 0.1

## log gamma noise
#python solve_denoising.py -prior glow -experiment denoise -dataset celeba -batchsize 4 -noise_scale 5. -noise_loc 1. -noise loggamma

# logistics noise
python solve_denoising.py -prior glow -experiment denoise -dataset celeba -batchsize 4 -noise_scale 0.05 -noise logistic
python solve_denoising.py -prior glow -experiment denoise -dataset celeba -batchsize 4 -noise_scale 0.05 -noise logistic -lr 0.5 -steps 100

# gamma noise: can noisy-init help?
#python solve_denoising.py -prior glow -experiment denoise -dataset celeba -batchsize 4 -noise_scale 5. -noise_loc 1. -noise gamma -init_strategy from-noisy


