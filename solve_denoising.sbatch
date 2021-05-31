#!/bin/bash

#SBATCH --job-name=solve_denosing
#SBATCH --gres=gpu:v100:1
#SBATCH --mem-per-cpu=8G
#SBATCH --mail-type=END

#SBATCH --time=1:00:00
#SBATCH --output=sbatch-logs/%x-%j.SLURM

echo $CUDA_VISIBLE_DEVICES

#module load CUDA/9.0 cuDNN/7.4.1-CUDA-10.0
module load Conda/3
conda activate glow

# use glow to add image noise
#python solve_denoising.py -prior glow -experiment denoise -dataset celeba -batchsize 4 -noise_scale 0.8 -noise glow
# use dcgan to add image noise
#python solve_denoising.py -prior glow -experiment denoise -dataset celeba -batchsize 4 -noise_scale 0.8 -noise dcgan

# use gaussian to add image noise
#python solve_denoising.py -prior glow -experiment denoise -dataset celeba -batchsize 4 -noise_scale 0.1 -noise gaussian
python solve_denoising.py -prior glow -experiment denoise -dataset celeba -batchsize 4 -noise_scale 0.2 -noise gaussian -size 128


