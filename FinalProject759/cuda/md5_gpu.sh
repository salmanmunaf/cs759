#!/usr/bin/env zsh
#SBATCH --job-name=md5_gpu
#SBATCH --partition=wacc
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:10:00
#SBATCH --output=md5.out
#SBATCH --error=md5.err

cd $SLURM_SUBMIT_DIR

make
./md5_gpu 39b18a809ace6b98510f4d59d09f5365 32 # H3lOw0
