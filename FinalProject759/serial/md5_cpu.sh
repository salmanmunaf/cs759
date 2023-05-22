#!/usr/bin/env zsh
#SBATCH --job-name=md5_cpu
#SBATCH --partition=wacc
#SBATCH --ntasks=1 --cpus-per-task=1
#SBATCH --time=0-00:03:00
#SBATCH --output=md5.out
#SBATCH --error=md5.err

cd $SLURM_SUBMIT_DIR

make
./md5_cpu 857fa0ed49e3bdbd4ba9ba569c68a97a 4 # H3lO