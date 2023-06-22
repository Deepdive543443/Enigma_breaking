#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=output.%j.test.out

module load Anaconda3/2022.10
module load cuDNN/7.6.4.38-gcccuda-2019b
source activate torch

python main.py --PROGRESS_BAR=0 --USE_COMPILE=0