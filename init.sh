srun --partition=gpu --qos=gpu --nodes=1 --gpus-per-node=1 --pty bash

srun --partition=gpu --qos=gpu --nodes=1 --gpus-per-node=1 --mem=34G --pty bash

module load Anaconda3/5.3.0
module load cuDNN/7.6.4.38-gcccuda-2019b
source activate torch
