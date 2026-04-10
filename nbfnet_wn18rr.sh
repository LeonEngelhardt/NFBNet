#!/bin/bash
#SBATCH --job-name=nbfnet_wn18rr
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --time=10:00:00
#SBATCH --output=nbfnet_wn18rr_%j.log

source ~/Seminar/NBFNet/nbfnet_env/bin/activate

export PYTHONUNBUFFERED=1

cd ~/Seminar/NBFNet

mkdir -p ~/experiments/wn18rr

python -m torch.distributed.launch --nproc_per_node=2 script/run.py -c config/inductive/wn18rr.yaml --gpus [0,1] --version v1

deactivate
