#!/bin/bash
#SBATCH --job-name=nbfnet_wn18rr
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=06:00:00
#SBATCH --output=nbfnet_wn18rr_v2_%j.log

source ~/Seminar/NBFNet/nbfnet_env/bin/activate

export PYTHONUNBUFFERED=1

cd ~/Seminar/NBFNet

mkdir -p ~/experiments/wn18rr_v2

python script/run.py -c config/inductive/wn18rr.yaml --gpus [0] --version v1

deactivate
