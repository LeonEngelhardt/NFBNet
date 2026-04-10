#!/bin/bash
#SBATCH --job-name=nbfnet_wn18rr_viz
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1          # Optional: 1 GPU, kann auf CPU geändert werden
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=nbfnet_wn18rr_viz_%j.log

source ~/Seminar/NBFNet/nbfnet_env/bin/activate

export PYTHONUNBUFFERED=1

cd ~/Seminar/NBFNet

mkdir -p ~/experiments/wn18rr_visualize

python script/visualize.py -c config/inductive/wn18rr_visualize.yaml --gpus [0] --checkpoint ~/experiments/wn18rr/model_epoch_20.pth

deactivate
