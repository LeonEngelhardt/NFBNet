#!/bin/bash
#SBATCH --job-name=nbfnet_wn18rr_viz
#SBATCH --partition=gpu_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --time=00:30:00
#SBATCH --output=nbfnet_wn18rr_viz_%j.log

source ~/Seminar/NBFNet/kg_reasoning_env/bin/activate
module load devel/cuda/11.8
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

export PYTHONUNBUFFERED=1

cd ~/Seminar/NBFNet

mkdir -p ~/experiments/wn18rr_visualize

CHECKPOINT_DIR=$(ls -td ~/experiments/InductiveKnowledgeGraphCompletion/WN18RRInductive/NBFNet/* 2>/dev/null | head -n 1)
CHECKPOINT_PATH="$CHECKPOINT_DIR/model_epoch_20.pth"

if [ -z "$CHECKPOINT_DIR" ] || [ ! -f "$CHECKPOINT_PATH" ]; then
  echo "Kein passender NBFNet-Checkpoint gefunden: $CHECKPOINT_PATH"
  exit 1
fi

python script/visualize.py -c config/inductive/wn18rr_visualize.yaml --gpus [0] --checkpoint "$CHECKPOINT_PATH"

deactivate
