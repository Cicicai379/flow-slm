#!/bin/bash
#SBATCH -p berkeleynlp
#SBATCH --nodelist=lorax
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name=training-spidr
#SBATCH --output=/data/cicicai/flow_slm/logs/train_%j.log
#SBATCH --error=/data/cicicai/flow_slm/logs/train_%j.err

source /usr/local/linux/mambaforge-3.11/bin/activate flow
export CUDA_HOME=/usr/local/cuda-12.9
export HF_HOME=/data/cicicai/hf_cache

python trainer_spidr.py \
      --conf conf/270m_spidr.yaml \
      --save_path /data/cicicai/flow_slm/checkpoints/test_run_spidr \
      --override "{'optimizer': {'lr': 1e-5, 'loss_function': 'FM'}, 'training': {'batch_size': 8}}" \
      --hf_training_data \
      --training_data "emilia" \
      --strategy "deepspeed_stage_2"