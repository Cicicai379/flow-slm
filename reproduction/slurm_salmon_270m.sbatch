#!/usr/bin/env bash
#SBATCH --job-name=flowslm-salmon-270m
#SBATCH --partition=berkeleynlp
#SBATCH --gpus=H200:1
#SBATCH --nodelist=lorax
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=reproduction/logs/salmon-270m-%j.log

set -euo pipefail

repo_dir=/accounts/projects/berkeleynlp/cicicai/flow-slm
python_bin=/scratch/users/cicicai/conda/envs/flow/bin/python
local_root=/data/cicicai/flow_slm/paper_reproduction

cd "$repo_dir"
export HF_HOME=${local_root}/hf_home
export HF_DATASETS_CACHE=${local_root}/datasets
export TRANSFORMERS_CACHE=${local_root}/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export MPLCONFIGDIR=/tmp/flowslm-matplotlib-${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/flowslm-triton-${SLURM_JOB_ID}

srun "$python_bin" -u reproduction/eval_salmon.py \
  --checkpoint ${local_root}/checkpoints/270m.bin \
  --config conf/270m.yaml \
  --batch-size 1
