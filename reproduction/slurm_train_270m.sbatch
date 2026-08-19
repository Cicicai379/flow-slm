#!/usr/bin/env bash
#SBATCH --job-name=flowslm-paper-270m
#SBATCH --partition=berkeleynlp
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --nodelist=lorax
#SBATCH --cpus-per-task=24
#SBATCH --mem=384G
#SBATCH --time=28-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:HUP@120
#SBATCH --output=reproduction/logs/train-270m-%j.log

set -euo pipefail

repo_dir=/accounts/projects/berkeleynlp/cicicai/flow-slm
python_bin=/scratch/users/cicicai/conda/envs/flow/bin/python
local_root=/data/cicicai/flow_slm/paper_reproduction
run_root=${local_root}/training/flow-slm-270m-mls
wandb_root=${local_root}/wandb

cd "$repo_dir"
mkdir -p "$run_root" "$wandb_root" "${local_root}/hf_datasets"

# All high-volume data, caches, checkpoints, and W&B buffers stay on Lorax NVMe.
export HF_HOME=/data/cicicai/hf_cache
export HF_DATASETS_CACHE=${local_root}/hf_datasets
export TRANSFORMERS_CACHE=${local_root}/transformers
export WANDB_DIR=${wandb_root}
export WANDB_CACHE_DIR=${wandb_root}/cache
export WANDB_CONFIG_DIR=${wandb_root}/config
export WANDB_MODE=online
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export MPLCONFIGDIR=/tmp/flowslm-matplotlib-${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/flowslm-triton-${SLURM_JOB_ID}
export TORCH_EXTENSIONS_DIR=${local_root}/torch_extensions
mkdir -p "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

# Refuse to spend GPU time if the run cannot be uploaded to W&B.
"$python_bin" - <<'PY'
import wandb
if not wandb.login(verify=True, timeout=30):
    raise SystemExit("W&B authentication failed")
print("W&B authentication verified", flush=True)
PY

run_id=flowslm-270m-mls-paper
run_name=Flow-SLM-270M-MLS-paper-reproduction

srun "$python_bin" -u trainer.py \
  --conf conf/270m.yaml \
  --save_path "$run_root" \
  --hf_training_data \
  --training_data MLSEn \
  --override "{'data': {'sr': 16000}}" \
  --strategy deepspeed_stage_3 \
  --wandb \
  --wandb_project flow-slm-paper-reproduction \
  --wandb_entity cicicai-university-of-california-berkeley \
  --wandb_run_name "$run_name" \
  --wandb_run_id "$run_id" \
  --wandb_dir "$wandb_root"
