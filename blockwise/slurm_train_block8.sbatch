#!/usr/bin/env bash
#SBATCH --job-name=flowslm-block8-270m
#SBATCH --partition=berkeleynlp
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --nodelist=lorax
#SBATCH --cpus-per-task=24
#SBATCH --mem=384G
#SBATCH --time=28-00:00:00
#SBATCH --requeue
#SBATCH --signal=B:HUP@120
#SBATCH --output=blockwise/logs/train-block8-%j.log

set -euo pipefail

repo_dir=/accounts/projects/berkeleynlp/cicicai/flow-slm
python_bin=/scratch/users/cicicai/conda/envs/flow/bin/python
paper_root=/data/cicicai/flow_slm/paper_reproduction
run_root=/data/cicicai/flow_slm/blockwise/runs/270m-block8-mls-v1
wandb_root=/data/cicicai/flow_slm/blockwise/wandb

cd "$repo_dir"
mkdir -p "$run_root" "$wandb_root"

export HF_HOME=/data/cicicai/hf_cache
export HF_DATASETS_CACHE=${paper_root}/hf_datasets
export TRANSFORMERS_CACHE=${paper_root}/transformers
export WANDB_DIR=${wandb_root}
export WANDB_CACHE_DIR=${wandb_root}/cache
export WANDB_CONFIG_DIR=${wandb_root}/config
export WANDB_MODE=online
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export MPLCONFIGDIR=/tmp/flowslm-block8-matplotlib-${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/flowslm-block8-triton-${SLURM_JOB_ID}
export TORCH_EXTENSIONS_DIR=/data/cicicai/flow_slm/blockwise/torch_extensions
mkdir -p "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

"$python_bin" - <<'PY'
import wandb
if not wandb.login(verify=True, timeout=30):
    raise SystemExit("W&B authentication failed")
print("W&B authentication verified", flush=True)
PY

srun "$python_bin" -u trainer_block.py \
  --conf blockwise/configs/270m_block8_mls.yaml \
  --save_path "$run_root" \
  --hf_training_data \
  --training_data MLSEn \
  --override "{'training': {'accumulate_grad_batches': 16}}" \
  --strategy deepspeed_stage_2 \
  --every_n_steps 5000 \
  --wandb \
  --wandb_project flow-slm-blockwise \
  --wandb_entity cicicai-university-of-california-berkeley \
  --wandb_run_name Flow-SLM-270M-Block8-MLS-v1 \
  --wandb_run_id flowslm-270m-block8-mls-v1 \
  --wandb_dir "$wandb_root"
