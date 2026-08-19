#!/usr/bin/env bash
#SBATCH --job-name=flowslm-smoke-original
#SBATCH --partition=berkeleynlp
#SBATCH --gpus=1
#SBATCH --nodelist=lorax
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=reproduction/logs/smoke-train-270m-%j.log

set -euo pipefail

repo_dir=/accounts/projects/berkeleynlp/cicicai/flow-slm
python_bin=/scratch/users/cicicai/conda/envs/flow/bin/python
local_root=/data/cicicai/flow_slm/paper_reproduction
smoke_root=${local_root}/smoke/original-${SLURM_JOB_ID}
wandb_root=${local_root}/wandb

cd "$repo_dir"
mkdir -p "$smoke_root" "$wandb_root"

export HF_HOME=/data/cicicai/hf_cache
export HF_DATASETS_CACHE=${local_root}/hf_datasets
export TRANSFORMERS_CACHE=${local_root}/transformers
export WANDB_DIR=$wandb_root
export WANDB_CACHE_DIR=${wandb_root}/cache
export WANDB_CONFIG_DIR=${wandb_root}/config
export WANDB_MODE=online
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export MPLCONFIGDIR=/tmp/flowslm-smoke-original-mpl-${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/flowslm-smoke-original-triton-${SLURM_JOB_ID}
export TORCH_EXTENSIONS_DIR=${local_root}/torch_extensions
mkdir -p "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

"$python_bin" - <<'PY'
import deepspeed
import wandb
assert wandb.login(verify=True, timeout=30)
print(f"DeepSpeed {deepspeed.__version__}; W&B authentication verified", flush=True)
PY

srun "$python_bin" -u trainer.py \
  --conf conf/270m.yaml \
  --save_path "$smoke_root" \
  --data_dir /data/cicicai/LibriSpeech/dev-clean \
  --train_id_file reproduction/smoke_train_ids.txt \
  --valid_id_file reproduction/smoke_valid_ids.txt \
  --override "{'data': {'sr': 16000}, 'training': {'batch_size': 1, 'num_workers': 1, 'accumulate_grad_batches': 1, 'num_warmup_steps': 1, 'max_steps': 3}}" \
  --strategy deepspeed_stage_3 \
  --every_n_steps 100 \
  --wandb \
  --wandb_project flow-slm-smoke \
  --wandb_entity cicicai-university-of-california-berkeley \
  --wandb_run_name "Flow-SLM-original-smoke-${SLURM_JOB_ID}" \
  --wandb_run_id "flowslm-original-smoke-${SLURM_JOB_ID}" \
  --wandb_dir "$wandb_root"
