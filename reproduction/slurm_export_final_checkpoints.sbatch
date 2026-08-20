#!/usr/bin/env bash
#SBATCH --job-name=flowslm-export-final
#SBATCH --partition=berkeleynlp
#SBATCH --nodes=1
#SBATCH --nodelist=lorax
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=reproduction/logs/export-final-%j.log

set -euo pipefail

python_bin=/scratch/users/cicicai/conda/envs/flow/bin/python
eval_root=/data/cicicai/flow_slm/evaluation
original_ckpt=/data/cicicai/flow_slm/paper_reproduction/training/flow-slm-270m-mls/270m/model-step=0085000.ckpt
block_ckpt=/data/cicicai/flow_slm/blockwise/runs/270m-block8-mls-v1/270m_block8_mls/model-step=0085000.ckpt

# DeepSpeed probes the CUDA toolchain while importing its CPU-only ZeRO
# consolidation utility, so expose the node's CUDA installation here too.
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export TRITON_CACHE_DIR=/data/cicicai/flow_slm/evaluation/cache/triton
mkdir -p "$TRITON_CACHE_DIR"

mkdir -p "${eval_root}/exports/original-85k" "${eval_root}/exports/block8-85k"

"$python_bin" "${original_ckpt}/zero_to_fp32.py" \
  "$original_ckpt" "${eval_root}/exports/original-85k" --max_shard_size 10GB

"$python_bin" "${block_ckpt}/zero_to_fp32.py" \
  "$block_ckpt" "${eval_root}/exports/block8-85k" --max_shard_size 10GB

find "${eval_root}/exports" -maxdepth 2 -type f -printf '%s %p\n' | sort
