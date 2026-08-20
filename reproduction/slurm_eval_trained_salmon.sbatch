#!/usr/bin/env bash
#SBATCH --job-name=flowslm-trained-salmon
#SBATCH --partition=berkeleynlp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --nodelist=lorax
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --output=reproduction/logs/trained-salmon-%j.log

set -euo pipefail

repo_dir=/accounts/projects/berkeleynlp/cicicai/flow-slm
python_bin=/scratch/users/cicicai/conda/envs/flow/bin/python
local_root=/data/cicicai/flow_slm/paper_reproduction
eval_root=/data/cicicai/flow_slm/evaluation

cd "$repo_dir"
export HF_HOME=${local_root}/hf_home
export HF_DATASETS_CACHE=${local_root}/datasets
export TRANSFORMERS_CACHE=${local_root}/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export MPLCONFIGDIR=/tmp/flowslm-salmon-mpl-${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/flowslm-salmon-triton-${SLURM_JOB_ID}
mkdir -p "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "${eval_root}/results"

srun "$python_bin" -u reproduction/eval_trained_salmon.py \
  --kind original \
  --checkpoint "${eval_root}/exports/original-85k/pytorch_model.bin" \
  --config "${local_root}/training/flow-slm-270m-mls/270m/conf.yaml" \
  --batch-size 1 \
  --output "${eval_root}/results/original-85k-salmon.json"

srun "$python_bin" -u reproduction/eval_trained_salmon.py \
  --kind block8 \
  --checkpoint "${eval_root}/exports/block8-85k/pytorch_model.bin" \
  --config /data/cicicai/flow_slm/blockwise/runs/270m-block8-mls-v1/270m_block8_mls/conf.yaml \
  --batch-size 1 --mc-samples 4 \
  --output "${eval_root}/results/block8-85k-salmon-flow-proxy.json"
