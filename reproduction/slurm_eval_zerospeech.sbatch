#!/usr/bin/env bash
#SBATCH --job-name=flowslm-zs21-eval
#SBATCH --partition=berkeleynlp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --nodelist=lorax
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=reproduction/logs/zerospeech-eval-%j.log

set -euo pipefail

repo_dir=/accounts/projects/berkeleynlp/cicicai/flow-slm
python_bin=/scratch/users/cicicai/conda/envs/flow/bin/python
local_root=/data/cicicai/flow_slm/paper_reproduction
eval_root=/data/cicicai/flow_slm/evaluation

cd "$repo_dir"
export HF_HOME=${local_root}/hf_home
export TRANSFORMERS_CACHE=${local_root}/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export MPLCONFIGDIR=/tmp/flowslm-zs21-mpl-${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/flowslm-zs21-triton-${SLURM_JOB_ID}
mkdir -p "$MPLCONFIGDIR" "$TRITON_CACHE_DIR" "${eval_root}/zerospeech_results"

srun "$python_bin" -u reproduction/eval_zerospeech_likelihood.py \
  --kind original \
  --checkpoint "${eval_root}/exports/original-85k/pytorch_model.bin" \
  --config "${local_root}/training/flow-slm-270m-mls/270m/conf.yaml" \
  --dataset-root "${eval_root}/zerospeech" \
  --output-root "${eval_root}/zerospeech_results" --batch-size 8

srun "$python_bin" -u reproduction/eval_zerospeech_likelihood.py \
  --kind block8 \
  --checkpoint "${eval_root}/exports/block8-85k/pytorch_model.bin" \
  --config /data/cicicai/flow_slm/blockwise/runs/270m-block8-mls-v1/270m_block8_mls/conf.yaml \
  --dataset-root "${eval_root}/zerospeech" \
  --output-root "${eval_root}/zerospeech_results" --batch-size 4 --mc-samples 2
