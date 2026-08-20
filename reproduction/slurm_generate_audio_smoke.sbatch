#!/usr/bin/env bash
#SBATCH --job-name=flowslm-audio-smoke
#SBATCH --partition=berkeleynlp
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --nodelist=lorax
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=reproduction/logs/audio-smoke-%j.log

set -euo pipefail

repo_dir=/accounts/projects/berkeleynlp/cicicai/flow-slm
python_bin=/scratch/users/cicicai/conda/envs/flow/bin/python
eval_root=/data/cicicai/flow_slm/evaluation
original_export=${eval_root}/exports/original-85k/pytorch_model.bin
block_export=${eval_root}/exports/block8-85k/pytorch_model.bin
prompt_root=/data/cicicai/LibriSpeech
prompt_csv=prompts/valid_libri_1_sample.csv

cd "$repo_dir"
export HF_HOME=/data/cicicai/hf_cache
export TRANSFORMERS_CACHE=/data/cicicai/flow_slm/paper_reproduction/transformers
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}
export MPLCONFIGDIR=/tmp/flowslm-audio-mpl-${SLURM_JOB_ID}
export TRITON_CACHE_DIR=/tmp/flowslm-audio-triton-${SLURM_JOB_ID}
mkdir -p "$MPLCONFIGDIR" "$TRITON_CACHE_DIR"

test -s "$original_export"
test -s "$block_export"

original_out=${eval_root}/audio/smoke/original
block_out=${eval_root}/audio/smoke/block8
mkdir -p "$original_out" "$block_out"

srun "$python_bin" -u inference.py \
  --ckpt_path "$original_export" \
  --conf_path /data/cicicai/flow_slm/paper_reproduction/training/flow-slm-270m-mls/270m/conf.yaml \
  --prompt_dir "$prompt_root" --prompt_csv "$prompt_csv" \
  --batch_size 1 --samples_per_prompt 1 --min_len 10 --max_len 10 \
  --ode_steps 32 --topp 0.95 --token_temperature 0.8 --temperature 0.8 \
  --cfg_scale 0.3 --solver euler --penalize_silence --penalize_weight 10 \
  --num_quantizers 16 --sr 16000 --seed 20260820 \
  --save_wav --output_dir "$original_out"

srun "$python_bin" -u inference_block.py \
  --ckpt_path "$block_export" \
  --conf_path /data/cicicai/flow_slm/blockwise/runs/270m-block8-mls-v1/270m_block8_mls/conf.yaml \
  --prompt_dir "$prompt_root" --prompt_csv "$prompt_csv" \
  --batch_size 1 --samples_per_prompt 1 --max_len 10 \
  --ode_steps 32 --topp 0.95 --token_temperature 0.8 --temperature 0.8 \
  --cfg_scale 0.3 --solver euler --penalize_silence --penalize_weight 10 \
  --num_quantizers 16 --sr 16000 --seed 20260820 \
  --save_wav --output_dir "$block_out"

find "${eval_root}/audio/smoke" -type f -name '*.wav' -printf '%s %p\n' | sort
