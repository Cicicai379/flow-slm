#!/usr/bin/env bash
#SBATCH --job-name=flowslm-audio-validate
#SBATCH --partition=berkeleynlp
#SBATCH --nodes=1
#SBATCH --nodelist=lorax
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=reproduction/logs/audio-validate-%j.log

set -euo pipefail

cd /accounts/projects/berkeleynlp/cicicai/flow-slm
/scratch/users/cicicai/conda/envs/flow/bin/python -u reproduction/validate_audio_outputs.py \
  --input-root /data/cicicai/flow_slm/evaluation/audio/smoke \
  --output-root reproduction/audio_continuations
