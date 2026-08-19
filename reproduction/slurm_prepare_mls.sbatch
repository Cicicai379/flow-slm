#!/usr/bin/env bash
#SBATCH --job-name=flowslm-prepare-mls
#SBATCH --partition=berkeleynlp
#SBATCH --nodelist=thidwick
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=7-00:00:00
#SBATCH --output=reproduction/logs/prepare-mls-%j.log

set -euo pipefail

repo_dir=/accounts/projects/berkeleynlp/cicicai/flow-slm
python_bin=/scratch/users/cicicai/conda/envs/flow/bin/python
local_root=/data/cicicai/flow_slm/paper_reproduction

cd "$repo_dir"
mkdir -p \
  /data/cicicai/hf_cache \
  "${local_root}/hf_datasets" \
  "${local_root}/transformers"

# Reuse the already-downloaded 196 GB from horton, then let Hugging Face resume
# missing blobs into thidwick-local storage. This job requests no GPUs.
rsync -a --ignore-existing \
  /net/horton/data/cicicai/hf_cache/ \
  /data/cicicai/hf_cache/
rsync -a --ignore-existing \
  /net/horton/data/cicicai/flow_slm/paper_reproduction/transformers/ \
  "${local_root}/transformers/"

export HF_HOME=/data/cicicai/hf_cache
export HF_DATASETS_CACHE=${local_root}/hf_datasets
export TRANSFORMERS_CACHE=${local_root}/transformers
export HF_HUB_DISABLE_TELEMETRY=1

"$python_bin" -u - <<'PY'
from datasets import load_dataset

for split in ("train", "dev"):
    dataset = load_dataset("parler-tts/mls_eng", split=split)
    if not dataset:
        raise RuntimeError(f"MLS-English {split} split is empty")
    print(f"MLS-English {split}: {len(dataset):,} rows", flush=True)

print("MLS-English staging complete", flush=True)
PY
