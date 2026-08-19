#!/usr/bin/env bash
#SBATCH --job-name=flowslm-stage-mls-lorax
#SBATCH --partition=berkeleynlp
#SBATCH --nodelist=lorax
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=reproduction/logs/stage-mls-lorax-%j.log

set -euo pipefail

source_cache=/net/thidwick/data/cicicai/flow_slm/paper_reproduction/hf_datasets
local_root=/data/cicicai/flow_slm/paper_reproduction
target_cache=${local_root}/hf_datasets
python_bin=/scratch/users/cicicai/conda/envs/flow/bin/python

mkdir -p "$target_cache"
echo "Starting MLS Arrow cache transfer at $(date --iso-8601=seconds)"
rsync -a --ignore-existing "${source_cache}/" "${target_cache}/"
echo "Transfer finished at $(date --iso-8601=seconds)"
du -sh "$target_cache"

export HF_HOME=/data/cicicai/hf_cache
export HF_DATASETS_CACHE=$target_cache
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

"$python_bin" - <<'PY'
from datasets import load_dataset

expected = {"train": 10_808_037, "dev": 3_807}
for split, expected_rows in expected.items():
    dataset = load_dataset("parler-tts/mls_eng", split=split)
    if len(dataset) != expected_rows:
        raise RuntimeError(
            f"MLS {split}: expected {expected_rows:,} rows, got {len(dataset):,}"
        )
    print(f"MLS {split}: {len(dataset):,} rows", flush=True)
print("Lorax MLS cache verified", flush=True)
PY
