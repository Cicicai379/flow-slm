#!/usr/bin/env bash
#SBATCH --job-name=flowslm-stage-zs21
#SBATCH --partition=berkeleynlp
#SBATCH --nodes=1
#SBATCH --nodelist=lorax
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=reproduction/logs/stage-zs21-%j.log

set -euo pipefail

eval_root=/data/cicicai/flow_slm/evaluation
archive=${eval_root}/downloads/sLM21.dataset.zip
dataset_root=${eval_root}/zerospeech
expected_md5=713dd4c2c4ab266c846cb2804c8d9e12

mkdir -p "${eval_root}/downloads" "$dataset_root"

if [[ ! -f "$archive" ]] || [[ "$(md5sum "$archive" | cut -d' ' -f1)" != "$expected_md5" ]]; then
  curl -L --fail --retry 5 --continue-at - \
    https://download.zerospeech.com/datasets/sLM21.dataset.zip \
    --output "$archive"
fi

echo "${expected_md5}  ${archive}" | md5sum --check
unzip -n "$archive" -d "$dataset_root"
test -s "$dataset_root/lexical/dev/gold.csv"
test -s "$dataset_root/syntactic/dev/gold.csv"
lexical_dev_count=$(find "$dataset_root/lexical/dev" -maxdepth 1 -type f -name '*.wav' | wc -l)
syntactic_dev_count=$(find "$dataset_root/syntactic/dev" -maxdepth 1 -type f -name '*.wav' | wc -l)
echo "ZeroSpeech staged: lexical_dev_wavs=${lexical_dev_count} syntactic_dev_wavs=${syntactic_dev_count}"
test "$lexical_dev_count" -eq 80000
test "$syntactic_dev_count" -eq 50400
