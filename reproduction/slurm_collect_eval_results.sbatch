#!/usr/bin/env bash
#SBATCH --job-name=flowslm-collect-eval
#SBATCH --partition=berkeleynlp
#SBATCH --nodes=1
#SBATCH --nodelist=lorax
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:15:00
#SBATCH --output=reproduction/logs/collect-eval-%j.log

set -euo pipefail

repo_dir=/accounts/projects/berkeleynlp/cicicai/flow-slm
eval_root=/data/cicicai/flow_slm/evaluation
result_dir=${repo_dir}/reproduction/results
mkdir -p "$result_dir"

cp "${eval_root}/results/original-85k-salmon.json" "$result_dir/"
cp "${eval_root}/results/block8-85k-salmon-flow-proxy.json" "$result_dir/"
cp "${eval_root}/zerospeech_results/original/zerospeech-dev-results.json" "$result_dir/original-85k-zerospeech-dev.json"
cp "${eval_root}/zerospeech_results/original/zerospeech-test-results.json" "$result_dir/original-85k-zerospeech-test.json"
cp "${eval_root}/zerospeech_results/block8/zerospeech-dev-results.json" "$result_dir/block8-85k-zerospeech-dev-flow-proxy.json"

sha256sum "$result_dir"/*.json
