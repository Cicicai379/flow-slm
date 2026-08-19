#!/usr/bin/env bash
#SBATCH --job-name=flowslm-retarget-lorax
#SBATCH --partition=berkeleynlp
#SBATCH --nodelist=lorax
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=reproduction/logs/retarget-lorax-%j.log

set -euo pipefail

for job_id in 3455965 3455966; do
    state=$(squeue -h -j "$job_id" -o '%T' || true)
    if [[ "$state" == "PENDING" ]]; then
        scontrol update JobId="$job_id" NodeList=lorax
        echo "Retargeted pending job $job_id to lorax"
    else
        echo "Left job $job_id unchanged (state=${state:-not-in-queue})"
    fi
done
